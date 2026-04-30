"""
Azure AI Language SDK tool implementations.

Used when LANGUAGE_TOOL_MODE=sdk (the default).
The agent declares these as function tools; the app intercepts required_action
runs, executes the appropriate function here, and submits the output back.

Switch to LANGUAGE_TOOL_MODE=mcp once the Foundry Agent Service supports
Streamable HTTP MCP transport (currently POST-only, Agent Service uses SSE GET).
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.ai.textanalytics import TextAnalyticsClient


# Pending file-uploaded documents are stored in-process so the agent can call
# analyze_sentiment without embedding hundreds of rows into the LLM context.
_pending_documents: list[str] | None = None


def set_pending_documents(documents: list[str]) -> None:
    """Store uploaded documents for the next tool call."""
    global _pending_documents
    _pending_documents = [str(d) for d in documents]


def _take_pending_documents() -> list[str] | None:
    """Return and clear pending uploaded documents."""
    global _pending_documents
    docs = _pending_documents
    _pending_documents = None
    return docs


def store_documents(documents=None) -> str:
    """Append documents to the pending store for later batch analysis.

    Same ``_pending_documents`` buffer used by the file-upload path.
    Call once or many times to accumulate rows, then call
    ``analyze_sentiment`` with NO arguments to process everything.
    """
    global _pending_documents
    if documents is None:
        return json.dumps({"stored": 0, "total": len(_pending_documents or [])})
    docs = _docs(documents)
    if _pending_documents is None:
        _pending_documents = docs
    else:
        _pending_documents.extend(docs)
    return json.dumps({"stored": len(docs), "total": len(_pending_documents)})


# ─── Client ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_client() -> TextAnalyticsClient:
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"].rstrip("/")
    return TextAnalyticsClient(endpoint=endpoint, credential=DefaultAzureCredential())


def _docs(documents: Any) -> list[str]:
    """Accept either a JSON string or a Python list."""
    if isinstance(documents, str):
        try:
            documents = json.loads(documents)
        except json.JSONDecodeError:
            documents = [documents]
    return [str(d) for d in documents]


def _chunked(lst: list[str], size: int = 10):
    """Yield successive chunks of `size` from `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _canonical_section2_theme(target: str) -> str | None:
    """Map noisy opinion targets into stable Section 2 theme clusters."""
    t = re.sub(r"\s+", " ", target.strip().lower())
    if not t:
        return None

    if any(k in t for k in ["billing", "payment", "invoice", "account", "subscription"]):
        return "Billing & Account"
    if any(k in t for k in ["knowledge base", "kb", "documentation", "article", "search", "manual"]):
        return "Documentation & Knowledge Base"
    if any(k in t for k in ["ui", "ux", "mobile", "app", "interface", "tool", "feature", "functionality"]):
        return "Product Experience"
    if any(k in t for k in ["response", "follow-up", "follow up", "communication", "resolution", "escalation", "ticket", "wait", "support", "engineer", "service"]):
        return "Support Operations"

    # Ignore one-word generic noise tokens to avoid misleading tiny rows.
    if len(t.split()) == 1 and t in {"issue", "problem", "team", "customer", "system"}:
        return None

    return "Other"


def _pretty_label(text: str) -> str:
    """Convert raw target text to a readable label for report tables."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned.title()


# ─── Tool implementations ─────────────────────────────────────────────────────

def analyze_sentiment(documents: Any = None) -> str:
    """Analyse sentiment and mine opinions for each document.

    Accepts explicit `documents` or uses pending uploaded documents when called
    with no arguments.
    """
    client = _get_client()
    docs = _docs(documents) if documents else (_take_pending_documents() or [])
    if not docs:
        return json.dumps({"error": "No documents provided."}, ensure_ascii=False)

    # Small payload: keep full per-document detail for richer chat responses.
    if len(docs) <= 10:
        output = []
        results = client.analyze_sentiment(docs, show_opinion_mining=True)
        for i, doc in enumerate(results):
            if doc.is_error:
                output.append({"index": i, "error": doc.error.message})
                continue
            sentences = []
            for sent in doc.sentences:
                s: dict[str, Any] = {
                    "text": sent.text,
                    "sentiment": sent.sentiment,
                    "confidence": {
                        "positive": round(sent.confidence_scores.positive, 3),
                        "neutral": round(sent.confidence_scores.neutral, 3),
                        "negative": round(sent.confidence_scores.negative, 3),
                    },
                    "opinions": [],
                }
                for mined in sent.mined_opinions:
                    s["opinions"].append({
                        "target": mined.target.text,
                        "sentiment": mined.target.sentiment,
                        "assessments": [
                            {"text": a.text, "sentiment": a.sentiment}
                            for a in mined.assessments
                        ],
                    })
                sentences.append(s)
            output.append({
                "index": i,
                "text": docs[i],
                "sentiment": doc.sentiment,
                "confidence": {
                    "positive": round(doc.confidence_scores.positive, 3),
                    "neutral": round(doc.confidence_scores.neutral, 3),
                    "negative": round(doc.confidence_scores.negative, 3),
                },
                "sentences": sentences,
            })
        return json.dumps(output, ensure_ascii=False)

    # Large payload: process all documents in 10-doc chunks and return compact
    # aggregate metrics so the LLM can reason over the full dataset.
    sentiments: list[str] = []
    errors = 0
    opinion_targets: dict[str, dict[str, int]] = {}
    opinion_assessments: dict[str, dict[str, int]] = {}
    # Theme -> document sentiment counts for responses mentioning the theme.
    theme_doc_sentiments: dict[str, dict[str, int]] = {}
    # Canonical Section 2 clusters aggregated across noisy targets.
    section2_cluster_mentions: dict[str, int] = {}
    section2_cluster_doc_sentiments: dict[str, dict[str, int]] = {}
    negative_samples: list[str] = []
    positive_samples: list[str] = []

    idx = 0
    for chunk in _chunked(docs, 10):
        results = client.analyze_sentiment(chunk, show_opinion_mining=True)
        for doc in results:
            if doc.is_error:
                errors += 1
                idx += 1
                continue

            sentiments.append(doc.sentiment)
            if doc.sentiment == "negative" and len(negative_samples) < 10:
                negative_samples.append(docs[idx][:200])
            elif doc.sentiment == "positive" and len(positive_samples) < 10:
                positive_samples.append(docs[idx][:200])

            doc_targets: set[str] = set()
            doc_clusters: set[str] = set()
            for sent in doc.sentences:
                for mined in sent.mined_opinions:
                    target = mined.target.text.lower()
                    t_sent = mined.target.sentiment
                    doc_targets.add(target)
                    opinion_targets.setdefault(target, {"positive": 0, "neutral": 0, "negative": 0})
                    opinion_targets[target][t_sent] = opinion_targets[target].get(t_sent, 0) + 1
                    opinion_assessments.setdefault(target, {})
                    for a in mined.assessments:
                        key = a.text.lower()
                        opinion_assessments[target][key] = opinion_assessments[target].get(key, 0) + 1

                    cluster = _canonical_section2_theme(target)
                    if cluster:
                        section2_cluster_mentions[cluster] = section2_cluster_mentions.get(cluster, 0) + 1
                        doc_clusters.add(cluster)

            # Distribute document-level sentiment to each theme mentioned by this document.
            for target in doc_targets:
                theme_doc_sentiments.setdefault(target, {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0})
                if doc.sentiment in theme_doc_sentiments[target]:
                    theme_doc_sentiments[target][doc.sentiment] += 1

            # Distribute document sentiment once per canonical cluster.
            for cluster in doc_clusters:
                section2_cluster_doc_sentiments.setdefault(cluster, {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0})
                if doc.sentiment in section2_cluster_doc_sentiments[cluster]:
                    section2_cluster_doc_sentiments[cluster][doc.sentiment] += 1
            idx += 1

    total = len(sentiments)
    pos = sentiments.count("positive")
    neu = sentiments.count("neutral")
    neg = sentiments.count("negative")
    mixed = sentiments.count("mixed")

    # Keep meaningful themes while avoiding collapse to a single row.
    # Threshold is adaptive (~0.5% of responses, minimum 3 mentions).
    min_theme_mentions = max(3, int(round(total * 0.005))) if total else 3
    sorted_targets = sorted(opinion_targets.items(), key=lambda x: sum(x[1].values()), reverse=True)
    filtered_targets = [(t, c) for t, c in sorted_targets if sum(c.values()) >= min_theme_mentions]
    # If filter is too strict for this dataset, fall back to top themes.
    if len(filtered_targets) < 8:
        filtered_targets = sorted_targets[:12]

    themes = []
    for target, counts in filtered_targets:
        t_total = sum(counts.values())
        doc_counts = theme_doc_sentiments.get(target, {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0})
        t_doc_total = sum(doc_counts.values())
        pos_pct = round((doc_counts.get("positive", 0) / t_doc_total) * 100, 1) if t_doc_total else 0
        neu_pct = round((doc_counts.get("neutral", 0) / t_doc_total) * 100, 1) if t_doc_total else 0
        neg_pct = round((doc_counts.get("negative", 0) / t_doc_total) * 100, 1) if t_doc_total else 0
        top_assessments = sorted(opinion_assessments.get(target, {}).items(), key=lambda x: x[1], reverse=True)[:5]
        themes.append({
            "target": target,
            "mentions": t_total,
            "responses_with_theme": t_doc_total,
            "positive": doc_counts.get("positive", 0),
            "neutral": doc_counts.get("neutral", 0),
            "negative": doc_counts.get("negative", 0),
            "mixed": doc_counts.get("mixed", 0),
            "positive_pct_within_theme": pos_pct,
            "neutral_pct_within_theme": neu_pct,
            "negative_pct_within_theme": neg_pct,
            "mixed_pct_within_theme": round((doc_counts.get("mixed", 0) / t_doc_total) * 100, 1) if t_doc_total else 0,
            "share_of_all_responses_pct": round((t_doc_total / total) * 100, 1) if total else 0,
            "positive_share_of_all_responses_pct": round((doc_counts.get("positive", 0) / total) * 100, 1) if total else 0,
            "neutral_share_of_all_responses_pct": round((doc_counts.get("neutral", 0) / total) * 100, 1) if total else 0,
            "negative_share_of_all_responses_pct": round((doc_counts.get("negative", 0) / total) * 100, 1) if total else 0,
            "top_assessments": [f"{txt} ({cnt})" for txt, cnt in top_assessments],
        })
        if len(themes) >= 20:
            break

    # Build Section 2 from canonical clusters (not raw target fragments).
    section2_rows = []
    for cluster, doc_counts in section2_cluster_doc_sentiments.items():
        c_doc_total = sum(doc_counts.values())
        if c_doc_total == 0:
            continue
        pos_pct = round((doc_counts.get("positive", 0) / c_doc_total) * 100, 1)
        neu_pct = round((doc_counts.get("neutral", 0) / c_doc_total) * 100, 1)
        neg_pct = round((doc_counts.get("negative", 0) / c_doc_total) * 100, 1)
        pos_count = doc_counts.get("positive", 0)
        neu_count = doc_counts.get("neutral", 0)
        neg_count = doc_counts.get("negative", 0)
        section2_rows.append({
            "theme": cluster,
            "mentions": section2_cluster_mentions.get(cluster, 0),
            "responses_with_theme": c_doc_total,
            "positive_count": pos_count,
            "neutral_count": neu_count,
            "negative_count": neg_count,
            "positive_pct": pos_pct,
            "neutral_pct": neu_pct,
            "negative_pct": neg_pct,
            "positive_display": f"{pos_count} ({pos_pct:.1f}%)",
            "neutral_display": f"{neu_count} ({neu_pct:.1f}%)",
            "negative_display": f"{neg_count} ({neg_pct:.1f}%)",
        })

    section2_rows = sorted(section2_rows, key=lambda r: r["responses_with_theme"], reverse=True)

    # Section 2 must be a single table with no high/low-volume split.
    min_section2_responses = 0
    section2_primary_rows = section2_rows
    section2_low_volume_rows: list[dict[str, Any]] = []

    # Build deterministic Section 3/4 tables from opinion-target sentiment counts.
    total_negative_target_mentions = sum(c.get("negative", 0) for c in opinion_targets.values())
    total_positive_target_mentions = sum(c.get("positive", 0) for c in opinion_targets.values())

    section3_negative_drivers = []
    for target, counts in sorted(opinion_targets.items(), key=lambda x: x[1].get("negative", 0), reverse=True):
        neg_count = counts.get("negative", 0)
        if neg_count <= 0:
            continue
        section3_negative_drivers.append({
            "issue_cluster": _pretty_label(target),
            "primary_theme": _canonical_section2_theme(target) or "Other",
            "mentions": neg_count,
            "share_of_negative_sentiment_pct": round((neg_count / total_negative_target_mentions) * 100, 1)
            if total_negative_target_mentions
            else 0,
        })
        if len(section3_negative_drivers) >= 5:
            break

    section4_positive_drivers = []
    for target, counts in sorted(opinion_targets.items(), key=lambda x: x[1].get("positive", 0), reverse=True):
        pos_count = counts.get("positive", 0)
        if pos_count <= 0:
            continue
        section4_positive_drivers.append({
            "positive_driver": _pretty_label(target),
            "mentions": pos_count,
            "share_of_positive_impact_pct": round((pos_count / total_positive_target_mentions) * 100, 1)
            if total_positive_target_mentions
            else 0,
        })
        if len(section4_positive_drivers) >= 5:
            break

    summary = {
        "total_documents": total,
        "errors": errors,
        "sentiment_distribution": {
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "mixed": mixed,
            "positive_pct": round(pos / total * 100, 1) if total else 0,
            "neutral_pct": round(neu / total * 100, 1) if total else 0,
            "negative_pct": round(neg / total * 100, 1) if total else 0,
        },
        "opinion_themes": themes,
        # Deterministic table payload for Section 2 rendering.
        "section_2_table_rows": [
            {
                "theme": t["theme"],
                "mentions": t["mentions"],
                "responses_with_theme": t["responses_with_theme"],
                "positive_count": t["positive_count"],
                "neutral_count": t["neutral_count"],
                "negative_count": t["negative_count"],
                "positive_pct": t["positive_pct"],
                "neutral_pct": t["neutral_pct"],
                "negative_pct": t["negative_pct"],
                "positive_display": t["positive_display"],
                "neutral_display": t["neutral_display"],
                "negative_display": t["negative_display"],
                "low_volume": False,
            }
            for t in section2_primary_rows
        ],
        # Kept for transparency so all raw rows remain available.
        "section_2_low_volume_rows": [
            {
                "theme": t["theme"],
                "mentions": t["mentions"],
                "responses_with_theme": t["responses_with_theme"],
                "positive_count": t["positive_count"],
                "neutral_count": t["neutral_count"],
                "negative_count": t["negative_count"],
                "positive_pct": t["positive_pct"],
                "neutral_pct": t["neutral_pct"],
                "negative_pct": t["negative_pct"],
                "positive_display": t["positive_display"],
                "neutral_display": t["neutral_display"],
                "negative_display": t["negative_display"],
                "low_volume": True,
            }
            for t in section2_low_volume_rows
        ],
        "section_2_min_responses_threshold": min_section2_responses,
        "section_2_reliable_row_count": len(section2_primary_rows),
        "section_2_low_volume_row_count": len(section2_low_volume_rows),
        "section_3_negative_drivers": section3_negative_drivers,
        "section_3_negative_denominator_mentions": total_negative_target_mentions,
        "section_4_positive_drivers": section4_positive_drivers,
        "section_4_positive_denominator_mentions": total_positive_target_mentions,
    }
    return json.dumps(summary, ensure_ascii=False)


def extract_key_phrases(documents: Any) -> str:
    """Extract the key phrases from each document."""
    client = _get_client()
    docs = _docs(documents)
    results = client.extract_key_phrases(docs)
    output = []
    for i, doc in enumerate(results):
        if doc.is_error:
            output.append({"index": i, "error": doc.error.message})
        else:
            output.append({"index": i, "text": docs[i], "key_phrases": list(doc.key_phrases)})
    return json.dumps(output, ensure_ascii=False)


def recognize_entities(documents: Any) -> str:
    """Recognise named entities (people, places, organisations, dates, …)."""
    client = _get_client()
    docs = _docs(documents)
    results = client.recognize_entities(docs)
    output = []
    for i, doc in enumerate(results):
        if doc.is_error:
            output.append({"index": i, "error": doc.error.message})
        else:
            output.append({
                "index": i,
                "text": docs[i],
                "entities": [
                    {
                        "text": e.text,
                        "category": e.category,
                        "subcategory": e.subcategory,
                        "confidence": round(e.confidence_score, 3),
                    }
                    for e in doc.entities
                ],
            })
    return json.dumps(output, ensure_ascii=False)


def detect_language(documents: Any) -> str:
    """Detect the language of each document."""
    client = _get_client()
    docs = _docs(documents)
    results = client.detect_language(docs)
    output = []
    for i, doc in enumerate(results):
        if doc.is_error:
            output.append({"index": i, "error": doc.error.message})
        else:
            output.append({
                "index": i,
                "text": docs[i],
                "language": doc.primary_language.name,
                "iso6391_name": doc.primary_language.iso6391_name,
                "confidence": round(doc.primary_language.confidence_score, 3),
            })
    return json.dumps(output, ensure_ascii=False)


def recognize_pii_entities(documents: Any) -> str:
    """Detect PII (names, emails, phone numbers, etc.) in each document."""
    client = _get_client()
    docs = _docs(documents)
    results = client.recognize_pii_entities(docs)
    output = []
    for i, doc in enumerate(results):
        if doc.is_error:
            output.append({"index": i, "error": doc.error.message})
        else:
            output.append({
                "index": i,
                "text": docs[i],
                "redacted_text": doc.redacted_text,
                "entities": [
                    {
                        "text": e.text,
                        "category": e.category,
                        "confidence": round(e.confidence_score, 3),
                    }
                    for e in doc.entities
                ],
            })
    return json.dumps(output, ensure_ascii=False)


# ─── Dispatch table ───────────────────────────────────────────────────────────

TOOL_DISPATCH: dict[str, Any] = {
    "analyze_sentiment": analyze_sentiment,
    "extract_key_phrases": extract_key_phrases,
    "recognize_entities": recognize_entities,
    "detect_language": detect_language,
    "recognize_pii_entities": recognize_pii_entities,
    "store_documents": store_documents,
}

# ─── Function tool definitions (OpenAI function-calling schema) ───────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": (
                "Analyse the sentiment (positive / neutral / negative) of one or more text "
                "documents and mine fine-grained opinions about specific aspects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents to analyse. Any size is accepted; internal batching is handled automatically.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_key_phrases",
            "description": "Extract the most important key phrases from one or more documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents (max 10 per call).",
                    }
                },
                "required": ["documents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recognize_entities",
            "description": (
                "Recognise named entities such as people, organisations, locations, dates, "
                "events, and products in one or more documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents (max 10 per call).",
                    }
                },
                "required": ["documents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_language",
            "description": "Detect the language of one or more documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents (max 10 per call).",
                    }
                },
                "required": ["documents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recognize_pii_entities",
            "description": (
                "Detect Personally Identifiable Information (PII) such as names, email "
                "addresses, phone numbers and ID numbers. Returns detected entities and "
                "a redacted version of the text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents (max 10 per call).",
                    }
                },
                "required": ["documents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_documents",
            "description": (
                "Store text documents for later batch analysis. Call one or more times "
                "to accumulate rows, then call analyze_sentiment with NO arguments to "
                "process the entire stored dataset. Use this when data is retrieved from Fabric."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents to store. Multiple calls accumulate.",
                    }
                },
                "required": ["documents"],
            },
        },
    },
]
