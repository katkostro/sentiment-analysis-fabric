"""
Create and register the Sentiment Analysis agent in Azure AI Foundry.

Supports two tool modes controlled by LANGUAGE_TOOL_MODE in .env:

  sdk  (default) — Agent calls Azure AI Language via function tools that the
                   app intercepts and executes locally using the Python SDK.
                   Works today with managed identity. Switch to this when
                   the MCP transport issue is unresolved.

  mcp            — Agent calls Language via the MCP server (requires the
                   Foundry Agent Service to support Streamable HTTP transport,
                   currently in preview and using SSE which is incompatible
                   with the Language MCP server's POST-only endpoint).

Run after deploying infrastructure:
  azd provision --environment sentiment-analysis-mcp

Then:
  python src/create_agent.py
"""

from __future__ import annotations

import os
import json
import sys

# Load .env from project root
# .env values OVERRIDE existing env vars to avoid stale terminal sessions
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip().strip('"')

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import McpTool, ToolSet, MCPToolResource, FunctionTool, FabricTool

# ─── Configuration ───────────────────────────────────────────────────────────

AI_SERVICES_ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
LANGUAGE_ENDPOINT = os.environ["AZURE_LANGUAGE_ENDPOINT"]
FOUNDRY_PROJECT = os.environ.get("FOUNDRY_PROJECT_NAME", "sentiment-analysis")
GPT_DEPLOYMENT = os.environ.get("GPT_DEPLOYMENT_NAME", "gpt-4o")

# sdk  → function tools executed locally via Language SDK (works today)
# mcp  → Language MCP server (requires Agent Service Streamable HTTP support)
TOOL_MODE = os.environ.get("LANGUAGE_TOOL_MODE", "sdk").lower()

LANGUAGE_MCP_URL = (
    f"{AI_SERVICES_ENDPOINT.rstrip('/')}/language/mcp?api-version=2025-11-15-preview"
)

# Fabric Data Agent (built-in tool) — requires a project connection
FABRIC_CONNECTION_NAME = os.environ.get("FABRIC_CONNECTION_NAME", "")

# ─── System prompts ───────────────────────────────────────────────────────────

_COMMON_INSTRUCTIONS = """
Data sources:
- Survey responses pre-extracted from Excel/CSV files by the UI (local upload)
- Survey data from Microsoft Fabric semantic models (via the Fabric data agent tool)

When you receive survey responses (from file upload):
- You will receive a numbered list of text responses already extracted from the file
- The user has already selected the correct column(s) and prepared the data
- If analyzing multiple columns, responses will be labeled like "[ColumnName] response text"
- Call the Language analysis tools on the responses (batch up to 10 at a time)
- DO NOT attempt to read files yourself - the responses are already provided as text
- Provide a comprehensive summary of the analysis results

When the user asks to query Fabric data:
- Use the Microsoft Fabric tool to retrieve survey data from the semantic model
- Then analyze the retrieved responses using the Language analysis tools
- Present results in the same structured format as file uploads

When analyzing multiple columns:
- Consider each column separately when identifying patterns
- Note any differences in sentiment or themes between columns
- Highlight if certain columns have more negative feedback

When presenting results, follow this exact structure:

1. **Customer Sentiment Overview** - Executive summary with key insight (e.g., "Dissatisfaction is concentrated, not widespread")

2. **Where Sentiment Breaks Down** - Table format:
    Theme | 🟢 Positive | 🟡 Neutral | 🔴 Negative
    If `section_2_table_rows` exists in the tool output, use it directly as the source of truth for this table.
    Render exactly one Section 2 table only.
    Do not create separate "High-Volume" or "Low-Volume" subsections.
    Do not smooth, normalize, or adjust values.
    Do not recompute Section 2 percentages yourself.
    Calculate percentages for each theme across sentiment categories using: positive_pct, neutral_pct, negative_pct.
    If fields `positive_display`, `neutral_display`, and `negative_display` exist, use them verbatim for the three sentiment cells.
    Do not change punctuation in those display fields.
    Only if display fields are missing, render each sentiment cell as: <count> (<percentage>%) using positive_count, neutral_count, negative_count.
    Do not include Mentions or Responses as table columns in Section 2.
    If needed, include counts context in one short note below the table.
    Use one decimal place exactly for every percentage.
    If `section_2_reliable_row_count` is 0, render an empty table with the required header and no data rows.
   Themes: Technical Support Quality, Communication, Tools, Documentation, Product Features, etc.
    Do not add extra commentary paragraphs after Section 2.

3. **Key Drivers of Negative Sentiment** - Table format:
   Issue Cluster | Primary Theme | Mentions | Share of Negative Sentiment
    If `section_3_negative_drivers` exists and is non-empty, use it directly as the source of truth.
    Do not recompute or reinterpret shares.
    Map fields exactly: issue_cluster, primary_theme, mentions, share_of_negative_sentiment_pct.
    If `section_3_negative_denominator_mentions` exists, add one sentence after the table:
    "Share of Negative Sentiment is calculated over <denominator> negative opinion mentions.".
    Use one decimal place exactly for percentages.
    Do not include sample responses in Section 3 unless the user explicitly asks for examples.
    Do not add extra commentary paragraphs after Section 3.

4. **Key Drivers of Positive Sentiment** - Table format:
    Positive Driver | Mentions | Share of Positive Impact
    If `section_4_positive_drivers` exists and is non-empty, use it directly as the source of truth.
    Do not recompute or reinterpret shares.
    Map fields exactly: positive_driver, mentions, share_of_positive_impact_pct.
    If `section_4_positive_denominator_mentions` exists, add one sentence after the table:
    "Share of Positive Impact is calculated over <denominator> positive opinion mentions.".
    Use one decimal place exactly for percentages.
    Do not include sample responses in Section 4 unless the user explicitly asks for examples.
    Do not add extra commentary paragraphs after Section 4.

5. **Insight-Driven Recommendations** - Numbered list with:
   - Title
   - Why: (reasoning)
   - Recommendation: (specific action)
   Prioritize by potential sentiment impact

- Use markdown tables for data presentation
- Include percentages and counts
- Provide narrative insights between sections
- Flag any PII found and suggest redaction
- Never state a percentage that is not present in tool output.
- If discussing neutral sentiment, explicitly label whether it is overall sentiment_distribution or a specific section table.
- In narrative text, never introduce new rounded percentages (for example, 60% instead of 57.6%). If citing a percentage, copy the exact one-decimal value from the table.
- For Sections 2, 3, and 4, output only the required tables and required denominator notes; no additional prose.
"""

AGENT_SYSTEM_PROMPT_SDK = (
    """You are a Survey Analysis Agent powered by Azure AI Language (SDK function tools).

You have access to these language analysis functions:
- analyze_sentiment      — sentiment (positive/neutral/negative) + opinion mining
- extract_key_phrases    — identify main topics and themes
- recognize_entities     — find people, places, organisations, dates
- recognize_pii_entities — detect and redact personal data
- detect_language        — identify the language of each response
- store_documents        — accumulate text rows for later batch analysis

You also have a fabric_dataagent tool for querying Microsoft Fabric semantic models.

IMPORTANT — default tool usage:
- For file uploads, call ONLY analyze_sentiment unless the user explicitly asks for
    key phrases, entities, PII, or language detection.
- When the upload message says to use uploaded dataset, call analyze_sentiment with
    NO arguments. The tool will process the full uploaded dataset automatically.
- Do NOT ask the user to resend rows and do NOT process only a subset.

How to decide which tools to use:
- If the user message contains numbered survey responses (text data) → analyze them directly with Language tools. Do NOT call the Fabric tool.
- If the user message asks to query, retrieve, or get data → call the fabric_dataagent tool FIRST to fetch the data, then analyze the results with Language tools.
- Never ask the user to clarify which data source to use — the message content makes it clear.

IMPORTANT — Fabric data workflow (MUST follow this pattern):
After the fabric_dataagent returns data, do NOT pass all rows directly to
analyze_sentiment. Instead:
  1. Call store_documents one or more times with batches of up to 10 rows each
     until every retrieved row has been stored.
  2. Then call analyze_sentiment with NO arguments — it will process the full
     stored dataset automatically, same as the file-upload path.
This ensures all rows are analyzed regardless of dataset size.

Always batch multiple documents into a single function call (up to 10 per call).
"""
    + _COMMON_INSTRUCTIONS
)

AGENT_SYSTEM_PROMPT_MCP = (
    """You are a Survey Analysis Agent powered by Azure AI Language via MCP.

You have direct access to all Azure AI Language capabilities as MCP tools:
- Sentiment Analysis with opinion mining (document & sentence level)
- Key Phrase Extraction — identify main topics and themes
- Named Entity Recognition (NER) — find people, places, organisations, dates
- PII Detection — detect and flag personal data for redaction
- Language Detection — identify the language of each response
"""
    + _COMMON_INSTRUCTIONS
)


# ─── Agent builders ───────────────────────────────────────────────────────────

def _build_sdk_agent(client: AIProjectClient) -> object:
    """Create agent with Language SDK function tools and optional Fabric Data Agent."""
    from language_tools import TOOL_DEFINITIONS  # local import to avoid SDK dep at top

    # raw_function_defs is a list of {type, function {...}} dicts
    tools = TOOL_DEFINITIONS.copy()
    
    # Add Fabric Data Agent as a built-in tool if connection is configured
    if FABRIC_CONNECTION_NAME:
        conn_id = client.connections.get(FABRIC_CONNECTION_NAME).id
        fabric = FabricTool(connection_id=conn_id)
        tools.extend(fabric.definitions)
        print(f"✓ Fabric Data Agent added (connection: {FABRIC_CONNECTION_NAME})")
    
    return client.agents.create_agent(
        model=GPT_DEPLOYMENT,
        name="sentiment-analysis-agent",
        instructions=AGENT_SYSTEM_PROMPT_SDK,
        tools=tools,
    )


def _build_mcp_agent(client: AIProjectClient) -> object:
    """Create agent with Language MCP server."""
    toolset = ToolSet()
    toolset.add(McpTool(server_label="azure_language", server_url=LANGUAGE_MCP_URL))

    resources = toolset.resources
    resources["mcp"] = [
        MCPToolResource(
            server_label="azure_language",
            headers={},
            require_approval="never",
        )
    ]
    return client.agents.create_agent(
        model=GPT_DEPLOYMENT,
        name="sentiment-analysis-agent",
        instructions=AGENT_SYSTEM_PROMPT_MCP,
        tools=toolset.definitions,
        tool_resources=resources,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    endpoint = f"{AI_SERVICES_ENDPOINT.rstrip('/')}/api/projects/{FOUNDRY_PROJECT}"
    print(f"Connecting to Foundry project: {endpoint}")
    print(f"Tool mode:                     {TOOL_MODE.upper()}")
    if TOOL_MODE == "mcp":
        print(f"Language MCP server:           {LANGUAGE_MCP_URL}")

    credential = DefaultAzureCredential()
    client = AIProjectClient(endpoint=endpoint, credential=credential)

    if FABRIC_CONNECTION_NAME:
        print(f"Fabric connection:             {FABRIC_CONNECTION_NAME}")

    if TOOL_MODE == "mcp":
        agent = _build_mcp_agent(client)
    else:
        # Add src/ to path so language_tools is importable
        sys.path.insert(0, os.path.dirname(__file__))
        agent = _build_sdk_agent(client)

    print("\n✅ Agent created successfully!")
    print(f"   Agent ID:   {agent.id}")
    print(f"   Agent Name: {agent.name}")
    print(f"   Model:      {agent.model}")
    print(f"   Tool mode:  {TOOL_MODE}")

    config = {
        "agent_id": agent.id,
        "agent_name": agent.name,
        "model": agent.model,
        "endpoint": endpoint,
        "tool_mode": TOOL_MODE,
    }
    with open("agent_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("\n📄 Agent config saved to agent_config.json")


if __name__ == "__main__":
    main()
