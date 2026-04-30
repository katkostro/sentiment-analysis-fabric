"""
Streamlit chat UI for the Sentiment Analysis Foundry Agent.

Run:
  streamlit run src/app.py
"""

from __future__ import annotations

import json
import io
import os
import re

# .env values OVERRIDE existing env vars to avoid stale terminal sessions
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8-sig") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip().strip('"')

import sys
import time

import streamlit as st
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import ToolOutput

# Ensure src/ is on path so language_tools is importable
sys.path.insert(0, os.path.dirname(__file__))

# ─── Application Insights Configuration ──────────────────────────────────────

# Initialize Application Insights if connection string is available
_appinsights_connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
if _appinsights_connection_string:
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from azure.ai.agents.telemetry import AIAgentsInstrumentor
        configure_azure_monitor(connection_string=_appinsights_connection_string)
        AIAgentsInstrumentor().instrument()
        print("✅ Application Insights monitoring enabled (with agents tracing)")
    except ImportError:
        print("⚠️  Application Insights packages not installed. Run: pip install -r requirements.txt")
else:
    print("ℹ️  Application Insights not configured (APPLICATIONINSIGHTS_CONNECTION_STRING not set)")


# ─── Excel reader ──────────────────────────────────────────────────────

_RESPONSE_COLS = ["response", "comment", "feedback", "text", "answer", "remarks", "survey"]
# Patterns that indicate an ID/code column — skip during auto-detection
_SKIP_COL_PATTERNS = ["id", "code", "ref", "key", "num", "date", "time", "score", "rating"]


def _load_dataframe(file_bytes: bytes) -> "pd.DataFrame":
    """Try every supported format and return a DataFrame."""
    import pandas as pd

    errors = []
    for engine in ["openpyxl", "xlrd"]:
        try:
            return pd.read_excel(io.BytesIO(file_bytes), engine=engine)
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception as exc:
            errors.append(f"csv/{enc}: {exc}")
    try:
        tables = __import__("pandas").read_html(io.BytesIO(file_bytes))
        if tables:
            return tables[0]
    except Exception as exc:
        errors.append(f"html: {exc}")

    drm_hint = ""
    if file_bytes[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        try:
            import olefile
            ole = olefile.OleFileIO(io.BytesIO(file_bytes))
            streams = [s for entry in ole.listdir() for s in entry]
            if "EncryptedPackage" in streams or "DRMEncryptedDataSpace" in streams:
                drm_hint = (
                    "\n\n**This file is protected by Microsoft Information Protection (DRM).**\n"
                    "To use it:\n"
                    "1. Open it in Excel\n"
                    "2. Go to **File \u2192 Save As** and save as a new `.xlsx` or CSV file\n"
                    "3. Upload the new unprotected file"
                )
        except Exception:
            pass
    raise ValueError("Could not open file." + drm_hint)


def _extract_fabric_rows(reply: str) -> list[str]:
    """Parse text rows from the agent's Fabric data-fetch response.

    Handles common formats returned by the LLM:
      - Numbered lines:  1. Some text  /  1) Some text
      - Bullet lines:    - Some text  /  * Some text
      - Pipe-delimited table rows (skip headers/separators)
      - Plain non-empty lines as fallback
    """
    lines = reply.strip().splitlines()
    rows: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip markdown table separators (|---|---|)
        if re.match(r"^\|[\s\-|]+\|$", stripped):
            continue
        # Skip lines that look like headers or labels
        if stripped.startswith("#") or stripped.startswith("**"):
            continue
        # Numbered list: "1. text" or "1) text"
        m = re.match(r"^\d+[\.\)]\s+(.+)", stripped)
        if m:
            rows.append(m.group(1).strip())
            continue
        # Bullet list
        m = re.match(r"^[-\*•]\s+(.+)", stripped)
        if m:
            rows.append(m.group(1).strip())
            continue
        # Pipe-delimited table row — take the longest cell as the text
        if "|" in stripped:
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            # Skip if all cells look like numbers/headers
            text_cells = [c for c in cells if len(c) > 10 and not re.match(r"^[\d\.\-\s%]+$", c)]
            if text_cells:
                rows.append(max(text_cells, key=len))
                continue
        # Fallback: any line with enough text content
        if len(stripped) > 15:
            rows.append(stripped)
    return rows


def _guess_text_column(df: "pd.DataFrame") -> str:
    """Pick the most likely free-text response column."""
    # 1. Column name matches a known keyword and NOT a skip pattern
    for candidate in _RESPONSE_COLS:
        for c in df.columns:
            col_lower = str(c).lower()
            if candidate in col_lower and not any(p in col_lower for p in _SKIP_COL_PATTERNS):
                return c
    # 2. Fall back: string column with longest average text length
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        return max(str_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())
    return str(df.columns[0])


def read_excel_responses(file_bytes: bytes, filename: str, col: str | None = None):
    """Load file and return (df, selected_col, all_columns)."""
    df = _load_dataframe(file_bytes)
    all_cols = [str(c) for c in df.columns]
    if col is None or col not in all_cols:
        col = _guess_text_column(df)
    # Normalise column name to string
    df.columns = all_cols
    return df, col, all_cols


# ─── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Survey Sentiment Agent",
    page_icon="📊",
    layout="wide",
)

# ─── Load agent config ────────────────────────────────────────────────────────

@st.cache_resource
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "agent_config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_client(endpoint: str) -> AIProjectClient:
    return AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())


# ─── Session state ────────────────────────────────────────────────────────────

def init_session(client: AIProjectClient) -> None:
    if "thread_id" not in st.session_state:
        thread = client.agents.threads.create()
        st.session_state.thread_id = thread.id
        st.session_state.messages = []  # [{role, content}]


def reset_thread(client: AIProjectClient) -> None:
    thread = client.agents.threads.create()
    st.session_state.thread_id = thread.id
    st.session_state.messages = []
    st.success("New conversation started.")


# ─── SDK tool-call loop ──────────────────────────────────────────────────────

def _count_docs_from_args(fn_args: dict) -> int:
    """Count documents from tool arguments."""
    docs = fn_args.get("documents", [])
    if isinstance(docs, str):
        try:
            docs = json.loads(docs)
        except (json.JSONDecodeError, TypeError):
            return 1
    return len(docs) if isinstance(docs, list) else 0


def _extract_processed_count(fn_name: str, fn_args: dict, result: str) -> int:
    """Estimate rows processed by analyze_sentiment tool calls."""
    if fn_name != "analyze_sentiment":
        return 0

    explicit = _count_docs_from_args(fn_args)
    if explicit:
        return explicit

    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return 0

    if isinstance(payload, dict):
        total = payload.get("total_documents")
        return int(total) if isinstance(total, int) else 0
    if isinstance(payload, list):
        return len(payload)
    return 0


def _extract_section2_rows(fn_name: str, result: str) -> list[dict]:
    """Extract deterministic Section 2 rows from analyze_sentiment summary payload."""
    if fn_name != "analyze_sentiment":
        return []
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(payload, dict):
        rows = payload.get("section_2_table_rows", [])
        return rows if isinstance(rows, list) else []
    return []


def _render_section2_markdown(rows: list[dict]) -> str:
    """Render Section 2 in a fixed format independent of agent markdown."""
    lines = [
        "2. Where Sentiment Breaks Down",
        "",
        "| Theme | 🟢 Positive | 🟡 Neutral | 🔴 Negative |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        theme = str(row.get("theme", "")).strip() or "Unknown"
        pos = str(row.get("positive_display") or f"{row.get('positive_count', 0)} ({float(row.get('positive_pct', 0)):.1f}%)")
        neu = str(row.get("neutral_display") or f"{row.get('neutral_count', 0)} ({float(row.get('neutral_pct', 0)):.1f}%)")
        neg = str(row.get("negative_display") or f"{row.get('negative_count', 0)} ({float(row.get('negative_pct', 0)):.1f}%)")
        lines.append(f"| {theme} | {pos} | {neu} | {neg} |")
    return "\n".join(lines)


def _inject_section2(reply: str, rows: list[dict]) -> str:
    """Replace agent-generated Section 2 with deterministic app-rendered table."""
    if not rows:
        return reply

    section2 = _render_section2_markdown(rows)
    pattern = re.compile(
        r"(?ims)^\s*\**\s*2\.\s*Where Sentiment Breaks Down.*?(?=^\s*\**\s*3\.\s*Key Drivers of Negative Sentiment\b|\Z)"
    )

    if pattern.search(reply):
        return pattern.sub(section2 + "\n\n", reply, count=1)

    section3_pattern = re.compile(r"(?ims)^\s*\**\s*3\.\s*Key Drivers of Negative Sentiment\b")
    m = section3_pattern.search(reply)
    if m:
        return reply[:m.start()] + section2 + "\n\n" + reply[m.start():]

    return reply.rstrip() + "\n\n" + section2

def _build_tool_outputs(run, client: AIProjectClient, thread_id: str, status_widget=None):
    """Execute Language SDK function tools and submit outputs."""
    from language_tools import TOOL_DISPATCH

    tool_outputs = []
    calls = run.required_action.submit_tool_outputs.tool_calls
    for i, call in enumerate(calls, 1):
        fn_name = call.function.name
        fn_args = json.loads(call.function.arguments or "{}")
        if status_widget:
            status_widget.update(
                label=f"Language Tools: {fn_name} ({i}/{len(calls)})",
                state="running",
            )
        t0 = time.time()
        try:
            fn = TOOL_DISPATCH.get(fn_name)
            if fn is None:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})
            else:
                result = fn(**fn_args)
        except Exception as exc:  # noqa: BLE001
            result = json.dumps({"error": str(exc)})

        processed = _extract_processed_count(fn_name, fn_args, result)
        section2_rows = _extract_section2_rows(fn_name, result)
        if section2_rows:
            st.session_state["_section2_rows"] = section2_rows

        if processed:
            st.session_state.setdefault("_rows_processed", 0)
            st.session_state["_rows_processed"] += processed
            if status_widget:
                status_widget.update(
                    label=(
                        f"Language Tools: {fn_name} ({i}/{len(calls)}) "
                        f"- {st.session_state['_rows_processed']} rows processed"
                    ),
                    state="running",
                )

        print(f"  ⚙️ {fn_name} took {time.time()-t0:.1f}s")
        tool_outputs.append(ToolOutput(tool_call_id=call.id, output=result))

    return client.agents.runs.submit_tool_outputs(
        thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs,
    )


def _wait_for_run(client: AIProjectClient, thread_id: str, run, status_widget=None, task: str = "chat") -> object:
    """Poll run to completion, handling SDK function tool calls."""
    terminal = {"completed", "failed", "cancelled", "expired"}
    deadline = time.time() + 300
    t_start = time.time()
    t_phase = t_start  # reset per phase

    # Task-specific labels for the initial server-side processing phase
    phase1_labels = {
        "fabric": "Foundry Agent \u2192 Fabric Agent: querying data",
        "file":   "Foundry Agent: processing file",
        "chat":   "Foundry Agent: thinking",
    }
    current_label = phase1_labels.get(task, phase1_labels["chat"])
    tools_ran = False

    while run.status not in terminal:
        if time.time() > deadline:
            try:
                client.agents.runs.cancel(thread_id=thread_id, run_id=run.id)
            except Exception:
                pass
            run._data["status"] = "failed"
            run._data["last_error"] = {"code": "timeout", "message": "Run exceeded 5-minute timeout."}
            rows_processed = int(st.session_state.pop("_rows_processed", 0))
            return run, rows_processed
        elapsed = int(time.time() - t_phase)
        if status_widget:
            status_widget.update(label=f"{current_label}... ({elapsed}s)", state="running")
        time.sleep(0.5)
        run = client.agents.runs.get(thread_id=thread_id, run_id=run.id)
        if run.status == "requires_action":
            print(f"\u23f1\ufe0f requires_action at {time.time()-t_start:.1f}s")
            t_phase = time.time()
            tools_ran = True
            run = _build_tool_outputs(run, client, thread_id, status_widget=status_widget)
            t_phase = time.time()
            current_label = "Foundry Agent: processing results"
        elif tools_ran and current_label != "Foundry Agent: generating response":
            current_label = "Foundry Agent: generating response"
            t_phase = time.time()
    total = int(time.time() - t_start)
    rows_processed = int(st.session_state.pop("_rows_processed", 0))
    print(f"\u23f1\ufe0f Run completed in {total}s (status: {run.status}, rows: {rows_processed})")
    if status_widget:
        done_label = f"Done ({total}s total)"
        if rows_processed:
            done_label += f" - {rows_processed} rows processed"
        status_widget.update(label=done_label, state="complete")
    return run, rows_processed


def _cancel_active_runs(client: AIProjectClient, thread_id: str) -> None:
    """Cancel any runs that are still in a non-terminal state on this thread."""
    terminal = {"completed", "failed", "cancelled", "expired"}
    try:
        runs = client.agents.runs.list(thread_id=thread_id)
        for run in runs:
            if run.status not in terminal:
                try:
                    client.agents.runs.cancel(thread_id=thread_id, run_id=run.id)
                    for _ in range(10):
                        time.sleep(0.5)
                        r = client.agents.runs.get(thread_id=thread_id, run_id=run.id)
                        if r.status in terminal:
                            break
                except Exception:
                    pass
    except Exception:
        pass


def send_message(
    client: AIProjectClient,
    agent_id: str,
    thread_id: str,
    content: str,
    status_widget=None,
    task: str = "chat",
) -> str:
    # Clear any previous run leftovers.
    st.session_state.pop("_section2_rows", None)
    _cancel_active_runs(client, thread_id)
    client.agents.messages.create(thread_id=thread_id, role="user", content=content)

    if status_widget:
        status_widget.update(label="Starting Foundry Agent...", state="running")
    run = client.agents.runs.create(thread_id=thread_id, agent_id=agent_id)
    run, rows_processed = _wait_for_run(client, thread_id, run, status_widget=status_widget, task=task)

    if run.status == "failed":
        return f"❌ Run failed: {run.last_error}"

    last = client.agents.messages.get_last_message_text_by_role(
        thread_id=thread_id, role="assistant",
    )
    reply = last.text.value if last else "(no response)"
    section2_rows = st.session_state.pop("_section2_rows", [])
    reply = _inject_section2(reply, section2_rows)
    if rows_processed:
        reply += f"\n\n---\n*Language service processed: {rows_processed} rows*"
    return reply


# ─── UI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    client = get_client(config["endpoint"])
    init_session(client)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📊 Survey Analysis")
        st.caption(f"Agent: `{config['agent_name']}`")
        st.caption(f"Model: `{config['model']}`")
        st.caption(f"Tools: `{config.get('tool_mode', 'sdk').upper()}`")
        st.divider()

        # Data source selector — Fabric is available when the agent has the Fabric tool
        fabric_enabled = bool(os.environ.get("FABRIC_CONNECTION_NAME"))
        if fabric_enabled:
            data_source = st.radio(
                "Data Source",
                ["Local File", "Fabric Semantic Model"],
                help="Choose between uploading a local file or querying Fabric data"
            )
        else:
            data_source = "Local File"
            
        st.subheader("📁 " + data_source)

        # Initialize variables
        fabric_query = None
        
        # Local file upload mode
        if data_source == "Local File":
            uploaded_file = st.file_uploader(
                "Excel / CSV file",
                type=["xlsx", "xls", "csv"],
                help="Upload a survey file to analyse",
            )
        else:
            # Fabric query mode
            uploaded_file = None
            fabric_query = st.text_area(
                "Natural Language Query",
                placeholder="e.g., Get all survey responses from last quarter",
                help="Describe what data you want to analyze from your Fabric semantic model",
                height=100
            )

        sel_cols = None
        file_bytes = None
        if uploaded_file:
            file_bytes = uploaded_file.read()
            try:
                df_preview, guessed_col, all_cols = read_excel_responses(file_bytes, uploaded_file.name)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

            sel_cols = st.multiselect(
                "Response columns (select 1-2)",
                options=all_cols,
                default=[guessed_col],
                max_selections=2,
                help="Select one or two columns to analyze",
            )
            # Show a quick preview of the selected columns
            if sel_cols:
                for col in sel_cols:
                    preview_vals = df_preview[col].dropna().astype(str).head(2).tolist()
                    st.caption(f"**{col}**: " + " / ".join(f'"{v[:50]}"' for v in preview_vals))

        if file_bytes and sel_cols and st.button("Analyse File", type="primary", use_container_width=True):
            try:
                df_final, _, _ = read_excel_responses(file_bytes, uploaded_file.name)
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
                st.stop()

            # Collect responses from all selected columns
            all_responses = []
            for col in sel_cols:
                col_responses = df_final[col].dropna().astype(str).tolist()
                all_responses.extend([(col, resp) for resp in col_responses])

            if not all_responses:
                st.warning("No responses found in the selected columns.")
                st.stop()

            # Store all uploaded responses for deterministic full-dataset processing.
            from language_tools import set_pending_documents
            docs_for_tool = [resp for _, resp in all_responses]
            set_pending_documents(docs_for_tool)

            if len(sel_cols) == 1:
                col_desc = f"column `{sel_cols[0]}`"
            else:
                col_desc = f"columns `{', '.join(sel_cols)}`"

            header = (
                f"File: **{uploaded_file.name}** — {len(docs_for_tool)} responses "
                f"from {col_desc}"
            )
            user_msg = (
                f"{header}\n\n"
                "Analyze the uploaded survey file now.\n"
                "Call analyze_sentiment with NO arguments so it uses the full uploaded dataset.\n"
                "Do NOT use extract_key_phrases or recognize_entities unless explicitly requested.\n\n"
                "Provide analysis in this structure:\n"
                "1. Customer Sentiment Overview (executive summary)\n"
                "2. Where Sentiment Breaks Down (table with themes and sentiment percentages)\n"
                "3. Key Drivers of Negative Sentiment (table with top 5 issue clusters)\n"
                "4. Key Drivers of Positive Sentiment (table with top strengths)\n"
                "5. Insight-Driven Recommendations (numbered, with Why/Recommendation format)"
            )

            st.session_state.messages.append({"role": "user", "content": user_msg})
            st.session_state["_pending_file_msg"] = user_msg
            st.rerun()

        # Fabric query mode — two-phase: fetch data, then analyze via same path as file upload
        if data_source == "Fabric Semantic Model" and fabric_query and st.button("Query & Analyze", type="primary", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": f"Query: {fabric_query}"})
            st.session_state["_pending_fabric_query"] = fabric_query
            st.rerun()

        st.divider()
        if st.button("🗑️ New Conversation", use_container_width=True):
            reset_thread(client)
            st.rerun()

        st.divider()
        st.caption(f"Thread: `{st.session_state.get('thread_id', '...')}`")

    # ── Main chat area ────────────────────────────────────────────────────────
    st.title("Survey Sentiment Agent")
    st.caption(
        "Powered by Azure AI Foundry · Azure Language MCP · GPT-4o"
    )

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Show placeholder when no messages yet
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            if fabric_enabled:
                st.markdown(
                    "👋 Hi! I'm your Survey Analysis Agent. You can:\n\n"
                    "- **Upload an Excel file** in the sidebar for analysis\n"
                    "- **Query Fabric data** using natural language\n"
                    "- **Type a message** below — paste responses directly or ask questions\n\n"
                    "Try: *'Analyse: Great service! / Very slow delivery / Best experience ever'*"
                )
            else:
                st.markdown(
                    "👋 Hi! I'm your Survey Analysis Agent. You can:\n\n"
                    "- **Upload an Excel file** in the sidebar for full analysis\n"
                    "- **Type a message** below — paste responses directly or ask questions\n\n"
                    "Try: *'Analyse: Great service! / Very slow delivery / Best experience ever'*"
                )

    # ── Run file analysis (outside sidebar so st.status renders in main area)
    if st.session_state.get("_pending_file_msg"):
        user_msg = st.session_state.pop("_pending_file_msg")
        with st.chat_message("assistant"):
            with st.status("Starting Foundry Agent...", expanded=True) as status:
                reply = send_message(client, config["agent_id"], st.session_state.thread_id, user_msg, status_widget=status, task="file")
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Run Fabric query (two-phase: fetch → extract → analyze) ────────────
    if st.session_state.get("_pending_fabric_query"):
        fabric_query = st.session_state.pop("_pending_fabric_query")
        with st.chat_message("assistant"):
            with st.status("Querying Fabric...", expanded=True) as status:
                # Phase 1: Ask agent to fetch data from Fabric only
                fetch_msg = (
                    f"Use the fabric_dataagent tool to query the semantic model: {fabric_query}\n\n"
                    "Return ALL the retrieved rows as a numbered list. Do NOT call any analysis tools yet."
                )
                status.update(label="Foundry Agent → Fabric Agent: querying data...", state="running")
                fetch_reply = send_message(
                    client, config["agent_id"], st.session_state.thread_id,
                    fetch_msg, status_widget=status, task="fabric",
                )

                # Phase 2: Extract rows from the agent's reply in Python
                status.update(label="Extracting rows from Fabric response...", state="running")
                rows = _extract_fabric_rows(fetch_reply)
                if not rows:
                    st.warning("Could not extract any rows from the Fabric response.")
                    st.markdown(fetch_reply)
                    st.session_state.messages.append({"role": "assistant", "content": fetch_reply})
                    st.rerun()

                # Phase 3: Same path as file upload — store in buffer, analyze
                from language_tools import set_pending_documents
                set_pending_documents(rows)

                analyze_msg = (
                    f"Fabric returned **{len(rows)} responses**. The data is now loaded.\n\n"
                    "Call analyze_sentiment with NO arguments so it uses the full dataset.\n"
                    "Do NOT use extract_key_phrases or recognize_entities unless explicitly requested.\n\n"
                    "Present results in this structure:\n"
                    "1. Customer Sentiment Overview (executive summary)\n"
                    "2. Where Sentiment Breaks Down (table with themes and sentiment percentages)\n"
                    "3. Key Drivers of Negative Sentiment (table with top 5 issue clusters)\n"
                    "4. Key Drivers of Positive Sentiment (table with top strengths)\n"
                    "5. Insight-Driven Recommendations (numbered, with Why/Recommendation format)"
                )
                status.update(label=f"Analyzing {len(rows)} rows...", state="running")
                reply = send_message(
                    client, config["agent_id"], st.session_state.thread_id,
                    analyze_msg, status_widget=status, task="file",
                )
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask the agent or paste survey responses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Starting Foundry Agent...", expanded=True) as status:
                reply = send_message(client, config["agent_id"], st.session_state.thread_id, prompt, status_widget=status, task="chat")
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
