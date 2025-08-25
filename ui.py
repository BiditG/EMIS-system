# ui.py
import os
import io
import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Data Assistant", page_icon="💬", layout="wide")
st.title("💬 AI Data Assistant (SQL Server + Cohere)")

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI /ask URL", value=DEFAULT_API, help="Your backend endpoint")
    show_sql = st.checkbox("Show generated SQL", value=False)
    show_preview = st.checkbox("Show preview table", value=True)
    st.caption("Tip: run the backend with: `uvicorn app:app --reload`")
    st.divider()
    if st.button("🧹 Clear conversation", key="clear_conv"):
        st.session_state.history = []

# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {q, answer, rows, sql, note, row_count}

# ---------------- Helper: call API ----------------
def ask_backend(question: str):
    try:
        resp = requests.post(api_url, json={"question": question}, timeout=90)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            st.error(f"Server error ({resp.status_code}): {detail}")
            return None
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API at {api_url}\n\n{e}")
        return None

# ---------------- Chat input ----------------
q = st.chat_input("Ask about your data (e.g., 'Total students by district in last 30 days')")
if q:
    with st.spinner("Thinking…"):
        result = ask_backend(q)
    if result:
        st.session_state.history.append({
            "q": q,
            "answer": result.get("answer", ""),
            "rows": result.get("preview") or [],
            "sql": result.get("sql", ""),
            "note": result.get("note", ""),
            "row_count": result.get("row_count", 0),
        })

# ---------------- Render conversation ----------------
for i, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        meta = f"Rows: **{turn.get('row_count', 0)}**"
        if turn.get("note"):
            meta += f" • {turn['note']}"
        st.caption(meta)

        # Optional SQL (unique key for expander)
        if show_sql and turn.get("sql"):
            with st.expander("🔎 Generated SQL", expanded=False):
                st.code(turn["sql"], language="sql")

        # Optional preview table + download (unique keys!)
        if show_preview and turn.get("rows"):
            df = pd.DataFrame(turn["rows"])
            st.dataframe(df, use_container_width=True, height=240, key=f"df_{i}")

            csv_io = io.StringIO()
            df.to_csv(csv_io, index=False)
            st.download_button(
                label="⬇️ Download preview as CSV",
                data=csv_io.getvalue().encode("utf-8"),
                file_name=f"preview_{i}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{i}",   # <-- unique key per message
            )

# ---------------- Quick examples ----------------
with st.expander("💡 Example questions", expanded=False):
    st.markdown(
        """
- **Totals:** `Total students by district in the last 30 days`  
- **Top-N:** `Top 10 schools by enrollment this year`  
- **Breakdowns:** `Teachers count by subject in Kathmandu`  
- **Trends:** `Monthly admissions by district for 2024`  
- **Lookups:** `List of schools opened after 2020 with capacity > 500`  
        """.strip()
    )
