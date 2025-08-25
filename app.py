# app.py
import os, re, json
import pyodbc, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import sqlglot

load_dotenv()

# ========================= Config =========================
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "200"))
QUERY_TIMEOUT_SECS = int(os.getenv("QUERY_TIMEOUT_SECS", "30"))

COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not set in environment")

def build_cnxn_str() -> str:
    driver = os.getenv('ODBC_DRIVER', 'ODBC Driver 17 for SQL Server')
    server = os.getenv('SQL_SERVER', 'localhost')
    port   = os.getenv('SQL_PORT', '')
    db     = os.getenv('SQL_DATABASE', 'CEHRDApp')
    trusted = os.getenv('SQL_TRUSTED_CONNECTION', 'yes').lower() in ('yes','true','1')

    if port and '\\' not in server:
        server_part = f"{server},{port}"
    else:
        server_part = server

    base = f"DRIVER={{{driver}}};SERVER={server_part};DATABASE={db};Encrypt=no;TrustServerCertificate=yes;"
    if trusted:
        return base + "Trusted_Connection=yes;"
    else:
        user = os.getenv('SQL_USER'); pwd  = os.getenv('SQL_PASSWORD')
        if not user or not pwd:
            raise RuntimeError("SQL_TRUSTED_CONNECTION=no requires SQL_USER and SQL_PASSWORD")
        return base + f"UID={user};PWD={pwd};"

CNXN_STR = build_cnxn_str()

# ====================== Load schema.json ===================
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "schema.json")
if not os.path.exists(SCHEMA_PATH):
    raise RuntimeError(f"{SCHEMA_PATH} not found. Put your schema.json next to app.py.")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    SCHEMA: dict[str, list[dict]] = json.load(f)

def table_names() -> list[str]:
    return list(SCHEMA.keys())

def format_schema_block(tables: list[str]) -> str:
    lines = []
    for t in tables:
        cols = ", ".join(f"{c['name']}:{c['type']}" for c in SCHEMA[t])
        lines.append(f"{t}({cols})")
    return "\n".join(lines)

def search_schema(keyword: str):
    kw = keyword.lower()
    hits = []
    for t, cols in SCHEMA.items():
        col_hits = [c["name"] for c in cols if kw in c["name"].lower()]
        if kw in t.lower() or col_hits:
            hits.append({"table": t, "columns": col_hits or [c["name"] for c in cols]})
    return hits

# ===================== Relevance heuristics =================
ALL_COLS = {(t, c["name"]): c["type"] for t, cols in SCHEMA.items() for c in cols}

def pick_relevant_tables(question: str, k: int = 18) -> list[str]:
    q = question.lower()
    tokens = re.findall(r"[a-z0-9_]+", q)
    scores = []
    for t, cols in SCHEMA.items():
        name_score = sum(tok in t.lower() for tok in tokens)
        col_score  = sum(any(tok == c["name"].lower() or tok in c["name"].lower() for tok in tokens) for c in cols)
        bonus = 0
        if "school" in q and ("school" in t.lower() or any("organization" in t.lower() for _ in [1])):
            bonus += 1
        if "district" in q and any("district" in c["name"].lower() for c in cols):
            bonus += 1
        if "student" in q and ("student" in t.lower() or any("student" in c["name"].lower() for c in cols)):
            bonus += 1
        scores.append((name_score + col_score + bonus, t))
    scores.sort(reverse=True)
    picked = [t for s, t in scores[:k] if s > 0]
    return picked or list(SCHEMA.keys())[:min(k, len(SCHEMA))]

# ==================== Fallback query builder =================
PREFERRED_NAME_COLUMNS = (
    "StudentName","FullName","FirstName","LastName",
    "SchoolName","OrganizationName","OrgName","Name","DisplayName",
    "DistrictName","MunicipalityName","VillageName"
)

def guess_distinct_list(question: str):
    """
    Try to satisfy prompts like:
      - "student names"
      - "names of schools"
      - "list districts"
    Returns (sql, reason) or None.
    """
    q = question.lower()
    want = None
    if re.search(r"\bstudent(s)?\b", q) and re.search(r"\bnames?\b", q):
        want = "student"
    elif re.search(r"\b(school|schools)\b", q) and re.search(r"\bnames?\b", q):
        want = "school"
    elif re.search(r"\b(district|districts)\b", q):
        want = "district"
    elif "names of" in q:
        want = "generic-name"

    if not want:
        return None

    # gather candidate (table, column) pairs that look like names
    candidates = []
    for t, cols in SCHEMA.items():
        for c in cols:
            col = c["name"]
            score = 0
            if col in PREFERRED_NAME_COLUMNS: score += 3
            if "name" in col.lower(): score += 1
            # intent boost
            tl = t.lower(); cl = col.lower()
            if want == "student" and ("student" in tl or "student" in cl): score += 2
            if want == "school" and ("school" in tl or "organization" in tl): score += 2
            if want == "district" and ("district" in tl or "district" in cl): score += 2
            if score > 0:
                candidates.append((score, t, col))
    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    _, best_t, best_c = candidates[0]

    # IMPORTANT FIX: DISTINCT comes BEFORE TOP in T-SQL
    sql = (
        f"SELECT DISTINCT TOP {ROW_LIMIT} {best_t}.{best_c} AS value "
        f"FROM {best_t} "
        f"WHERE {best_t}.{best_c} IS NOT NULL AND LTRIM(RTRIM({best_t}.{best_c})) <> '' "
        f"ORDER BY {best_t}.{best_c};"
    )
    return sql, f"Auto list of distinct {best_c} from {best_t}"

# ==================== Cohere wrapper =======================
def cohere_chat(prompt: str) -> str:
    import cohere
    co = cohere.Client(COHERE_API_KEY)
    model = COHERE_MODEL
    # Support both SDK styles
    try:
        resp = co.chat(model=model, message=prompt, temperature=0.2)
        return getattr(resp, "text", str(resp))
    except TypeError:
        resp = co.chat(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return getattr(resp, "text", str(resp))

PROMPT_NL2SQL = """
You are an expert analyst generating T-SQL (Microsoft SQL Server).

Task: Convert the user's question to ONE safe SELECT query.

STRICT RULES:
- Return ONLY the SQL inside a ```sql fenced code block.
- SELECT-only. No INSERT/UPDATE/DELETE/ALTER/DROP/TRUNCATE/MERGE/EXEC.
- ONE statement only (no multiple statements).
- Use fully-qualified schema.table when possible.
- Include TOP {ROW_LIMIT}.
- Use ONLY tables/columns from the schema below. If the question is unclear,
  return exactly: UNCLEAR_REQUEST

Relevant database schema (subset):
{SCHEMA_BLOCK}

User question:
{QUESTION}

Return ONLY the SQL in a code block, or UNCLEAR_REQUEST.
""".strip()

PROMPT_SUMMARY = """
You are a careful analyst. Summarize the result for a non-technical user (3–6 sentences),
using only the rows shown. Include key numbers/dates and one brief caveat if relevant.
If zero rows, say what was searched and suggest a narrower filter. No code.

Question: {QUESTION}
Rows (JSON preview): {ROWS_JSON}
total_rows={ROW_COUNT}

Return only prose.
""".strip()

def build_nl2sql_prompt(q: str) -> str:
    tables = pick_relevant_tables(q, k=18)
    schema_block = format_schema_block(tables)
    return PROMPT_NL2SQL.format(ROW_LIMIT=ROW_LIMIT, SCHEMA_BLOCK=schema_block, QUESTION=q)

# ====================== SQL Inspector ======================
BAD_TOKENS = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|MERGE|EXEC|sp_|xp_)\b", re.I)

def extract_sql(text: str) -> str | None:
    t = text.strip()
    if t == "UNCLEAR_REQUEST":
        return None
    m = re.search(r"```sql(.*?)```", t, re.S | re.I)
    if m:
        return m.group(1).strip()
    if t.upper().startswith("SELECT"):
        return t
    return None

def is_safe_select(sql: str) -> bool:
    """
    Relaxed inspector: allow any single SELECT statement
    unless it contains known dangerous tokens.
    """
    sql_clean = sql.strip().upper()

    # must start with SELECT
    if not sql_clean.startswith("SELECT"):
        return False

    # block obvious bad tokens
    if BAD_TOKENS.search(sql_clean):
        return False

    # block multiple statements (more than one ';')
    if sql_clean.count(";") > 1:
        return False

    return True


def enforce_top(sql: str) -> str:
    if re.search(r"\bTOP\s+\d+\b", sql, re.I):
        return sql
    return re.sub(r"^\s*SELECT\b", f"SELECT TOP {ROW_LIMIT}", sql, flags=re.I)

# ========================= API =============================
class AskReq(BaseModel):
    question: str

class AskResp(BaseModel):
    answer: str
    row_count: int
    preview: list[dict] | None = None
    note: str | None = None
    sql: str | None = None

app = FastAPI(title="AI Data Assistant (SQL Server + Cohere)")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/tables")
def get_tables():
    return {"tables": table_names()}

@app.get("/schema")
def get_schema():
    return {"schema": SCHEMA}

@app.get("/schema/search")
def schema_search(contains: str = Query(..., description="keyword to search in table/column names")):
    return {"matches": search_schema(contains)}

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    q = req.question.strip()

    # Fast-path: list tables
    if re.search(r"\b(what|list|show).*(table|tables)\b", q.lower()):
        return AskResp(
            answer="Here are the tables I can see:\n- " + "\n- ".join(table_names()),
            row_count=0, preview=None, note="From schema.json", sql=None
        )

    # 1) NL→SQL via Cohere grounded on relevant schema
    try:
        sql_text = cohere_chat(build_nl2sql_prompt(q))
    except Exception as e:
        raise HTTPException(500, f"Cohere error (NL→SQL): {e}")

    sql = extract_sql(sql_text)

    # 1a) If unclear, try DISTINCT list fallback (e.g., "student names")
    if not sql:
        fb = guess_distinct_list(q)
        if fb:
            sql, why = fb
            try:
                with pyodbc.connect(CNXN_STR, timeout=QUERY_TIMEOUT_SECS) as cn:
                    df = pd.read_sql(sql, cn)
            except Exception as e:
                return AskResp(answer="The database rejected the fallback query.",
                               row_count=0, preview=None, note=f"DB error: {e}", sql=sql)
            row_count = int(df.shape[0])
            rows = df.head(min(20, ROW_LIMIT)).to_dict(orient="records")
            return AskResp(
                answer=f"{why}. Returned {row_count} rows.",
                row_count=row_count, preview=rows,
                note=f"Showing up to {min(20, ROW_LIMIT)} preview rows.",
                sql=sql
            )
        # Otherwise guide the user
        some = ", ".join(table_names()[:10])
        return AskResp(
            answer=("Your question is a bit broad. Please specify table and a filter/date range "
                    "or say precisely what list you want (e.g., 'student names in district X')."),
            row_count=0, preview=None, note=f"Try tables like → {some}", sql=None
        )

    # 1b) Inspect & enforce safety
    sql = enforce_top(sql)
    if not is_safe_select(sql):
        fb = guess_distinct_list(q)
        if fb:
            sql, why = fb
            try:
                with pyodbc.connect(CNXN_STR, timeout=QUERY_TIMEOUT_SECS) as cn:
                    df = pd.read_sql(sql, cn)
            except Exception as e:
                return AskResp(answer="The database rejected the fallback query.",
                               row_count=0, preview=None, note=f"DB error: {e}", sql=sql)
            row_count = int(df.shape[0])
            rows = df.head(min(20, ROW_LIMIT)).to_dict(orient="records")
            return AskResp(
                answer=f"{why}. Returned {row_count} rows.",
                row_count=row_count, preview=rows,
                note=f"Showing up to {min(20, ROW_LIMIT)} preview rows.",
                sql=sql
            )
        return AskResp(
            answer=f"I blocked an unsafe query for: {q}. Please specify a clearer table/date filter.",
            row_count=0, preview=None,
            note="Inspector enforces SELECT-only with a single statement.",
            sql=sql
        )

    # 2) Execute model SQL
    try:
        with pyodbc.connect(CNXN_STR, timeout=QUERY_TIMEOUT_SECS) as cn:
            df = pd.read_sql(sql, cn)
    except Exception as e:
        return AskResp(
            answer=("I tried to answer but the database rejected the query. "
                    "Please narrow the time range or specify columns/tables."),
            row_count=0, preview=None, note=f"DB error: {e}", sql=sql
        )

    row_count = int(df.shape[0])
    preview_rows = df.head(min(20, ROW_LIMIT)).to_dict(orient="records")

    # 3) Summarize
    rows_json = json.dumps(preview_rows, ensure_ascii=False)[:8000]
    try:
        answer = cohere_chat(
            f"""You are a careful analyst. Summarize clearly (3–6 sentences).
Question: {q}
Rows (JSON preview): {rows_json}
total_rows={row_count}
Return only prose."""
        ).strip()
    except Exception as e:
        raise HTTPException(500, f"Cohere error (summary): {e}")

    return AskResp(
        answer=answer,
        row_count=row_count,
        preview=preview_rows if row_count <= 200 else None,
        note=f"Showing up to {min(20, ROW_LIMIT)} preview rows.",
        sql=sql
    )
