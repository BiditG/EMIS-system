import json, re
from difflib import get_close_matches

def load_schema_json(path="schema.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_relevant_tables(question: str, schema: dict, k: int = 15) -> list[str]:
    q = question.lower()
    scores = []
    for t, cols in schema.items():
        name_score = sum(tok in t.lower() for tok in re.findall(r"[a-z0-9_]+", q))
        col_score  = sum(c["name"].lower() in q for c in cols)
        scores.append((name_score + col_score, t))
    scores.sort(reverse=True)
    picked = [t for s,t in scores[:k] if s > 0]
    if not picked:  # fallback: fuzzy match table names
        tokens = re.findall(r"[a-z]{3,}", q)
        cands = []
        for tok in tokens:
            cands += get_close_matches(tok, list(schema.keys()), n=3, cutoff=0.5)
        picked = list(dict.fromkeys(cands))[:k] or list(schema.keys())[:k]
    return picked

def format_schema_block(schema: dict, tables: list[str]) -> str:
    lines = []
    for t in tables:
        cols = ", ".join(f"{c['name']}:{c['type']}" for c in schema[t])
        lines.append(f"{t}({cols})")
    return "\n".join(lines)
