import os, json
import pyodbc, pandas as pd
from dotenv import load_dotenv



load_dotenv()

def cnxn_str() -> str:
    driver = os.getenv("ODBC_DRIVER", "ODBC Driver 18 for SQL Server")
    server = os.getenv("SQL_SERVER", "localhost")
    port   = os.getenv("SQL_PORT")  # may be None
    db     = os.getenv("SQL_DATABASE", "CEHRDApp")
    trusted = os.getenv("SQL_TRUSTED_CONNECTION","no").lower() in ("yes","true","1")

    # If server is a named instance (has "\"), do NOT append a port.
    server_part = server if ("\\" in server or not port) else f"{server},{port}"

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server_part}",
        f"DATABASE={db}",
    ]
    if trusted:
        parts.append("Trusted_Connection=yes")
    else:
        parts.append(f"UID={os.getenv('SQL_USER')}")
        parts.append(f"PWD={os.getenv('SQL_PASSWORD')}")
    # For local dev with Driver 18
    parts.append("Encrypt=no")
    parts.append("TrustServerCertificate=yes")
    return ";".join(parts) + ";"


def fetch_schema():
    q = """
    SELECT c.TABLE_SCHEMA, c.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS c
    JOIN INFORMATION_SCHEMA.TABLES t
      ON t.TABLE_SCHEMA=c.TABLE_SCHEMA AND t.TABLE_NAME=c.TABLE_NAME
    WHERE t.TABLE_TYPE='BASE TABLE'
    ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION;
    """
    with pyodbc.connect(cnxn_str(), timeout=30) as cn:
        df = pd.read_sql(q, cn)
    schema = {}
    for (schema_name, table), g in df.groupby(["TABLE_SCHEMA","TABLE_NAME"]):
        cols = [{"name": r.COLUMN_NAME, "type": r.DATA_TYPE} for _, r in g.iterrows()]
        schema[f"{schema_name}.{table}"] = cols
    return schema

if __name__ == "__main__":
    schema = fetch_schema()
    with open("schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"Wrote schema.json with {len(schema)} tables.")
