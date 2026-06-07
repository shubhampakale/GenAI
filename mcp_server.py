import os
from fastmcp import FastMCP
import psycopg2
import json

mcp = FastMCP("DSA-DB")

@mcp.tool()
def query(sql: str) -> str:
    """Execute SQL on DSA database"""
    conn = psycopg2.connect(
        host="localhost", port=5432, dbname="dsa_db",
        user="postgres", password="admin"
    )
    try:
        cur = conn.cursor()
        cur.execute(sql)
        if cur.description:
            columns = [d[0] for d in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
            return json.dumps(results, indent=2)
        conn.commit()
        return "Query OK"
    finally:
        conn.close()

if __name__ == "__main__":
    print("DSA MCP Server Started!")
    mcp.run()