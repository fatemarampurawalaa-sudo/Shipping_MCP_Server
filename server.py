"""Shipping Operations MCP Server (demo data only).

Gives Claude 5 safe tools to answer questions about a made-up shipping company.
Safety: read-only database, customer names masked, every tool call written to an audit log.
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd
from mcp.server.mcpserver import MCPServer

FOLDER = Path(__file__).parent
DB = FOLDER / "shipping_demo.db"
AUDIT_LOG = FOLDER / "audit_log.csv"
REPORTS = FOLDER / "reports"

server = MCPServer("Shipping Operations (demo)")


# ---------- Helpers (not tools) ----------
def load(sql, params=()):
    """Read data in READ-ONLY mode, so nothing can ever be changed or deleted."""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


CUSTOMERS = load("SELECT DISTINCT customer FROM shipments")["customer"]
MASK = dict(zip(CUSTOMERS, CUSTOMERS.str[0] + "*****"))


def audit(tool, details):
    """Add one row to the audit log: when, which tool, what was asked."""
    entry = pd.DataFrame([{"time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "tool": tool, "details": details}])
    entry.to_csv(AUDIT_LOG, mode="a", header=not AUDIT_LOG.exists(), index=False)


def as_text(df):
    """Mask customer names, keep max 50 rows, return as text for Claude."""
    return df.head(50).replace(MASK).to_string(index=False)


# ---------- Tools Claude can use ----------
@server.tool()
def run_sql(sql: str) -> str:
    """Run a read-only SQL SELECT query on the shipping database.
    Tables: ports(port, country), vessels(vessel, type, capacity_tons),
    shipments(shipment_id, vessel, from_port, to_port, cargo, customer, departure,
    planned_arrival, delay_days, delay_reason, status, cost_usd).
    status is one of: Delivered, Delayed, On time. Customer names are masked."""
    audit("run_sql", sql)
    if not sql.strip().lower().startswith("select"):
        return "Blocked: only SELECT questions are allowed (read-only)."
    return as_text(load(sql))


@server.tool()
def delayed_shipments(port: str = "") -> str:
    """List shipments that are currently delayed, longest delay first.
    Optionally give an arrival port name (e.g. 'Jeddah') to filter."""
    audit("delayed_shipments", port or "all ports")
    sql = ("SELECT shipment_id, vessel, from_port, to_port, cargo, delay_days, delay_reason "
           "FROM shipments WHERE status = 'Delayed'")
    if port:
        return as_text(load(sql + " AND to_port = ? ORDER BY delay_days DESC", (port,)))
    return as_text(load(sql + " ORDER BY delay_days DESC"))


@server.tool()
def port_performance() -> str:
    """Delay scorecard for every arrival port: number of shipments,
    how many had a delay, and average delay in days (worst port first)."""
    audit("port_performance", "all ports")
    return as_text(load("""
        SELECT to_port AS port, COUNT(*) AS shipments,
               SUM(delay_days > 0) AS had_delay, ROUND(AVG(delay_days), 1) AS avg_delay_days
        FROM shipments
        GROUP BY to_port
        ORDER BY avg_delay_days DESC
    """))


@server.tool()
def delay_risk(shipment_id: str) -> str:
    """Predict the delay risk of one shipment (e.g. 'SH-1002') from past delivered shipments.
    Returns a risk %, a level (LOW / MEDIUM / HIGH) and the reasons."""
    audit("delay_risk", shipment_id)
    ship = load("SELECT * FROM shipments WHERE shipment_id = ?", (shipment_id,))
    if ship.empty:
        return f"Shipment {shipment_id} not found."
    ship = ship.iloc[0]

    history = load("SELECT * FROM shipments WHERE status = 'Delivered'")
    history["late"] = history["delay_days"] > 0

    port_rate = history[history["to_port"] == ship["to_port"]]["late"].mean()
    vessel_rate = history[history["vessel"] == ship["vessel"]]["late"].mean()
    cargo_rate = history[history["cargo"] == ship["cargo"]]["late"].mean()

    risk = (port_rate + vessel_rate + cargo_rate) / 3
    level = "HIGH" if risk > 0.40 else "MEDIUM" if risk > 0.30 else "LOW"

    return (f"Shipment {shipment_id}: {ship['vessel']} from {ship['from_port']} to {ship['to_port']} "
            f"({ship['cargo']}), status: {ship['status']}\n"
            f"Delay risk: {risk:.0%} -> {level}\n"
            f"Reasons: {ship['to_port']} port late {port_rate:.0%} of the time | "
            f"vessel {ship['vessel']} late {vessel_rate:.0%} | {ship['cargo']} cargo late {cargo_rate:.0%}")


@server.tool()
def save_report() -> str:
    """Create today's shipping operations report and save it as a file in the reports folder."""
    audit("save_report", "daily operations report")
    status = load("SELECT status, COUNT(*) AS shipments FROM shipments GROUP BY status")
    delays = load("""SELECT shipment_id, vessel, to_port, delay_days, delay_reason FROM shipments
                     WHERE status = 'Delayed' ORDER BY delay_days DESC LIMIT 5""")
    reasons = load("""SELECT delay_reason, COUNT(*) AS shipments FROM shipments
                      WHERE delay_reason != 'None' GROUP BY delay_reason ORDER BY shipments DESC""")

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    report = (f"DAILY SHIPPING OPERATIONS REPORT - {today}\n\n"
              f"1. Shipment status\n{status.to_string(index=False)}\n\n"
              f"2. Top 5 delayed shipments\n{delays.to_string(index=False)}\n\n"
              f"3. Main delay reasons\n{reasons.to_string(index=False)}\n\n"
              f"4. Port performance\n{port_performance()}\n")

    os.makedirs(REPORTS, exist_ok=True)
    file_name = REPORTS / f"operations_report_{today}.txt"
    file_name.write_text(report)
    return f"Report saved: {file_name}\n\n{report}"


if __name__ == "__main__":
    server.run()
