# 🚢 Shipping Operations: MCP Server + Live Fleet Tracker

A shipping-operations AI project with two parts:
1. **An MCP server** that gives an AI assistant 5 safe tools to answer questions about shipments
2. **A live fleet tracker**: ships sailing real sea lanes on a world map, each one checked by a delay-risk model

> ⚠️ **Demo project only.** All ships, shipments and customers are made-up data.

## 🗺️ Live Fleet Tracker (`app.py`)
- About 80 ships at sea at any moment, moving between 15 world ports
- Ships follow **real sea lanes** (Suez Canal, Strait of Malacca, Panama Canal, Cape of Good Hope), found with a shortest-path search over sea waypoints
- Every ship is coloured by status: 🟢 on schedule · 🟡 high delay risk · 🔴 running late
- Live counters plus a **"ships needing attention"** panel with the delay reason
- Auto-refreshing map (Streamlit fragment), play/pause and speed control

## 🔌 MCP Server (`server.py`)
MCP (Model Context Protocol) is the standard way to connect AI assistants to company data and tools.

| Tool | What it does |
|---|---|
| `run_sql` | Answers any question with a **read-only** SQL query |
| `delayed_shipments` | Lists delayed shipments and the reason, optionally for one port |
| `port_performance` | Delay scorecard for every port |
| `delay_risk` | Predicts if a shipment will be late, with the reasons |
| `save_report` | Writes a daily operations report to a file |

Example questions for the AI: *"Which port causes the most delays?"* · *"Will SH-1002 arrive on time?"* · *"Write today's operations report."*

## 🔒 Safety by design
- **Read-only:** the database is opened in read-only mode, and non-SELECT queries are blocked
- **Data masking:** customer names are masked (`A*****`) before the AI sees them
- **Audit log:** every tool call is recorded in `audit_log.csv` (time, tool, request)
- **Safe queries:** parameterised SQL (`?`) and a 50-row limit on results

## 📈 Delay-risk model
Explainable by design: risk = the average historical late-rate of the **arrival port**, the **vessel** and the **cargo type**. Every prediction comes with its reasons, e.g. *"Singapore port late 75% · vessel late 50% · electronics late 32% → 52% HIGH"*.

## ▶️ Run it
```bash
pip install -r requirements.txt
python build_database.py      # creates the demo database
streamlit run app.py          # live fleet tracker
python server.py              # MCP server (connect from any MCP client)
```

## Tech stack
Python · MCP (Model Context Protocol) · SQLite · pandas · Plotly · Streamlit
