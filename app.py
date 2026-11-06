"""Live Fleet Tracker: demo ships sailing between world ports (made-up data only)."""
import heapq
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB = Path(__file__).parent / "shipping_demo.db"

# ---------- Map points ----------
PORTS = {
    "Jebel Ali": (25.01, 55.06), "Singapore": (1.26, 103.84), "Rotterdam": (51.95, 4.14),
    "Shanghai": (31.23, 121.90), "Mumbai": (18.95, 72.84), "Hamburg": (53.54, 9.97),
    "Los Angeles": (33.73, -118.26), "Busan": (35.10, 129.04), "Antwerp": (51.26, 4.40),
    "Colombo": (6.95, 79.84), "Jeddah": (21.48, 39.17), "Salalah": (16.94, 54.00),
    "Hong Kong": (22.29, 114.17), "New York": (40.67, -74.05), "Durban": (-29.87, 31.03),
}

# Sea waypoints, so ships follow real sea lanes instead of sailing over land
SEA = {
    "Hormuz": (26.3, 56.6), "Gulf of Oman": (24.5, 59.0), "Arabian Sea": (16.0, 62.0),
    "Horn of Africa": (12.0, 52.5), "Gulf of Aden": (12.5, 47.0), "Bab el-Mandeb": (12.6, 43.4),
    "North Red Sea": (27.0, 34.5), "Suez": (30.5, 32.4), "East Med": (33.5, 28.0),
    "Central Med": (36.5, 15.0), "Gibraltar": (36.0, -5.8), "Portugal": (39.0, -10.5),
    "Biscay": (46.0, -7.5), "English Channel": (50.0, -2.0), "Dover": (51.2, 1.8),
    "North Sea": (53.8, 6.5), "North Atlantic": (45.0, -40.0), "West Atlantic": (30.0, -72.0),
    "Mid Atlantic": (5.0, -25.0), "South Atlantic": (-20.0, 0.0), "Cape of Good Hope": (-36.0, 20.0),
    "South of Madagascar": (-28.0, 47.0), "Somalia Coast": (0.0, 52.0), "Mozambique Channel": (-18.0, 41.0),
    "South India": (8.0, 75.5), "South of Sri Lanka": (5.5, 80.5), "North Malacca": (6.0, 96.5),
    "South Malacca": (2.5, 101.0), "South China Sea S": (5.0, 107.0), "South China Sea N": (15.0, 113.0),
    "Taiwan Strait": (24.0, 119.5), "East China Sea": (29.0, 123.5), "Korea Strait": (33.0, 128.0),
    "South of Japan": (30.0, 134.0), "West Pacific": (35.0, 165.0), "East Pacific": (35.0, -150.0),
    "Mexico Pacific": (15.0, -100.0), "Panama": (8.5, -79.8), "Caribbean": (13.0, -78.0),
    "Windward Passage": (20.0, -73.8),
}
POINTS = {**PORTS, **SEA}

SEA_LANES = [
    ("Jebel Ali", "Hormuz"), ("Hormuz", "Gulf of Oman"), ("Gulf of Oman", "Arabian Sea"),
    ("Mumbai", "Arabian Sea"), ("Salalah", "Arabian Sea"), ("Salalah", "Gulf of Aden"),
    ("Arabian Sea", "Gulf of Aden"), ("Arabian Sea", "Horn of Africa"), ("Horn of Africa", "Gulf of Aden"),
    ("Gulf of Aden", "Bab el-Mandeb"), ("Bab el-Mandeb", "Jeddah"), ("Jeddah", "North Red Sea"),
    ("North Red Sea", "Suez"), ("Suez", "East Med"), ("East Med", "Central Med"),
    ("Central Med", "Gibraltar"), ("Gibraltar", "Portugal"), ("Portugal", "Biscay"),
    ("Biscay", "English Channel"), ("English Channel", "Dover"), ("Dover", "Rotterdam"),
    ("Dover", "Antwerp"), ("Rotterdam", "North Sea"), ("North Sea", "Hamburg"),
    ("English Channel", "North Atlantic"), ("Portugal", "North Atlantic"), ("North Atlantic", "New York"),
    ("New York", "West Atlantic"), ("West Atlantic", "Windward Passage"), ("Windward Passage", "Caribbean"),
    ("Caribbean", "Panama"), ("Panama", "Mexico Pacific"), ("Mexico Pacific", "Los Angeles"),
    ("Los Angeles", "East Pacific"), ("East Pacific", "West Pacific"), ("West Pacific", "South of Japan"),
    ("South of Japan", "East China Sea"), ("East China Sea", "Shanghai"), ("East China Sea", "Korea Strait"),
    ("Korea Strait", "Busan"), ("East China Sea", "Taiwan Strait"), ("Taiwan Strait", "Hong Kong"),
    ("Hong Kong", "South China Sea N"), ("South China Sea N", "South China Sea S"),
    ("South China Sea S", "Singapore"), ("Singapore", "South Malacca"), ("South Malacca", "North Malacca"),
    ("North Malacca", "South of Sri Lanka"), ("Colombo", "South of Sri Lanka"), ("Colombo", "South India"),
    ("Mumbai", "South India"), ("South India", "South of Sri Lanka"), ("Arabian Sea", "South of Sri Lanka"),
    ("South of Sri Lanka", "South of Madagascar"), ("South of Madagascar", "Durban"),
    ("Durban", "Mozambique Channel"), ("Mozambique Channel", "Somalia Coast"), ("Somalia Coast", "Horn of Africa"),
    ("Durban", "Cape of Good Hope"), ("Cape of Good Hope", "South Atlantic"), ("South Atlantic", "Mid Atlantic"),
    ("Mid Atlantic", "Portugal"), ("Mid Atlantic", "West Atlantic"),
]


def to_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def km(a, b):
    """Distance in km between two map points (a, b are names)."""
    dot = np.clip(to_xyz(*POINTS[a]) @ to_xyz(*POINTS[b]), -1, 1)
    return 6371 * np.arccos(dot)


def between(a, b, f):
    """The point a fraction f of the way from a to b along the curve of the Earth."""
    p, q = to_xyz(*POINTS[a]), to_xyz(*POINTS[b])
    angle = np.arccos(np.clip(p @ q, -1, 1))
    if angle < 1e-9:
        return POINTS[a]
    v = (np.sin((1 - f) * angle) * p + np.sin(f * angle) * q) / np.sin(angle)
    return np.degrees(np.arcsin(v[2])), np.degrees(np.arctan2(v[1], v[0]))


@st.cache_data
def sea_route(start, end):
    """Shortest sea route between two ports (Dijkstra over the sea lanes)."""
    neighbours = {name: [] for name in POINTS}
    for a, b in SEA_LANES:
        neighbours[a].append(b)
        neighbours[b].append(a)
    queue, seen = [(0.0, start, [start])], set()
    while queue:
        dist, here, path = heapq.heappop(queue)
        if here == end:
            return path
        if here in seen:
            continue
        seen.add(here)
        for nxt in neighbours[here]:
            if nxt not in seen:
                heapq.heappush(queue, (dist + km(here, nxt), nxt, path + [nxt]))
    return [start, end]


def position_on_route(route, f):
    """Where a ship is after completing fraction f (0 to 1) of its route."""
    legs = [km(a, b) for a, b in zip(route, route[1:])]
    target = f * sum(legs)
    for (a, b), leg in zip(zip(route, route[1:]), legs):
        if target <= leg:
            return between(a, b, target / leg if leg else 0)
        target -= leg
    return POINTS[route[-1]]


def route_line(route, steps=12):
    """Many small points along a route, for drawing a smooth sea lane."""
    lats, lons = [], []
    for a, b in zip(route, route[1:]):
        for f in np.linspace(0, 1, steps):
            lat, lon = between(a, b, f)
            lats.append(lat)
            lons.append(lon)
    return lats, lons


# ---------- Data ----------
@st.cache_data
def load_shipments():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql("SELECT * FROM shipments", conn)
    conn.close()
    df["departure"] = pd.to_datetime(df["departure"])
    df["planned_arrival"] = pd.to_datetime(df["planned_arrival"])
    df["actual_arrival"] = df["planned_arrival"] + pd.to_timedelta(df["delay_days"], unit="D")

    # Delay risk: same idea as the delay_risk MCP tool (history of port, vessel, cargo)
    history = df[df["status"] == "Delivered"].assign(late=lambda d: d["delay_days"] > 0)
    overall = history["late"].mean()
    rates = [df[col].map(history.groupby(col)["late"].mean()).fillna(overall)
             for col in ["to_port", "vessel", "cargo"]]
    df["risk"] = sum(rates) / 3
    return df


ships = load_shipments()
START, END = pd.Timestamp("2026-09-15"), pd.Timestamp("2026-11-15")  # the simulation loops between these dates
SPEEDS = {"Slow": 3, "Normal": 8, "Fast": 16}  # simulated hours per update
COLORS = {"On schedule": "#2ecc71", "High risk": "#f5b041", "Delayed": "#eb4034"}

if "sim_time" not in st.session_state:
    st.session_state.sim_time = pd.Timestamp("2026-10-15 06:00")


def ships_at_sea(t):
    df = ships[(ships["departure"] <= t) & (t < ships["actual_arrival"])].copy()
    df["progress"] = (t - df["departure"]) / (df["actual_arrival"] - df["departure"])
    df["state"] = np.where(t > df["planned_arrival"], "Delayed",
                           np.where(df["risk"] >= 0.40, "High risk", "On schedule"))
    df["days_late"] = ((t - df["planned_arrival"]).dt.total_seconds() / 86400).clip(lower=0).round(1)
    df["route"] = [sea_route(a, b) for a, b in zip(df["from_port"], df["to_port"])]
    positions = [position_on_route(r, f) for r, f in zip(df["route"], df["progress"])]
    df["lat"] = [p[0] for p in positions]
    df["lon"] = [p[1] for p in positions]
    return df


def stat_card(label, value, color):
    return (f"<div style='background:#111a33;border-radius:12px;padding:10px 16px'>"
            f"<div style='color:#8c9bb9;font-size:0.85rem'>{label}</div>"
            f"<div style='color:{color};font-size:1.7rem;font-weight:700'>{value}</div></div>")


# ---------- Page ----------
st.set_page_config(page_title="Live Fleet Tracker", page_icon="🚢", layout="wide")
st.title("🚢 Live Fleet Tracker")
st.caption("A demo shipping fleet sailing real sea lanes between 15 world ports. Every ship is checked by the "
           "delay-risk model: 🟢 on schedule · 🟡 high delay risk · 🔴 running late.")
st.warning("Demo project: all ships, shipments and customers are made-up data.", icon="⚠️")


def live_view():
    if st.session_state.get("live_on"):
        st.session_state.sim_time += pd.Timedelta(hours=SPEEDS[st.session_state.get("live_speed", "Normal")])
        if st.session_state.sim_time > END:
            st.session_state.sim_time = START
    t = st.session_state.sim_time
    sea = ships_at_sea(t)
    arrived = ships[ships["actual_arrival"] <= t]

    st.markdown(f"#### 📅 {t:%d %b %Y · %H:%M}")
    cards = [("Ships at sea", len(sea), "#e1e8f5"),
             ("On schedule", int((sea["state"] == "On schedule").sum()), COLORS["On schedule"]),
             ("High delay risk", int((sea["state"] == "High risk").sum()), COLORS["High risk"]),
             ("Running late", int((sea["state"] == "Delayed").sum()), COLORS["Delayed"]),
             ("Delivered", len(arrived), "#e1e8f5")]
    for col, (label, value, color) in zip(st.columns(5), cards):
        col.markdown(stat_card(label, f"{value:,}", color), unsafe_allow_html=True)

    left, right = st.columns([2.3, 1])
    with left:
        fig = go.Figure()
        lane_lat, lane_lon = [], []
        for route in {tuple(r) for r in sea["route"]}:
            lats, lons = route_line(route)
            lane_lat += lats + [None]
            lane_lon += lons + [None]
        fig.add_trace(go.Scattergeo(lat=lane_lat, lon=lane_lon, mode="lines", hoverinfo="skip",
                                    line=dict(width=1, color="rgba(120,160,230,0.25)"), showlegend=False))
        fig.add_trace(go.Scattergeo(
            lat=[p[0] for p in PORTS.values()], lon=[p[1] for p in PORTS.values()], text=list(PORTS),
            mode="markers+text", textposition="top center", name="Port", hoverinfo="text",
            textfont=dict(color="#8c9bb9", size=10), marker=dict(size=6, color="#e1e8f5", symbol="square")))
        for state, size in [("On schedule", 8), ("High risk", 10), ("Delayed", 12)]:
            d = sea[sea["state"] == state]
            fig.add_trace(go.Scattergeo(
                lat=d["lat"], lon=d["lon"], mode="markers", name=state,
                marker=dict(size=size, color=COLORS[state], opacity=0.9,
                            line=dict(width=1 if state == "Delayed" else 0, color="white")),
                text=[f"{r.shipment_id} · {r.vessel}<br>{r.from_port} → {r.to_port} · {r.cargo}<br>"
                      f"{r.progress:.0%} of voyage · delay risk {r.risk:.0%}" for r in d.itertuples()],
                hoverinfo="text"))
        fig.update_geos(projection_type="natural earth", showland=True, landcolor="#1c2a4a",
                        showocean=True, oceancolor="#0b1224", showcountries=True, countrycolor="#33476e",
                        showcoastlines=True, coastlinecolor="#33476e", showframe=False, bgcolor="#0b1224",
                        lataxis_range=[-45, 65])
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0b1224",
                          legend=dict(orientation="h", x=0.01, y=0.02, font=dict(color="#e1e8f5"),
                                      bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, key="fleet_map", config={"displayModeBar": False})

    with right:
        st.markdown("<div style='font-weight:700;font-size:1.1rem;margin-bottom:6px'>🚨 Ships needing attention</div>",
                    unsafe_allow_html=True)
        alerts = pd.concat([sea[sea["state"] == "Delayed"].sort_values("days_late", ascending=False),
                            sea[sea["state"] == "High risk"].sort_values("risk", ascending=False)]).head(5)
        for _, a in alerts.iterrows():
            color = COLORS[a["state"]]
            if a["state"] == "Delayed":
                label, detail = "RUNNING LATE", f"{a['days_late']:.1f} days late · {a['delay_reason']}"
            else:
                label, detail = "HIGH RISK", f"delay risk {a['risk']:.0%} · {a['progress']:.0%} of voyage"
            st.markdown(
                f"<div style='border-left:6px solid {color};background:#111a33;color:#e1e8f5;padding:8px 12px;"
                f"border-radius:8px;margin-bottom:8px'>"
                f"<span style='color:{color};font-weight:700;font-size:0.8rem'>{label}</span>"
                f"<span style='color:#8c9bb9;font-size:0.8rem'> · {a['shipment_id']}</span><br>"
                f"<b>{a['vessel']} · {a['from_port']} → {a['to_port']}</b><br>"
                f"<span style='color:#8c9bb9;font-size:0.8rem'>{detail}</span></div>",
                unsafe_allow_html=True)


c1, c2, c3 = st.columns([1.2, 1.5, 1])
c1.toggle("▶️ Live tracking", key="live_on")
c2.select_slider("Speed", options=list(SPEEDS), value="Normal", key="live_speed")
if c3.button("🔄 Reset"):
    st.session_state.sim_time = pd.Timestamp("2026-10-15 06:00")
st.fragment(run_every=1.5 if st.session_state.get("live_on") else None)(live_view)()
