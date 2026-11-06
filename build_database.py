"""Creates the demo shipping database (made-up data only)."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path(__file__).parent / "shipping_demo.db"
rng = np.random.default_rng(42)

ports = pd.DataFrame({
    "port": ["Jebel Ali", "Singapore", "Rotterdam", "Shanghai", "Mumbai", "Hamburg", "Los Angeles", "Busan",
             "Antwerp", "Colombo", "Jeddah", "Salalah", "Hong Kong", "New York", "Durban"],
    "country": ["UAE", "Singapore", "Netherlands", "China", "India", "Germany", "USA", "South Korea",
                "Belgium", "Sri Lanka", "Saudi Arabia", "Oman", "China", "USA", "South Africa"],
})

vessels = pd.DataFrame({
    "vessel": ["Ocean Star", "Blue Horizon", "Sea Falcon", "Golden Wave", "Silver Tide", "Coral Queen",
               "Desert Pearl", "Northern Light", "Atlas Carrier", "Pacific Dream", "Red Sea Runner", "Iron Gull",
               "Emerald Bay", "Storm Petrel", "Sun Voyager", "Arctic Fox", "Harbor King", "Jade Express",
               "Wind Dancer", "Neptune Pride"],
    "type": rng.choice(["Container", "Bulk", "Tanker"], 20),
    "capacity_tons": rng.integers(20000, 200000, 20),
})

n = 300
from_idx = rng.integers(0, 15, n)
to_idx = (from_idx + rng.integers(1, 15, n)) % 15
departure = pd.Series(pd.Timestamp("2026-09-01") + pd.to_timedelta(rng.integers(0, 65, n), unit="D"))
planned_days = rng.integers(5, 30, n)
delay_reason = rng.choice(["None", "Weather", "Port congestion", "Customs check", "Engine issue"], n,
                          p=[0.65, 0.12, 0.12, 0.07, 0.04])
delay_days = np.where(delay_reason == "None", 0, rng.integers(1, 10, n))
planned_arrival = departure + pd.to_timedelta(planned_days, unit="D")
actual_arrival = planned_arrival + pd.to_timedelta(delay_days, unit="D")
today = pd.Timestamp("2026-11-06")

shipments = pd.DataFrame({
    "shipment_id": "SH-" + pd.Series(range(1001, 1001 + n)).astype(str),
    "vessel": rng.choice(vessels["vessel"], n),
    "from_port": ports["port"].values[from_idx],
    "to_port": ports["port"].values[to_idx],
    "cargo": rng.choice(["Electronics", "Furniture", "Machinery", "Clothing", "Food", "Auto parts", "Chemicals"], n),
    "customer": rng.choice(["Alpha Traders", "Blue Lake Imports", "Cedar Retail", "Delta Foods", "Eagle Motors",
                            "Fresh Mart", "Green Valley Co", "Horizon Electronics"], n),
    "departure": departure.dt.strftime("%Y-%m-%d"),
    "planned_arrival": planned_arrival.dt.strftime("%Y-%m-%d"),
    "delay_days": delay_days,
    "delay_reason": delay_reason,
    "status": np.where(actual_arrival < today, "Delivered", np.where(delay_days > 0, "Delayed", "On time")),
    "cost_usd": rng.integers(5000, 80000, n),
})

conn = sqlite3.connect(DB)
ports.to_sql("ports", conn, if_exists="replace", index=False)
vessels.to_sql("vessels", conn, if_exists="replace", index=False)
shipments.to_sql("shipments", conn, if_exists="replace", index=False)
conn.close()

print("Database created:", DB)
print("Ports:", len(ports), "| Vessels:", len(vessels), "| Shipments:", len(shipments))
