"""
Synthetic retail data generator.

Generates 12 full months of data ending on the last day of the previous month,
and loads it into PostgreSQL.

PLANTED STORY (so "Why did sales decline?" has a real, findable answer):
  1. Electronics promo: a 15% discount on Electronics ran from PROMO_START until the end
     of two months ago. It also boosted electronics demand. It ended -> electronics
     revenue fell last month.
  2. Store renovation: "Pune Camp" ran at ~20% of normal traffic for 10 days last month
     (renovation) -> it becomes the lowest-selling store last month.
  3. Festive season: Oct-Dec has higher traffic (a mild seasonal pattern).

Usage:
  python scripts/generate_data.py --dry-run   # generate + print summary, no DB needed
  python scripts/generate_data.py             # generate + load into Postgres (wipes tables first)
"""
import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 800

# (store_id, name, city, region, avg orders/day)
STORES = [
    (1, "Mumbai Central",        "Mumbai",    "West",  14),
    (2, "Delhi Connaught",       "Delhi",     "North", 13),
    (3, "Bengaluru Indiranagar", "Bengaluru", "South", 12),
    (4, "Hyderabad Banjara",     "Hyderabad", "South", 10),
    (5, "Pune Camp",             "Pune",      "West",   9),
]

# (name, category, price, cost as fraction of price)
PRODUCTS = [
    ("Wireless Earbuds", "Electronics", 2499, 0.62), ("Bluetooth Speaker", "Electronics", 1999, 0.60),
    ("Smartwatch", "Electronics", 4999, 0.65), ("Power Bank 10000mAh", "Electronics", 1499, 0.58),
    ("USB-C Fast Charger", "Electronics", 899, 0.55), ("Noise-Cancelling Headphones", "Electronics", 5999, 0.66),
    ("Basmati Rice 5kg", "Grocery", 650, 0.80), ("Olive Oil 1L", "Grocery", 799, 0.78),
    ("Masala Tea 500g", "Grocery", 320, 0.70), ("Almonds 500g", "Grocery", 550, 0.75),
    ("Instant Noodles Pack", "Grocery", 160, 0.72), ("Peanut Butter 1kg", "Grocery", 350, 0.70),
    ("Cotton T-Shirt", "Apparel", 599, 0.45), ("Slim Fit Jeans", "Apparel", 1799, 0.50),
    ("Hoodie", "Apparel", 1499, 0.48), ("Running Sneakers", "Apparel", 2999, 0.55),
    ("Cotton Kurta", "Apparel", 1299, 0.47), ("Socks (3-pack)", "Apparel", 299, 0.40),
    ("Cookware Set", "Home", 2499, 0.60), ("Cotton Bedsheet", "Home", 999, 0.50),
    ("LED Desk Lamp", "Home", 699, 0.52), ("Steel Water Bottle", "Home", 449, 0.45),
    ("Storage Box Set", "Home", 799, 0.50), ("Towel Set", "Home", 899, 0.48),
    ("Face Wash", "Beauty", 249, 0.40), ("Moisturizer", "Beauty", 499, 0.42),
    ("Shampoo", "Beauty", 399, 0.45), ("Sunscreen SPF50", "Beauty", 449, 0.42),
    ("Perfume 100ml", "Beauty", 1499, 0.45), ("Lipstick", "Beauty", 599, 0.40),
]

CATEGORIES = ["Electronics", "Grocery", "Apparel", "Home", "Beauty"]
BASE_CATEGORY_WEIGHTS = {"Electronics": 0.15, "Grocery": 0.30, "Apparel": 0.20, "Home": 0.20, "Beauty": 0.15}
SEASONALITY = {10: 1.25, 11: 1.20, 12: 1.10}   # festive season boost; other months = 1.0

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Rohan", "Ananya", "Diya", "Meera", "Kavya", "Priya",
               "Arjun", "Sneha", "Rahul", "Neha", "Karan", "Pooja", "Siddharth", "Riya", "Manish", "Tanvi"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Reddy", "Patil", "Nair", "Gupta", "Mehta", "Kulkarni", "Singh",
              "Joshi", "Rao", "Das", "Khan", "Bose", "Shetty", "Desai", "Menon", "Chopra", "Kapoor"]


def generate():
    rng = np.random.default_rng(SEED)

    today = date.today()
    first_this_month = today.replace(day=1)
    end_date = first_this_month - timedelta(days=1)                       # last day of previous month
    start_date = (pd.Timestamp(first_this_month) - pd.DateOffset(months=12)).date()
    last_month_start = end_date.replace(day=1)

    promo_start = (pd.Timestamp(start_date) + pd.DateOffset(months=3)).date()
    promo_end = last_month_start - timedelta(days=1)                      # promo ends before last month
    reno_start = last_month_start + timedelta(days=7)                     # Pune renovation: 10 days
    reno_end = last_month_start + timedelta(days=16)

    stores = [(sid, name, city, region, start_date - timedelta(days=1000 + sid * 90))
              for sid, name, city, region, _ in STORES]
    store_rate = {name: rate for _, name, _, _, rate in STORES}

    products = [(i + 1, n, c, float(p), round(p * cr, 2)) for i, (n, c, p, cr) in enumerate(PRODUCTS)]
    prods_by_cat = {c: [p for p in products if p[2] == c] for c in CATEGORIES}

    city_list = [s[2] for s in STORES]
    customers = []
    for cid in range(1, N_CUSTOMERS + 1):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        city = str(rng.choice(city_list))
        signup = start_date - timedelta(days=int(rng.integers(1, 365)))
        customers.append((cid, name, city, signup))

    orders, items = [], []
    order_id = item_id = 0
    d = start_date
    while d <= end_date:
        in_promo = promo_start <= d <= promo_end
        w = dict(BASE_CATEGORY_WEIGHTS)
        if in_promo:
            w["Electronics"] *= 2.0
        probs = np.array([w[c] for c in CATEGORIES])
        probs = probs / probs.sum()

        day_factor = (1.3 if d.weekday() >= 5 else 1.0) * SEASONALITY.get(d.month, 1.0)

        for store_id, store_name, *_ in stores:
            rate = store_rate[store_name] * day_factor
            if store_name == "Pune Camp" and reno_start <= d <= reno_end:
                rate *= 0.2
            for _ in range(int(rng.poisson(rate))):
                order_id += 1
                customer_id = int(rng.integers(1, N_CUSTOMERS + 1))
                n_items = min(1 + int(rng.poisson(1.2)), 6)
                total = 0.0
                for cat in rng.choice(CATEGORIES, size=n_items, p=probs):
                    prod = prods_by_cat[cat][int(rng.integers(len(prods_by_cat[cat])))]
                    qty = int(rng.choice([1, 2, 3], p=[0.75, 0.20, 0.05]))
                    disc = 15.0 if (in_promo and cat == "Electronics") else 0.0
                    line_total = round(qty * prod[3] * (1 - disc / 100), 2)
                    item_id += 1
                    items.append((item_id, order_id, prod[0], qty, prod[3], disc, line_total))
                    total += line_total
                orders.append((order_id, customer_id, store_id, d, round(total, 2)))
        d += timedelta(days=1)

    info = dict(start=start_date, end=end_date, promo_start=promo_start, promo_end=promo_end,
                reno_start=reno_start, reno_end=reno_end)
    return stores, products, customers, orders, items, info


def print_summary(stores, products, customers, orders, items, info):
    o = pd.DataFrame(orders, columns=["order_id", "customer_id", "store_id", "order_date", "total_amount"])
    it = pd.DataFrame(items, columns=["item_id", "order_id", "product_id", "qty", "unit_price", "disc", "line_total"])
    cat = {p[0]: p[2] for p in products}
    it["category"] = it["product_id"].map(cat)
    it = it.merge(o[["order_id", "order_date", "store_id"]], on="order_id")
    it["month"] = pd.to_datetime(it["order_date"]).dt.to_period("M")
    o["month"] = pd.to_datetime(o["order_date"]).dt.to_period("M")
    names = {s[0]: s[1] for s in stores}

    print(f"Date range: {info['start']} -> {info['end']}")
    print(f"Electronics promo: {info['promo_start']} -> {info['promo_end']}")
    print(f"Pune renovation:   {info['reno_start']} -> {info['reno_end']}")
    print(f"Rows: {len(stores)} stores, {len(products)} products, {len(customers)} customers, "
          f"{len(orders)} orders, {len(items)} order_items\n")
    print("Monthly revenue:")
    print(o.groupby("month")["total_amount"].sum().round(0).to_string(), "\n")
    print("Monthly Electronics revenue:")
    print(it[it.category == "Electronics"].groupby("month")["line_total"].sum().round(0).to_string(), "\n")
    last = o["month"].max()
    print(f"Revenue by store in {last}:")
    print(o[o.month == last].groupby("store_id")["total_amount"].sum().rename(index=names).sort_values().round(0).to_string())


def load_to_postgres(stores, products, customers, orders, items):
    import psycopg2
    from psycopg2.extras import execute_values
    from dotenv import load_dotenv
    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "retail_bi"),
        user=os.getenv("POSTGRES_ADMIN_USER", "bi_admin"),
        password=os.getenv("POSTGRES_ADMIN_PASSWORD", "bi_admin_pass"),
    )
    with conn, conn.cursor() as cur:
        cur.execute("TRUNCATE order_items, orders, customers, products, stores CASCADE")
        execute_values(cur, "INSERT INTO stores VALUES %s", stores)
        execute_values(cur, "INSERT INTO products VALUES %s", products)
        execute_values(cur, "INSERT INTO customers VALUES %s", customers)
        execute_values(cur, "INSERT INTO orders VALUES %s", orders, page_size=5000)
        execute_values(cur, "INSERT INTO order_items VALUES %s", items, page_size=5000)
    conn.close()
    print("Loaded data into PostgreSQL.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="generate and summarise only; no database needed")
    args = ap.parse_args()

    data = generate()
    print_summary(*data)
    if not args.dry_run:
        load_to_postgres(*data[:5])
