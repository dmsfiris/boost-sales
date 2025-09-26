import sys, pandas as pd
csv = "data/sales.csv"
store_id, item_id = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv)
df = df[(df.store_id.astype(str)==store_id) & (df.item_id.astype(str)==item_id)]
if df.empty:
    print("No rows for that store/item"); sys.exit(1)
q = df["price"].quantile([0.1,0.5,0.9]).round(2)
print({"count": int(len(df)), "p10": float(q.loc[0.1]), "p50": float(q.loc[0.5]), "p90": float(q.loc[0.9])})
