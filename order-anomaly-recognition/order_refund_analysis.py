import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns, random, os
from datetime import datetime, timedelta
from pathlib import Path

# ---------- 全局参数 ----------
N_ORDER = 5_000
TODAY = datetime(2025, 6, 26)
OUT = Path("data"); OUT.mkdir(parents=True, exist_ok=True)


# ---------- 1) 构造示例数据 ----------
def rand_date(start, end):
    return start + timedelta(days=random.randint(0, (end-start).days))


def mock_orders():
    random.seed(2); np.random.seed(2)
    chn  = ["Organic", "Douyin", "WeChat", "Bilibili"]
    cats = ["Electronics", "Beauty", "Clothes", "Home"]
    s, e = datetime(2024, 7, 1), datetime(2025, 3, 31)
    rows = []
    for i in range(1, N_ORDER+1):
        odate = rand_date(s, e)
        rows.append(
            dict(order_id=f"O{i:06d}",
                 user_id=f"U{np.random.randint(1,1001):05d}",
                 order_date=odate,
                 order_channel=random.choice(chn),
                 product_category=random.choice(cats),
                 order_amount=round(np.random.uniform(20,300),2))
        )
    return pd.DataFrame(rows)


def mock_refunds(order_df, ratio=0.16):
    subset = order_df.sample(frac=ratio, random_state=3)
    return pd.DataFrame({
        "order_id"    : subset["order_id"],
        "refund_date" : subset["order_date"] + pd.to_timedelta(
            np.random.randint(1,10,len(subset)), unit="d"),
        "refund_amount": subset["order_amount"]
    })


orders = mock_orders()
refunds = mock_refunds(orders)


# ---------- 2) 合并退款标记 ----------
data = orders.merge(refunds[["order_id", "refund_amount"]],
                    on="order_id", how="left")
data["is_refunded"] = data["refund_amount"].notna().astype(int)


# ---------- 3) 按用户聚合 ----------
u_stat = (
    data.groupby("user_id")
        .agg(
        orders_cnt=("order_id",      "nunique"),
        refunds_cnt=("is_refunded",  "sum"),
        total_amt  =("order_amount", "sum"),
        refund_amt =("refund_amount","sum")
    )
        .fillna(0)
)
u_stat["refund_rate"] = (u_stat["refunds_cnt"] / u_stat["orders_cnt"]).round(3)


# ---------- 4) 高退款用户筛选 ----------
high_refund = u_stat[(u_stat.refund_rate >= 0.5) & (u_stat.refunds_cnt >= 3)]
high_refund.to_csv(OUT / "high_refund_users.csv")


# ---------- 5) 渠道 / 品类退款率 ----------
def refund_rate(df, col):
    table = (
        df.groupby(col)
            .agg(orders=("order_id","nunique"), refunds=("is_refunded","sum"))
            .assign(rate=lambda x: (x.refunds / x.orders).round(3))
            .sort_values("rate", ascending=False)
    )
    return table


channel_rate = refund_rate(data, "order_channel")
cate_rate = refund_rate(data, "product_category")

channel_rate.to_csv(OUT/"channel_refund_rate.csv")
cate_rate.to_csv(OUT/"category_refund_rate.csv")

# ---------- 6) 可视化（英文标题） ----------
sns.set_theme(style="whitegrid")      # 可选美化风格

# 6.1 高退款用户柱状图
plt.figure(figsize=(10, 5))
(high_refund.sort_values("refund_rate", ascending=True)
 .tail(15)["refund_rate"]
 .plot(kind="barh", color="#d9534f"))
plt.title("Top Refund-Rate Users")
plt.xlabel("Refund Rate")
plt.tight_layout(); plt.savefig(OUT/"top_refund_users.png"); plt.close()

# 6.2 渠道退款率
plt.figure(figsize=(6, 4))
channel_rate["rate"].plot(kind="bar", color="#5bc0de")
plt.title("Refund Rate by Channel"); plt.ylabel("Rate")
plt.tight_layout(); plt.savefig(OUT/"channel_refund_bar.png"); plt.close()

# 6.3 品类退款率
plt.figure(figsize=(6, 4))
cate_rate["rate"].plot(kind="bar", color="#5bc0de")
plt.title("Refund Rate by Category"); plt.ylabel("Rate")
plt.tight_layout(); plt.savefig(OUT/"category_refund_bar.png"); plt.close()

print("▲ Analysis finished. All files saved under /data")
