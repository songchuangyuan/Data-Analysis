import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns, random, os
from datetime import datetime, timedelta
from pathlib import Path

# ───── 参数配置 ─────
TODAY, N = datetime(2025, 6, 26), 500
SILENT_DAYS, NEW_DAYS = 30, 14
OUT = Path("data"); OUT.mkdir(parents=True, exist_ok=True)


# ───── 1. 生成示例数据 ─────
def rand_date(a, b):
    return a + timedelta(days=random.randint(0,(b-a).days),

                         seconds=random.randint(0,86400))
def mock_data():
    random.seed(1); np.random.seed(1)
    chn = ["Organic","Douyin","WeChat","Xiaohongshu","Bilibili"]
    rs, re = datetime(2024,1,1), datetime(2025,4,1)
    data=[]
    for i in range(1, N+1):
        reg = rand_date(rs,re)
        last= reg+timedelta(days=random.randint(0,(TODAY-reg).days))
        orders = np.random.poisson(2.5)
        data.append(dict(
            user_id=f"U{i:05d}", register_date=reg, last_active_date=last,
            register_channel=random.choice(chn), total_orders=orders,
            total_revenue=round(np.random.uniform(15,120),2)))
    return pd.DataFrame(data)


# ───── 2. 打标签 ─────
def tag(df):
    df["days_since_last_active"] = (TODAY-df["last_active_date"]).dt.days
    df["days_since_register"] = (TODAY-df["register_date"]).dt.days
    df["is_silent"] = df["days_since_last_active"]>SILENT_DAYS
    df["is_new"] = df["days_since_register"]<NEW_DAYS
    df["is_paid"] = df["total_orders"]>0

    def seg(r):
        if r.is_new:
            return "New-Inactive"
        if r.is_silent and r.is_paid:
            return "Silent-Paid"
        if r.is_silent and not r.is_paid:
            return "Silent-Free"
        return "Recently-Active"
    df["user_segment"] = df.apply(seg, axis=1); return df


# ───── 3. Cohort 留存 ─────
def cohort(df):
    df["reg_month"] = df["register_date"].dt.to_period("M").astype(str)
    df["act_month"] = df["last_active_date"].dt.to_period("M").astype(str)
    ctab = (df.groupby(["reg_month", "act_month"]).size()
            .unstack(fill_value=0).sort_index())
    return ctab.divide(ctab.sum(axis=1), axis=0).round(3)[sorted(ctab)]


# ───── 4. 画图（全英文） ─────
def bar(summary, path):
    plt.figure(figsize=(8, 5))
    summary["Users"].plot(kind="bar", color="#3477b0")
    plt.title("User Segments Distribution"); plt.ylabel("Users"); plt.xlabel("")
    plt.tight_layout(); plt.savefig(path); plt.close()


def heat(ret, path):
    plt.figure(figsize=(12, 6))
    sns.heatmap(ret, annot=True, fmt=".0%", cmap="YlGnBu",
                cbar_kws={"label": "Retention"})
    plt.title("Cohort Retention Heat-map")
    plt.xlabel("Active Month"); plt.ylabel("Registration Month")
    plt.tight_layout(); plt.savefig(path); plt.close()


# ───── 5. 主流程 ─────
if __name__ == "__main__":
    # 生成样例数据
    df = tag(mock_data())
    # 段汇总
    seg_summary = (df.groupby("user_segment").size()
                   .to_frame("Users").sort_values("Users", ascending=False))
    seg_summary["Ratio%"] = (seg_summary["Users"]/N*100).round(2)

    # 热力留存分析
    retention = cohort(df)

    # 导出
    df.to_csv(OUT/"sample_user_data.csv", index=False)
    df[df.is_silent].to_csv(OUT/"silent_users.csv", index=False)
    retention.to_csv(OUT/"retention_matrix.csv")
    # 图
    bar(seg_summary, OUT/"user_category_bar.png")
    heat(retention, OUT/"retention_heatmap.png")
    print("用户分布统计已完成，请查看 'user_category_bar.png'")
    print("Cohort 留存矩阵已生成，请查看 'retention_heatmap.png'")
