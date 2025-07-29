import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# 1. 读取数据（示例 Excel 文件名：demo_user_data.xlsx）
df = pd.read_excel('demo_user_data.xlsx')

# 查看基本信息
print(df.info())

# 2. 清洗与转换字段
# 统一时间字段格式
df['注册时间'] = pd.to_datetime(df['注册时间'], errors='coerce')
df['最后活跃时间'] = pd.to_datetime(df['最后活跃时间'], errors='coerce')

# 去除缺失、重复
df = df.dropna(subset=['用户ID', '注册时间'])
df = df.drop_duplicates(subset=['用户ID'])

# 3. 构造分析字段
df['活跃天数'] = (df['最后活跃时间'] - df['注册时间']).dt.days
df['注册月'] = df['注册时间'].dt.to_period('M').astype(str)
df['是否参与'] = df['活动参与'].map({'是': 1, '否': 0})

# 4. 分组聚合指标
summary = df.groupby('注册月').agg({
    '用户ID': 'count',
    '是否参与': 'mean',
    '消费金额': 'mean',
    '活跃天数': 'mean'
}).reset_index()

summary.rename(columns={
    '用户ID': '注册用户数',
    '是否参与': '活动参与率',
    '消费金额': '平均消费',
    '活跃天数': '平均活跃天数'
}, inplace=True)

# 5. 导出 Excel 报表
summary.to_excel('分析结果报告.xlsx', index=False)
print("✅ 报表已保存为：分析结果报告.xlsx")

# 6. 可视化趋势图
plt.figure(figsize=(10, 5))
plt.plot(summary['注册月'], summary['注册用户数'], marker='o', label='注册用户数')
plt.plot(summary['注册月'], summary['活动参与率'], marker='s', label='活动参与率')
plt.xticks(rotation=45)
plt.title('Trend of Registered Users and Activity Participation Rate')
plt.xlabel('Registration Month')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
# plt.show()  # 显示图表（可选）
plt.savefig('趋势图.png')
print("✅ 图表已保存为：趋势图.png")

# 可视化柱状图
summary.plot(x='Registration Month', y='Average consumption', kind='bar', legend=False, title='Average monthly consumption')
plt.tight_layout()
# plt.show()  # 显示图表（可选）
plt.savefig('各月平均消费.png', dpi=300, bbox_inches='tight', transparent=False)
print("✅ 图表已保存为：各月平均消费.png")
