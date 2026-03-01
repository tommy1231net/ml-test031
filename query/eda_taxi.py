import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. ファイルの存在確認（アップロード忘れ防止）
train_file = 'chicago_taxi_train_2022.csv'  # アップロードしたファイル名に合わせて変更してください
if not os.path.exists(train_file):
    print(f"Error: {train_file} not found in current directory.")
    exit()

# 2. 効率的な読み込み
print(f"Loading {train_file}...")
# カテゴリ型を指定してメモリ消費を抑える
dtype_dict = {
    'label': 'category',
    'pickup_area': 'category',
    'dropoff_area': 'category',
    'company_id': 'category'
}

df = pd.read_csv(
    train_file, 
    dtype=dtype_dict, 
    parse_dates=['trip_start_timestamp', 'trip_end_timestamp']
)

# 3. データの基本統計量の確認
print("\n--- Data Information ---")
print(df.info())

print("\n--- Summary Statistics (Numerical) ---")
print(df.describe())

# 4. 【重要】ターゲット（ラベル）の分布確認と保存
print("\nCalculating label distribution...")
label_dist = df['label'].value_counts(normalize=True) * 100
print(label_dist)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
plt.title('Payment Type Distribution (2022 Training Data)')
plt.savefig('label_distribution.png')  # 画面表示の代わりにファイル保存
print("Saved: label_distribution.png")

# 5. 【リーケージチェック】チップ(tips)と支払い方法の関係
print("\n--- Leakage Check: Ratio of trips with tips > 0 per Payment Type ---")
# 支払方法ごとに、チップが0より大きい行の割合を算出
df['has_tip'] = df['tips'] > 0
tip_ratio = df.groupby('label')['has_tip'].mean().sort_values(ascending=False)
print(tip_ratio)

plt.figure(figsize=(10, 6))
tip_ratio.plot(kind='bar', color='skyblue')
plt.title('Ratio of Trips with Tips > 0 by Payment Type')
plt.ylabel('Ratio')
plt.savefig('tip_leakage_check.png')
print("Saved: tip_leakage_check.png")