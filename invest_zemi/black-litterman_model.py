import yfinance as yf
import numpy as np
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 準備
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# 銘柄コードの入力
tickers = input("銘柄コードをカンマ区切りで入力してください（例: 7203,9432,9984）: ").split(',')
tickers = [ticker + '.T' for ticker in tickers]

# 日付の設定
end_date = datetime.now()
start_date = datetime(2020, 1, 1)

# データの取得
stock_data = get_stock_data(tickers, start_date, end_date)

# データ取得の確認
print("\nデータ取得の確認:")
print(f"取得期間: {start_date.date()} から {end_date.date()}")
print(f"取得銘柄数: {len(stock_data.columns)}")
print(f"データポイント数: {len(stock_data)}")

# 各銘柄のデータ欠損をチェック
missing_data = stock_data.isnull().sum()
print("\n各銘柄の欠損データ数:")
print(missing_data)

if missing_data.sum() > 0:
    print("\n警告: 一部の銘柄にデータの欠損があります。")
else:
    print("\nすべての銘柄のデータが正常に取得されました。")

# データの先頭と末尾を表示
print("\nデータの先頭5行:")
print(stock_data.head())
print("\nデータの末尾5行:")
print(stock_data.tail())

# 市場データの準備
returns = stock_data.pct_change().dropna()
market_cap = stock_data.iloc[-1] * stock_data.iloc[-1]  # 簡易的な時価総額の計算
weights = market_cap / market_cap.sum()
correlation_matrix = returns.corr()
volatility = returns.std() * np.sqrt(252)  # 年間ボラティリティ
cov_matrix = np.diag(volatility) @ correlation_matrix @ np.diag(volatility)

# パラメータの設定
risk_aversion = 2.5
tau = 0.05

# Reverse Optimization
pi = risk_aversion * np.dot(cov_matrix, weights)

# 投資家のViewの設定
P = np.eye(len(tickers))  # 各銘柄に対する個別のView
Q = np.array([0.15, 0.10, 0.12] + [0.08] * (len(tickers) - 3))  # 期待リターン
omega = np.diag([0.05] * len(tickers))  # Viewの不確実性

# 新たな期待リターンの計算
temp = np.linalg.inv(np.dot(tau * P, np.dot(cov_matrix, P.T)) + omega)
new_return = pi + np.dot(np.dot(tau * cov_matrix, np.dot(P.T, temp)), (Q - np.dot(P, pi)))

# 新ポートフォリオの計算
new_weights = np.linalg.inv(risk_aversion * cov_matrix) @ new_return

# プロット
plt.figure(figsize=(12, 6))
plt.bar(range(len(tickers)), weights, alpha=0.5, label='元のウェイト')
plt.bar(range(len(tickers)), new_weights, alpha=0.5, label='新しいウェイト')
plt.xticks(range(len(tickers)), [t.replace('.T', '') for t in tickers], rotation=45)
plt.ylabel('ウェイト')
plt.title('ポートフォリオウェイトの比較')
plt.legend()
plt.tight_layout()
plt.show()

# 結果の表示
print("均衡期待リターン:")
for ticker, ret in zip(tickers, pi):
    print(f"{ticker}: {ret:.4f}")

print("\n新たな期待リターン:")
for ticker, ret in zip(tickers, new_return):
    print(f"{ticker}: {ret:.4f}")
