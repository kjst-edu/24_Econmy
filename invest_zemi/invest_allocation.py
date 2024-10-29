import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime

def get_optimal_portfolio(tickers):
    # 銘柄名に.Tを追加
    tickers = [ticker + '.T' for ticker in tickers]

    # 現在の日付を取得
    end_date = datetime.now().strftime('%Y-%m-%d')

    # 銘柄のデータを取得（2020年1月1日から現在まで）
    data = yf.download(tickers, start='2020-01-01', end=end_date)['Adj Close']

    # データが取得できたか確認
    if data.isnull().all().any():
        print("一部の銘柄のデータが取得できませんでした。以下の銘柄を確認してください:")
        for ticker in tickers:
            if data[ticker].isnull().all():
                print(ticker)
        return None, None  # データ取得に失敗した場合はNoneを返す

    # リターンの計算
    returns = data.pct_change().dropna()

    # 各銘柄の期待リターンと共分散行列を計算
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # 最適化関数の定義
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # 初期重み
    num_assets = len(tickers)
    initial_weights = np.array(num_assets * [1. / num_assets])

    # 重みの合計が1になる制約
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # 各重みが0以上である制約
    bounds = tuple((0, 1) for asset in range(num_assets))

    # 最適化を実行
    optimal_solution = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # 最適な重みを返す
    optimal_weights = optimal_solution.x

    # ポートフォリオのリスクを計算
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

    return optimal_weights, portfolio_risk

# 銘柄をコードで入力
tickers_input = input("投資する銘柄コードをカンマで区切って入力してください（例: 7203, 6758, 9984）: ")
tickers = [ticker.strip() for ticker in tickers_input.split(",")]

# 最適な配分を計算
optimal_weights, portfolio_risk = get_optimal_portfolio(tickers)

if optimal_weights is not None:
    print("最適な配分（%）:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight * 100:.2f}%")  # パーセンテージ形式で表示
    print(f"ポートフォリオのリスク（標準偏差）: {portfolio_risk:.4f}")
