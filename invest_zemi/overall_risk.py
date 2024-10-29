import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_portfolio_risk(tickers):
    # 銘柄名に.Tを追加
    tickers = [ticker + '.T' for ticker in tickers]

    # 10年前の日付と現在の日付を設定
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10)

    # 銘柄のデータを取得
    data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Adj Close']

    # データが取得できたか確認
    if data.isnull().all().any():
        print("一部の銘柄のデータが取得できませんでした。以下の銘柄を確認してください:")
        for ticker in tickers:
            if data[ticker].isnull().all():
                print(ticker)
        return None  # データ取得に失敗した場合はNoneを返す

    # リターンの計算
    returns = data.pct_change().dropna()

    # 各銘柄の標準偏差を計算
    std_devs = returns.std()

    # 相関行列を計算
    correlation_matrix = returns.corr()

    # 各銘柄の重みを均等に設定
    weights = np.array([1 / len(tickers)] * len(tickers))

    # ポートフォリオのリスク計算
    portfolio_variance = np.dot(weights, np.dot(correlation_matrix * np.outer(std_devs, std_devs), weights))
    portfolio_risk = np.sqrt(portfolio_variance)

    return portfolio_risk

# 銘柄をコードで入力
tickers_input = input("投資する銘柄コードをカンマで区切って入力してください（例: 7203, 6758, 9984）: ")
tickers = [ticker.strip() for ticker in tickers_input.split(",")]

# ポートフォリオのリスクを計算
risk = get_portfolio_risk(tickers)

if risk is not None:
    print(f"ポートフォリオのリスク（標準偏差）: {risk:.4f}")
