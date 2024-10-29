import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# 銘柄のリスクを計算する関数
def calculate_risk_metrics(ticker_symbol, risk_free_rate=0.01):
    # 銘柄コードに「.T」がない場合は追加
    if not ticker_symbol.endswith(".T"):
        ticker_symbol += ".T"

    # 10年前から現在までのデータを取得
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 10)
    data = yf.download(ticker_symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Adj Close']
    
    if data.empty:
        print(f"{ticker_symbol}のデータを取得できませんでした。")
        return

    # 日次リターンを計算
    daily_returns = data.pct_change().dropna()

    # 週次リターンを計算
    weekly_returns = data.resample('W').ffill().pct_change().dropna()

    # 月次リターンを計算
    monthly_returns = data.resample('M').ffill().pct_change().dropna()

    # 年次リターンを計算
    yearly_returns = data.resample('Y').ffill().pct_change().dropna()

    # 10年リターンを計算
    ten_years_returns = data.resample('10Y').ffill().pct_change().dropna()

    # 指標を計算するための関数
    def calculate_metrics(returns, risk_free_rate):
        mean_return = returns.mean().item()
        volatility = returns.std().item()
        if np.isnan(mean_return) or np.isnan(volatility):
            return None, None, None
        sharpe_ratio = (mean_return - risk_free_rate) / volatility
        return mean_return, volatility, sharpe_ratio

    # 各リターンの指標を計算
    daily_metrics = calculate_metrics(daily_returns, risk_free_rate)
    weekly_metrics = calculate_metrics(weekly_returns, risk_free_rate)
    monthly_metrics = calculate_metrics(monthly_returns, risk_free_rate)
    yearly_metrics = calculate_metrics(yearly_returns, risk_free_rate)
    ten_years_metrics = calculate_metrics(ten_years_returns, risk_free_rate)

    # 結果を出力
    print(f"銘柄: {ticker_symbol}")

    # 日次結果
    if daily_metrics[0] is not None:
        print(f"日次 - 平均リターン: {daily_metrics[0]:.4f}, 標準偏差: {daily_metrics[1]:.4f}, シャープレシオ: {daily_metrics[2]:.4f}")

    # 週次結果
    if weekly_metrics[0] is not None:
        print(f"週次 - 平均リターン: {weekly_metrics[0]:.4f}, 標準偏差: {weekly_metrics[1]:.4f}, シャープレシオ: {weekly_metrics[2]:.4f}")

    # 月次結果
    if monthly_metrics[0] is not None:
        print(f"月次 - 平均リターン: {monthly_metrics[0]:.4f}, 標準偏差: {monthly_metrics[1]:.4f}, シャープレシオ: {monthly_metrics[2]:.4f}")

    # 年次結果
    if yearly_metrics[0] is not None:
        print(f"年次 - 平均リターン: {yearly_metrics[0]:.4f}, 標準偏差: {yearly_metrics[1]:.4f}, シャープレシオ: {yearly_metrics[2]:.4f}")

    # 10年次結果
    if ten_years_metrics[0] is not None:
        print(f"10年次 - 平均リターン: {ten_years_metrics[0]:.4f}, 標準偏差: {ten_years_metrics[1]:.4f}, シャープレシオ: {ten_years_metrics[2]:.4f}")

# 使用例
ticker = input("リスクを計算したい銘柄のコードを入力してください（例：7203（トヨタ）など）: ")
calculate_risk_metrics(ticker)
