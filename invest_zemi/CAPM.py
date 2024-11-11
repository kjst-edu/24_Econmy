import yfinance as yf
import requests

# FRED APIを使用してリスクフリーレート（米国10年債利回り）を取得
def get_risk_free_rate():
    """
    米国10年債利回りをFRED APIから取得します。
    
    Returns:
    float: リスクフリーレート（年率）
    """
    api_key = "54d31670a9f0e88b7626d33d3601c81f"
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DGS10",  # 米国10年債利回り
        "api_key": api_key,
        "file_type": "json",
        "frequency": "d",  # 日次データ
        "sort_order": "desc",  # 最新データを取得
        "limit": 1  # 最新の1データを取得
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'observations' in data and len(data['observations']) > 0:
        # リスクフリーレートをパーセントから小数に変換
        return float(data['observations'][0]['value']) / 100
    else:
        print("リスクフリーレートのデータ取得に失敗しました。")
        return None

def get_stock_beta(ticker):
    """
    指定された銘柄のβを取得します。
    
    Parameters:
    ticker (str): 銘柄のティッカーシンボル
    
    Returns:
    float: 銘柄のβ
    """
    stock = yf.Ticker(ticker)
    return stock.info.get('beta')

def get_market_risk_premium():
    """
    マーケットリスクプレミアムを設定します。
    ここでは一般的な長期平均として推定値を使用します。
    
    Returns:
    float: マーケットリスクプレミアム
    """
    return 0.05  # 5%として仮定

def calculate_capm(risk_free_rate, beta, market_risk_premium):
    """
    CAPMモデルによる期待収益率を計算します。
    
    Parameters:
    risk_free_rate (float): リスクフリーレート
    beta (float): 銘柄のβ
    market_risk_premium (float): マーケットリスクプレミアム
    
    Returns:
    float: CAPMによる期待収益率
    """
    return risk_free_rate + beta * market_risk_premium

def investment_decision(expected_return, target_return):
    """
    CAPMの期待収益率と目標収益率を比較し、投資判断を行います。
    
    Parameters:
    expected_return (float): CAPMによる期待収益率
    target_return (float): 投資家が求める目標収益率
    
    Returns:
    str: 投資判断の結果
    """
    if expected_return >= target_return:
        return "投資すべきです。期待収益率が目標収益率を上回っています。"
    else:
        return "投資を控えるべきです。期待収益率が目標収益率を下回っています。"

# ユーザー入力
ticker_input = input("銘柄番号を入力してください (例: 7203): ")
ticker = f"{ticker_input}.T"  # 日本の銘柄として「.T」を付加
target_return = float(input("目標収益率を入力してください (例: 0.08 は 8% に相当): "))

# 各パラメータを取得
risk_free_rate = get_risk_free_rate()
beta = get_stock_beta(ticker)
market_risk_premium = get_market_risk_premium()

if risk_free_rate is None or beta is None:
    print("リスクフリーレートまたはβの取得に失敗しました。")
else:
    # CAPMによる期待収益率を計算
    expected_return = calculate_capm(risk_free_rate, beta, market_risk_premium)
    print(f"{ticker} のCAPMによる期待収益率: {expected_return:.2%}")

    # 投資判断
    decision = investment_decision(expected_return, target_return)
    print(decision)
