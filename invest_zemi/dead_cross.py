import yfinance as yf
import pandas as pd
import time
import matplotlib.pyplot as plt

# デッドクロスを検出する関数
def check_dead_cross(stock_symbol):
    # 過去1年分のデータを取得
    stock_data = yf.download(stock_symbol, period="1y")
    
    # データが取得できているか確認
    if stock_data.empty:
        print(f"{stock_symbol} のデータが見つかりませんでした。")
        return False
    
    # 50日移動平均と200日移動平均を計算
    stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()

    # 最後の2日間のデータを取得
    last_two_days = stock_data.tail(2)

    # 移動平均のデータが足りない場合はエラーを回避
    if len(last_two_days) < 2 or pd.isna(last_two_days['50_MA'].iloc[-1]) or pd.isna(last_two_days['200_MA'].iloc[-1]):
        print(f"{stock_symbol} の移動平均データが不足しています。")
        return False

    # 昨日と今日の移動平均を比較してデッドクロスを検出
    dead_cross = (last_two_days['50_MA'].iloc[-2] > last_two_days['200_MA'].iloc[-2] and 
                  last_two_days['50_MA'].iloc[-1] < last_two_days['200_MA'].iloc[-1])

    if dead_cross:
        print(f"デッドクロス発生: {stock_symbol}")

    return dead_cross

# ユーザーに銘柄コードを入力してもらう
stock_symbol = input("銘柄コードを入力してください（例: AAPL, MSFT, 7974など）: ")


# 銘柄コードに「.T」を追加
if not stock_symbol.endswith(".T"):
    stock_symbol += ".T"

# 定期的にチェックするループ（15分ごとに確認）
while True:
    print(f"{stock_symbol} の株価を分析中...")
    if check_dead_cross(stock_symbol):
        print("通知: デッドクロスが発生しました！")
    
    # グラフを表示
    stock_data = yf.download(stock_symbol, period="1y")
    stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()

    plt.figure(figsize=(12,6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.plot(stock_data['50_MA'], label='50 Day MA')
    plt.plot(stock_data['200_MA'], label='200 Day MA')
    plt.title(f"{stock_symbol} の株価と移動平均")
    plt.legend()
    plt.show()

    # 15分ごとにチェック（900秒）
    time.sleep(900)
