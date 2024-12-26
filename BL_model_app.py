from shiny import reactive, render
from shiny.express import input, ui, module
import re
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime


#何を入力しても数字の部分だけ取り出し4桁ごとに区切ってリスト化

def extract_numbers(input_string):
    # 数字のみを抽出
    numeric_string = re.sub(r'\D', '', input_string)
    # 4桁ごとに分割してリスト化
    result = [numeric_string[i:i+4] for i in range(0, len(numeric_string), 4)]
    return result


# 銘柄データ取得関数
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# ブラックリッターマンモデルの計算関数
def black_litterman(tickers, stock_data, risk_aversion, tau, P, Q, omega, market_cap=950e12):
    # 日次リターン計算
    returns = stock_data.pct_change().dropna()
    
    stock = yf.Ticker(ticker)

    # 株価の最新データを取得
    current_price = stock.history(period="1d")["Adj Close"].iloc[-1]

    # 発行済株式数を取得
    shares_outstanding = stock.info["sharesOutstanding"]

    # 時価総額を計算
    market_cap_camp = current_price * shares_outstanding

    weights = market_cap_camp / market_cap  # 市場ウェイトの計算
    
    # 相関行列の計算
    correlation_matrix = returns.corr()
    volatility = returns.std() * np.sqrt(252)  # 年間ボラティリティ
    cov_matrix = np.diag(volatility) @ correlation_matrix @ np.diag(volatility)  # 共分散行列
    
    # Reverse Optimizationで均衡ポートフォリオの期待リターンを計算
    pi = risk_aversion * np.dot(cov_matrix, weights)
    
    # 新たな期待リターンの計算
    temp = np.linalg.inv(np.dot(tau * P, np.dot(cov_matrix, P.T)) + omega)
    new_return = pi + np.dot(np.dot(tau * cov_matrix, np.dot(P.T, temp)), (Q - np.dot(P, pi)))
    
    # 新ポートフォリオの計算
    new_weights = np.linalg.inv(risk_aversion * cov_matrix) @ new_return
    
    return weights, new_weights, pi, new_return

# 可視化関数
@reactive.calc
def plot_weights_comparison(tickers, weights, new_weights):
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tickers)), weights, alpha=0.5, label='元のウェイト')
    plt.bar(range(len(tickers)), new_weights, alpha=0.5, label='新しいウェイト')
    plt.xticks(range(len(tickers)), [t.replace('.T', '') for t in tickers], rotation=45)
    plt.ylabel('ウェイト')
    plt.title('ポートフォリオウェイトの比較')
    plt.legend()
    plt.tight_layout()

    # Save the plot to a temporary file
    fd, path = tempfile.mkstemp(suffix='.svg')
    plt.savefig(path, format='svg')
    plt.close()  # Close the plot to release resources

    return path

# メイン関数
def main(tickers, risk_aversion, tau, list_p, list_q, list_omega):
    # 日付の設定
    end_date = datetime.now()
    start_date = datetime(2020, 1, 1)

    # データの取得
    stock_data = get_stock_data(tickers, start_date, end_date)


    # P（投資家のView）
    P = np.array(list_p)
    
    # Q（期待リターン）
    Q = np.array(list_q)
    
    # omega（Viewの不確実性）
    omega = np.diag(list_omega)

    # ブラックリッターマンモデルの計算
    weights, new_weights, pi, new_return = black_litterman(tickers, stock_data, risk_aversion, tau, P, Q, omega)

    return tickers, pi, new_return


    with ui.layout_sidebar(fillable=True):
        with ui.sidebar(open='desktop'):
            ui.input_text_area("tickers", "株式コードを入力して下さい", placeholder="0000, 1111, 1234 （各コードの区切りの形式に指定なし（スペースでもコンマでも何もなしでも可））")
            ui.input_slider('risk_aversion', 'リスク回避係数', min=0, max=15, value=2.5, step=0.1)
            ui.input_slider('tau', '非直感的調整係数', min=0, max=1, value=0.05, step=0.01)
            ui.input_text_area('p', 'ビュー（株式コードと同じ順番で入力してください）', placeholder='-1, 0, 1のいずれか（強気なら1、弱気なら-1）')
            ui.input_text_area('q', '期待リターン（株式コードと同じ順番で入力してください、年率、%表記、0以上）', placeholder='例15%→15')
            ui.input_text_area('omega', 'ビューの不確実性（株式コードと同じ順番で入力してください、0以上、小さいほど信頼性が高いとみなされる）', placeholder='0以上、0.01~0.10が一般的')





    # 結果の表示
    print("\n均衡期待リターン:")
    for ticker, ret in zip(tickers, pi):
        print(f"{ticker}: {ret:.4f}")

    print("\n新たな期待リターン:")
    for ticker, ret in zip(tickers, new_return):
        print(f"{ticker}: {ret:.4f}")

risk_aversion = float(input("リスク回避係数を入力してください（例: 2.5）: "))
tau = float(input("非直感的調整係数を入力してください（例: 0.05）: "))
    
    # P（投資家のView）
P = np.array([list(map(int, input(f"{ticker}のViewを入力（例えば、強気なら1, 弱気なら-1）: ").split())) for ticker in tickers])
    
    # Q（期待リターン）
Q = np.array([float(input(f"{ticker}の期待リターンを入力（例: 0.15）: ")) for ticker in tickers])
    
    # omega（Viewの不確実性）
omega = np.diag([float(input(f"{ticker}のViewの不確実性を入力（例: 0.05）: ")) for ticker in tickers])

"""
銘柄コードをカンマ区切りで入力してください（例: 7203,9432,9984）: 7203, 9432, 9984
リスク回避係数を入力してください（例: 2.5）: 2.5
非直感的調整係数を入力してください（例: 0.05）: 0.05
7203.TのViewを入力（例えば、強気なら1, 弱気なら-1）: 1
9432.TのViewを入力（例えば、強気なら1, 弱気なら-1）: -1
9984.TのViewを入力（例えば、強気なら1, 弱気なら-1）: 0
7203.Tの期待リターンを入力（例: 0.15）: 0.15
9432.Tの期待リターンを入力（例: 0.15）: 0.10
9984.Tの期待リターンを入力（例: 0.15）: 0.12
7203.TのViewの不確実性を入力（例: 0.05）: 0.05
9432.TのViewの不確実性を入力（例: 0.05）: 0.05
9984.TのViewの不確実性を入力（例: 0.05）: 0.05
"""

