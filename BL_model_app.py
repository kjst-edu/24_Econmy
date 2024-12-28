from shiny import reactive, render
from shiny.express import input, ui, module
import re
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

#何を入力しても数字の部分だけ取り出し4桁ごとに区切ってリスト化

def extract_numbers(input_string):
    # 数字のみを抽出
    numeric_string = re.sub(r'\D', '', input_string)
    # 4桁ごとに分割してリスト化
    result = [numeric_string[i:i+4] for i in range(0, len(numeric_string), 4)]
    return result


# 銘柄データ取得関数
def get_stock_data(tickers, start_date, end_date):
    for i in range(len(tickers)):
        ticker = tickers[i]
        stock = yf.download(f'{ticker}.T', start=start_date, end=end_date)['Adj Close']
        if i == 0:
            stocks = stock
        else:
            stocks = pd.concat([stocks, stock], axis=1)
    return stocks

# ブラックリッターマンモデルの計算関数
def black_litterman(stock_data, risk_aversion, tau, P, Q, omega, ticker_objects, market_cap=950e12):
    # 日次リターン計算
    returns = stock_data.astype(float).pct_change().dropna()
    
    weights = []
    for i in range(len(ticker_objects)):
        ticker_object = ticker_objects[i]
        current_price = stock_data.iloc[-1, i]
        # 発行済株式数を取得
        shares_outstanding = ticker_object.info["sharesOutstanding"]

        # 時価総額を計算
        market_cap_camp = current_price * shares_outstanding

        weight = market_cap_camp / market_cap  # 市場ウェイトの計算

        weights.append(weight)
    
    weights = np.array(weights)
    
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




# メイン関数
def main(tickers, risk_aversion, tau, p, q, omega, ticker_objects):
    # 日付の設定
    end_date = datetime.now()
    start_date = datetime(2020, 1, 1)

    tickers = extract_numbers(tickers)

    # データの取得
    stock_data = get_stock_data(tickers, start_date, end_date)

    print(stock_data.head())
    risk_aversion = float(risk_aversion)
    tau = float(tau)

    
    list_omega = np.diag([float(x) for x in omega.split(', ')])
    list_p = [float(x) for x in p.split(', ')]
    list_q = np.array([float(x) for x in q.split(', ')])

    P = np.zeros((len(list_p), len(list_p)))
    for i in range(len(list_p)):
        P[i, i] = list_p[i]
            
    P = np.array(P)
    P = P.reshape(len(list_q), -1)

        
    # ブラックリッターマンモデルの計算
    weights, new_weights, pi, new_return = black_litterman(stock_data, risk_aversion, tau, P, list_q/100, list_omega, ticker_objects)

    return [pi, new_return, weights, new_weights]
    



def get_company_tickers(string):
    ticker_objects = []
    tickers = extract_numbers(string)
    for ticker in tickers:
        ticker_object = yf.Ticker(f'{ticker}.T')
        ticker_objects.append(ticker_object)
    return ticker_objects


company_tickers = reactive.value([])

@reactive.effect
@reactive.event(input.tickers)
def __():
    if input.tickers() != '':
        tickers = get_company_tickers(input.tickers())
        company_tickers.set(tickers)

result_bl = reactive.value([])

@reactive.effect
@reactive.event(input.start_calc)
def _():
    if input.tickers() != '' or input.p() != '' or input.q() != '' or input.omega() != '':
        result = main(input.tickers(), input.risk_aversion(), input.tau(), input.p(), input.q(), input.omega(), company_tickers())
        result_bl.set(result)


# 可視化関数
@reactive.calc
def figpath_pie():
    """
    Creates a pie chart based on portfolio weights.
    
    Parameters:
        weights (np.array): Array of portfolio weights.
        labels (list or None): List of labels for the assets. If None, default labels will be used.
        title (str): Title of the pie chart.
    """
    import japanize_matplotlib
    weights = result_bl()[3]
    labels = [x.info.get('longName', '企業が見つかりません') for x in company_tickers()]
    # Validate input
    if not isinstance(weights, (list, np.ndarray)):
        raise ValueError("weights must be a list or numpy array.")
    
    # Normalize weights to ensure they sum to 1 (if necessary)
    weights = np.maximum(weights, 0)  # Replace negatives with 0
    if weights.sum() != 1.0:
        weights = weights / np.sum(weights)
    
    # Create default labels if not provided
    
    # Create the pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(
        weights,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab20.colors[:len(weights)],
        textprops={'fontsize': 10}
    )
    plt.title('最適ポートフォリオ', fontsize=16, y=1.1)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

    import tempfile
    # Save the plot to a temporary file
    fd, path = tempfile.mkstemp(suffix='.svg')
    plt.savefig(path, format='svg')
    plt.close()  # Close the plot to release resources

    return path


with ui.layout_sidebar(fillable=True):
    with ui.sidebar(open='desktop'):
        ui.input_text_area("tickers", "株式コードを入力して下さい", value='4902, 7203, 9432, 9984', placeholder="0000, 1111, 1234")
        ui.input_slider('risk_aversion', 'リスク回避係数', min=0, max=15, value=2.5, step=0.1)
        ui.input_slider('tau', '非直感的調整係数', min=0, max=1, value=0.05, step=0.01)
        ui.input_text_area('p', 'ビュー（株式コードと同じ順番で入力してください）', value = '-1, 1, 0, 1', placeholder='-1, 0, 1のいずれか（強気なら1、弱気なら-1）')
        ui.input_text_area('q', '期待リターン（株式コードと同じ順番で入力してください、年率、%表記、0以上）', value = '12, 20, 5, 30', placeholder='例15%→15')
        ui.input_text_area('omega', 'ビューの不確実性（株式コードと同じ順番で入力してください、0以上、小さいほど信頼性が高いとみなされる）', value = '0.01, 0.05, 0.1, 0.07', placeholder='0以上、0.01~0.10が一般的')
        ui.input_action_button('start_calc', '計算開始')

    with ui.card(full_screen=True):
        ui.card_header("分析中の企業")
        @render.text
        def company_names():
            if input.tickers() == '':
                return '銘柄コードを入力してください'
            else:
                company_names = [x.info.get('longName', '企業が見つかりません') for x in company_tickers()]
                return ', '.join(company_names)#改行したい

   
    with ui.card(full_screen=True):
        ui.card_header('分析結果')

        @render.text
        def result_text1():
            if result_bl() == ['入力エラー']:
                return '入力エラー'
            else:
                return f'pi : {result_bl()[0][0]}'

        @render.text
        def result_text2():
            if result_bl() == ['入力エラー']:
                return None
            else:
                return f'new return : {result_bl()[1]}'

        @render.text
        def result_text3():
            if result_bl() == ['入力エラー']:
                return None
            else:
                return f'weights : {result_bl()[2]}'

        @render.text
        def result_text4():
            if result_bl() == ['入力エラー']:
                return None
            else:
                return f'new weights : {result_bl()[3]}'

    with ui.card(full_screen=True):
        @render.image
        def image():
            if input.tickers() != '' or input.p() != '' or input.q() != '' or input.omega() != '':
                return {"src": str(figpath_pie()), 
                    "width": "600px", "format":"svg"}
            else:
                return None
      
#stock_dataをreactive.valueに、入力の数が一致しないとき（銘柄コードとp, qなど）、銘柄コードが4桁でないとき、結果の表示（表形式で）




