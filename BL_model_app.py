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
        if stock.empty:
            pass
        else:
            if i == 0:
                stocks = stock
            else:
                stocks = pd.concat([stocks, stock], axis=1)
    return stocks

# ブラックリッターマンモデルの計算関数
def black_litterman(stock_data, risk_aversion, tau, P, Q, omega, ticker_objects, market_cap=950e12):
    print(stock_data)
    # 日次リターン計算
    returns = stock_data.astype(float).pct_change().dropna()
    
    weights = []
    for i in range(len(stock_data.columns)):
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
def main(tickers, risk_aversion, tau, p, q, omega, ticker_objects, stock_data):
    risk_aversion = float(risk_aversion)
    tau = float(tau)

    P = np.zeros((len(p), len(p)))
    for i in range(len(p)):
        P[i, i] = p[i]
            
    P = np.array(P)
    P = P.reshape(len(q), -1)

        
    # ブラックリッターマンモデルの計算
    weights, new_weights, pi, new_return = black_litterman(stock_data, risk_aversion, tau, P, q/100, omega, ticker_objects)

    return [pi, new_return, weights, new_weights]
    


def get_company_tickers(string):
    ticker_objects = []
    company_names = []
    error_tickers = []
    tickers = extract_numbers(string)
    for ticker in tickers:
        ticker_object = yf.Ticker(f'{ticker}.T')
        if len(ticker) != 4 or ticker_object.history(period="1d").empty:
            ticker_object = 'ticker_error'
            name = '企業が見つかりません'
            error_tickers.append(ticker)
        else:
            name = ticker_object.info.get('longName', '企業が見つかりません')
        ticker_objects.append(ticker_object)
        company_names.append(name)
    return ticker_objects, company_names, error_tickers


company_tickers = reactive.value([])
@reactive.effect
@reactive.event(input.get_data)
def __():
    if input.tickers() != '':
        tickers, names, error_tickers = get_company_tickers(input.tickers())
        company_tickers.set([tickers, names, error_tickers])

stock_data = reactive.value(pd.DataFrame())
@reactive.effect
@reactive.event(input.get_data)
def _():
    if input.tickers() != '' and input.p() != '' and input.q() != '' and input.omega() != '':
        end_date = datetime.now()
        start_date = datetime(2020, 1, 1)
        tickers_list = extract_numbers(input.tickers())
        tickers = [x for x in tickers_list if x not in company_tickers()[2]]
        stock_data.set(get_stock_data(tickers, start_date, end_date))

input_params = reactive.value([])
@reactive.effect
@reactive.event(input.start_calc)
def ___():
    if input.tickers() != '' and input.p() != '' and input.q() != '' and input.omega() != '':
        try:
            list_omega = np.diag([float(x) for x in input.omega().split(', ')])
            list_p = [float(x) for x in input.p().split(', ')]
            list_q = np.array([float(x) for x in input.q().split(', ')])
            if len(list_omega) == len(list_p) and len(list_p) == len(list_q):
                input_params.set([list_p, list_q, list_omega])
            else:
                input_params.set([])
        except ValueError:
            input_params.set([])

@reactive.event(input.start_calc)
def result_bl(): 
    tickers_list = extract_numbers(input.tickers())
    tickers = [x for x in tickers_list if x not in company_tickers()[2]]
    if (input.tickers() != '' and input.p() != '' and input.q() != '' and
     input.omega() != '' and len(input_params()) != 0 and len(tickers) == len(input_params()[0])):
        p, q, omega = input_params()
        result = main(tickers, input.risk_aversion(), input.tau(), p, q, omega, company_tickers()[0], stock_data())
        company_names = company_tickers()[1]
        company_names = [x for x in company_names if '企業が見つかりません' not in x]
        result_df = pd.DataFrame({'分析中の企業':company_names, '均衡リターン（pi）':result[2],
         '調整後の期待リターン':result[3], 'ウェイト':result[0], '新しいウェイト':result[1]})
        return result_df
    else:
        return pd.DataFrame()




# 可視化関数
@reactive.event(input.start_calc)
def figpath_pie():
    """
    Creates a pie chart based on portfolio weights.
    
    Parameters:
        weights (np.array): Array of portfolio weights.
        labels (list or None): List of labels for the assets. If None, default labels will be used.
        title (str): Title of the pie chart.
    """
    if result_bl().empty:
        return None
    else:
        import japanize_matplotlib
        weights = list(result_bl()['新しいウェイト'])
        company_names = [x for x in company_tickers()[1] if '企業が見つかりません' not in x]
        labels = company_names
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
        ui.input_text_area("tickers", "銘柄コードを入力して下さい", value='4902, 7203, 9432, 9984', placeholder="0000, 1111, 1234")
        ui.input_action_button('get_data', '銘柄データの取得')
        ui.input_slider('risk_aversion', 'リスク回避係数', min=0, max=15, value=2.5, step=0.1)
        ui.input_slider('tau', '非直感的調整係数', min=0, max=1, value=0.05, step=0.01)
        'ビュー、期待リターン、ビューの不確実性　は銘柄コードと同じ順番でコンマと半角スペースで区切ってください'
        ui.input_text_area('p', 'ビュー', value = '-1, 1, 0, 1', placeholder='-1, 0, 1のいずれか（強気なら1、弱気なら-1）')
        ui.input_text_area('q', '期待リターン（年率、%表記、0以上）', value = '12, 20, 5, 30', placeholder='例15%→15')
        ui.input_text_area('omega', 'ビューの不確実性（0以上、小さいほど信頼性が高いとみなされる）', value = '0.01, 0.05, 0.1, 0.07', placeholder='0以上、0.01~0.10が一般的')
        ui.input_action_button('start_calc', '計算開始')

    with ui.card():
        ui.card_header("分析中の企業")
        @render.text
        def company_names():
            if input.tickers() == '' or len(company_tickers()) == 0:
                return '銘柄コードを入力してください'
            else:
                company_names = company_tickers()[1]
                return ', '.join(company_names)#改行したい
        
        @render.text
        def error_tickers():
            if input.tickers() == '':
                return '銘柄コードを入力してください'
            elif len(company_tickers()) == 0 or len(company_tickers()[2]) == 0:
                return None
            else:
                error_tickers = ', '.join(company_tickers()[2])
                return f'以下の銘柄コードに問題があります。：{error_tickers}'
   
    with ui.card():
        ui.card_header('分析結果')
        @render.data_frame
        def table():
            return render.DataGrid(result_bl())

        @render.text
        def error_message():
            if result_bl().empty:
                return 'エラーが発生しました：入力する値を確認してください'
            else:
                None


    with ui.card(full_screen=True):
        @render.image
        def image():
            if input.tickers() != '' and input.p() != '' and input.q() != '' and input.omega() != '' and figpath_pie() is not None:
                return {"src": str(figpath_pie()), 
                    "width": "600px", "format":"svg"}
            else:
                return None
        
        @render.text
        def error_message_():
            if figpath_pie() is None:
                return 'エラーが発生しました：入力する値を確認してください'
            else:
                None

      
#stock_dataをreactive.valueに、入力の数が一致しないとき（銘柄コードとp, qなど）、銘柄コードが4桁でないとき、結果の表示（表形式で）、plotly




