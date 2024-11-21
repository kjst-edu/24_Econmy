import yfinance as yf
import requests
from shiny import reactive, render
from shiny.express import input, ui, module

# 54d31670a9f0e88b7626d33d3601c81f
@module
def calc_capm(input, output, session):
    # FRED APIを使用してリスクフリーレート（米国10年債利回り）を取得
    def get_risk_free_rate(api_key):
        """
        米国10年債利回りをFRED APIから取得します。
        
        Returns:
        float: リスクフリーレート（年率）
        """

        if api_key is None:
            return 'api_error'

        else:
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
                return 'acquisition_error'

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

        market_risk_premium = float(market_risk_premium)/100
        if beta is None:
            return 'beta_error'
        elif (risk_free_rate == 'api_error') or (risk_free_rate == 'acquisition_error'):
            return risk_free_rate
        else:
            expected_return = float(risk_free_rate) + float(beta) * (market_risk_premium)
            return expected_return*100
        

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

    risk_free_rate = reactive.value()
    @reactive.effect
    @reactive.event(input.api_key_)
    def _():
        risk_free_rate.set(get_risk_free_rate(input.api_key_()))

    beta = reactive.value()

    @reactive.effect
    @reactive.event(input.ticker)
    def __():
        beta.set(get_stock_beta(f'{input.ticker()}.T'))


    @reactive.event(input.ticker)
    def get_company_name():
        ticker_code = input.ticker() + ".T"
        try:
            ticker = yf.Ticker(ticker_code)
            company_name = ticker.info.get("longName", "企業名が見つかりません")
        except Exception as e:
            company_name = "取得エラー"
        return company_name

    @reactive.calc
    def capm():
        return calculate_capm(risk_free_rate(), beta(), input.market_risk_premium())

    with ui.layout_sidebar():
        with ui.sidebar():
            ui.input_password("api_key_", "API KEY", placeholder="FRED APIのキーを入力してください")
            ui.input_slider('return_rate', '目標収益率', min=0, max=20, post='%', value=10, step=0.01)
            ui.input_slider('market_risk_premium', 'マーケットリスクプレミアム（一般的には5%）', min=0, max=20, value=5, post='%', step = 0.1)

        @render.text
        def company_name_():
            return f"企業名: {get_company_name()}"  

        @render.text
        def return_rate_():
            if capm() == 'api_error':
                return 'API KEYに問題があります。'
            elif capm() == 'acquisition_error':
                return 'データを取得できませんでした'
            elif capm() == 'beta_error':
                return '株データを取得できませんでした'
            else:
                return f'期待収益率 : {capm()}%'

        @render.text
        def decision():
            if type(capm()) == str:
                return f'{capm}'
            else:
                return f'{investment_decision(capm(), input.return_rate())}'

