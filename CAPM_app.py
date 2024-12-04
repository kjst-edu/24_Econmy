import yfinance as yf
import requests
import numpy as np
from datetime import datetime, timedelta
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
    @reactive.event(input.start_button)
    def _():
        risk_free_rate.set(get_risk_free_rate(input.api_key_()))

    beta = reactive.value()

    @reactive.effect
    @reactive.event(input.start_button)
    def __():
        beta.set(get_stock_beta(f'{input.ticker_capm()}.T'))


    @reactive.event(input.start_button)
    def get_company_name():
        ticker_code = input.ticker_capm() + ".T"
        try:
            ticker = yf.Ticker(ticker_code)
            company_name = ticker.info.get("longName", "企業名が見つかりません")
        except Exception as e:
            company_name = "取得エラー"
        return company_name

    @reactive.calc
    def capm():
        return calculate_capm(risk_free_rate(), beta(), input.market_risk_premium())

    #平均リターン、標準偏差、シャープレシオ

    def calculate_metrics(returns, risk_free_rate):
        mean_return = returns.mean().item()
        volatility = returns.std().item()
        if np.isnan(mean_return) or np.isnan(volatility):
            return None, None, None
        sharpe_ratio = (mean_return - float(risk_free_rate)) / volatility
        return mean_return, volatility, sharpe_ratio

    def calculate_risk_metrics(ticker_symbol, risk_free_rate=0.01):
        # 銘柄コードに「.T」がない場合は追加
        ticker_symbol = f'{ticker_symbol}.T'

        # 10年前から現在までのデータを取得
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * 6)
        data = yf.download(ticker_symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Adj Close']
        
        if data.empty:
            print(f"{ticker_symbol}のデータを取得できませんでした。")
            return

        # 日次リターンを計算
        daily_returns = data.pct_change().dropna()

        # 週次リターンを計算
        weekly_returns = data.resample('W').ffill().pct_change().dropna()

        # 月次リターンを計算
        monthly_returns = data.resample('ME').ffill().pct_change().dropna()

        # 年次リターンを計算
        yearly_returns = data.resample('YE').ffill().pct_change().dropna()

        # 10年リターンを計算
        long_returns = data.resample('5YE').ffill().pct_change().dropna()

        return_dict = {}
        return_dict['daily_metrics'] = calculate_metrics(daily_returns, risk_free_rate)
        return_dict['weekly_metrics'] = calculate_metrics(weekly_returns, risk_free_rate)
        return_dict['monthly_metrics'] = calculate_metrics(monthly_returns, risk_free_rate)
        return_dict['yearly_metrics'] = calculate_metrics(yearly_returns, risk_free_rate)
        return_dict['long_metrics'] = calculate_metrics(long_returns, risk_free_rate)

        return return_dict

    @reactive.event(input.start_button)
    def calc_risk():
        return calculate_risk_metrics(input.ticker_capm(), risk_free_rate())

    with ui.layout_sidebar():
        with ui.sidebar():
            ui.input_text('ticker_capm', '株価コード', placeholder='0000')
            ui.input_password("api_key_", "API KEY", placeholder="FRED APIのキーを入力してください")
            ui.input_slider('return_rate', '目標収益率', min=0, max=20, post='%', value=10, step=0.01)
            ui.input_slider('market_risk_premium', 'マーケットリスクプレミアム（一般的には5%）', min=0, max=20, value=5, post='%', step = 0.1)
            ui.input_action_button('start_button', '計算開始')

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
        
        with ui.card():
            @render.text
            def company_name2():
                return f"{get_company_name()}のリスク"
            
            @render.text
            def beta_text():
                return f'ベータ：{beta()}'
                
            @render.text
            def daily_text():
                daily = calc_risk()['daily_metrics']
                return f"日次 - 平均リターン: {daily[0]:.4f}, 標準偏差: {daily[1]:.4f}, シャープレシオ: {daily[2]:.4f}"
            @render.text
            def weekly_text():
                weekly = calc_risk()['weekly_metrics']
                return f"週次 - 平均リターン: {weekly[0]:.4f}, 標準偏差: {weekly[1]:.4f}, シャープレシオ: {weekly[2]:.4f}"
            @render.text
            def monthly_text():
                monthly = calc_risk()['monthly_metrics']
                return f"月次 - 平均リターン: {monthly[0]:.4f}, 標準偏差: {monthly[1]:.4f}, シャープレシオ: {monthly[2]:.4f}"
            @render.text
            def yearly_text():
                yearly = calc_risk()['yearly_metrics']
                return f"年次 - 平均リターン: {yearly[0]:.4f}, 標準偏差: {yearly[1]:.4f}, シャープレシオ: {yearly[2]:.4f}"
            @render.text
            def long_text():
                long = calc_risk()['long_metrics']
                return f"５年次 - 平均リターン: {long[0]:.4f}, 標準偏差: {long[1]:.4f}, シャープレシオ: {long[2]:.4f}"

