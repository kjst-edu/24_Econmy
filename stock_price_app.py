import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime as dt, timedelta
import yfinance as yf
import mplfinance as mpf
from shiny import reactive, render
from shiny.express import input, ui, module
import csv
import requests
import shutil
import os
import zipfile
import sys
import math
import urllib.request
import glob
import smtplib
from email.mime.text import MIMEText

#api_key : 5c09c7cab09f4d98b34aa3f247f7e36d


#株価の表示
"""
ここで書いたコードをapp.pyで動かすので@module付けてdef stock_price(input, output, session):
という関数の中にすべてのコードを入れる必要がある。関数の名前は何でもよいがすでにあるものと同じだとエラーになる
引数(input, output, session)は変更不可
"""

@module
def stock_price(input, output, session):
    def get_stock_price(ticker, start, end=None):
        ticker = f'{ticker}.T'
        if end is None:
            end = dt.today()

        start_date = start
        end_date = end
        start_date_100_days_ago = start_date - timedelta(days=150) #そのまま計算すると移動平均の最初がnanになるため
        return_df = yf.download(ticker, start=start_date_100_days_ago, end=end_date)
        return_df = return_df.stack(future_stack=True).reset_index()

        moving_average = [7, 10, 20, 30, 50, 100]
        for i in range(len(moving_average)):
            days=moving_average[i]
            return_df[f'{days}日間移動平均'] = return_df['Close'].rolling(days).mean()

        return return_df.set_index('Date')
    
    #入力の一年前の日付を取得
    def last_year():
        current_date = dt.today().date()
        return current_date.replace(year=current_date.year-1)

    #ダウンロードしたデータがたまるので削除
    def delete_csv():
        extracted_folder = "invest_zemi/ignored_folder"  # フォルダパスを修正（パス区切りは/推奨）
        target_filename = 'stock_data.csv' # 削除対象のCSVファイル名リスト

        if os.path.exists(extracted_folder):
            file_path = os.path.join(extracted_folder, target_filename)
                # 削除対象かどうか確認
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            print("Specified CSV files have been deleted.")
        else:
            print(f"{extracted_folder} does not exist.")

    #グラフを作成するためのデータをデータフレーム（df）として保存
    df = reactive.value(pd.DataFrame())
    @reactive.effect
    @reactive.event(input.start_search_data)
    def _():
        df.set(get_stock_price(input.ticker(), input.start(), input.end()))
        df().to_csv(r'invest_zemi/ignored_folder/stock_price_data.csv')
    

    #会社名を取得
    def get_company_name():
        ticker_code = input.ticker() + ".T"
        try:
            ticker = yf.Ticker(ticker_code)
            company_name = ticker.info.get("longName", "企業名が見つかりません")
        except Exception as e:
            company_name = "取得エラー"
        return company_name
    

    #グラフ作成
    @reactive.calc
    def figpath():
        import tempfile
        fd, path = tempfile.mkstemp(suffix = '.svg')

        start = input.start()
        end = input.end()
        data = df()

        data_for_display = data[(data.index.date>start) & (data.index.date<end)].drop('Ticker', axis=1)

        apd = []
        for days in list(input.moving_average()):
            apd.append(mpf.make_addplot(data_for_display[f'{days}日間移動平均']))  


        #ろうそく足
        fig, axes = mpf.plot(data_for_display ,returnfig=True,  type='candle', volume=True, addplot=apd)

        axes[0].legend([None] * (len(apd) + 2))
        handles = axes[0].get_legend().legend_handles
        axes[0].legend(handles=handles[2:],
                    labels=list(input.moving_average()))    

        fig.savefig(path)

        return path
    
        #ここより上の部分で表示用のデータを作成したりグラフを作成したりしている。アプリの裏側の部分、下からが表側の部分

    #サイドバーの表示（入力や出力（グラフなど）の処理はここで）
    with ui.layout_sidebar(fillable=True):
        with ui.sidebar(open='desktop'):
            #銘柄コードの入力部分や検索ボタンなどユーザが入力する部分
            ui.input_action_button('start_search_data', 'データの検索開始')
            ui.input_text("ticker", "Enter Stock Code", placeholder="0000")#株価コード
            ui.input_date("start", "Start Date", value=last_year(), min='1970-01-01', max=dt.today().date())#日付（開始）
            ui.input_date("end", "End Date", value = dt.today().date(), min='1970-01-01', max=dt.today().date())#日付（終了）
            ui.input_checkbox_group(
                "moving_average", "Moving Average",
                choices=[7, 10, 20, 30, 50, 100],
                selected=[7],
                inline=True)#移動平均のチェックボックス

        #企業名を表示
        @render.text
        def company_name():
            return f"企業名: {get_company_name()}"

        #株価のグラフの表示
        with ui.card(full_screen=True):
            @render.image
            def image():
                if df().empty or isinstance(df(), str):
                    return None
                else:
                    return {"src": str(figpath()), 
                        "width": "600px", "format":"svg"}

        #CSVのダウンロード
        with ui.card(full_screen=True):
            @render.download(label='データをダウンロード（CSV)')
            def download_stock_price():
                path = r"invest_zemi/ignored_folder/stock_price_data.csv"
                return path
        


@module
def golden_cross(input, output, session):

    def get_stock_price_for_gc(ticker):
        ticker = f'{ticker}.T'
        start_date = dt.today().date()
        start_date_150_days_ago = start_date - timedelta(days=150)
        return_df = yf.download(ticker, start=start_date_150_days_ago, end=dt.today().date(), interval="1h")
        return_df = return_df.stack(future_stack=True).reset_index()

        return_df['date'] = pd.to_datetime(return_df['Datetime'].dt.date)
        moving_average = [7, 10, 20, 30, 50, 100]
        for days in moving_average:
            return_df.loc[:, f'{days}日間移動平均'] = return_df.apply(
                lambda row: return_df[
                    (return_df['date'] <= row['date']) &  # 現在の日付以下
                    (return_df['date'] > row['date'] - pd.Timedelta(days=days))  # 現在の日付からn日間の範囲
                ]['Close'].mean(),
                axis=1
            )
        return return_df.set_index('date')

    def get_company_name():
        ticker_code = input.ticker_gc() + ".T"
        try:
            ticker = yf.Ticker(ticker_code)
            company_name = ticker.info.get("longName", "企業名が見つかりません")
        except Exception as e:
            company_name = "取得エラー"
        return company_name
        
    @reactive.calc
    def df_for_gc():
        reactive.invalidate_later(60*60)
        if len(input.ticker_gc()) != 4:
            return 'ticker_error'
        else:
            return get_stock_price_for_gc(input.ticker_gc())


        #ゴールデン・デッドクロス
    def check_golden_dead_cross():
        print('check')
        if len(input.gc_dc_li()) != 2:
            print('ゴールデンクロス・デッドクロスの検出用のデータを確認してください')
        else:
            # 過去1年分のデータを取得
            stock_data = df_for_gc()
            # データが取得できているか確認
            if isinstance(stock_data, str) and stock_data == 'ticker_error':
                print(f"{get_company_name()} のデータが見つかりませんでした。")
                return False
            
            else:
                # 最後の2日間のデータを取得
                last_two_days = stock_data.tail(2)

                short_mv = f'{input.gc_dc_li()[0]}日間移動平均'
                long_mv = f'{input.gc_dc_li()[1]}日間移動平均'
                # 昨日と今日の移動平均を比較してゴールデンクロスを検出
                golden_cross = (last_two_days[short_mv].iloc[-2] < last_two_days[long_mv].iloc[-2] and 
                    last_two_days[short_mv].iloc[-1] > last_two_days[long_mv].iloc[-1])

                dead_cross = (last_two_days[short_mv].iloc[-2] > last_two_days[long_mv].iloc[-2] and 
                    last_two_days[short_mv].iloc[-1] < last_two_days[long_mv].iloc[-1])

                if golden_cross:
                    return 'golden_cross'
                elif dead_cross:
                    return 'dead_cross'
                else:
                    'none'

    @reactive.calc
    def golden_dead_cross():
        return check_golden_dead_cross()
    
    @reactive.calc
    def figpath_():
        import tempfile
        fd, path = tempfile.mkstemp(suffix = '.svg')

        data_for_display = df_for_gc()

        apd = []
        for days in list(input.moving_average_gc()):
            apd.append(mpf.make_addplot(data_for_display[f'{days}日間移動平均']))  


        #ろうそく足
        fig, axes = mpf.plot(data_for_display ,returnfig=True,  type='candle', volume=True, addplot=apd)

        axes[0].legend([None] * (len(apd) + 2))
        handles = axes[0].get_legend().legend_handles
        axes[0].legend(handles=handles[2:],
                    labels=list(input.moving_average_gc()))    

        fig.savefig(path)

        return path
    
    with ui.layout_sidebar(fillable=True):
        with ui.sidebar(open='desktop'):
            ui.input_text("ticker_gc", "Enter Stock Code", placeholder="0000")
            ui.input_checkbox_group(
                'gc_dc_li', 'ゴールデンクロス・デッドクロスの検出に使用するデータ（二つ選択）',
                choices=[7, 10, 20, 30, 50, 100],
                selected=[7, 100],
                inline=True
            )
            ui.input_checkbox_group(
                "moving_average_gc", "移動平均",
                choices=[7, 10, 20, 30, 50, 100],
                selected=[7],
                inline=True)

        @render.text
        @reactive.calc
        def search_text():
            return (f"{get_company_name()} の株価を分析中...")

        @render.text
        @reactive.calc
        def start_sarching():
            if input.ticker_gc() is None:
                return input.ticker()
            elif golden_dead_cross() == 'golden_cross':
                return ("通知: ゴールデンクロスが発生しました！")
            elif golden_dead_cross() == 'dead_cross':
                return ("通知: デッドクロスが発生しました！")
            elif golden_dead_cross() is False:
                return ("データが見つかりませんでした")
            else:
                return ("通知: ゴールデンクロスもデッドクロスも発生していません")

