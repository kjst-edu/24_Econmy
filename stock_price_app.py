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

#api_key : 5c09c7cab09f4d98b34aa3f247f7e36d


#株価の表示

@module
def stock_price(input, output, session):
    def get_stock_price(ticker, moving_average, start, end=None):
        ticker = f'{ticker}.T'
        if end is None:
            end = dt.today()

        start_date = start
        end_date = end
        start_date_100_days_ago = start_date - timedelta(days=150)
        return_df = yf.download(ticker, start=start_date_100_days_ago, end=end_date)
        return_df = return_df.stack(future_stack=True).reset_index()

        moving_average = [int(x) for x in moving_average]
        for i in range(len(moving_average)):
            days=moving_average[i]
            return_df[f'moving_average_{days}'] = return_df['Close'].rolling(days).mean()

        return return_df.set_index('Date')

    def last_year():
        current_date = dt.today().date()
        return current_date.replace(year=current_date.year-1)

    #サイドバーの表示
    with ui.sidebar():
        ui.input_text("ticker", "Enter Stock Code", placeholder="0000")
        ui.input_date("start", "Start Date", value=last_year(), min='1970-01-01', max=dt.today().date())
        ui.input_date("end", "End Date", value = dt.today().date(), min='1970-01-01', max=dt.today().date())
        ui.input_checkbox_group(
            "moving_average", "Moving Average",
            choices=[7, 10, 20, 30, 50, 100],
            selected=[7],
            inline=True)




    @reactive.calc
    def df():
        return get_stock_price(input.ticker(), list(input.moving_average()), input.start(), input.end())

    @reactive.calc
    def get_company_name():
        ticker_code = input.ticker() + ".T"
        try:
            ticker = yf.Ticker(ticker_code)
            company_name = ticker.info.get("longName", "企業名が見つかりません")
        except Exception as e:
            company_name = "取得エラー"
        return company_name

    @render.text
    def company_name():
        return f"企業名: {get_company_name()}"   

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
            apd.append(mpf.make_addplot(data_for_display[f'moving_average_{days}']))  


        #ろうそく足
        fig, axes = mpf.plot(data_for_display ,returnfig=True,  type='candle', volume=True, addplot=apd)

        axes[0].legend([None] * (len(apd) + 2))
        handles = axes[0].get_legend().legend_handles
        axes[0].legend(handles=handles[2:],
                    labels=list(input.moving_average()))    

        fig.savefig(path)

        return path


    with ui.card(full_screen=True):
        
        @render.image
        def image():
            return {"src": str(figpath()), 
                    "width": "500px", "format":"svg"}