import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime as dt, timedelta
import yfinance as yf
import mplfinance as mpf
from shiny import reactive, render
from shiny.express import input, ui


def get_stock_price(ticker, moving_average, start, end=None):
    ticker = f'{ticker}.T'
    if end is None:
        end = dt.today()

    start_date = dt.strptime(start, '%Y-%m-%d').date()
    end_date = dt.strptime(end, '%Y-%m-%d').date()
    start_date_100_days_ago = start_date - timedelta(days=150)
    return_df = yf.download(ticker, start=start_date_100_days_ago, end=end_date)
    return_df = return_df.stack(future_stack=True).reset_index()

    moving_average = [int(x) for x in moving_average]
    for i in range(len(moving_average)):
        days=moving_average[i]
        return_df[f'moving_average_{days}'] = return_df['Close'].rolling(days).mean()

    return return_df.set_index('Date')

with ui.sidebar():

    ui.input_text("ticker", "Enter ticker", "4902")
    ui.input_text("start", "Start Date", "2023-07-01")
    ui.input_text("end", "End Date", "2024-07-31")
    ui.input_checkbox_group(
        "moving_average", "Moving Average",
        choices=[7, 10, 20, 30, 50, 100],
        selected=[7],
        inline=True
        )


@reactive.calc
def df():
    return get_stock_price(input.ticker(), list(input.moving_average()), input.start(), input.end())

@reactive.Calc
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

    start = dt.strptime(input.start(), '%Y-%m-%d').date()
    end = dt.strptime(input.end(), '%Y-%m-%d').date()
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
                "width": "800px", "format":"svg"}