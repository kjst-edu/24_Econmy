from functools import partial
from pathlib import Path

from shiny import reactive, req
from shiny.ui import page_navbar
from shiny.express import input, render, ui

from stock_price_app import stock_price, golden_cross
from financial_statement_app import fin_statement
from CAPM_app import calc_capm
from BL_model_app import black_litterman_app



ui.page_opts(
    title="アプリ名(未定)",  
    page_fn=partial(page_navbar, id="page"),  
)

with ui.nav_panel("株価"): 
    stock_price("main")

with ui.nav_panel("財務諸表"):
    fin_statement("main")

with ui.nav_panel('ブラックリッターマンモデル'):
    black_litterman_app("main")

with ui.nav_panel("CAPM"):
    calc_capm("main")

with ui.nav_panel('ゴールデンクロス・デッドクロス'):
    golden_cross('main')

with ui.nav_panel('API KEY 最終的に削除'):
    @render.text
    def text_1():
        return f"edinet api key : 5c09c7cab09f4d98b34aa3f247f7e36d    (財務諸表の方)"
    
    @render.text
    def text_2():
        return "FRED api key : 54d31670a9f0e88b7626d33d3601c81f   (CAPMの方)"