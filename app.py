from functools import partial
from pathlib import Path

from shiny import reactive, req
from shiny.ui import page_navbar
from shiny.express import input, render, ui

from stock_price_app import stock_price
from financial_statement_app import fin_statement


ui.page_opts(
    title="アプリ名(未定)",  
    page_fn=partial(page_navbar, id="page"),  
)

with ui.nav_panel("Stock Price"): 
    stock_price("main")

with ui.nav_panel("CAPM"):
    fin_statement("main")