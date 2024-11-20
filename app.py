from functools import partial
from pathlib import Path

from shiny import reactive, req
from shiny.ui import page_navbar
from shiny.express import input, render, ui

from stock_price_app import stock_price
from financial_statement_app import fin_statement


ui.page_opts(
    title="Module Demo",  
    page_fn=partial(page_navbar, id="page"),  
)

with ui.nav_panel("Panel A"):  
    stock_price("main")

with ui.nav_panel("Panel B"):
    fin_statement("main")