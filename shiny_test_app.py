from shiny import reactive, render
from shiny.express import input, ui, module
import re
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os

@render.data_frame
def _():
    li = pd.DataFrame(columns = ["a", "b", "c", "d"])
    return li

ui.input_action_button('a', 'A')

@reactive.event(input.a)
def make_dir():
    print('makedir')
    os.makedirs("invest_zemi\ignored_folder", exist_ok=True)
    return 'made'

@render.text
def __():
    return make_dir()

"""
#https://shiny.posit.co/py/docs/user-interfaces.html
from shiny.express import render, ui
from faicons import icon_svg as icon

with ui.value_box(showcase=icon("piggy-bank")):
    "Total sales"
    "$1,000,000"

with ui.value_box(showcase=icon("person")):
    "Total customers"
    @render.ui
    def customers():
        return f"{1000:,}"

"Hover this icon: "

with ui.tooltip():
    icon("circle-info")
    "Tooltip message"

ui.br()

"Click this icon: "

with ui.popover(title="Popover title"):
    icon("circle-info")
    "Popover message"

"""