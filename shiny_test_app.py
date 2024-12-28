from shiny import reactive, render
from shiny.express import input, ui, module
import re
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

ui.input_text_area('text', '入力')

@render.text
def _():
    a = input.text().split(',')
    return a