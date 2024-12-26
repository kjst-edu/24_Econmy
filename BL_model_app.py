from shiny import reactive, render
from shiny.express import input, ui, module
import re

ui.input_text_area('a', 'b','aaa')
"""
何を入力しても数字の部分だけ取り出し4桁ごとに区切ってリスト化
このファイルを実行すると実験できる
"""
def extract_numbers(input_string):
    # 数字のみを抽出
    numeric_string = re.sub(r'\D', '', input_string)
    # 4桁ごとに分割してリスト化
    result = [numeric_string[i:i+4] for i in range(0, len(numeric_string), 4)]
    return result

@reactive.calc
def extract_num():
    return extract_numbers(input.a())
@render.text
def _():
    return extract_num()
