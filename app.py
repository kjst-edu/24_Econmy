import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime as dt, timedelta
import yfinance as yf
import mplfinance as mpf
from shiny import reactive, render
from shiny.express import input, ui
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

    ui.input_password("api_key", "API KEY", placeholder="Enter EDINET API KEY")
    ui.input_action_button('start_search', 'Search Documents')
    ui.input_checkbox_group(
        "cols", "Data for Display",
        choices=['売上高','経常利益', '純資産額', '総資産額', 
        '１株当たり純資産額', '１株当たり配当額', '自己資本比率', 
        '自己資本利益率', '株価収益率', 
        '営業活動によるキャッシュ・フロー', '投資活動によるキャッシュ・フロー',
        '財務活動によるキャッシュ・フロー', '従業員数'],
        selected=['売上高'],
        inline=True
        )


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


def get_edinet_code(code):
    with open('invest_zemi\EdinetcodeDlInfo.csv') as f:
        reader= csv.reader(f)
        #先頭行をスキップする
        next(reader)
        # 2行目以を項目行する
        col = next(reader)
        # 3行目以降をデータとする
        # rowは1行づつの配列となる。内包表記でまとめる
        data = [row for row in reader]
    
    edinet_codelist = pd.DataFrame(data, columns=col)
    return edinet_codelist[edinet_codelist['証券コード'] == f'{code}0']['ＥＤＩＮＥＴコード'].values[0]

def get_docID(code, date, api_key):
    api_key = str(api_key)
    if (api_key == "Enter EDINET API code") or (api_key is None):
        return 0
    else:
        # APIエンドポイント
        api_url = 'https://disclosure.edinet-fsa.go.jp/api/v2/documents.json'

        # パラメータ（例: 企業コードや提出日）
        params = {
            'date': date,  # 提出日
            'type': 2,             # 提出書類の種類（有価証券報告書など）
            "Subscription-Key":str(api_key)
        }

        # APIリクエスト
        response = requests.get(api_url, params=params)

        # レスポンスの確認
        if response.status_code == 200:
            data = response.json()
        elif response.status_code == 401:
            return 1
        else:
            print("Error:", response.status_code)
            return None

        # ドキュメント情報を抽出
        documents = data['results']
        if documents == []:
            return None
        
        # 結果をDataFrameに変換
        df = pd.DataFrame(documents)

        # 特定のカラムだけを選択
        df_filtered = df[['docID', 'edinetCode', 'filerName', 'docDescription']]

        # 有価証券報告書のみにフィルタリング
        df_financial = df_filtered[df_filtered['docDescription'].str.contains('有価証券報告書', na=False)]
        
        # EDINETコードを取得してフィルタリング
        edinet_code = get_edinet_code(code)
        docID_data = df_financial[df_financial['edinetCode'] == edinet_code]
            
        # docIDを返す
        return docID_data['docID'].values[0]

def get_nearest_quarter_end(current_date):
    # 現在の月を取得
    month = current_date.month

    # 四半期末日を設定
    if month <= 5:
        return current_date.replace(year=current_date.year - 1, month=12, day=31)
    else:
        return current_date.replace(month=6, day=30)


def search_docID(code, api_key):
    api_key = str(api_key)
    if len(code) >= 4:
        flag = True
    else:
        flag = False
    current_date = dt.today().date()
    date = get_nearest_quarter_end(current_date)
    days = 0

    while flag:
        print(date)
        try:
            docID = get_docID(code, str(date), api_key)
            if docID is not None:
                flag = False
                print("取得したdocID:", docID)
            
            elif docID == 1:
                print('Please check the API code again')
                break
            
            elif docID == 0:
                print('Enter EDINET API code')
                break
            else:
                # ドキュメントが見つからない場合、日付を1日前に戻す
                date -= timedelta(days=1)
        except IndexError:
            date -= timedelta(days=1)

        days += 1
        if days > 365:
            break
    return docID

def get_fs(docID, api_key):
    api_key = str(api_key)
    url = f'https://api.edinet-fsa.go.jp/api/v2/documents/{docID}?type=5&Subscription-Key={api_key}'
    try:
            # ZIPファイルのダウンロード
        with urllib.request.urlopen(url) as res:
            content = res.read()
        output_path = os.path.join("invest_zemi\ignored_folder", f'{docID}.zip')
        with open(output_path, 'wb') as file_out:
            file_out.write(content)
    except urllib.error.HTTPError as e:
        if e.code >= 400:
            sys.stderr.write(e.reason + '\n')
        else:
            raise e    
    
    # 解凍フォルダのパスを設定（.gitignoreで無視されるフォルダの中に保存）
    output_dir = os.path.join("invest_zemi\ignored_folder", f"{docID}_extracted")

    # 解凍処理
    zip_path = os.path.join("invest_zemi\ignored_folder", f'{docID}.zip')
    try:
        # ZIP形式であり、ファイルサイズが0以上であるかチェック
        if zipfile.is_zipfile(zip_path) and os.path.getsize(zip_path) > 0:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # ファイル数を確認して空かどうかを判定
                if len(zip_ref.namelist()) > 0:
                    os.makedirs(output_dir, exist_ok=True)  # 保存先フォルダが存在しない場合に作成
                    zip_ref.extractall(output_dir)
                    print(f"Extracted {zip_path} to {output_dir}")
                else:
                    print(f"{zip_path} is an empty zip file.")
        else:
            print(f"{zip_path} is not a valid or empty zip file.")
    except zipfile.BadZipFile:
        print(f"{zip_path} is corrupted or not a valid zip file.")

    # フォルダ内の'-asr-'が含まれるCSVファイルを検索
    base_folder_path = r'invest_zemi\ignored_folder'

    # パターンを指定してファイルパスを取得
    folder_pattern = os.path.join(base_folder_path, '*_extracted', 'XBRL_TO_CSV', '*-asr-*.csv')
    matching_folders = glob.glob(folder_pattern)
    file = matching_folders[0]

    # 見つかったCSVファイルを読み込む

    try:
        return_df = pd.read_csv(file, encoding='utf-16', sep='\t')
        print(f"Loaded {file} successfully.")
        # DataFrameの確認や処理が必要であればここに追加
        # 例: print(df.head())など
    except Exception as e:
        print(f"Error loading {file}: {e}")
    
    return return_df

def delete_files():
    extracted_folder = "invest_zemi\ignored_folder"

    # フォルダ内の全データを削除
    if os.path.exists(extracted_folder):
        shutil.rmtree(extracted_folder)
        os.makedirs(extracted_folder)  # 空のフォルダを再作成
        print(f"Deleted all contents in: {extracted_folder}")
    else:
        print(f"{extracted_folder} does not exist.")

def convert_number(value):
    # 数値に変換できるかを確認
    try:
        # 文字列が浮動小数点数として解釈されるか試す
        num = float(value)
        # 小数点以下があるかを確認し、適切な型に変換
        if num.is_integer():
            return int(num)  # 小数点以下が0ならintに変換
        else:
            return num  # 小数点以下があればfloatのまま
    except ValueError:
        # 数値に変換できない場合はそのまま返す
        return value

def make_data_for_display(cols, df):
    years = ['四期前', '三期前', '前々期', '前期', '当期']
    data_for_display = pd.DataFrame({'相対年度':years}).reset_index(drop=True)

    for col in cols:
        data = [convert_number(x) for x in df[df['項目名'].str.contains(f'{col}', na=False)][:5]['値'].reset_index(drop=True)]
        data_for_display[col] = data
    
    return data_for_display

def get_financialstatement(code, api_key):
    if  (api_key == "Enter EDINET API code") or (api_key is None):
        raise ValueError('There is a problem with the API code')
    else:
        api_key = str(api_key)
        docID = search_docID(code, api_key)
        data = get_fs(docID, api_key)
        delete_files()
        
        return data

data = reactive.value(pd.DataFrame())

@reactive.effect
@reactive.event(input.start_search)
def _():
    data.set(get_financialstatement(input.ticker(), input.api_key()))


@reactive.calc
def fs_df():
    return make_data_for_display(list(input.cols()), data())



@reactive.calc
def figpath2():
    import tempfile
    fd, path = tempfile.mkstemp(suffix = '.svg')
    data = fs_df()

    remove_cols = data.columns[data.dtypes==object].tolist()
    remove_cols.remove('相対年度')

    r = math.ceil((len(data.columns)-1)/3)
    import japanize_matplotlib
    sns.set(font="IPAexGothic")
    fig, axes = plt.subplots(r, 3, figsize=(12, r*3))
    i, j = 0, 0
    if r == 1:
        for col in data.columns:
            if col != '相対年度':
                sns.lineplot(data=data, x='相対年度', y=col, ax=axes[i])
                axes[i].set_title(f'{col}')
                axes[i].set_ylabel('')
                i += 1

        plt.tight_layout()

    else:
        for col in data.columns:
            if col != '相対年度':
                sns.lineplot(data=data, x='相対年度', y=col, ax=axes[j, i])
                axes[j, i].set_title(f'{col}')
                axes[j, i].set_ylabel('')
            
                if i % 3 == 2:
                    i = 0
                    j += 1
                else:
                    i += 1
        plt.tight_layout()

    fig.savefig(path)

    return path


with ui.card(full_screen=True):
    
    @render.image
    def image2():
        return {"src": str(figpath2()), 
                "width": "800px", "format":"svg"}

