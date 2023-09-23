import json
import requests

import numpy as np
import pandas as pd
import yfinance as yf 
import streamlit as st 

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def getETFList():
    etfList = pd.read_csv('./ETFYearly_2023.csv')
    return etfList['Code'].str.replace("=", "").str.replace('"', "").to_list()

def getETFDividends(code, start_date, end_date):
    base_url = "https://www.twse.com.tw/rwd/zh/ETF/etfDiv"
    url = f"{base_url}?stkNo={code}&startDate={start_date}&endDate={end_date}&response=html"
    
    html = requests.get(url).text
    json_context = json.loads(html)
    table_body = json_context['data']
    table_colName = json_context['fields']
    
    data = pd.DataFrame(table_body, columns=table_colName)
    
    raw_date = [
        list(item)
        for item in (
            data['除息交易日']
            .str
            .findall("(?P<year>\d{2,3})年(?P<month>\d{1,2})月(?P<day>\d{1,2})")
            .explode()
            .to_list()
        )
    ]
    
    for _date in raw_date:
        _date[0] = str(int(_date[0]) + 1911)
    
    dateList = [
        pd.to_datetime( "-".join(date) +  " 00:00:00+0800",  utc=False)
        for date in raw_date
    ]
    
    data['Date'] = dateList
    filtered_df = data[['收益分配金額 (每1受益權益單位)', 'Date']]
    filtered_df.columns = ['Dividend_per_share', 'Date']
    filtered_df = filtered_df.set_index("Date")
    filtered_df = filtered_df.astype({"Dividend_per_share": float})
    return filtered_df, data

def getETFHistory(etf_code, start_date, end_date, period='1d'):
    etf = yf.Ticker(f'{etf_code}.tw')
    ohlc = etf.history(start=start_date, end=end_date, period=period)
    
    return etf, ohlc

def OHLCFeatureExtraction(data):
    data['is_red'] = (data.Open - data.Close) > 0
    data['is_green'] = (data.Open - data.Close) < 0
    data['is_equi'] = (data.Open - data.Close) == 0
    data['daily_mean'] = np.round((data.Open + data.Close) / 2, 1)
    return data

def DividendWithOHLC(ticker_data, dividend_data):
    merged_data = pd.merge(
        ticker_data.reset_index(),
        dividend_data.reset_index(),
    )

    return merged_data

def calculateUnrealizedProfit(data, trading_date, trading_volume):
    
    trading_df = data[data.index.day.isin(trading_date)].copy()
    trading_df['holdings_per_trading_volume'] = np.floor(trading_volume / trading_df['daily_mean'])
    trading_df['cost'] = np.round(trading_df['holdings_per_trading_volume'] * trading_df['daily_mean'])
    
    trading_df['cum_cost'] = trading_df['cost'].cumsum()
    trading_df['cum_holdings'] = trading_df['holdings_per_trading_volume'].cumsum()
    trading_df['unrealized_gains'] = np.round(trading_df['cum_holdings'] * trading_df['daily_mean']) 
    trading_df['PE_ratio'] = (trading_df['unrealized_gains'] / trading_df['cum_cost'] ) * 100
    
    return trading_df

def calculateDividendsProfit(trading_data, dividend_data):
    merged_data = (
        dividend_data
        .reset_index()
        .merge(
            trading_data.reset_index(),
            how='outer',
        )
        .set_index('Date')
        .sort_index()
        .fillna(0)
    )
    
    merged_data['cum_holdings'] = merged_data['holdings_per_trading_volume'].cumsum()
    merged_data['Dividend_profit'] = np.round(merged_data['Dividend_per_share'] * merged_data['cum_holdings'])
    merged_data['cum_dividend_profit'] = merged_data['Dividend_profit'].cumsum()

    return merged_data

def calculateReinvestment(data):
    data['Dividend_reinvestment_holding'] = np.floor(data['Dividend_profit'] / data['daily_mean'])
    data['cost_reinvestment'] = np.round(data['Dividend_reinvestment_holding'] * data['daily_mean'])
    
    data['Holding_include_reinvest'] = data['holdings_per_trading_volume'] + data['Dividend_reinvestment_holding']
    data['cost_include_reinvest'] = data['cost'] + data['cost_reinvestment']
    
    data['cum_holding_include_reinvest'] = data['Holding_include_reinvest'].cumsum()
    data['cun_cost_include_reinvest'] = data['cost_include_reinvest'].cumsum()
    
    data['unrealized_gains_include_reinvest'] = np.round(data['cum_holding_include_reinvest'] * data['daily_mean'])
    data['PE_ratio_include_reinvest'] = (data['unrealized_gains_include_reinvest'] / data['cun_cost_include_reinvest'] ) * 100
    
    return data

def plotOHLCTicks(etf_code, ohlc_data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Ohlc(
            x=ohlc_data.index,
            open=ohlc_data['Open'],
            high=ohlc_data['High'],
            low=ohlc_data['Low'],
            close=ohlc_data['Close'],
            name=f'{etf_code}',
            increasing_line_color='red',
            decreasing_line_color='green',
            legendgroup="1",  # this can be any string, not just "group"
            legendgrouptitle_text="K線",
        ),
        secondary_y=False,
    )

    ohlc_red = ohlc_data[ohlc_data.is_red == True]
    ohlc_green = ohlc_data[ohlc_data.is_green == True]
    ohlc_yellow = ohlc_data[ohlc_data.is_equi == True]

    fig.add_trace(
        go.Bar(
            x=ohlc_red.index,
            y=ohlc_red['Volume'],
            name='價漲',
            legendgroup="2",  # this can be any string, not just "group"
            legendgrouptitle_text="成交量圖",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=ohlc_green.index,
            y=ohlc_green['Volume'],
            name='價跌',
            legendgroup="2",  # this can be any string, not just "group"
            legendgrouptitle_text="成交量圖",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=ohlc_yellow.index,
            y=ohlc_yellow['Volume'],
            name='持平',
            marker_color='yellow',
            legendgroup="2",  # this can be any string, not just "group"
            legendgrouptitle_text="成交量圖",
        ),
        secondary_y=True,
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]), #hide weekends
            dict(values=["2015-12-25", "2016-01-01"])  # hide Christmas and New Year's
        ]
    )

    fig.update_yaxes(title_text="金額 （台幣）", secondary_y=False)
    fig.update_yaxes(title_text="成交量", secondary_y=True)

    fig.update_layout(
        title=f'{etf_code} 歷史交易資訊',
        xaxis_title='日期',
        yaxis_title='金額（台幣）'
    )
    
    return fig
    
def plotUnrealizedProfit(etf_code, trading_data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=trading_data.index,
            y=trading_data.cum_cost,
            name='累計投入成本 (A)',
            legendgroup="left",  # this can be any string, not just "group"
            legendgrouptitle_text="左軸 (單位：台幣)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=trading_data.index,
            y=trading_data.unrealized_gains,
            name='持股現值 (B)',
            legendgroup="left",  # this can be any string, not just "group"
            legendgrouptitle_text="左軸 (單位：台幣)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
            go.Scatter(
                x=trading_data.index,
                y=trading_data.PE_ratio,
                name='未實現損益比 B/A',
                legendgroup="right",  # this can be any string, not just "group"
                legendgrouptitle_text="右軸 (100%)",
            ),
            secondary_y=True,
        )
        
    fig.update_layout(
        title=f'{etf_code} 定期定額收益計算'
    )

    fig.update_yaxes(title_text="金額 （台幣）", secondary_y=False)
    fig.update_yaxes(title_text="成本－未實現損益比（%）", secondary_y=True)
    fig.update_xaxes(title_text='日期')
    return fig
    
def plotDividendsProfit(etf_code, dividends_data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=dividends_data.index,
            y=dividends_data.cum_dividend_profit,
            name='累計配息金額',
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dividends_data.index,
            y=dividends_data.cum_holdings,
            name='累計持股數量',
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f'{etf_code} 累計配息金額以及持股數量'
    )

    fig.update_yaxes(title_text="金額 （台幣）", secondary_y=False)
    fig.update_yaxes(title_text="持股數量（股）", secondary_y=True)
    fig.update_xaxes(title_text='日期')
    return fig

def main():
    
    st.set_page_config(layout="wide")
    
    st.title("台股ETF收益計算器")
    with st.form("input"):
        etfList = getETFList()
        
        etf_code = st.selectbox("請選擇一擋ETF:", etfList, key=1)
        
        col11, col12 = st.columns(2)
        
        with col11:
            start_date = st.date_input("請選擇起始時間：")
        
        with col12:
            end_date = st.date_input("請選擇結束時間：")
        
        col21, col22 = st.columns(2)
        
        with col21:
            trading_date = st.multiselect(
                "請選擇交易日:",
                list(range(1, 32)),
                [1]
            )
        
        with col22:
            trading_volume = st.number_input(
                "請填寫交易金額:",
                min_value=1,
                step=1,
            )
        
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        start_date = str(start_date)
        end_date = str(end_date)
        
        start_date_list = start_date.split("-")
        end_date_list = end_date.split("-")
        
        dividend_df, _ = getETFDividends(
            etf_code, 
            start_date="".join(start_date_list), 
            end_date="".join(end_date_list)
        )
        
        etf, ohlc_data = getETFHistory(etf_code, start_date=start_date, end_date=end_date)
        
        ohlc_data = OHLCFeatureExtraction(ohlc_data)
        
        dividend_df_ohlc = DividendWithOHLC(ohlc_data, dividend_df)
        trading_data = calculateUnrealizedProfit(ohlc_data, trading_date, trading_volume)
        
        dividend_data = calculateDividendsProfit(trading_data, dividend_df_ohlc)
        
        plotFuncList = [plotOHLCTicks, plotUnrealizedProfit, plotDividendsProfit]
        plotMaterialsList = [ohlc_data, trading_data, dividend_data]
        for _tab, func, data in zip(st.tabs(["K線圖", "預估損益圖", "預估配息收益圖"]), plotFuncList, plotMaterialsList):
            with _tab:
                _figure = func(etf_code, data)
                st.plotly_chart(_figure, use_container_width=True)


if __name__ == "__main__":
    main()