# Python project to display stock market analysis using Streamlit

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

def get_stockmarket_data(symbol, start_date, end_date):
    stockmarket_data = yf.download(symbol, start=start_date, end=end_date)
    return stockmarket_data

def plot_stockmarket_price(stockmarket_data):
    fig = px.line(stockmarket_data, x = stockmarket_data.index, y = 'Close', title = 'Stock Price over Time')
    st.plotly_chart(fig)

def plot_stock_returns(stockmarket_data):
    stockmarket_data['Daily Return'] = stockmarket_data['Close'].pct_change()
    fig = px.line(stockmarket_data, x = stockmarket_data.index, y = 'Daily Return', title = 'Daily Returns over Time')
    st.plotly_chart(fig)

def main():
    st.title("Stock Market Analysis App")

    symbol = st.text_input("Enter Stock Symbol (eg. AAPL):", value = 'AAPL').upper()
    start_date = st.date_input("Select Start Date:", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("Select End Date:", pd.to_datetime('2022-01-01'))

    stock_data = get_stockmarket_data(symbol, start_date, end_date)
    st.subheader(f"Stock Data for {symbol}")
    st.write(stock_data.head())

    plot_stockmarket_price(stock_data)
    plot_stock_returns(stock_data)

if __name__ == '__main__':
    main()