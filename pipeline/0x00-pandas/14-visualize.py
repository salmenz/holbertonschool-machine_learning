#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop('Weighted_Price', axis=1)

df = df.rename(columns={'Timestamp': 'Date'})

df['Date'] = pd.to_datetime(df['Date'], unit='s')

df = df[df['Date'] >= '2017-01-01']

df = df.set_index('Date')

df['Close'].fillna(method='ffill', inplace=True)
df["High"].fillna(method="ffill", inplace=True)
df["Low"].fillna(method="ffill", inplace=True)
df["Open"].fillna(method="ffill", inplace=True)
df.fillna(method='ffill')

df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)
