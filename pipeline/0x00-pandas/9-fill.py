#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Setting the missing values to the previous Close values
df["Close"].fillna(method="ffill", inplace=True)
df["High"].fillna(method="ffill", inplace=True)
df["Low"].fillna(method="ffill", inplace=True)
df["Open"].fillna(method="ffill", inplace=True)
df.fillna(method='ffill')

# Setting the missing values to 0
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)

print(df.head())
print(df.tail())
