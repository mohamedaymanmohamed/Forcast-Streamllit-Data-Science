import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("📈 Sales Prediction")

df = pd.read_csv('sales_data.csv')
df['Month'] = pd.to_datetime(df['Month'])
df['Month_Num'] = np.arange(len(df))

st.subheader("📊  Main Data")
st.dataframe(df[['Month', 'Sales']])

X = df[['Month_Num']]
y = df['Sales']
model = LinearRegression()
model.fit(X, y)

n_months = st.slider(" Number of Months:", 1, 12, 6)

future_months = np.arange(len(df), len(df) + n_months).reshape(-1, 1)
future_preds = model.predict(future_months)
future_dates = pd.date_range(start=df['Month'].iloc[-1] + pd.offsets.MonthBegin(1), periods=n_months, freq='MS')

df_future = pd.DataFrame({'Month': future_dates, 'Sales': future_preds})
df_all = pd.concat([df[['Month', 'Sales']], df_future], ignore_index=True)

st.subheader("📉 Forcast Chart ")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Month'], df['Sales'], label='Actual Sales', marker='o')
ax.plot(df_future['Month'], df_future['Sales'], label='Forecast', linestyle='--', marker='x')
ax.set_xlabel('Month')
ax.set_ylabel('Sales')
ax.set_title('Sales Forecast')
ax.legend()
st.pyplot(fig)

st.subheader("🔮 Expected values:")
st.dataframe(df_future)
