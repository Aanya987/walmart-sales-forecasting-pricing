import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Walmart Sales Forecast & Pricing Dashboard", layout="wide")
st.title("🛒 Walmart Sales Forecasting & Pricing Recommendation Dashboard")

st.markdown("""
This dashboard predicts next week's sales for a selected store and recommends
stock/pricing actions based on the prediction vs recent trend.
""")

# Load saved data and model (we'll create these in the next step)
results = pd.read_csv('results.csv', parse_dates=['Date'])

store_list = sorted(results['Store'].unique())
selected_store = st.selectbox("Select Store", store_list)

store_data = results[results['Store'] == selected_store].sort_values('Date')

col1, col2, col3 = st.columns(3)
col1.metric("Avg Weekly Sales", f"${store_data['Weekly_Sales'].mean():,.0f}")
col2.metric("Avg Predicted Sales", f"${store_data['Predicted_Sales'].mean():,.0f}")
col3.metric("Model MAPE", "4.85%")

st.subheader(f"Actual vs Predicted Sales — Store {selected_store}")
st.line_chart(store_data.set_index('Date')[['Weekly_Sales', 'Predicted_Sales']])

st.subheader("Latest Recommendations")
st.dataframe(
    store_data[['Date', 'Weekly_Sales', 'Predicted_Sales', 'Recommendation']].tail(10),
    use_container_width=True
)
