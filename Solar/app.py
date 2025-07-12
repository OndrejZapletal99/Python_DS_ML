import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="PV Output Prediction", layout="wide")
st.title("☀️ PV Output Prediction")

st.markdown("""
This application predicts photovoltaic (PV) power output based on the current weather forecast for Valašské Meziříčí.
The prediction is displayed **from tomorrow until the end of the available forecast**. For each day, you can compare the prediction with the actual output for the same day last year.
""")

@st.cache_resource
def load_model_and_encoder():
    model = joblib.load('./Model/final_model_gbm.joblib')
    encoder = joblib.load('./Model/encoder_preciptype.joblib')
    return model, encoder

model, ohe = load_model_and_encoder()

@st.cache_data
def load_forecast():
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Vala%C5%A1sk%C3%A9%20Mezi%C5%99%C3%AD%C4%8D%C3%AD%2C%20Zl%C3%ADnsk%C3%BD%20kraj%2C%20%C4%8Cesk%C3%A1%20Republika?unitGroup=metric&include=days&key=H5X3D8N2WKAAQ68QDPHD5UA8J&contentType=json"
    response = requests.get(url)
    data = response.json()
    days = data.get("days", [])
    df_forecast = pd.json_normalize(days)
    return df_forecast

df_forecast = load_forecast()

#  Filter from tomorrow onwards 
today = datetime.now().date()
df_forecast['datetime'] = pd.to_datetime(df_forecast['datetime']).dt.date
df_pred = df_forecast[df_forecast['datetime'] > today]

if df_pred.empty:
    st.error("No forecast data available from tomorrow onwards.")
    st.stop()

# Select and preprocess columns for prediction
columns = [
    'cloudcover', 'datetime', 'dew', 'feelslike', 'feelslikemax', 'feelslikemin', 'humidity',
    'precip', 'precipcover', 'preciptype', 'pressure', 'snow', 'snowdepth', 'solarenergy',
    'solarradiation', 'sunrise', 'sunset', 'temp', 'tempmax', 'tempmin', 'uvindex', 'visibility',
    'winddir', 'windgust', 'windspeed'
]
df_forecast_filtered = df_pred[columns].copy()
df_forecast_filtered['datetime'] = pd.to_datetime(df_forecast_filtered['datetime'])
df_forecast_filtered['month'] = df_forecast_filtered['datetime'].dt.month
df_forecast_filtered['dayofyear'] = df_forecast_filtered['datetime'].dt.dayofyear
df_forecast_filtered['sin_month'] = np.sin(2 * np.pi * df_forecast_filtered['month'] / 12)
df_forecast_filtered['cos_month'] = np.cos(2 * np.pi * df_forecast_filtered['month'] / 12)
df_forecast_filtered['sin_dayofyear'] = np.sin(2 * np.pi * df_forecast_filtered['dayofyear'] / 365)
df_forecast_filtered['cos_dayofyear'] = np.cos(2 * np.pi * df_forecast_filtered['dayofyear'] / 365)
df_forecast_filtered['year'] = df_forecast_filtered['datetime'].dt.year

def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h*3600 + m*60 + s

df_forecast_filtered['sunset_seconds'] = df_forecast_filtered['sunset'].apply(time_to_seconds)
df_forecast_filtered['sunrise_seconds'] = df_forecast_filtered['sunrise'].apply(time_to_seconds)
seconds_in_day = 24*3600
df_forecast_filtered['sunset_sin'] = np.sin(2 * np.pi * df_forecast_filtered['sunset_seconds'] / seconds_in_day)
df_forecast_filtered['sunset_cos'] = np.cos(2 * np.pi * df_forecast_filtered['sunset_seconds'] / seconds_in_day)
df_forecast_filtered['sunrise_sin'] = np.sin(2 * np.pi * df_forecast_filtered['sunrise_seconds'] / seconds_in_day)
df_forecast_filtered['sunrise_cos'] = np.cos(2 * np.pi * df_forecast_filtered['sunrise_seconds'] / seconds_in_day)
df_forecast_filtered['day_length_seconds'] = df_forecast_filtered['sunset_seconds'] - df_forecast_filtered['sunrise_seconds']
df_forecast_filtered['day_length_hours'] = df_forecast_filtered['day_length_seconds'] / 3600

df_forecast_final = df_forecast_filtered.drop(columns=['sunset', 'sunrise', 'month', 'dayofyear'])
df_forecast_final = df_forecast_final.drop(columns=['datetime'])

df_forecast_final['preciptype'] = df_forecast_final['preciptype'].fillna("['none']").apply(str)

def encode_precip(df, ohe):
    df = df.copy()
    encoded = ohe.transform(df[['preciptype']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(['preciptype']),
        index=df.index
    )
    return pd.concat([df.drop(columns=['preciptype']), encoded_df], axis=1)

X_forecast_enc = encode_precip(df_forecast_final, ohe)

desired_order = [
    'winddir', 'feelslikemin', 'dew', 'windgust', 'precipcover', 'uvindex',
    'snow', 'solarenergy', 'visibility', 'precip', 'pressure', 'tempmin',
    'cloudcover', 'tempmax', 'windspeed', 'feelslikemax', 'temp',
    'feelslike', 'solarradiation', 'humidity', 'snowdepth', 'sin_month',
    'cos_month', 'sin_dayofyear', 'cos_dayofyear', 'year', 'sunset_seconds',
    'sunrise_seconds', 'sunset_sin', 'sunset_cos', 'sunrise_sin',
    'sunrise_cos', 'day_length_seconds', 'day_length_hours',
    "preciptype_['none']", "preciptype_['rain', 'snow']",
    "preciptype_['rain']", "preciptype_['snow']"
]
X_forecast_enc = X_forecast_enc[desired_order]

# Prediction
y_pred = model.predict(X_forecast_enc)
result_df = df_pred[['datetime']].copy()
result_df['PV(kWh)_pred'] = y_pred

#  Load and filter last year's data 
df_last_year_filtered = pd.DataFrame()
try:
    df_last_year = pd.read_csv('Data/SEMS_data.csv', parse_dates=["Datum"])
    df_last_year['Datum'] = pd.to_datetime(df_last_year['Datum'], dayfirst=True, errors='coerce')
    
    # Find the same days and months, but last year
    days_months = result_df['datetime'].apply(lambda d: (d.month, d.day)).tolist()
    
    df_last_year['month'] = df_last_year['Datum'].dt.month
    df_last_year['day'] = df_last_year['Datum'].dt.day
    
    mask = df_last_year.apply(lambda row: (row['month'], row['day']) in days_months, axis=1)
    df_last_year_filtered = df_last_year[mask].copy() # Use .copy() to prevent SettingWithCopyWarning
    df_last_year_filtered = df_last_year_filtered.sort_values(by='Datum').reset_index(drop=True)

    # Merge for combined graph
    # Create a helper column for month and day comparison
    result_df['month_day'] = result_df['datetime'].apply(lambda d: (d.month, d.day))
    df_last_year_filtered['month_day'] = df_last_year_filtered['Datum'].apply(lambda d: (d.month, d.day))

    # Merge based on month and day
    combined_df = pd.merge(result_df, df_last_year_filtered[['month_day', 'PV(kWh)']], on='month_day', how='left')
    combined_df.rename(columns={'PV(kWh)': 'PV(kWh)_last_year'}, inplace=True)
    combined_df.drop(columns=['month_day'], inplace=True)

    # Add relative day for combined graph
    combined_df['relative_day'] = [f"T+{i+1}" for i in range(len(combined_df))]

except Exception as e:
    st.warning(f"Could not load or process historical data for comparison: {e}")
    combined_df = result_df.copy()
    combined_df['PV(kWh)_last_year'] = np.nan
    combined_df['relative_day'] = [f"T+{i+1}" for i in range(len(combined_df))]


#  Display total sums 
total_pred = result_df["PV(kWh)_pred"].sum()
total_last_year = df_last_year_filtered["PV(kWh)"].sum() if not df_last_year_filtered.empty else 0

col1, col2 = st.columns(2)
with col1:
    st.success(f"**Total Predicted Output:**  {total_pred:.1f} kWh")
with col2:
    if total_last_year > 0:
        st.info(f"**Total Actual Output Last Year (same period):** {total_last_year:.1f} kWh")
    else:
        st.warning("Actual output data for comparison last year is not available.")

#  GRAPH 1: Predicted Output 
with st.expander("📈 Show Predicted Output Graph", expanded=True):
    st.markdown("### Predicted PV Output from Tomorrow Onwards")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(result_df["datetime"].astype(str), result_df["PV(kWh)_pred"], color="#1976D2")
    for x, y in zip(result_df["datetime"].astype(str), result_df["PV(kWh)_pred"]):
        ax1.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Predicted PV [kWh]")
    ax1.set_title("Predicted Output (Model)")
    ax1.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    st.dataframe(result_df, use_container_width=True)



#  GRAPH 2: Actual Output Last Year 
with st.expander("📊 Show Actual Output Last Year Graph", expanded=False):
    st.markdown("### Actual Output for the Same Days Last Year (for comparison)")
    if not df_last_year_filtered.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(df_last_year_filtered["Datum"].dt.strftime('%Y-%m-%d'), df_last_year_filtered["PV(kWh)"], color="#43A047")
        for x, y in zip(df_last_year_filtered["Datum"].dt.strftime('%Y-%m-%d'), df_last_year_filtered["PV(kWh)"]):
            ax2.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("PV Output [kWh]")
        ax2.set_title("Actual Output Last Year")
        ax2.set_ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        st.dataframe(df_last_year_filtered[["Datum", "PV(kWh)"]].reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Actual output data for these days last year is not available.")



#  GRAPH 3: Combined Comparison 
with st.expander("📈📊 Show Combined Comparison", expanded=False):
    st.markdown("### Comparison of Predicted and Actual Output (T+ Days)")
    if not combined_df.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        
        # Bar width and position
        bar_width = 0.35
        r1 = np.arange(len(combined_df))
        r2 = [x + bar_width for x in r1]

        ax3.bar(r1, combined_df["PV(kWh)_pred"], color="#1976D2", width=bar_width, label='Predicted PV')
        ax3.bar(r2, combined_df["PV(kWh)_last_year"], color="#43A047", width=bar_width, label='Last Year PV')

        # Labels on bars
        for i, (pred_val, last_year_val) in combined_df[['PV(kWh)_pred', 'PV(kWh)_last_year']].iterrows():
            ax3.text(r1[i], pred_val + 0.2, f"{pred_val:.1f}", ha='center', fontsize=9, color='black')
            if not pd.isna(last_year_val):
                ax3.text(r2[i], last_year_val + 0.2, f"{last_year_val:.1f}", ha='center', fontsize=9, color='black')

        ax3.set_xlabel("Relative Day (T+X)")
        ax3.set_ylabel("PV Output [kWh]")
        ax3.set_title("Predicted vs. Actual Output Comparison")
        ax3.set_xticks([r + bar_width / 2 for r in range(len(combined_df))])
        ax3.set_xticklabels(combined_df['relative_day'])
        ax3.set_ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        st.dataframe(combined_df[['datetime', 'relative_day', 'PV(kWh)_pred', 'PV(kWh)_last_year']].reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Cannot display combined graph as comparison data is missing.")

#  Download Prediction Button 
st.markdown("""
    <style>
    .stDownloadButton>button {
        background-color: #43A047;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
        border: none;
        margin-top: 1em;
        margin-bottom: 1em;
    }
    .stDownloadButton>button:hover {
        background-color: #388E3C;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

st.download_button(
    label="⬇️ Download CSV with Prediction",
    data=result_df.to_csv(index=False).encode('utf-8'),
    file_name=f"pv_prediction_{result_df['datetime'].min().strftime('%Y%m%d')}_{result_df['datetime'].max().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)