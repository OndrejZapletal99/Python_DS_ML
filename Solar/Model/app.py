import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime, timedelta
from pathlib import Path

# --- Model & encoder path setting ---
script_dir = Path(__file__).parent
model_path = script_dir / "final_model_gbm.joblib"
encoder_path = script_dir / "encoder_preciptype.joblib"
data_path = script_dir / "SEMS_data.csv"

# --- Language selection ---
lang = st.radio("Language / Jazyk", ("English", "Čeština"), horizontal=True)

# --- Text dictionaries ---
TEXT = {
    "title": {
        "English": "☀️ PV Output Prediction",
        "Čeština": "☀️ Predikce výroby FVE"
    },
    "intro": {
        "English": (
            "This application predicts photovoltaic (PV) power output based on the current weather forecast for Valašské Meziříčí.\n"
            "The prediction is displayed **from tomorrow until the end of the available forecast**. For each day, you can compare the prediction with the actual output for the same day last year."
        ),
        "Čeština": (
            "Tato aplikace predikuje výrobu fotovoltaické elektrárny (FVE) na základě aktuální předpovědi počasí pro Valašské Meziříčí.\n"
            "Predikce je zobrazena **od zítřka do konce dostupné předpovědi**. Pro každý den můžete porovnat predikci se skutečnou výrobou za stejný den minulého roku."
        )
    },
    "total_pred": {
        "English": "**Total Predicted Output:**  {val:.1f} kWh",
        "Čeština": "**Celková predikovaná výroba:**  {val:.1f} kWh"
    },
    "total_actual": {
        "English": "**Total Actual Output Last Year (same period):** {val:.1f} kWh",
        "Čeština": "**Skutečná výroba minulý rok (stejné období):** {val:.1f} kWh"
    },
    "total_actual_missing": {
        "English": "Actual output data for comparison last year is not available.",
        "Čeština": "Skutečná data pro porovnání za minulý rok nejsou k dispozici."
    },
    "graph1_title": {
        "English": "Predicted PV Output from Tomorrow Onwards",
        "Čeština": "Predikovaná výroba FVE od zítřka"
    },
    "graph1_ylabel": {
        "English": "Predicted PV [kWh]",
        "Čeština": "Predikovaná výroba [kWh]"
    },
    "graph1_xtitle": {
        "English": "Date",
        "Čeština": "Datum"
    },
    "graph1_tab": {
        "English": "📈 Show Predicted Output Graph",
        "Čeština": "📈 Zobrazit graf predikované výroby"
    },
    "graph2_title": {
        "English": "Actual Output for the Same Days Last Year (for comparison)",
        "Čeština": "Skutečná výroba za stejné dny minulého roku (pro porovnání)"
    },
    "graph2_ylabel": {
        "English": "PV Output [kWh]",
        "Čeština": "Výroba FVE [kWh]"
    },
    "graph2_xtitle": {
        "English": "Date",
        "Čeština": "Datum"
    },
    "graph2_tab": {
        "English": "📊 Show Actual Output Last Year Graph",
        "Čeština": "📊 Zobrazit graf skutečné výroby minulý rok"
    },
    "graph2_missing": {
        "English": "Actual output data for these days last year is not available.",
        "Čeština": "Skutečná data pro tyto dny minulého roku nejsou k dispozici."
    },
    "graph3_title": {
        "English": "Comparison of Predicted and Actual Output (T+ Days)",
        "Čeština": "Porovnání predikované a skutečné výroby (T+ dny)"
    },
    "graph3_ylabel": {
        "English": "PV Output [kWh]",
        "Čeština": "Výroba FVE [kWh]"
    },
    "graph3_xtitle": {
        "English": "Relative Day (T+X)",
        "Čeština": "Relativní den (T+X)"
    },
    "graph3_tab": {
        "English": "📈📊 Show Combined Comparison",
        "Čeština": "📈📊 Zobrazit kombinované porovnání"
    },
    "graph3_missing": {
        "English": "Cannot display combined graph as comparison data is missing.",
        "Čeština": "Nelze zobrazit kombinovaný graf, protože chybí data pro porovnání."
    },
    "download": {
        "English": "⬇️ Download CSV with Prediction",
        "Čeština": "⬇️ Stáhnout CSV s predikcí"
    }
}

# --- Page config and header ---
st.set_page_config(page_title=TEXT["title"][lang], layout="wide")
st.title(TEXT["title"][lang])
st.markdown(TEXT["intro"][lang])

@st.cache_resource
def load_model_and_encoder():
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
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
    df_last_year = pd.read_csv(data_path, parse_dates=["Datum"])
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
    st.success(TEXT["total_pred"][lang].format(val=total_pred))
with col2:
    if total_last_year > 0:
        st.info(TEXT["total_actual"][lang].format(val=total_last_year))
    else:
        st.warning(TEXT["total_actual_missing"][lang])

#  GRAPH 1: Predicted Output 
with st.expander(TEXT["graph1_tab"][lang], expanded=True):
    st.markdown(f"### {TEXT['graph1_title'][lang]}")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(result_df["datetime"].astype(str), result_df["PV(kWh)_pred"], color="#1976D2")
    for x, y in zip(result_df["datetime"].astype(str), result_df["PV(kWh)_pred"]):
        ax1.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')
    ax1.set_xlabel(TEXT["graph1_xtitle"][lang])
    ax1.set_ylabel(TEXT["graph1_ylabel"][lang])
    ax1.set_title(TEXT["graph1_title"][lang])
    ax1.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    st.dataframe(result_df[["datetime", "PV(kWh)_pred"]], use_container_width=True)

#  GRAPH 2: Actual Output Last Year 
with st.expander(TEXT["graph2_tab"][lang], expanded=False):
    st.markdown(f"### {TEXT['graph2_title'][lang]}")
    if not df_last_year_filtered.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(df_last_year_filtered["Datum"].dt.strftime('%Y-%m-%d'), df_last_year_filtered["PV(kWh)"], color="#43A047")
        for x, y in zip(df_last_year_filtered["Datum"].dt.strftime('%Y-%m-%d'), df_last_year_filtered["PV(kWh)"]):
            ax2.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')
        ax2.set_xlabel(TEXT["graph2_xtitle"][lang])
        ax2.set_ylabel(TEXT["graph2_ylabel"][lang])
        ax2.set_title(TEXT["graph2_title"][lang])
        ax2.set_ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        st.dataframe(df_last_year_filtered[["Datum", "PV(kWh)"]].reset_index(drop=True), use_container_width=True)
    else:
        st.warning(TEXT["graph2_missing"][lang])

#  GRAPH 3: Combined Comparison 
with st.expander(TEXT["graph3_tab"][lang], expanded=False):
    st.markdown(f"### {TEXT['graph3_title'][lang]}")
    if not combined_df.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bar_width = 0.35
        r1 = np.arange(len(combined_df))
        r2 = [x + bar_width for x in r1]
        ax3.bar(r1, combined_df["PV(kWh)_pred"], color="#1976D2", width=bar_width, label='Predicted PV' if lang == "English" else "Predikce FVE")
        ax3.bar(r2, combined_df["PV(kWh)_last_year"], color="#43A047", width=bar_width, label='Last Year PV' if lang == "English" else "Minulý rok FVE")
        for i, (pred_val, last_year_val) in combined_df[['PV(kWh)_pred', 'PV(kWh)_last_year']].iterrows():
            ax3.text(r1[i], pred_val + 0.2, f"{pred_val:.1f}", ha='center', fontsize=9, color='black')
            if not pd.isna(last_year_val):
                ax3.text(r2[i], last_year_val + 0.2, f"{last_year_val:.1f}", ha='center', fontsize=9, color='black')
        ax3.set_xlabel(TEXT["graph3_xtitle"][lang])
        ax3.set_ylabel(TEXT["graph3_ylabel"][lang])
        ax3.set_title(TEXT["graph3_title"][lang])
        ax3.set_xticks([r + bar_width / 2 for r in range(len(combined_df))])
        ax3.set_xticklabels(combined_df['relative_day'])
        ax3.set_ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        st.dataframe(combined_df[['datetime', 'relative_day', 'PV(kWh)_pred', 'PV(kWh)_last_year']].reset_index(drop=True), use_container_width=True)
    else:
        st.warning(TEXT["graph3_missing"][lang])

#  Download Prediction Button 
st.download_button(
    label=TEXT["download"][lang],
    data=result_df.to_csv(index=False).encode('utf-8'),
    file_name=f"pv_prediction_{result_df['datetime'].min().strftime('%Y%m%d')}_{result_df['datetime'].max().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

#