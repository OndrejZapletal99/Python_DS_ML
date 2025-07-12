import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Predikce výroby fotovoltaiky", layout="wide")
st.title("☀️ Predikce výroby fotovoltaiky")

st.markdown("""
Tato aplikace predikuje výrobu fotovoltaiky na základě aktuální předpovědi počasí pro Valašské Meziříčí.
Predikce je zobrazena **od zítřka až do konce dostupné předpovědi**. Pro každý den je možné porovnat predikci s reálnou výrobou stejného dne v loňském roce.
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

# --- Filtruj od zítřka dál ---
today = datetime.now().date()
df_forecast['datetime'] = pd.to_datetime(df_forecast['datetime']).dt.date
df_pred = df_forecast[df_forecast['datetime'] > today]

if df_pred.empty:
    st.error("Žádná předpověď od zítřka dál není v datech k dispozici.")
    st.stop()

# Výběr a úprava sloupců
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

# Predikce
y_pred = model.predict(X_forecast_enc)
result_df = df_pred[['datetime']].copy()
result_df['PV(kWh)_pred'] = y_pred

# --- GRAF: Predikovaná výroba ---
with st.expander("📈 Zobrazit graf predikované výroby", expanded=True):
    st.markdown("**Predikovaná výroba fotovoltaiky od zítřka dál**")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(result_df["datetime"].astype(str), result_df["PV(kWh)_pred"], color="#1976D2")
    for x, y in zip(result_df["datetime"].astype(str), result_df["PV(kWh)_pred"]):
        ax1.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Predikce PV [kWh]")
    ax1.set_title("Predikovaná výroba (model)")
    ax1.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    st.dataframe(result_df, use_container_width=True)

total_pred = result_df["PV(kWh)_pred"].sum()
st.success(f"**Celková predikovaná výroba ({result_df['datetime'].min().strftime('%d.%m.%Y')} až {result_df['datetime'].max().strftime('%d.%m.%Y')}): {total_pred:.1f} kWh**")

# --- GRAF: Skutečná výroba loni ---
with st.expander("📊 Zobrazit graf skutečné výroby loni", expanded=False):
    st.markdown("**Skutečná výroba za stejné dny v loňském roce (pro srovnání)**")
    try:
        df_last_year = pd.read_csv('Data/SEMS_data.csv', parse_dates=["Datum"])
        df_last_year['Datum'] = pd.to_datetime(df_last_year['Datum'], dayfirst=True, errors='coerce')
        # Najdi stejné dny a měsíce, ale minulý rok
        days_months = result_df['datetime'].apply(lambda d: (d.month, d.day)).tolist()
        df_last_year['month'] = df_last_year['Datum'].dt.month
        df_last_year['day'] = df_last_year['Datum'].dt.day
        mask = df_last_year.apply(lambda row: (row['month'], row['day']) in days_months, axis=1)
        df_last_year_filtered = df_last_year[mask]
        if not df_last_year_filtered.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.bar(df_last_year_filtered["Datum"].dt.strftime('%Y-%m-%d'), df_last_year_filtered["PV(kWh)"], color="#43A047")
            for x, y in zip(df_last_year_filtered["Datum"].dt.strftime('%Y-%m-%d'), df_last_year_filtered["PV(kWh)"]):
                ax2.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')
            ax2.set_xlabel("Datum")
            ax2.set_ylabel("Výroba PV [kWh]")
            ax2.set_title("Skutečná výroba loni")
            ax2.set_ylim(bottom=0)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            st.dataframe(df_last_year_filtered[["Datum", "PV(kWh)"]].reset_index(drop=True), use_container_width=True)
            st.info(f"**Celková skutečná výroba ({df_last_year_filtered['Datum'].min().strftime('%d.%m.%Y')} až {df_last_year_filtered['Datum'].max().strftime('%d.%m.%Y')}): {df_last_year_filtered['PV(kWh)'].sum():.1f} kWh**")
        else:
            st.warning("Data o skutečné výrobě pro tyto dny loni nejsou k dispozici.")
    except Exception as e:
        st.warning("Nepodařilo se načíst nebo zpracovat historická data pro srovnání.")

# --- Tlačítko pro stažení predikce ---
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
    label="⬇️ Stáhnout CSV s predikcí",
    data=result_df.to_csv(index=False).encode('utf-8'),
    file_name=f"predikce_pv_{result_df['datetime'].min().strftime('%Y%m%d')}_{result_df['datetime'].max().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)