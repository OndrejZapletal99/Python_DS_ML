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
history_data_path = script_dir / "prediction_history.csv" # Path to historical prediction data

# --- Text dictionary (only English) ---
TEXT = {
    "title": "☀️ PV Output Prediction",
    "intro": (
        "This application predicts photovoltaic (PV) power output based on the current weather forecast for Valašské Meziříčí.\n"
        "The prediction is displayed **from tomorrow until the end of the available forecast**. For each day, you can compare the prediction with the actual output for the same day last year."
    ),
    "total_pred": "**Total Predicted Output:** {val:.1f} kWh",
    "total_actual": "**Total Actual Output Last Year (same period):** {val:.1f} kWh",
    "total_actual_missing": "Actual output data for comparison last year is not available.",
    "graph1_title": "Predicted PV Output from Tomorrow Onwards",
    "graph1_ylabel": "Predicted PV [kWh]",
    "graph1_xtitle": "Date",
    "graph1_tab": "📈 Predicted Output Graph", # Simplified for subheader
    "graph2_title": "Actual Output for the Same Days Last Year (for comparison)",
    "graph2_ylabel": "PV Output [kWh]",
    "graph2_xtitle": "Date",
    "graph2_tab": "📊 Actual Output Last Year Graph", # Simplified for subheader
    "graph3_title": "Comparison of Predicted and Actual Output (T+ Days)",
    "graph3_ylabel": "PV Output [kWh]",
    "graph3_xtitle": "Relative Day (T+X)",
    "graph3_tab": "📈📊 Combined Comparison Graph", # Simplified for subheader
    "graph3_missing": "Cannot display combined graph as comparison data is missing.",
    "download": "⬇️ Download CSV with Prediction",
    # --- New texts for Evaluation section ---
    "evaluation_section_title": "📊 Prediction Evaluation",
    "evaluation_intro": "Here you can analyze how the predicted PV output for a specific future date has changed over time. Select a date to see its prediction history.",
    "select_date_label": "Select Date to Evaluate:",
    "evaluation_graph_title": "Historical Prediction for {selected_date}",
    "evaluation_graph_ylabel": "Predicted PV [kWh]",
    "evaluation_graph_xtitle": "Prediction Download Date",
    "evaluation_data_missing": "No historical prediction data available for the selected date or file not found.",
    "evaluation_file_error": "Could not load historical prediction data. Please ensure 'prediction_history.csv' exists and is correctly formatted."
}

# --- Page config and header ---
st.set_page_config(page_title=TEXT["title"], layout="wide")
st.title(TEXT["title"])
st.markdown(TEXT["intro"])

@st.cache_resource(ttl=3600) # Cache will reset every hour for model and encoder
def load_model_and_encoder():
    """Loads the pre-trained model and OneHotEncoder."""
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

model, ohe = load_model_and_encoder()

@st.cache_data(ttl=3600) # Cache will reset every hour for forecast data
def load_forecast():
    """Fetches weather forecast data from Visual Crossing API."""
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Vala%C5%A1sk%C3%A9%20Mezi%C5%99%C3%AD%C4%8D%C3%AD%2C%20Zl%C3%ADnsk%C3%BD%20kraj%2C%20%C4%8Ceska%CC%81%20Republika?unitGroup=metric&include=days&key=H5X3D8N2WKAAQ68QDPHD5UA8J&contentType=json"
    response = requests.get(url)
    data = response.json()
    days = data.get("days", [])
    df_forecast = pd.json_normalize(days)
    return df_forecast

df_forecast = load_forecast()

# Filter forecast data from tomorrow onwards
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
    """Converts a time string (HH:MM:SS) to total seconds."""
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
    """Applies OneHotEncoding to 'preciptype' column."""
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

# Load and filter last year's data for comparison
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


# Display total sums
total_pred = result_df["PV(kWh)_pred"].sum()
total_last_year = df_last_year_filtered["PV(kWh)"].sum() if not df_last_year_filtered.empty else 0

col1, col2 = st.columns(2)
with col1:
    st.success(TEXT["total_pred"].format(val=total_pred))
with col2:
    if total_last_year > 0:
        st.info(TEXT["total_actual"].format(val=total_last_year))
    else:
        st.warning(TEXT["total_actual_missing"])

# Main expander for "Prediction Results" section
with st.expander("Forecasted PV Output & Comparison", expanded=True):
    st.markdown("Here you can see the predicted PV output for the upcoming days and compare it with last year's actual output.")
    
    # Sub-section for Predicted Output Graph (using subheader and markdown separator)
    st.markdown("---") 
    st.subheader(TEXT["graph1_tab"]) 
    st.markdown(f"#### {TEXT['graph1_title']}") 
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(result_df["datetime"].astype(str), result_df["PV(kWh)_pred"], color="#1976D2")
    for x, y in zip(result_df["datetime"].astype(str), result_df["PV(kWh)_pred"]):
        ax1.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')
    ax1.set_xlabel(TEXT["graph1_xtitle"])
    ax1.set_ylabel(TEXT["graph1_ylabel"])
    ax1.set_title(TEXT["graph1_title"])
    ax1.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    st.dataframe(result_df[["datetime", "PV(kWh)_pred"]], use_container_width=True)

    # Sub-section for Actual Output Last Year Graph
    st.markdown("---") 
    st.subheader(TEXT["graph2_tab"]) 
    st.markdown(f"#### {TEXT['graph2_title']}") 
    if not df_last_year_filtered.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(df_last_year_filtered["Datum"].dt.strftime('%Y-%m-%d'), df_last_year_filtered["PV(kWh)"], color="#43A047")
        for x, y in zip(df_last_year_filtered["Datum"].dt.strftime('%Y-%m-%d'), df_last_year_filtered["PV(kWh)"]):
            ax2.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')
        ax2.set_xlabel(TEXT["graph2_xtitle"])
        ax2.set_ylabel(TEXT["graph2_ylabel"])
        ax2.set_title(TEXT["graph2_title"])
        ax2.set_ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        st.dataframe(df_last_year_filtered[["Datum", "PV(kWh)"]].reset_index(drop=True), use_container_width=True)
    else:
        st.warning(TEXT["graph2_missing"])

    # Sub-section for Combined Comparison Graph
    st.markdown("---") 
    st.subheader(TEXT["graph3_tab"]) 
    st.markdown(f"#### {TEXT['graph3_title']}") 
    if not combined_df.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bar_width = 0.35
        r1 = np.arange(len(combined_df))
        r2 = [x + bar_width for x in r1]
        ax3.bar(r1, combined_df["PV(kWh)_pred"], color="#1976D2", width=bar_width, label='Predicted PV')
        ax3.bar(r2, combined_df["PV(kWh)_last_year"], color="#43A047", width=bar_width, label='Last Year PV')
        for i, (pred_val, last_year_val) in combined_df[['PV(kWh)_pred', 'PV(kWh)_last_year']].iterrows():
            ax3.text(r1[i], pred_val + 0.2, f"{pred_val:.1f}", ha='center', fontsize=9, color='black')
            if not pd.isna(last_year_val):
                ax3.text(r2[i], last_year_val + 0.2, f"{last_year_val:.1f}", ha='center', fontsize=9, color='black')
        ax3.set_xlabel(TEXT["graph3_xtitle"])
        ax3.set_ylabel(TEXT["graph3_ylabel"])
        ax3.set_title(TEXT["graph3_title"])
        ax3.set_xticks([r + bar_width / 2 for r in range(len(combined_df))])
        ax3.set_xticklabels(combined_df['relative_day'])
        ax3.set_ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        st.dataframe(combined_df[['datetime', 'relative_day', 'PV(kWh)_pred', 'PV(kWh)_last_year']].reset_index(drop=True), use_container_width=True)
    else:
        st.warning(TEXT["graph3_missing"])

# Download Prediction Button (green color)
st.download_button(
    label=TEXT["download"],
    data=result_df.to_csv(index=False).encode('utf-8'),
    file_name=f"pv_prediction_{result_df['datetime'].min().strftime('%Y%m%d')}_{result_df['datetime'].max().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    help="Click to download the predicted PV output as a CSV file.",
    type="primary" # 'primary' sets the button to the theme's primary color (often green or blue)
)


## 📊 Prediction Evaluation

st.markdown("---") # Separator for the new section
st.header(TEXT["evaluation_section_title"])
st.markdown(TEXT["evaluation_intro"])

@st.cache_data(ttl=3600)
def load_prediction_history():
    """Loads and preprocesses the prediction history data."""
    try:
        df_history = pd.read_csv(history_data_path, parse_dates=["datetime", "DownloadDate"])
        # Convert datetime columns to date type for comparison without time
        df_history['datetime'] = df_history['datetime'].dt.date
        df_history['DownloadDate'] = df_history['DownloadDate'].dt.date
        return df_history
    except FileNotFoundError:
        st.error(TEXT["evaluation_file_error"])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"{TEXT['evaluation_file_error']} Error: {e}")
        return pd.DataFrame()

df_history = load_prediction_history()

# Main expander for "Evaluation" section
with st.expander("View Historical Prediction Changes", expanded=True):
    if not df_history.empty:
        # Get unique dates for selection from history (dates for which predictions exist)
        unique_prediction_dates = sorted(df_history['datetime'].unique())
        
        if unique_prediction_dates:
            # Convert dates to string for display in the selectbox
            selected_date_str = st.selectbox(
                TEXT["select_date_label"],
                options=[d.strftime('%Y-%m-%d') for d in unique_prediction_dates]
            )
            selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()

            # Sub-section for the historical graph (using subheader and markdown separator)
            st.markdown("---")
            st.subheader(TEXT["evaluation_graph_title"].format(selected_date=selected_date_str)) 
            
            # Filter history for the selected date
            filtered_history = df_history[df_history['datetime'] == selected_date].copy()
            
            if not filtered_history.empty:
                # Display the graph
                # --- Vykreslení grafu s reálnou hodnotou ---
                fig_eval, ax_eval = plt.subplots(figsize=(10, 5))

                # Predikce v čase
                ax_eval.plot(
                    filtered_history["DownloadDate"].astype(str),
                    filtered_history["PV(kWh)_pred"],
                    marker='o',
                    color='#FF8C00',
                    label="Predicted PV"
                )

                # 🟢 Načti reálnou hodnotu z SEMS_data.csv
                try:
                    df_actual = pd.read_csv(data_path, parse_dates=["Datum"])
                    df_actual['Datum'] = pd.to_datetime(df_actual['Datum'], dayfirst=True, errors='coerce')
                    actual_value_row = df_actual[df_actual['Datum'].dt.date == selected_date]

                    if not actual_value_row.empty:
                        actual_val = actual_value_row['PV(kWh)'].values[0]
                        ax_eval.axhline(y=actual_val, color='green', linestyle='--', linewidth=2, label=f"Actual PV = {actual_val:.1f} kWh")
                    else:
                        st.info("No actual data found for the selected date.")
                except Exception as e:
                    st.warning(f"Error loading actual production data: {e}")

                # Formátování grafu
                for x, y in zip(filtered_history["DownloadDate"].astype(str), filtered_history["PV(kWh)_pred"]):
                    ax_eval.text(x, y + 0.2, f"{y:.1f}", ha='center', fontsize=9, color='black')

                ax_eval.set_xlabel(TEXT["evaluation_graph_xtitle"])
                ax_eval.set_ylabel(TEXT["evaluation_graph_ylabel"])
                ax_eval.set_title(TEXT["evaluation_graph_title"].format(selected_date=selected_date_str))
                ax_eval.set_ylim(bottom=0)
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                ax_eval.legend()
                plt.tight_layout()
                st.pyplot(fig_eval)
                
                # Display data in a table
                st.dataframe(filtered_history[['DownloadDate', 'PV(kWh)_pred']].sort_values(by='DownloadDate', ascending=False).reset_index(drop=True), use_container_width=True)
            else:
                st.warning(TEXT["evaluation_data_missing"])
        else:
            st.warning(TEXT["evaluation_data_missing"])
    else:
        st.warning(TEXT["evaluation_data_missing"])