import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
#BUAT MODEL SEDERHANA
# Path
DATA_PATH = pd("Data/Online Retail.xlsx")
MODEL_PATH = pd("Model/model_sales.pkl")

# Baca data
df = pd.read_excel(DATA_PATH)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Sales'] = df['Quantity'] * df['UnitPrice']

# Agregasi penjualan bulanan
df['Bulan'] = df['InvoiceDate'].dt.to_period('M')
monthly_sales = df.groupby('Bulan')['Sales'].sum().reset_index()

# Buat fitur waktu
monthly_sales['Bulan_ordinal'] = monthly_sales['Bulan'].apply(lambda p: p.ordinal)
base_month_ordinal = monthly_sales['Bulan_ordinal'].min()
monthly_sales['t'] = monthly_sales['Bulan_ordinal'] - base_month_ordinal
monthly_sales['month_num'] = monthly_sales['Bulan'].dt.month
monthly_sales['sin_12'] = np.sin(2*np.pi*monthly_sales['month_num']/12.0)
monthly_sales['cos_12'] = np.cos(2*np.pi*monthly_sales['month_num']/12.0)

# Model features
model_features = ['t', 'sin_12', 'cos_12']
X = monthly_sales[model_features]
y = monthly_sales['Sales']

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Simpan model
with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, model_features, base_month_ordinal), f)

print(f"Model berhasil dibuat di {MODEL_PATH}")

def load_data():
    return pd.read_excel("Data/Online Retail.xlsx", engine="openpyxl")

# Konfigurasi halaman (harus di awal skrip)
st.set_page_config(
    page_title="Dashboard Analisis Penjualan",
    page_icon="📈",
    layout="wide",  # Layout 'wide' membuat konten lebih lebar
    initial_sidebar_state="expanded"  # Sidebar langsung terbuka (visible) saat user pertama kali mengakses app
)