import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Konfigurasi halaman (harus di awal skrip)
st.set_page_config(
    page_title="Sales Dashboard | I Made Bayu Satria Wardhana",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/online_retail_store.csv")

# Load data penjualan
df_sales = load_data()
df_sales.columns = df_sales.columns.str.lower().str.replace(' ', '_')
df_sales['invoicedate'] = pd.to_datetime(df_sales['invoicedate'])  # Mengubah ke datetime

#load model
with open("Model/model_sales.pkl", "rb") as f:
    sales_prediction_model, model_features, base_month_ordinal = pickle.load(f)

#buat judul
st.title("E-Commerce Sales Dashboard")
st.markdown("**Analisis performa penjualan** untuk memantau tren, produk terlaris, dan prediksi penjualan.")
st.markdown("---")

# Pilihan halaman
pilihan_halaman = st.sidebar.radio(
    "Pilih Halaman:",
    ("Overview Dashboard",)
)

# Filter halaman Overview
if pilihan_halaman == "Overview Dashboard":
    st.sidebar.subheader("Filter Data")

    # Filter tanggal
    min_date = df_sales['invoicedate'].min().date()
    max_date = df_sales['invoicedate'].max().date()

    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter negara
    country_list = sorted(df_sales['country'].dropna().unique())
    selected_countries = st.sidebar.multiselect(
        "Pilih Negara:",
        options=country_list,
        default=country_list
    )

    # Filter produk
    product_list = sorted(df_sales['description'].dropna().unique())
    selected_products = st.sidebar.multiselect(
        "Pilih Produk:",
        options=product_list,
        default=product_list[:10]  
    )

    # Terapkan filter
    filtered_df = df_sales.copy()

    if len(date_range) == 2:
        start_date_filter = pd.to_datetime(date_range[0])
        end_date_filter = pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df['invoicedate'] >= start_date_filter) &
            (filtered_df['invoicedate'] <= end_date_filter)
        ]

    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    filtered_df = filtered_df[filtered_df['description'].isin(selected_products)]

    #Hitung Total Penjualan
    filtered_df['TotalSales'] = filtered_df['quantity'] * filtered_df['unitprice']

    #Ringkasan Metrik
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Penjualan", f"£ {filtered_df['TotalSales'].sum():,.2f}")

    with col2:
        st.metric("Total Pesanan", filtered_df['invoiceno'].nunique())

    with col3:
        st.metric("Jumlah Pelanggan", filtered_df['customerid'].nunique())

    with col4:
        avg_order_value = filtered_df['TotalSales'].sum() / filtered_df['invoiceno'].nunique()
        st.metric("Rata-rata Order", f"£ {avg_order_value:,.2f}")


    st.markdown("---")

    #Tren Penjualan
    st.subheader("Tren Penjualan dari Waktu ke Waktu")
    sales_over_time = filtered_df.groupby('invoicedate')['TotalSales'].sum().reset_index()
    fig_sales = px.line(sales_over_time, x='invoicedate', y='TotalSales',
                    title='Total Penjualan per Hari', markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)

    st.markdown("---")

    #Top Produk & Distribusi Negara
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 Produk Terlaris")
    top_products = filtered_df.groupby('description')['TotalSales'].sum().nlargest(10).reset_index()
    fig_top_products = px.bar(top_products, x='TotalSales', y='description',
                              orientation='h', title="Top Produk", color='TotalSales',
                              color_continuous_scale='Blues')
    fig_top_products.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_top_products, use_container_width=True)

    with col_right:
        st.subheader("Distribusi Penjualan per Negara")
    sales_by_country = filtered_df.groupby('country')['TotalSales'].sum().reset_index()
    fig_country = px.pie(sales_by_country, values='TotalSales', names='country',
                         title='Proporsi Penjualan per Negara', hole=0.3)
    st.plotly_chart(fig_country, use_container_width=True)

    st.markdown("---")

    #Data Mentah
    with st.expander("Lihat Data Mentah"):
        st.dataframe(filtered_df)

    #Slider untuk memilih berapa banyak baris data yang ingin ditampilkan
    num_rows_to_display = st.slider(
            "Jumlah Baris Data yang Ditampilkan:",
            min_value=10, 
            max_value=200,  
            value=50,  
            step=10  
        )

        #Tampilkan tabel data sesuai jumlah baris yang dipilih
    st.dataframe(filtered_df.head(num_rows_to_display))

        #Tampilkan statistik deskriptif
    st.write("Statistik Deskriptif:")
    st.dataframe(filtered_df.describe())

    