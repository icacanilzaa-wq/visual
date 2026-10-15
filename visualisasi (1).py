import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import plotly.graph_objects as go
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Analisis Regresi Interaktif",
    page_icon="📊",
    layout="wide"
)

# Header Utama
st.title("📊 Dashboard Analisis Regresi Lineer Berganda")
st.caption("Aplikasi Interaktif Analisis Regresi Data Fiskal")

# 1. READ DATASET
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Alam-A99/DATAAKSI/main/dataset_fiskal_97_hipoFull.csv"
    df_all = pd.read_csv(url)
    return df_all

df_all = load_data()

# SIDEBAR: PENGATURAN PILIHAN DATA & VARIABEL
st.sidebar.header("⚙️ Pengaturan Model Regresi")

# 1. Pilih Kota/Kabupaten
list_kota = df_all['Kota'].unique() if 'Kota' in df_all.columns else []
selected_kota = st.sidebar.selectbox("Pilih Kota/Kabupaten:", options=list_kota, index=list_kota.tolist().index('Kota Surabaya') if 'Kota Surabaya' in list_kota else 0)

# Filter Data berdasarkan Kota yang dipilih
df = df_all[df_all['Kota'] == selected_kota].copy()

# Pilih Kolom Numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.sidebar.markdown("---")
# 2. Pilih Variabel Dependent (Y)
default_y = 'KEMISKINAN' if 'KEMISKINAN' in numeric_cols else numeric_cols[0]
var_y = st.sidebar.selectbox("Pilih Variabel Terikat (Y):", options=numeric_cols, index=numeric_cols.index(default_y) if default_y in numeric_cols else 0)

# 3. Pilih Variabel Independent (X1 dan X2)
available_x = [col for col in numeric_cols if col != var_y]

default_x1 = 'bm_ekonomi' if 'bm_ekonomi' in available_x else available_x[0]
var_x1 = st.sidebar.selectbox("Pilih Variabel Bebas 1 (X1):", options=available_x, index=available_x.index(default_x1) if default_x1 in available_x else 0)

available_x2 = [col for col in available_x if col != var_x1]
default_x2 = 'DBH_PAJAK' if 'DBH_PAJAK' in available_x2 else available_x2[0]
var_x2 = st.sidebar.selectbox("Pilih Variabel Bebas 2 (X2):", options=available_x2, index=available_x2.index(default_x2) if default_x2 in available_x2 else 0)

st.sidebar.markdown("---")
st.sidebar.metric("Total Data Terpilih", f"{len(df)} baris")

# 2. PROSES MODEL REGRESI
X = df[[var_x1, var_x2]].dropna()
Y = df.loc[X.index, var_y]

X_sm = sm.add_constant(X)
model = sm.OLS(Y, X_sm).fit()

# Ringkasan Performa Model (Metric Cards)
col1, col2, col3, col4 = st.columns(4)
col1.metric("R-Squared (R²)", f"{model.rsquared:.4f}")
col2.metric("Adj. R-Squared", f"{model.rsquared_adj:.4f}")
col3.metric("F-Statistic", f"{model.fvalue:.2f}")
col4.metric("Prob (F-Statistic)", f"{model.f_pvalue:.4f}")

st.markdown("---")

# TABS UNTUK INTERFACE
tab1, tab2, tab3, tab4 = st.tabs(["📉 Visualisasi 3D", "📋 Uji Asumsi Klasik", "📑 Ringkasan OLS", "🔍 Data Frame"])

# TAB 1: VISUALISASI 3D INTERAKTIF
with tab1:
    st.subheader(f"Plot Bidang Regresi 3D: {var_y} terhadap {var_x1} & {var_x2} ({selected_kota})")
    
    # Grid Bidang Regresi
    x_range = np.linspace(df[var_x1].min(), df[var_x1].max(), 20)
    y_range = np.linspace(df[var_x2].min(), df[var_x2].max(), 20)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = model.params['const'] + (model.params[var_x1] * x_grid) + (model.params[var_x2] * y_grid)

    fig = go.Figure()

    # Permukaan Bidang
    fig.add_trace(go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        name='Bidang Regresi',
        colorscale='Viridis',
        opacity=0.8,
        showscale=False
    ))

    # Titik Data Aktual Asli
    fig.add_trace(go.Scatter3d(
        x=df[var_x1],
        y=df[var_x2],
        z=df[var_y],
        mode='markers',
        name='Data Aktual',
        marker=dict(size=6, color='red', line=dict(color='white', width=1))
    ))

    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title=var_x1,
            yaxis_title=var_x2,
            zaxis_title=var_y
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

# TAB 2: UJI ASUMSI KLASIK
with tab2:
    st.subheader("Pemeriksaan Uji Asumsi Klasik")
    
    # Kalkulasi Uji Klasik
    shapiro_stat, shapiro_p = stats.shapiro(model.resid)
    
    vif_data = pd.DataFrame()
    vif_data["Variabel"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
    dw_value = sm.stats.stattools.durbin_watson(model.resid)

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 1. Normalitas (Shapiro-Wilk)")
        st.write(f"**P-Value:** `{shapiro_p:.4f}`")
        if shapiro_p > 0.05:
            st.success("✅ **Lolos** (Residul berdistribusi normal, p-value > 0.05)")
        else:
            st.error("❌ **Tidak Lolos** (Residul tidak berdistribusi normal)")

        st.markdown("### 2. Heteroskedastisitas (Breusch-Pagan)")
        st.write(f"**P-Value:** `{bp_test[1]:.4f}`")
        if bp_test[1] > 0.05:
            st.success("✅ **Lolos** (Bebas Heteroskedastisitas, p-value > 0.05)")
        else:
            st.error("❌ **Terjadi Heteroskedastisitas**")

    with col_b:
        st.markdown("### 3. Multikolinearitas (VIF)")
        st.dataframe(vif_data, use_container_width=True, hide_index=True)
        if vif_data['VIF'].max() < 10:
            st.success("✅ **Lolos** (Tidak ada multikolinearitas, VIF < 10)")
        else:
            st.error("❌ **Terjadi Multikolinearitas**")

        st.markdown("### 4. Autokorelasi (Durbin-Watson)")
        st.write(f"**Nilai DW:** `{dw_value:.4f}`")
        if 1.5 <= dw_value <= 2.5:
            st.success("✅ **Lolos** (Mendekati nilai 2, bebas autokorelasi)")
        else:
            st.warning("⚠️ **Indikasi Autokorelasi**")

# TAB 3: SUMMARY OLS
with tab3:
    st.subheader("Persamaan Regresi")
    const_val = model.params['const']
    x1_val = model.params[var_x1]
    x2_val = model.params[var_x2]
    
    st.code(
        f"{var_y} = {const_val:.4f} + ({x1_val:.4f} * {var_x1}) + ({x2_val:.4f} * {var_x2})",
        language="text"
    )

    st.subheader("Ringkasan Statistik Model OLS")
    st.text(model.summary().as_text())

# TAB 4: DATASET PREVIEW
with tab4:
    st.subheader(f"Tabel Data {selected_kota}")
    st.dataframe(df, use_container_width=True)
