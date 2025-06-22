import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import plotly.graph_objects as go

# --- Funciones de procesamiento y estrategias ---
def regression_data_processing(df, var='kms'):
    filtered_df = df.dropna(subset=[var])
    X = filtered_df[[var]].values
    y = filtered_df['price'].values
    model = LinearRegression()
    model.fit(X, y)
    filtered_df['predicted_price'] = model.predict(X)
    return filtered_df, model

def strategy_discount_line(df, discount_factor=0.25):
    df = df.copy()
    df['discount_line'] = df['predicted_price'] * (1- discount_factor)
    return df['price'] < df['discount_line']

def strategy_residual_threshold(df, threshold_euros=1000):
    df = df.copy()
    df['residual'] = df['predicted_price'] - df['price']
    return df['residual'] > threshold_euros

def strategy_top_percentile_residuals(df, percentile=90):
    df = df.copy()
    df['residual'] = df['predicted_price'] - df['price']
    threshold = np.percentile(df['residual'], percentile)
    return df['residual'] > threshold

def strategy_residual_zscore(df, z_threshold=1.5):
    df = df.copy()
    df['residual'] = df['predicted_price'] - df['price']
    z_mean = df['residual'].mean()
    z_std = df['residual'].std()
    df['z_score'] = (df['residual'] - z_mean) / z_std
    return df['z_score'] > z_threshold


def plot_strategy_plotly(df, model, var='kms', discount_factor=0.25, strategy_mask=None, strategy_name="Estrategia"):
    X_range = np.linspace(df[var].min(), df[var].max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_range)
    y_discount = y_pred * (1-discount_factor)

    # Gráfico interactivo
    fig = go.Figure()

    # Datos generales
    fig.add_trace(go.Scatter(
        x=df.loc[~strategy_mask, var],
        y=df.loc[~strategy_mask, 'price'],
        mode='markers',
        name='Datos',
        marker=dict(color='lightblue', opacity=0.5)
    ))

    # Datos que cumplen la estrategia
    fig.add_trace(go.Scatter(
        x=df.loc[strategy_mask, var],
        y=df.loc[strategy_mask, 'price'],
        mode='markers',
        name='Buena oferta',
        marker=dict(color='red', size=10, symbol='circle')
    ))

    # Línea de regresión
    fig.add_trace(go.Scatter(
        x=X_range.flatten(),
        y=y_pred,
        mode='lines',
        name='Regresión lineal',
        line=dict(color='blue')
    ))

    # Línea de descuento
    fig.add_trace(go.Scatter(
        x=X_range.flatten(),
        y=y_discount,
        mode='lines',
        name=f'Oferta (~{int(discount_factor*100)}% menos)',
        line=dict(color='green', dash='dash')
    ))

    fig.update_layout(
        title=f"Relación {var} vs Precio - {strategy_name}",
        xaxis_title=var,
        yaxis_title='Precio (€)',
        legend_title='Leyenda',
        template='plotly_white',
        width=1100,  # << NUEVO ancho explícito
        height=600
    )

    return fig


# --- Streamlit App ---
st.title("Análisis de precios de coches")

# Carga de datos
df = pd.read_csv('./cochesnet_full.csv', 
                 dtype={'year': 'Int64', 'kms': 'Int64', 'cv': 'Int64', 'engine': 'str'})

st.subheader("Selección de filtros")

brands = df['brand'].unique()
st.multiselect("Selecciona una o varias marcas", options=brands, key='brand_filter_multi')
selected_brands = st.session_state['brand_filter_multi'] if 'brand_filter_multi' in st.session_state else []
if selected_brands:
    df = df[df['brand'].isin(selected_brands)]

models = df['model'].unique()
st.multiselect("Selecciona uno o varios modelos", options=models, key='model_filter_multi')
selected_models = st.session_state['model_filter_multi'] if 'model_filter_multi' in st.session_state else []
if selected_models:
    df = df[df['model'].isin(selected_models)]
 
# Filtro de engine primero
engine_options = np.append(np.sort(np.array([float(x) for x in df['engine'].unique() if x != 'No data'])), 'No data')
engine_filter = st.selectbox(
    "Filtrar por cilindrada (engine)",
    options=np.insert(engine_options, 0, ''),
    format_func=lambda x: "Selecciona..." if x == '' else str(x)
)
if engine_filter and engine_filter != '':
    df = df[df['engine'] == str(engine_filter)]

# Ahora las versiones solo muestran las disponibles tras filtrar por engine
versions = df['version'].unique()
st.multiselect("Selecciona una o varias versiones", options=versions, key='version_filter_multi')
selected_versions = st.session_state['version_filter_multi'] if 'version_filter_multi' in st.session_state else []
if selected_versions:
    df = df[df['version'].isin(selected_versions)]


var = st.selectbox("Variable para regresión", options=['kms', 'cv', 'year'])

# Filtros adicionales
remove_outliers = st.checkbox("Eliminar outliers (precio fuera de 1.5*IQR)", value=False)
if remove_outliers:
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df['price'] >= lower) & (df['price'] <= upper)]

years = df['year'].dropna().unique()
years = np.sort(years)
min_year = int(years.min()) if len(years) > 0 else 2000
max_year = int(years.max()) if len(years) > 0 else 2025

if min_year < max_year:
    year_range = st.slider(
        "Selecciona rango de años",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
else:
    st.info(f"Solo hay datos para el año {min_year}.")
    df = df[df['year'] == min_year]

df, model = regression_data_processing(df, var=var)

# Filtrar por marca, modelo, versión y cilindrada (engine)
# if selected_brands:
#     df = df[df['brand'].isin(selected_brands)]
# if selected_models:
#     df = df[df['model'].isin(selected_models)]
# if selected_versions:
#     df = df[df['version'].isin(selected_versions)]


# Estrategias
st.subheader("Estrategia de filtrado")
strategy_name = st.selectbox(
    "Selecciona una estrategia",
    options=[
        "Descuento Línea",
        "Residuo > umbral",
        "Residuo percentil",
        "Z-score del residuo"
    ]
)

if strategy_name == "Descuento Línea":
    discount = st.slider("Descuento (%)", 0, 100, 25)
    strategy_mask = strategy_discount_line(df, discount / 100)
    fig = plot_strategy_plotly(df, model, var=var, discount_factor=discount/100,
                                    strategy_mask=strategy_mask, strategy_name=strategy_name)

elif strategy_name == "Residuo > umbral":
    threshold = st.number_input("Umbral (€)", value=1000)
    strategy_mask = strategy_residual_threshold(df, threshold)
    fig = plot_strategy_plotly(df, model, var=var, discount_factor=0.25,
                                    strategy_mask=strategy_mask, strategy_name=strategy_name)

elif strategy_name == "Residuo percentil":
    percentile = st.slider("Percentil (%)", 0, 100, 90)
    strategy_mask = strategy_top_percentile_residuals(df, percentile)
    fig = plot_strategy_plotly(df, model, var=var, discount_factor=0.25,
                                    strategy_mask=strategy_mask, strategy_name=strategy_name)

elif strategy_name == "Z-score del residuo":
    zscore = st.slider("Z-score mínimo", 0.0, 3.0, 1.5)
    strategy_mask = strategy_residual_zscore(df, zscore)
    fig = plot_strategy_plotly(df, model, var=var, discount_factor=0.25,
                                    strategy_mask=strategy_mask, strategy_name=strategy_name)

st.plotly_chart(fig, use_container_width=True)

# Mostrar las filas que cumplen la estrategia
st.subheader("Ofertas detectadas")
st.dataframe(df[strategy_mask].sort_values(by='price', ascending=True).reset_index(drop=True))

