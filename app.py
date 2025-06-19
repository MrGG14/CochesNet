import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# --- Funciones de procesamiento y estrategias ---
def regression_data_processing(df, engine_cc_filter, var='kms'):
    filtered_df = df[df['engine'] == str(engine_cc_filter)].copy()
    filtered_df = filtered_df.dropna(subset=[var])
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

# def plot_strategy_separately(df, model, var, discount_factor=0.85, strategy_mask=None, strategy_name="Estrategia"):
#     X_range = np.linspace(df[var].min(), df[var].max(), 100).reshape(-1, 1)
#     y_pred = model.predict(X_range)
#     y_discount = y_pred * discount_factor

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.scatter(df.loc[~strategy_mask, var], df.loc[~strategy_mask, 'price'], alpha=0.5, label='Datos')
#     ax.scatter(df.loc[strategy_mask, var], df.loc[strategy_mask, 'price'], color='red', label='Buena oferta')
#     ax.plot(X_range, y_pred, color='blue', label='Regresión lineal')
#     ax.plot(X_range, y_discount, color='green', linestyle='--', label=f'Oferta (~{int((1 - discount_factor)*100)}% menos)')
#     ax.set_xlabel('Kilómetros')
#     ax.set_ylabel('Precio (€)')
#     ax.set_title(f'Relación {var} vs Precio - {strategy_name}')
#     ax.legend()
#     ax.grid(True)
#     return fig


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

df = pd.read_csv('./cochesnet_z3_infered.csv', 
                 dtype={'year': 'Int64', 'kms': 'Int64', 'cv': 'Int64', 'engine': 'str'})

st.subheader("Parámetros del modelo")
engine_filter = st.selectbox("Filtrar por cilindrada (engine)", np.append(np.sort(np.array([float(x) for x in df['engine'].unique() if not pd.isna(x)])), np.nan))

var = st.selectbox("Variable para regresión", options=['kms', 'cv', 'year'])

df_filtered, model = regression_data_processing(df, engine_filter, var=var)

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
    strategy_mask = strategy_discount_line(df_filtered, discount / 100)
    fig = plot_strategy_plotly(df_filtered, model, var=var, discount_factor=discount/100,
                                    strategy_mask=strategy_mask, strategy_name=strategy_name)

elif strategy_name == "Residuo > umbral":
    threshold = st.number_input("Umbral (€)", value=1000)
    strategy_mask = strategy_residual_threshold(df_filtered, threshold)
    fig = plot_strategy_plotly(df_filtered, model, var=var, discount_factor=0.85,
                                    strategy_mask=strategy_mask, strategy_name=strategy_name)

elif strategy_name == "Residuo percentil":
    percentile = st.slider("Percentil (%)", 0, 100, 90)
    strategy_mask = strategy_top_percentile_residuals(df_filtered, percentile)
    fig = plot_strategy_plotly(df_filtered, model, var=var, discount_factor=0.85,
                                    strategy_mask=strategy_mask, strategy_name=strategy_name)

elif strategy_name == "Z-score del residuo":
    zscore = st.slider("Z-score mínimo", 0.0, 3.0, 1.5)
    strategy_mask = strategy_residual_zscore(df_filtered, zscore)
    fig = plot_strategy_plotly(df_filtered, model, var=var, discount_factor=0.85,
                                    strategy_mask=strategy_mask, strategy_name=strategy_name)

st.plotly_chart(fig, use_container_width=True)

# Mostrar las filas que cumplen la estrategia
st.subheader("Ofertas detectadas")
st.dataframe(df_filtered[strategy_mask].sort_values(by='price', ascending=True).reset_index(drop=True))

