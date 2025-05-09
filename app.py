import streamlit as st
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# ==============================
# TITULO Y DESCRIPCIÓN
# ==============================
st.title("🧮 Calculadora de Priors LogNormal")

st.markdown("""
Esta aplicación te ayuda a calcular los parámetros `loc` y `scale` de una distribución **LogNormal**
a partir de la media y la desviación estándar reales que necesitas.

🔍 **¿Para qué sirve?**
Ideal para configurar priors en modelos de Marketing Mix Modeling (MMM), como Meridian u otros,
donde necesitas definir una distribución LogNormal a partir de tu conocimiento previo (por ejemplo, ROI esperado).

---
""")

# ==============================
# ENTRADAS DEL USUARIO
# ==============================
media_real = st.number_input(
    "👉 Media real deseada (debe ser > 0):", 
    min_value=0.0001, 
    value=2.0, 
    step=0.1,
    help="La media real que quieres para la distribución (por ejemplo, ROI esperado)."
)
desviacion_real = st.number_input(
    "👉 Desviación estándar real deseada (debe ser > 0):", 
    min_value=0.0001, 
    value=1.0, 
    step=0.1,
    help="La desviación estándar real que quieres para la distribución (incertidumbre esperada)."
)

# ==============================
# CÁLCULO AL HACER CLIC
# ==============================
if st.button("Calcular loc y scale"):
    try:
        # Calcular scale (σ_log)
        scale = np.sqrt(np.log(1 + (desviacion_real / media_real) ** 2))
        
        # Calcular loc (μ_log)
        loc = np.log(media_real) - 0.5 * scale ** 2

        st.success(f"✅ **loc (μ_log): {loc:.4f}**")
        st.success(f"✅ **scale (σ_log): {scale:.4f}**")

        # Crear la distribución LogNormal para verificar los valores
        dist = tfp.distributions.LogNormal(loc=loc, scale=scale)

        st.info(f"📈 **Verificación:** Media real: {dist.mean().numpy():.4f}, Desviación real: {dist.stddev().numpy():.4f}")

        # ==============================
        # GRÁFICA DE LA DISTRIBUCIÓN
        # ==============================
        x_max = media_real + 4 * desviacion_real
        x = np.linspace(0, x_max, 500)
        y = dist.prob(x)

        fig, ax = plt.subplots()
        ax.plot(x, y, label="LogNormal")
        ax.set_title("Distribución LogNormal generada")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad de probabilidad")
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
