import streamlit as st
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# ==============================
# TITULO Y DESCRIPCIÓN
# ==============================
st.title("🧮 Calculadora de Priors LogNormal")

st.markdown("""
Esta aplicación calcula los parámetros `loc` y `scale` de una distribución **LogNormal**
a partir de la media y la desviación estándar reales que defines.

🔍 **¿Para qué sirve?**
Perfecta para configurar priors en modelos de Marketing Mix Modeling (MMM), como Meridian o cualquier otro que use LogNormal.

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
    help="Por ejemplo, el ROI esperado o cualquier parámetro positivo."
)
desviacion_real = st.number_input(
    "👉 Desviación estándar real deseada (debe ser > 0):", 
    min_value=0.0001, 
    value=1.0, 
    step=0.1,
    help="La incertidumbre que asumes sobre ese parámetro."
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

        # Crear la distribución LogNormal con scipy.stats.lognorm
        # En scipy: s=scale (σ_log), scale=np.exp(loc), loc=0 siempre
        dist = lognorm(s=scale, scale=np.exp(loc))

        # Comprobamos media y desviación real
        media_real_check = dist.mean()
        desviacion_real_check = dist.std()
        st.info(f"📈 **Verificación:** Media real: {media_real_check:.4f}, Desviación real: {desviacion_real_check:.4f}")

        # ==============================
        # GRÁFICA DE LA DISTRIBUCIÓN
        # ==============================
        x_max = media_real + 4 * desviacion_real
        x = np.linspace(0.001, x_max, 500)
        y = dist.pdf(x)

        fig, ax = plt.subplots()
        ax.plot(x, y, label="LogNormal PDF")
        ax.set_title("Distribución LogNormal generada")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad de probabilidad")
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

