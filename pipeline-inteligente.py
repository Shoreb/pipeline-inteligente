# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

# warnings: Para suprimir avisos no críticos que ensucian la salida del programa.
import warnings
warnings.filterwarnings('ignore')

# os: Módulo del sistema operativo. Lo usamos para construir rutas de archivos
# de forma compatible entre Windows, Mac y Linux (os.path.join).
import os

# datetime: Para trabajar con fechas. Necesario para generar fechas sintéticas
# y crear features temporales como "días desde registro".
from datetime import datetime, timedelta
 
# --- CONFIGURACIÓN VISUAL GLOBAL ---
# Establece el estilo visual por defecto de seaborn para todas las gráficas del proyecto.
# 'whitegrid' añade una cuadrícula blanca sobre fondo gris claro, ideal para datos cuantitativos.
sns.set_style("whitegrid")
 
# Configura el tamaño por defecto de las figuras de matplotlib: ancho=12, alto=8 pulgadas.
# Esto evita tener que especificar figsize en cada plt.figure() del proyecto.
plt.rcParams['figure.figsize'] = (12, 8)


