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


# =============================================================================
# 2. FUNCIONES DE EXTRACCIÓN (EXTRACT)
# =============================================================================

def cargar_dataset_base():
    nombre_archivo = "datos/datos_consolidados_40_registros.csv"

    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(directorio_script, nombre_archivo)

    try:
        df = pd.read_csv(ruta_archivo, encoding='utf-8')

        # --- CONFIRMACIÓN VISUAL DE CARGA EXITOSA ---
        print(f"  ✓ Archivo cargado: {nombre_archivo}")
        print(f"  ✓ Registros encontrados: {len(df)} filas × {len(df.columns)} columnas")
        print(f"  ✓ Columnas: {list(df.columns)}")
 
        # .head(3): Muestra las primeras 3 filas para inspección rápida.
        # Es más informativo que solo el número de filas.
        print("\n  Vista previa (primeras 3 filas):")
        print(df.head(3).to_string(index=False))
 
        # .dtypes: Muestra el tipo de dato de cada columna (int64, float64, object).
        # 'object' en pandas generalmente significa texto (str).
        print("\n  Tipos de datos detectados:")
        for col, dtype in df.dtypes.items():
            print(f"    - {col}: {dtype}")
 
        return df
    
    except FileNotFoundError:
        # Error específico: el archivo no existe en esa ruta.
        # Damos un mensaje accionable (no solo "error") que le dice al usuario qué hacer.
        print(f"  ✗ ERROR: No se encontró el archivo '{nombre_archivo}'")
        print(f"  ✗ Ruta buscada: {ruta_archivo}")
        print("  ✗ Asegúrate de que el CSV esté en la misma carpeta que este script.")
        return None
 
    except Exception as e:
        # Exception genérica: captura cualquier otro error inesperado (permisos, CSV corrupto, etc.)
        print(f"  ✗ ERROR inesperado al cargar el dataset: {e}")
        return None
    
def validar_estructura_datos(df):

    print("  Ejecutando validaciones de estructura...")
 
    # Inicializamos el reporte como diccionario.
    # Un diccionario es ideal aquí: clave = nombre de validación, valor = resultado.
    # Esto permite que otras funciones del pipeline consulten resultados específicos.
    reporte = {
        "total_filas": len(df),
        "total_columnas": len(df.columns),
        "columnas_presentes": list(df.columns),
        "validaciones": {}
    }
 
    # -----------------------------------------------------------------------
    # VALIDACIÓN 1: Columnas requeridas
    # -----------------------------------------------------------------------
    # Estas son las columnas que el reto define como obligatorias.
    # Si falta alguna, el pipeline no puede continuar — mejor abortar aquí.
    columnas_requeridas = ["id", "nombre", "categoria", "precio", "origen", "fecha_registro"]
 
    # set(): Convierte listas a conjuntos para comparación eficiente.
    # La diferencia de conjuntos (A - B) nos da los elementos en A que no están en B.
    columnas_faltantes = set(columnas_requeridas) - set(df.columns)
 
    if columnas_faltantes:
        print(f"  ✗ COLUMNAS FALTANTES: {columnas_faltantes}")
        reporte["validaciones"]["columnas_requeridas"] = {
            "estado": "FALLIDO",
            "faltantes": list(columnas_faltantes)
        }
    else:
        print(f"  ✓ Todas las columnas requeridas presentes: {columnas_requeridas}")
        reporte["validaciones"]["columnas_requeridas"] = {"estado": "OK"}
 
    # -----------------------------------------------------------------------
    # VALIDACIÓN 2: Valores nulos por columna
    # -----------------------------------------------------------------------
    # .isnull(): Devuelve DataFrame booleano (True donde hay NaN).
    # .sum(): Cuenta los True por columna → número de nulos por columna.
    nulos_por_columna = df.isnull().sum()
 
    # Solo nos interesan columnas que TIENEN nulos (> 0).
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]
 
    if len(columnas_con_nulos) > 0:
        print(f"  ⚠ Valores nulos detectados:")
        for col, cantidad in columnas_con_nulos.items():
            porcentaje = (cantidad / len(df)) * 100
            print(f"    - {col}: {cantidad} nulos ({porcentaje:.1f}%)")
        reporte["validaciones"]["valores_nulos"] = {
            "estado": "ADVERTENCIA",
            "detalle": columnas_con_nulos.to_dict()
        }
    else:
        print("  ✓ Sin valores nulos en el dataset")
        reporte["validaciones"]["valores_nulos"] = {"estado": "OK", "total_nulos": 0}
 
    # -----------------------------------------------------------------------
    # VALIDACIÓN 3: Duplicados
    # -----------------------------------------------------------------------
    # .duplicated(): Marca con True cada fila que es duplicado de otra anterior.
    # .sum() cuenta cuántas filas duplicadas hay.
    total_duplicados = df.duplicated().sum()
 
    if total_duplicados > 0:
        print(f"  ⚠ Filas duplicadas encontradas: {total_duplicados}")
        reporte["validaciones"]["duplicados"] = {
            "estado": "ADVERTENCIA",
            "cantidad": int(total_duplicados)
        }
    else:
        print("  ✓ Sin filas duplicadas")
        reporte["validaciones"]["duplicados"] = {"estado": "OK", "cantidad": 0}
 
    # -----------------------------------------------------------------------
    # VALIDACIÓN 4: Tipos de datos esperados
    # -----------------------------------------------------------------------
    # En pandas, texto se almacena como dtype "object".
    # Verificamos que precio sea numérico (int o float) — si es "object", hay un problema.
    if "precio" in df.columns:
        if pd.api.types.is_numeric_dtype(df["precio"]):
            print(f"  ✓ Columna precio es numérica (dtype: {df['precio'].dtype})")
            reporte["validaciones"]["tipo_precio"] = {"estado": "OK"}
        else:
            print(f"  ✗ Columna precio NO es numérica (dtype: {df['precio'].dtype})")
            print("    Puede contener simbolos como $, comas o espacios. Se corregira en limpieza.")
            reporte["validaciones"]["tipo_precio"] = {
                "estado": "ADVERTENCIA",
                "dtype_actual": str(df["precio"].dtype)
            }
 
    # -----------------------------------------------------------------------
    # VALIDACIÓN 5: Sanity check de negocio — rango de precios
    # -----------------------------------------------------------------------
    # Un sanity check verifica que los datos tienen sentido en el mundo real.
    # Precios negativos o cero son señal de error de captura de datos.
    if "precio" in df.columns and pd.api.types.is_numeric_dtype(df["precio"]):
        precio_min = df["precio"].min()
        precio_max = df["precio"].max()
        precios_invalidos = (df["precio"] <= 0).sum()
 
        print(f"  ✓ Rango de precios: ${precio_min:,.2f} — ${precio_max:,.2f}")
        reporte["validaciones"]["rango_precios"] = {
            "estado": "OK" if precios_invalidos == 0 else "ADVERTENCIA",
            "min": float(precio_min),
            "max": float(precio_max),
            "precios_invalidos": int(precios_invalidos)
        }
 
        if precios_invalidos > 0:
            print(f"  ⚠ Precios con valor <= 0 encontrados: {precios_invalidos}")
 
    # -----------------------------------------------------------------------
    # RESUMEN FINAL DEL REPORTE
    # -----------------------------------------------------------------------
    estados = [v.get("estado") for v in reporte["validaciones"].values()]
    hay_fallo = "FALLIDO" in estados
    hay_advertencia = "ADVERTENCIA" in estados
 
    print("\n  -- RESUMEN DE VALIDACION --")
    print(f"  Filas: {reporte['total_filas']} | Columnas: {reporte['total_columnas']}")
 
    if hay_fallo:
        print("  Estado general: ✗ FALLIDO — Corrige los errores antes de continuar")
    elif hay_advertencia:
        print("  Estado general: ⚠ CON ADVERTENCIAS — Revisar antes de continuar")
    else:
        print("  Estado general: ✓ APROBADO — Dataset listo para procesar")
 
    reporte["estado_general"] = "FALLIDO" if hay_fallo else ("ADVERTENCIA" if hay_advertencia else "OK")
 
    return reporte
