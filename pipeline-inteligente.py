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

# =============================================================================
# 3. FUNCIONES DE TRANSFORMACIÓN Y EXPANSIÓN (TRANSFORM)
# =============================================================================
def limpiar_datos(df):
 
    print("  Iniciando proceso de limpieza...")
 
    # -----------------------------------------------------------------------
    # PASO 1: Copiar el DataFrame
    # -----------------------------------------------------------------------
    # REGLA DE ORO: Nunca modificar el DataFrame original.
    # .copy() crea una copia independiente en memoria.
    # Sin .copy(), pandas puede modificar el original por referencia
    # y producir el temido SettingWithCopyWarning.
    df_limpio = df.copy()
    filas_iniciales = len(df_limpio)
 
    # -----------------------------------------------------------------------
    # PASO 2: Limpiar columnas de texto
    # -----------------------------------------------------------------------
    # .strip() elimina espacios al inicio y al final: "  Ropa  " → "Ropa"
    # .str es el accessor de pandas para operaciones vectorizadas sobre strings.
    # Lo aplicamos a todas las columnas de tipo object (texto).
    columnas_texto = df_limpio.select_dtypes(include=['object']).columns
    for col in columnas_texto:
        # fillna('') evita que .str.strip() falle en celdas con NaN
        df_limpio[col] = df_limpio[col].fillna('').str.strip()
        # Reemplazar string vacío por NaN real para manejo uniforme
        df_limpio[col] = df_limpio[col].replace('', np.nan)
 
    print(f"  ✓ Espacios eliminados en {len(columnas_texto)} columnas de texto")
 
    # -----------------------------------------------------------------------
    # PASO 3: Estandarizar columnas de texto a Title Case
    # -----------------------------------------------------------------------
    # Title Case: primera letra de cada palabra en mayúscula.
    # "electrónica" → "Electrónica" | "COLOMBIA" → "Colombia"
    # Esto evita duplicados por capitalización: "Ropa" ≠ "ropa" para el modelo.
    columnas_categoricas = ['categoria', 'origen', 'nombre']
    for col in columnas_categoricas:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].str.title()
 
    print("  ✓ Texto estandarizado a Title Case (categoria, origen, nombre)")
 
    # -----------------------------------------------------------------------
    # PASO 4: Corregir y limpiar columna 'precio'
    # -----------------------------------------------------------------------
    # Problema común: el CSV puede traer precios como "$1,200.50" (string).
    # pd.to_numeric con errors='coerce' convierte lo que pueda a número
    # y convierte lo que no pueda a NaN (en vez de explotar con error).
    if 'precio' in df_limpio.columns:
        # Si precio es string, eliminar símbolos de moneda y separadores de miles
        if df_limpio['precio'].dtype == object:
            df_limpio['precio'] = (
                df_limpio['precio']
                .astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
 
        # Convertir a numérico. errors='coerce' → valores no convertibles = NaN
        df_limpio['precio'] = pd.to_numeric(df_limpio['precio'], errors='coerce')
 
        # Corregir precios negativos o cero: reemplazar por la mediana de la categoría.
        # La mediana es más robusta que la media ante outliers.
        precios_invalidos = (df_limpio['precio'] <= 0) | df_limpio['precio'].isna()
        if precios_invalidos.sum() > 0:
            mediana_global = df_limpio['precio'].median()
            df_limpio.loc[precios_invalidos, 'precio'] = mediana_global
            print(f"  ⚠ {precios_invalidos.sum()} precios inválidos corregidos con mediana: ${mediana_global:,.2f}")
        else:
            print(f"  ✓ Precios válidos. Rango: ${df_limpio['precio'].min():,.2f} — ${df_limpio['precio'].max():,.2f}")
 
        # Redondear a 2 decimales para consistencia monetaria
        df_limpio['precio'] = df_limpio['precio'].round(2)
 
    # -----------------------------------------------------------------------
    # PASO 5: Estandarizar columna 'fecha_registro'
    # -----------------------------------------------------------------------
    # pd.to_datetime convierte strings a objetos datetime de pandas.
    # errors='coerce' → fechas inválidas se convierten en NaT (Not a Time).
    # dayfirst=True → interpreta DD/MM/YYYY (formato latinoamericano).
    if 'fecha_registro' in df_limpio.columns:
        df_limpio['fecha_registro'] = pd.to_datetime(
            df_limpio['fecha_registro'],
            errors='coerce',
            dayfirst=True
        )
 
        fechas_invalidas = df_limpio['fecha_registro'].isna().sum()
        if fechas_invalidas > 0:
            # Rellenar fechas inválidas con la fecha más frecuente del dataset
            fecha_moda = df_limpio['fecha_registro'].mode()[0]
            df_limpio['fecha_registro'].fillna(fecha_moda, inplace=True)
            print(f"  ⚠ {fechas_invalidas} fechas inválidas corregidas con moda: {fecha_moda.date()}")
        else:
            print(f"  ✓ Fechas estandarizadas. Rango: {df_limpio['fecha_registro'].min().date()} — {df_limpio['fecha_registro'].max().date()}")
 
    # -----------------------------------------------------------------------
    # PASO 6: Eliminar duplicados
    # -----------------------------------------------------------------------
    # keep='first': Si hay filas idénticas, conserva la primera y elimina el resto.
    # Alternativa keep='last': conserva la más reciente. keep=False: elimina todas.
    filas_antes = len(df_limpio)
    df_limpio = df_limpio.drop_duplicates(keep='first')
    duplicados_eliminados = filas_antes - len(df_limpio)
 
    if duplicados_eliminados > 0:
        print(f"  ⚠ {duplicados_eliminados} filas duplicadas eliminadas")
    else:
        print("  ✓ Sin duplicados encontrados")
 
    # -----------------------------------------------------------------------
    # PASO 7: Manejar nulos restantes en columnas categóricas
    # -----------------------------------------------------------------------
    # Para columnas de texto, rellenamos nulos con el valor más frecuente (moda).
    # Alternativa: eliminar la fila. Elegimos moda porque el dataset es pequeño (40 filas).
    for col in ['categoria', 'origen']:
        if col in df_limpio.columns and df_limpio[col].isna().sum() > 0:
            moda_col = df_limpio[col].mode()[0]
            df_limpio[col].fillna(moda_col, inplace=True)
            print(f"  ⚠ Nulos en '{col}' rellenados con moda: '{moda_col}'")
 
    # -----------------------------------------------------------------------
    # RESUMEN FINAL
    # -----------------------------------------------------------------------
    filas_finales = len(df_limpio)
    print(f"\n  -- RESUMEN DE LIMPIEZA --")
    print(f"  Filas entrada : {filas_iniciales}")
    print(f"  Filas salida  : {filas_finales}")
    print(f"  Filas perdidas: {filas_iniciales - filas_finales}")
    print(f"  ✓ Limpieza completada exitosamente")
 
    return df_limpio
 
 
def expandir_dataset_500_registros(df_original):
    """
    Expandir el dataset de 40 a 500 registros usando Data Augmentation
    con Ruido Gaussiano controlado por categoría.
 
    TÉCNICA ELEGIDA: Gaussian Noise Augmentation con preservación de proporciones
    """
    print("  Aplicando Data Augmentation con Ruido Gaussiano...")
    print(f"  Registros originales: {len(df_original)}")
 
    # Fijar semilla aleatoria para REPRODUCIBILIDAD.
    # Con la misma semilla, np.random siempre genera los mismos números.
    # Esto es esencial en ciencia de datos: otro científico debe poder replicar tus resultados.
    np.random.seed(42)
 
    # -----------------------------------------------------------------------
    # PASO 1: Calcular cuántos registros generar por categoría
    # -----------------------------------------------------------------------
    # Objetivo: 500 registros totales, manteniendo las mismas PROPORCIONES por categoría.
    # Si Electrónica representa el 30% de 40 registros → debe ser el 30% de 500.
 
    OBJETIVO_TOTAL = 500
    registros_a_generar = OBJETIVO_TOTAL - len(df_original)  # 460 nuevos registros
 
    # value_counts() cuenta cuántas filas hay por categoría.
    # normalize=True devuelve proporciones (0.0 a 1.0) en lugar de conteos absolutos.
    proporciones = df_original['categoria'].value_counts(normalize=True)
 
    print("\n  Distribucion original por categoria:")
    for cat, prop in proporciones.items():
        print(f"    - {cat}: {prop:.1%} ({int(df_original['categoria'].value_counts()[cat])} registros)")
 
    # -----------------------------------------------------------------------
    # PASO 2: Calcular estadísticas por categoría para generar precios realistas
    # -----------------------------------------------------------------------
    # groupby('categoria')['precio']: agrupa el DataFrame por categoría y accede a la columna precio.
    # .agg(['mean', 'std', 'min', 'max']): calcula múltiples estadísticas de una vez.
    stats_por_categoria = df_original.groupby('categoria')['precio'].agg(['mean', 'std', 'min', 'max'])
 
    # Rellenar std=NaN (ocurre si una categoría tiene solo 1 registro)
    # Usamos el 10% de la media como std estimada en ese caso.
    stats_por_categoria['std'] = stats_por_categoria['std'].fillna(
        stats_por_categoria['mean'] * 0.10
    )
 
    print("\n  Estadísticas de precio por categoría:")
    for cat, row in stats_por_categoria.iterrows():
        print(f"    - {cat}: media=${row['mean']:,.2f}, std=${row['std']:,.2f}, rango=[${row['min']:,.2f}, ${row['max']:,.2f}]")
 
    # -----------------------------------------------------------------------
    # PASO 3: Calcular valores únicos por columna para sampling
    # -----------------------------------------------------------------------
    # Para columnas como 'nombre' y 'origen', no generamos texto nuevo.
    # En su lugar, hacemos sampling (muestreo con reemplazo) de los valores existentes.
    valores_origen = df_original['origen'].dropna().unique().tolist()
    valores_nombres_por_cat = df_original.groupby('categoria')['nombre'].apply(list).to_dict()
 
    # Rango de fechas para generar fechas sintéticas coherentes
    fecha_min = df_original['fecha_registro'].min()
    fecha_max = df_original['fecha_registro'].max()
    rango_dias = (fecha_max - fecha_min).days
    # Si todas las fechas son iguales, usar un rango de 365 días hacia atrás
    if rango_dias == 0:
        rango_dias = 365
 
    # -----------------------------------------------------------------------
    # PASO 4: Generar registros sintéticos categoría por categoría
    # -----------------------------------------------------------------------
    # FACTOR_RUIDO: controla cuánta variación tienen los nuevos precios.
    # 0.15 significa que el ruido es el 15% de la desviación estándar de la categoría.
    # Muy pequeño (0.01): datos casi idénticos a los originales.
    # Muy grande (0.5+): datos irreales, fuera del rango creíble.
    FACTOR_RUIDO = 0.15
 
    registros_sinteticos = []  # Lista donde acumulamos los nuevos registros
    id_contador = df_original['id'].max() + 1  # IDs nuevos continúan desde el máximo existente
 
    for categoria, proporcion in proporciones.items():
        # Cuántos registros generar para esta categoría
        n_registros_cat = round(registros_a_generar * proporcion)
 
        # Estadísticas de precio para esta categoría específica
        media_precio = stats_por_categoria.loc[categoria, 'mean']
        std_precio   = stats_por_categoria.loc[categoria, 'std']
        min_precio   = stats_por_categoria.loc[categoria, 'min']
        max_precio   = stats_por_categoria.loc[categoria, 'max']
 
        # Obtener filas originales de esta categoría para sampling de otros campos
        filas_categoria = df_original[df_original['categoria'] == categoria]
 
        for _ in range(n_registros_cat):
 
            # GENERACIÓN DE PRECIO SINTÉTICO con ruido gaussiano
            # np.random.normal(loc, scale): genera 1 muestra de N(media, std*factor)
            # np.clip: recorta el valor para que no salga del rango original [min, max]
            # Esto garantiza que no generemos precios negativos o absurdamente altos.
            precio_sintetico = np.random.normal(
                loc=media_precio,
                scale=std_precio * FACTOR_RUIDO
            )
            precio_sintetico = float(np.clip(precio_sintetico, min_precio * 0.8, max_precio * 1.2))
            precio_sintetico = round(precio_sintetico, 2)
 
            # SAMPLING de nombre: elegir aleatoriamente uno de los nombres reales de esa categoría
            nombres_disponibles = valores_nombres_por_cat.get(categoria, ['Producto Genérico'])
            nombre_sintetico = np.random.choice(nombres_disponibles)
 
            # SAMPLING de origen: elegir un origen de los que existen en el dataset
            origen_sintetico = np.random.choice(valores_origen)
 
            # GENERACIÓN de fecha sintética dentro del rango existente
            dias_offset = np.random.randint(0, rango_dias + 1)
            fecha_sintetica = fecha_min + timedelta(days=int(dias_offset))
 
            # Construir el registro completo como diccionario
            # Las claves deben coincidir EXACTAMENTE con las columnas del DataFrame original
            nuevo_registro = {
                'id'              : id_contador,
                'nombre'          : nombre_sintetico,
                'categoria'       : categoria,
                'precio'          : precio_sintetico,
                'origen'          : origen_sintetico,
                'fecha_registro'  : fecha_sintetica
            }
 
            registros_sinteticos.append(nuevo_registro)
            id_contador += 1
 
    # -----------------------------------------------------------------------
    # PASO 5: Combinar datos originales + datos sintéticos
    # -----------------------------------------------------------------------
    # pd.DataFrame(lista_de_dicts): convierte lista de diccionarios en DataFrame.
    df_sintetico = pd.DataFrame(registros_sinteticos)
 
    # pd.concat: concatena DataFrames verticalmente (axis=0 = por filas).
    # ignore_index=True: resetea el índice del resultado (0, 1, 2, ..., 499).
    df_expandido = pd.concat([df_original, df_sintetico], axis=0, ignore_index=True)
 
    # -----------------------------------------------------------------------
    # PASO 6: Ajuste fino — asegurar exactamente 500 registros
    # -----------------------------------------------------------------------
    # El redondeo por categoría puede darnos 498 o 502 registros.
    # Ajustamos tomando o descartando filas del bloque sintético.
    if len(df_expandido) > OBJETIVO_TOTAL:
        df_expandido = df_expandido.iloc[:OBJETIVO_TOTAL]
    elif len(df_expandido) < OBJETIVO_TOTAL:
        faltantes = OBJETIVO_TOTAL - len(df_expandido)
        extras = df_sintetico.sample(n=faltantes, replace=True, random_state=42)
        df_expandido = pd.concat([df_expandido, extras], axis=0, ignore_index=True)
 
    # -----------------------------------------------------------------------
    # RESUMEN FINAL
    # -----------------------------------------------------------------------
    print("\n  Registros generados : {len(registros_sinteticos)}")
    print(f"  Total final         : {len(df_expandido)} registros")
    print("\n  Distribución final por categoría:")
    dist_final = df_expandido['categoria'].value_counts()
    for cat, count in dist_final.items():
        pct = count / len(df_expandido)
        print(f"    - {cat}: {count} registros ({pct:.1%})")
    print("\n  ✓ Dataset expandido a {len(df_expandido)} registros exitosamente")
 
    return df_expandido
 
 
def crear_variables_derivadas(df):
 
    print("  Creando variables derivadas (feature engineering)...")
 
    df_features = df.copy()
 
    # -----------------------------------------------------------------------
    # FEATURE GROUP 1: Variables de fecha
    # -----------------------------------------------------------------------
    # Las fechas como objeto datetime tienen atributos .dt que extraen componentes.
    # .dt.month → número del mes (1-12)
    # .dt.year  → año como entero
    # .dt.dayofweek → día de la semana (0=Lunes, 6=Domingo)
    # .dt.day   → día del mes (1-31)
 
    if 'fecha_registro' in df_features.columns:
        # Asegurar que la columna es datetime (puede haber llegado como string)
        df_features['fecha_registro'] = pd.to_datetime(df_features['fecha_registro'], errors='coerce')
 
        df_features['mes_registro']        = df_features['fecha_registro'].dt.month
        df_features['anio_registro']       = df_features['fecha_registro'].dt.year
        df_features['dia_semana_registro'] = df_features['fecha_registro'].dt.dayofweek
        df_features['dia_mes_registro']    = df_features['fecha_registro'].dt.day
 
        # DÍAS TRANSCURRIDOS: cuántos días han pasado desde el registro hasta hoy.
        # Esto convierte una fecha absoluta en un número continuo útil para el modelo.
        # pd.Timestamp.now() → momento actual. .dt.days extrae la cantidad de días del timedelta.
        fecha_referencia = df_features['fecha_registro'].max()  # Usamos la fecha máxima como referencia
        df_features['dias_desde_registro'] = (
            fecha_referencia - df_features['fecha_registro']
        ).dt.days
 
        print("  ✓ Features de fecha: mes, año, día_semana, día_mes, días_desde_registro")
 
    # -----------------------------------------------------------------------
    # FEATURE GROUP 2: Precio relativo a la media de su categoría
    # -----------------------------------------------------------------------
    # Esta feature responde: ¿Este producto es caro o barato DENTRO de su categoría?
    # precio_relativo = precio / media_precio_categoria
    # - valor > 1 → más caro que la media de su categoría
    # - valor < 1 → más barato que la media de su categoría
    # - valor = 1 → exactamente en la media
    # Es una normalización por grupo, muy útil para el modelo.
 
    if 'precio' in df_features.columns and 'categoria' in df_features.columns:
        # .transform('mean'): calcula la media por grupo y la "expande" de vuelta
        # al tamaño original del DataFrame (una media por fila según su categoría).
        # Es diferente a .agg(): agg colapsa filas, transform las mantiene.
        media_por_categoria = df_features.groupby('categoria')['precio'].transform('mean')
        df_features['precio_relativo_categoria'] = (df_features['precio'] / media_por_categoria).round(4)
 
        print("  ✓ Feature creada: precio_relativo_categoria")
 
    # -----------------------------------------------------------------------
    # FEATURE GROUP 3: Indicador de precio alto (variable binaria)
    # -----------------------------------------------------------------------
    # Una variable binaria (0 o 1) que indica si el precio supera la mediana global.
    # Las variables binarias (también llamadas dummies o flags) son muy útiles
    # porque capturan umbrales que la regresión lineal no detecta por sí sola.
 
    if 'precio' in df_features.columns:
        mediana_global = df_features['precio'].median()
        df_features['es_precio_alto'] = (df_features['precio'] > mediana_global).astype(int)
        # .astype(int) convierte True/False → 1/0, que el modelo puede usar directamente
        print(f"  ✓ Feature creada: es_precio_alto (umbral: ${mediana_global:,.2f})")
 
    # -----------------------------------------------------------------------
    # FEATURE GROUP 4: Indicador de origen nacional vs importado
    # -----------------------------------------------------------------------
    # Binarizamos el origen en: nacional (Colombia) = 1, importado = 0.
    # Esta simplificación puede capturar el efecto precio de los productos importados.
    if 'origen' in df_features.columns:
        df_features['es_nacional'] = (
            df_features['origen'].str.lower().str.contains('colombia', na=False)
        ).astype(int)
        print("  ✓ Feature creada: es_nacional (1=Colombia, 0=importado)")
 
    # -----------------------------------------------------------------------
    # RESUMEN
    # -----------------------------------------------------------------------
    columnas_nuevas = [c for c in df_features.columns if c not in df.columns]
    print("\n  Total features creadas: {len(columnas_nuevas)}")
    print(f"  Nuevas columnas: {columnas_nuevas}")
    print(f"  Shape final: {df_features.shape[0]} filas × {df_features.shape[1]} columnas")
 
    return df_features
 
 
def codificar_variables_categoricas(df):
    """
    Convertir columnas de texto (categóricas) a números para que el modelo ML pueda usarlas.
 
    TÉCNICA USADA: LabelEncoder
    Convierte cada categoría única a un entero:
    - "Electrónica" → 0
    - "Hogar"       → 1
    - "Ropa"        → 2
    (el orden es alfabético, asignado automáticamente)
    """
    print("  Codificando variables categóricas...")
 
    df_encoded = df.copy()
 
    # Diccionario para guardar los encoders entrenados.
    # BUENA PRÁCTICA: guardar el encoder permite:
    # 1. Decodificar resultados ("0" → "Electrónica") para informes
    # 2. Aplicar el mismo encoding a datos nuevos en producción
    # 3. Mantener consistencia: "Ropa" siempre será el mismo número
    
    encoders = {}
 
    # -----------------------------------------------------------------------
    # CODIFICACIÓN: columna 'categoria'
    # -----------------------------------------------------------------------
    if 'categoria' in df_encoded.columns:
 
        # Crear instancia del encoder (todavía no ha aprendido nada)
        le_categoria = LabelEncoder()
 
        # .fit_transform(serie): APRENDE las categorías únicas Y las convierte en números
        # Equivale a: le.fit(serie) seguido de le.transform(serie), pero en un solo paso.
        # El resultado es un array numpy que asignamos a la nueva columna.
        df_encoded['categoria_encoded'] = le_categoria.fit_transform(
            df_encoded['categoria'].fillna('Desconocido')
        )
 
        # Guardar el encoder entrenado en el diccionario
        encoders['categoria'] = le_categoria
 
        # Mostrar el mapeo aprendido para trazabilidad
        mapeo_cat = dict(zip(le_categoria.classes_, le_categoria.transform(le_categoria.classes_)))
        print(f"  ✓ 'categoria' codificada → categoria_encoded")
        print(f"    Mapeo: {mapeo_cat}")
 
    # -----------------------------------------------------------------------
    # CODIFICACIÓN: columna 'origen'
    # -----------------------------------------------------------------------
    if 'origen' in df_encoded.columns:
 
        le_origen = LabelEncoder()
        df_encoded['origen_encoded'] = le_origen.fit_transform(
            df_encoded['origen'].fillna('Desconocido')
        )
        encoders['origen'] = le_origen
 
        mapeo_orig = dict(zip(le_origen.classes_, le_origen.transform(le_origen.classes_)))
        print(f"  ✓ 'origen' codificada → origen_encoded")
        print(f"    Mapeo: {mapeo_orig}")
 
    # -----------------------------------------------------------------------
    # VERIFICACIÓN: columnas numéricas disponibles para el modelo
    # -----------------------------------------------------------------------
    # Identificar todas las columnas numéricas que el modelo podría usar.
    # Excluimos 'id' (es un identificador, no una feature) y 'precio' (es el target y).
    columnas_numericas = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    columnas_modelo    = [c for c in columnas_numericas if c not in ['id', 'precio']]
 
    print("\n  Variables disponibles para el modelo ML:")
    for col in columnas_modelo:
        print(f"    - {col}")
 
    print("\n  ✓ Encoding completado. Shape: {df_encoded.shape[0]} × {df_encoded.shape[1]}")
 
    # Retornamos AMBOS: el DataFrame codificado y los encoders guardados.
    # El pipeline principal los usará por separado.
    return df_encoded, encoders

 
# =============================================================================
# 4. FUNCIONES DE VALIDACIÓN Y CALIDAD (VALIDATE)
# =============================================================================
 
def detectar_outliers(df):
    """
    Identificar valores atípicos usando dos métodos estadísticos complementarios.
 
    MÉTODO 1 — IQR (Rango Intercuartílico):
        - Q1 = percentil 25, Q3 = percentil 75
        - IQR = Q3 - Q1  (el 50% central de los datos)
        - Límite inferior = Q1 - 1.5 × IQR
        - Límite superior = Q3 + 1.5 × IQR
        - Todo valor fuera de esos límites = outlier
        - Ventaja: robusto, no asume distribución normal
        - Uso típico: boxplots, análisis exploratorio
 
    MÉTODO 2 — Z-score:
        - z = (valor - media) / desviación_estándar
        - Si |z| > 3 → el valor está a más de 3 desviaciones de la media
        - Por la regla empírica, el 99.7% de datos normales caen dentro de ±3σ
        - Ventaja: intuitivo y fácil de comunicar a negocio
        - Limitación: asume distribución aproximadamente normal
 
    ¿QUÉ HACEMOS CON LOS OUTLIERS?
    En este proyecto los DETECTAMOS y REPORTAMOS pero NO los eliminamos.
    Razón: con solo 40 registros originales, eliminar outliers reduciría
    demasiado la información. Los reportamos para que el analista decida.
 
    Args:
        df (pd.DataFrame): Dataset expandido con variables numéricas
 
    Returns:
        dict: Reporte de outliers por columna y método
    """
 
    print("  Detectando outliers con métodos IQR y Z-score...")
 
    # Solo analizamos columnas numéricas que tengan sentido de negocio.
    # Excluimos id (identificador), columnas encoded (ya son categóricas convertidas)
    # y variables binarias (solo tienen 0 y 1, no pueden ser outliers).
    columnas_excluir = ['id', 'categoria_encoded', 'origen_encoded',
                        'es_precio_alto', 'es_nacional', 'dia_semana_registro',
                        'mes_registro', 'anio_registro', 'dia_mes_registro']
 
    columnas_analizar = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in columnas_excluir
    ]
 
    reporte_outliers = {}
    total_outliers_encontrados = 0
 
    for col in columnas_analizar:
        # Eliminar NaN antes de calcular — algunos métodos fallan con NaN
        serie = df[col].dropna()
 
        if len(serie) < 4:
            # Con menos de 4 valores no podemos calcular cuartiles confiables
            continue
 
        reporte_col = {}
 
        # -------------------------------------------------------------------
        # MÉTODO 1: IQR
        # -------------------------------------------------------------------
        Q1  = serie.quantile(0.25)   # Percentil 25: el 25% de datos está por debajo
        Q3  = serie.quantile(0.75)   # Percentil 75: el 75% de datos está por debajo
        IQR = Q3 - Q1                # Rango del 50% central de los datos
 
        limite_inferior_iqr = Q1 - 1.5 * IQR
        limite_superior_iqr = Q3 + 1.5 * IQR
 
        # Máscara booleana: True en cada fila que es outlier por IQR
        mascara_iqr = (serie < limite_inferior_iqr) | (serie > limite_superior_iqr)
        outliers_iqr = serie[mascara_iqr]
 
        reporte_col['IQR'] = {
            'Q1'               : round(float(Q1), 2),
            'Q3'               : round(float(Q3), 2),
            'IQR'              : round(float(IQR), 2),
            'limite_inferior'  : round(float(limite_inferior_iqr), 2),
            'limite_superior'  : round(float(limite_superior_iqr), 2),
            'total_outliers'   : int(mascara_iqr.sum()),
            'porcentaje'       : round(float(mascara_iqr.sum() / len(serie) * 100), 2),
            'valores'          : outliers_iqr.tolist()[:10]  # Máximo 10 para no saturar el reporte
        }
 
        # -------------------------------------------------------------------
        # MÉTODO 2: Z-score
        # -------------------------------------------------------------------
        # (valor - media) / std: cuántas desviaciones estándar está del centro
        media = serie.mean()
        std   = serie.std()
 
        if std == 0:
            # Si std=0 todos los valores son iguales, no hay outliers posibles
            reporte_col['Z_score'] = {'total_outliers': 0, 'nota': 'std=0, todos los valores son iguales'}
        else:
            z_scores     = (serie - media) / std
            # abs() → valor absoluto: nos da la distancia sin importar si es mayor o menor
            mascara_z    = z_scores.abs() > 3
            outliers_z   = serie[mascara_z]
 
            reporte_col['Z_score'] = {
                'media'          : round(float(media), 2),
                'std'            : round(float(std), 2),
                'umbral'         : 3,
                'total_outliers' : int(mascara_z.sum()),
                'porcentaje'     : round(float(mascara_z.sum() / len(serie) * 100), 2),
                'valores'        : outliers_z.tolist()[:10]
            }
 
        reporte_outliers[col] = reporte_col
        total_col = reporte_col['IQR']['total_outliers']
        total_outliers_encontrados += total_col
 
        # Imprimir resumen por columna
        pct = reporte_col['IQR']['porcentaje']
        estado = "⚠" if total_col > 0 else "✓"
        print(f"  {estado} {col}: {total_col} outliers por IQR ({pct}%) | "
              f"Rango válido: [{limite_inferior_iqr:,.2f}, {limite_superior_iqr:,.2f}]")
 
    # -----------------------------------------------------------------------
    # RESUMEN GLOBAL
    # -----------------------------------------------------------------------
    print("\n  Total outliers detectados (IQR): {total_outliers_encontrados}")
    print("  DECISIÓN: Outliers documentados pero NO eliminados.")
    print("  Justificación: dataset pequeño, outliers pueden ser valores reales de negocio.")
 
    reporte_outliers['_resumen'] = {
        'columnas_analizadas'      : len(columnas_analizar),
        'total_outliers_iqr'       : total_outliers_encontrados,
        'decision'                 : 'conservar',
        'justificacion'            : 'Dataset pequeño. Outliers pueden representar productos premium reales.'
    }
 
    return reporte_outliers
 
 
def validar_calidad_expansion(df_original, df_expandido):
    """
    Verificar que el dataset expandido es estadísticamente coherente con el original.
 
    PRINCIPIO: Un buen Data Augmentation no cambia la naturaleza del dataset.
    Si los precios originales tenían media=$500, el expandido también debería
    tener media cercana a $500. Si la distribución cambia mucho, los datos
    sintéticos introducen sesgos que contaminarán el modelo ML.
 
    QUÉ COMPARAMOS:
    1. Estadísticas descriptivas de precio (media, mediana, std, min, max)
    2. Proporciones por categoría
    3. Proporciones por origen
    4. Rango de fechas
 
    MÉTRICA DE CALIDAD USADA: Diferencia porcentual
        diff% = |valor_expandido - valor_original| / valor_original × 100
    Si diff% < 15% → expansión aceptable
    Si diff% > 15% → revisar técnica de augmentation
 
    Args:
        df_original  (pd.DataFrame): Dataset de 40 registros originales
        df_expandido (pd.DataFrame): Dataset de 500 registros expandidos
 
    Returns:
        dict: Reporte comparativo con métricas de calidad
    """
 
    print("  Comparando distribuciones: original vs expandido...")
 
    reporte_calidad = {
        'n_original' : len(df_original),
        'n_expandido': len(df_expandido),
        'comparaciones': {}
    }
 
    UMBRAL_TOLERANCIA = 15.0  # % máximo de diferencia aceptable
 
    # -----------------------------------------------------------------------
    # COMPARACIÓN 1: Estadísticas descriptivas de precio
    # -----------------------------------------------------------------------
    # .describe() calcula en una sola llamada: count, mean, std, min, 25%, 50%, 75%, max
    if 'precio' in df_original.columns and 'precio' in df_expandido.columns:
 
        stats_orig = df_original['precio'].describe()
        stats_exp  = df_expandido['precio'].describe()
 
        metricas_precio = {}
        print("\n  Comparación de precio (original → expandido):")
        print(f"  {'Métrica':<12} {'Original':>12} {'Expandido':>12} {'Diferencia':>12} {'Estado':>8}")
        print("  " + "-" * 58)
 
        for metrica in ['mean', '50%', 'std', 'min', 'max']:
            val_orig = stats_orig[metrica]
            val_exp  = stats_exp[metrica]
 
            # Diferencia porcentual: qué tanto cambió en términos relativos
            if val_orig != 0:
                diff_pct = abs(val_exp - val_orig) / abs(val_orig) * 100
            else:
                diff_pct = 0.0
 
            estado = "✓ OK" if diff_pct <= UMBRAL_TOLERANCIA else "⚠ REVISAR"
            nombre_metrica = 'mediana' if metrica == '50%' else metrica
 
            print(f"  {nombre_metrica:<12} {val_orig:>12,.2f} {val_exp:>12,.2f} {diff_pct:>11.1f}% {estado:>8}")
 
            metricas_precio[nombre_metrica] = {
                'original' : round(float(val_orig), 2),
                'expandido': round(float(val_exp), 2),
                'diff_pct' : round(diff_pct, 2),
                'estado'   : 'OK' if diff_pct <= UMBRAL_TOLERANCIA else 'REVISAR'
            }
 
        reporte_calidad['comparaciones']['precio'] = metricas_precio
 
    # -----------------------------------------------------------------------
    # COMPARACIÓN 2: Proporciones por categoría
    # -----------------------------------------------------------------------
    # value_counts(normalize=True): proporción de cada categoría (suma = 1.0)
    if 'categoria' in df_original.columns and 'categoria' in df_expandido.columns:
 
        prop_orig = df_original['categoria'].value_counts(normalize=True)
        prop_exp  = df_expandido['categoria'].value_counts(normalize=True)
 
        print("\n  Proporciones por categoría (original → expandido):")
        print(f"  {'Categoría':<20} {'Original':>10} {'Expandido':>10} {'Δ puntos':>10} {'Estado':>8}")
        print("  " + "-" * 60)
 
        comp_categorias = {}
        for cat in prop_orig.index:
            p_orig = prop_orig.get(cat, 0.0)
            p_exp  = prop_exp.get(cat, 0.0)
            # Para proporciones usamos diferencia absoluta en puntos porcentuales
            delta_pp = abs(p_exp - p_orig) * 100
            estado   = "✓ OK" if delta_pp <= 5.0 else "⚠ REVISAR"  # Tolerancia de 5pp para proporciones
 
            print(f"  {cat:<20} {p_orig:>9.1%} {p_exp:>10.1%} {delta_pp:>9.1f}pp {estado:>8}")
 
            comp_categorias[cat] = {
                'original' : round(float(p_orig), 4),
                'expandido': round(float(p_exp), 4),
                'delta_pp' : round(delta_pp, 2),
                'estado'   : 'OK' if delta_pp <= 5.0 else 'REVISAR'
            }
 
        reporte_calidad['comparaciones']['categorias'] = comp_categorias
 
    # -----------------------------------------------------------------------
    # COMPARACIÓN 3: Proporciones por origen
    # -----------------------------------------------------------------------
    if 'origen' in df_original.columns and 'origen' in df_expandido.columns:
 
        prop_orig_o = df_original['origen'].value_counts(normalize=True)
        prop_exp_o  = df_expandido['origen'].value_counts(normalize=True)
 
        print("\n  Proporciones por origen (original → expandido):")
        comp_origen = {}
        for orig_val in prop_orig_o.index:
            p_o = prop_orig_o.get(orig_val, 0.0)
            p_e = prop_exp_o.get(orig_val, 0.0)
            delta = abs(p_e - p_o) * 100
            estado = "✓ OK" if delta <= 5.0 else "⚠ REVISAR"
            print(f"    {orig_val:<18}: {p_o:.1%} → {p_e:.1%}  Δ={delta:.1f}pp  {estado}")
            comp_origen[orig_val] = {'original': round(float(p_o), 4),
                                     'expandido': round(float(p_e), 4),
                                     'delta_pp' : round(delta, 2)}
 
        reporte_calidad['comparaciones']['origen'] = comp_origen
 
    # -----------------------------------------------------------------------
    # VEREDICTO GLOBAL
    # -----------------------------------------------------------------------
    # Revisamos cuántas métricas de precio están fuera de tolerancia
    metricas_precio_dict = reporte_calidad['comparaciones'].get('precio', {})
    n_fuera_tolerancia = sum(
        1 for v in metricas_precio_dict.values()
        if isinstance(v, dict) and v.get('estado') == 'REVISAR'
    )
 
    if n_fuera_tolerancia == 0:
        veredicto = "APROBADO"
        print("\n  ✓ VEREDICTO: Expansión APROBADA — distribuciones estadísticamente coherentes")
    elif n_fuera_tolerancia <= 2:
        veredicto = "ACEPTABLE"
        print("\n  ⚠ VEREDICTO: Expansión ACEPTABLE — {n_fuera_tolerancia} métrica(s) fuera de tolerancia")
    else:
        veredicto = "REVISAR"
        print("\n  ✗ VEREDICTO: Expansión REQUIERE REVISIÓN — {n_fuera_tolerancia} métricas fuera de tolerancia")
 
    reporte_calidad['veredicto'] = veredicto
    return reporte_calidad
 

def generar_reporte_calidad(df):
    """
    Generar reporte ejecutivo completo de calidad del dataset expandido.
 
    PROPÓSITO:
    Este reporte es el "pasaporte de calidad" del dataset.
    Antes de entrenar cualquier modelo, un Data Engineer debe poder
    responder: ¿Cuántos registros? ¿Cuántos nulos? ¿Qué distribuciones?
    ¿Hay duplicados? ¿Los rangos tienen sentido de negocio?
 
    SECCIONES DEL REPORTE:
    1. Dimensiones generales
    2. Estadísticas descriptivas por columna numérica
    3. Distribución de variables categóricas
    4. Conteo de nulos y duplicados
    5. Resumen ejecutivo con semáforo de calidad
 
    Args:
        df (pd.DataFrame): Dataset expandido y transformado
 
    Returns:
        dict: Reporte completo de calidad listo para exportar o imprimir
    """
 
    print("  Generando reporte completo de calidad de datos...")
 
    separador = "  " + "=" * 54
 
    # -----------------------------------------------------------------------
    # SECCIÓN 1: Dimensiones generales
    # -----------------------------------------------------------------------
    print(separador)
    print("  REPORTE DE CALIDAD — DATASET EXPANDIDO")
    print(separador)
    print(f"  Filas totales   : {len(df):,}")
    print(f"  Columnas totales: {len(df.columns)}")
    print(f"  Memoria usada   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
 
    reporte = {
        'dimensiones': {
            'filas'  : len(df),
            'columnas': len(df.columns),
            'memoria_kb': round(df.memory_usage(deep=True).sum() / 1024, 2)
        }
    }
 
    # -----------------------------------------------------------------------
    # SECCIÓN 2: Estadísticas descriptivas — columnas numéricas
    # -----------------------------------------------------------------------
    # .describe() calcula count, mean, std, min, 25%, 50%, 75%, max
    # Transponemos (.T) para que cada columna numérica sea una fila del reporte
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_reporte   = [c for c in columnas_numericas if c not in ['id']]
 
    print("\n  ESTADÍSTICAS DESCRIPTIVAS (columnas numéricas clave):")
    print(f"  {'Columna':<28} {'Media':>10} {'Mediana':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 72)
 
    stats_numericas = {}
    for col in columnas_reporte:
        serie = df[col].dropna()
        if len(serie) == 0:
            continue
        media   = serie.mean()
        mediana = serie.median()
        std     = serie.std()
        minimo  = serie.min()
        maximo  = serie.max()
 
        print(f"  {col:<28} {media:>10,.2f} {mediana:>10,.2f} {std:>10,.2f} {minimo:>10,.2f} {maximo:>10,.2f}")
 
        stats_numericas[col] = {
            'media'  : round(float(media), 2),
            'mediana': round(float(mediana), 2),
            'std'    : round(float(std), 2),
            'min'    : round(float(minimo), 2),
            'max'    : round(float(maximo), 2),
            'nulos'  : int(df[col].isna().sum())
        }
 
    reporte['estadisticas_numericas'] = stats_numericas
 
    # -----------------------------------------------------------------------
    # SECCIÓN 3: Distribución de variables categóricas
    # -----------------------------------------------------------------------
    columnas_categoricas = ['categoria', 'origen']
    stats_categoricas    = {}
 
    for col in columnas_categoricas:
        if col not in df.columns:
            continue
 
        conteos     = df[col].value_counts()
        proporciones = df[col].value_counts(normalize=True)
        n_unicos    = df[col].nunique()   # .nunique() → número de valores únicos
 
        print("\n  DISTRIBUCIÓN — {col.upper()} ({n_unicos} valores únicos):")
        col_stats = {}
        for valor in conteos.index:
            cnt = conteos[valor]
            pct = proporciones[valor]
            barra = "█" * int(pct * 30)   # Barra visual proporcional (máx 30 chars)
            print(f"    {valor:<18} {cnt:>5} registros  {pct:>6.1%}  {barra}")
            col_stats[valor] = {'count': int(cnt), 'pct': round(float(pct), 4)}
 
        stats_categoricas[col] = {
            'n_unicos': n_unicos,
            'distribucion': col_stats
        }
 
    reporte['estadisticas_categoricas'] = stats_categoricas
 
    # -----------------------------------------------------------------------
    # SECCIÓN 4: Calidad — nulos, duplicados, valores únicos
    # -----------------------------------------------------------------------
    nulos_total      = df.isnull().sum().sum()       # Suma de todos los NaN del DataFrame
    duplicados_total = df.duplicated().sum()
    nulos_por_col    = df.isnull().sum()
    cols_con_nulos   = nulos_por_col[nulos_por_col > 0]
 
    print("\n  CALIDAD DE DATOS:")
    print(f"    Valores nulos totales  : {nulos_total:,}")
    print(f"    Filas duplicadas       : {duplicados_total:,}")
    print(f"    Columnas con nulos     : {len(cols_con_nulos)}")
 
    if len(cols_con_nulos) > 0:
        for col, n in cols_con_nulos.items():
            pct = n / len(df) * 100
            print(f"      - {col}: {n} nulos ({pct:.1f}%)")
 
    reporte['calidad'] = {
        'nulos_totales'    : int(nulos_total),
        'duplicados'       : int(duplicados_total),
        'cols_con_nulos'   : cols_con_nulos.to_dict()
    }
 
    # -----------------------------------------------------------------------
    # SECCIÓN 5: Semáforo de calidad ejecutivo
    # -----------------------------------------------------------------------
    # Asignamos un puntaje de calidad de 0 a 100 basado en 4 criterios
    puntaje = 100
 
    if nulos_total > 0:
        penalizacion_nulos = min(30, int(nulos_total / len(df) * 100))
        puntaje -= penalizacion_nulos
 
    if duplicados_total > 0:
        puntaje -= min(20, int(duplicados_total / len(df) * 100))
 
    # Verificar que hay exactamente 500 registros
    if len(df) != 500:
        puntaje -= 10
 
    # Verificar que el precio no tiene valores negativos
    if 'precio' in df.columns:
        precios_invalidos = (df['precio'] <= 0).sum()
        if precios_invalidos > 0:
            puntaje -= 20
 
    # Asignar nivel de calidad
    if puntaje >= 90:
        nivel = "EXCELENTE"
        icono = "✓"
    elif puntaje >= 75:
        nivel = "BUENO"
        icono = "✓"
    elif puntaje >= 60:
        nivel = "ACEPTABLE"
        icono = "⚠"
    else:
        nivel = "REQUIERE MEJORA"
        icono = "✗"
 
    print(separador)
    print(f"  {icono} PUNTAJE DE CALIDAD: {puntaje}/100 — {nivel}")
    print(separador)
 
    reporte['semaforo_calidad'] = {
        'puntaje': puntaje,
        'nivel'  : nivel
    }
 
    return reporte

# =============================================================================
# 5. FUNCIONES DE MODELADO ML (LOAD + MODEL)
# =============================================================================
 
def preparar_datos_ml(df):
    """
    Seleccionar features, definir target y dividir en conjuntos de entrenamiento y prueba.
 
    CONCEPTOS CLAVE:
    ┌─────────────────────────────────────────────────────────┐
    │  X (features / predictoras): lo que el modelo RECIBE   │
    │  y (target / objetivo):      lo que el modelo PREDICE  │
    └─────────────────────────────────────────────────────────┘
 
    VARIABLE OBJETIVO: precio
    Queremos predecir el precio de un producto dado su categoría,
    origen y otras características. Esto tiene valor de negocio real:
    permite estimar precios de nuevos productos antes de lanzarlos.
 
    TRAIN / TEST SPLIT:
    Dividimos el dataset en dos bloques separados:
    - Train (80%): el modelo ve estos datos y ajusta sus coeficientes
    - Test  (20%): el modelo NUNCA vio estos datos durante el entrenamiento
    La métrica en Test nos dice si el modelo generaliza o solo memorizó.
 
    ¿POR QUÉ 80/20 Y NO 70/30?
    Con 500 registros, 80/20 da 400 para entrenar (más datos = mejor modelo)
    y 100 para evaluar (suficiente para métricas confiables).
    Con datasets pequeños (<1000), 80/20 es preferible a 70/30.
 
    Args:
        df (pd.DataFrame): Dataset expandido, limpio y con features codificadas
 
    Returns:
        tuple: (X_train, X_test, y_train, y_test, features_usadas)
    """
 
    print("  Preparando datos para Machine Learning...")
 
    # -----------------------------------------------------------------------
    # PASO 1: Definir variables predictoras (X)
    # -----------------------------------------------------------------------
    # Seleccionamos las columnas que el modelo usará para predecir.
    # CRITERIOS DE SELECCIÓN:
    # - Deben ser numéricas (el modelo las necesita así)
    # - No deben ser el target (precio) ni el identificador (id)
    # - No deben ser la versión original de una columna ya codificada
    #   (no usamos 'categoria' si ya tenemos 'categoria_encoded')
 
    # Lista de features candidatas — solo las que sabemos que son numéricas y útiles
    features_candidatas = [
        'categoria_encoded',       # Categoría del producto (número)
        'origen_encoded',          # País de origen (número)
        'mes_registro',            # Mes del registro (1-12)
        'dia_semana_registro',     # Día de la semana (0-6)
        'dias_desde_registro',     # Antigüedad del registro
        'precio_relativo_categoria', # Posición de precio dentro de su categoría
        'es_nacional',             # Bandera: producto nacional vs importado
    ]
 
    # Filtrar solo las que realmente existen en el DataFrame
    # (puede que algunas no se hayan creado si faltaban columnas de origen)
    features_usadas = [f for f in features_candidatas if f in df.columns]
 
    print(f"  Features seleccionadas ({len(features_usadas)}):")
    for f in features_usadas:
        print(f"    - {f}")
 
    # -----------------------------------------------------------------------
    # PASO 2: Separar X e y
    # -----------------------------------------------------------------------
    # X: DataFrame con solo las columnas predictoras
    # y: Serie con la columna objetivo (precio)
    X = df[features_usadas].copy()
    y = df['precio'].copy()
 
    # Eliminar filas con NaN en X o y — sklearn no acepta valores faltantes
    # .dropna() sobre X nos da los índices válidos
    indices_validos = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[indices_validos]
    y = y.loc[indices_validos]
 
    print(f"\n  Shape de X (features): {X.shape}  →  {X.shape[0]} muestras × {X.shape[1]} features")
    print(f"  Shape de y (target)  : {y.shape}  →  {y.shape[0]} valores de precio")
    print(f"  Precio objetivo — media: ${y.mean():,.2f}  |  rango: [${y.min():,.2f}, ${y.max():,.2f}]")
 
    # -----------------------------------------------------------------------
    # PASO 3: Dividir en Train y Test
    # -----------------------------------------------------------------------
    # train_test_split parámetros explicados:
    # - X, y          : los datos a dividir (se dividen igual para mantener correspondencia)
    # - test_size=0.2 : 20% para test, 80% para train
    # - random_state=42: semilla para reproducibilidad — siempre la misma división
    # - shuffle=True  : mezcla los datos antes de dividir (evita sesgos de orden)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
 
    print(f"\n  División Train/Test (80/20):")
    print(f"    Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"    Test : {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.0f}%)")
    print(f"\n  ✓ Datos listos para entrenar el modelo")
 
    return X_train, X_test, y_train, y_test, features_usadas
 
 
def entrenar_modelo_regresion(X_train, y_train):
    """
    Instanciar y entrenar el modelo de Regresión Lineal.
 
    EL PROCESO DE ENTRENAMIENTO:
    sklearn usa el método de Mínimos Cuadrados Ordinarios (OLS):
    Encuentra los valores de β que minimizan la suma de errores al cuadrado:
        minimizar Σ(y_real - y_predicho)²
    """
 
    print("  Entrenando modelo de Regresión Lineal...")
    print(f"  Muestras de entrenamiento: {X_train.shape[0]}")
    print(f"  Features utilizadas      : {X_train.shape[1]}")
 
    # -----------------------------------------------------------------------
    # INSTANCIAR EL MODELO
    # -----------------------------------------------------------------------
    # LinearRegression() crea el modelo pero AÚN NO ha aprendido nada.
    # Es como contratar a un empleado: está listo para trabajar, pero
    # todavía no ha visto ningún dato.
    #
    # Parámetros disponibles (usamos defaults que son correctos para este caso):
    # - fit_intercept=True : calcula β₀ (intercepto). Casi siempre debe ser True.
    # - positive=False     : permite coeficientes negativos (precio puede bajar con ciertas features)
    modelo = LinearRegression(fit_intercept=True)
 
    # -----------------------------------------------------------------------
    # ENTRENAR EL MODELO — .fit()
    # -----------------------------------------------------------------------
    # .fit(X_train, y_train): EL momento donde ocurre el aprendizaje.
    # sklearn calcula internamente la solución OLS:
    #     β = (XᵀX)⁻¹ Xᵀy
    # En milisegundos, el modelo encuentra los mejores coeficientes.
    modelo.fit(X_train, y_train)
 
    # -----------------------------------------------------------------------
    # INSPECCIONAR LO QUE APRENDIÓ EL MODELO
    # -----------------------------------------------------------------------
    print(f"\n  Intercepto (β₀): ${modelo.intercept_:,.2f}")
    print("  Coeficientes aprendidos (βᵢ):")
 
    # modelo.coef_ : array con un coeficiente por cada feature
    # zip(columnas, coefs) empareja cada nombre de columna con su coeficiente
    for feature, coef in zip(X_train.columns, modelo.coef_):
        signo     = "▲" if coef > 0 else "▼"   # ▲ sube precio, ▼ baja precio
        impacto   = "alto" if abs(coef) > 50 else "medio" if abs(coef) > 10 else "bajo"
        print(f"    {signo} {feature:<30}: {coef:>10,.4f}  (impacto {impacto})")
 
    print(f"\n  ✓ Modelo entrenado exitosamente")
    print("  Interpretación: cada coeficiente indica cuánto cambia el precio")
    print("  predicho al aumentar esa feature en 1 unidad, manteniendo el resto fijo.")
 
    return modelo
 
 
def evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    """
    Calcular y comparar métricas de rendimiento del modelo en Train y Test.
    """
 
    print("  Evaluando rendimiento del modelo...")
 
    # -----------------------------------------------------------------------
    # PASO 1: Generar predicciones
    # -----------------------------------------------------------------------
    # .predict(X): aplica la ecuación aprendida β₀ + β₁x₁ + β₂x₂ + ...
    # Devuelve un array numpy con un precio predicho por cada fila de X.
    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)
 
    # -----------------------------------------------------------------------
    # PASO 2: Calcular métricas en Train
    # -----------------------------------------------------------------------
    r2_train   = r2_score(y_train, y_pred_train)
    # mean_squared_error devuelve MSE; aplicamos sqrt() para obtener RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train  = mean_absolute_error(y_train, y_pred_train)
 
    # -----------------------------------------------------------------------
    # PASO 3: Calcular métricas en Test
    # -----------------------------------------------------------------------
    r2_test    = r2_score(y_test, y_pred_test)
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
 
    # -----------------------------------------------------------------------
    # PASO 4: Mostrar tabla comparativa
    # -----------------------------------------------------------------------
    print(f"\n  {'Métrica':<8} {'Train':>12} {'Test':>12} {'Diferencia':>12}")
    print("  " + "-" * 46)
    print(f"  {'R²':<8} {r2_train:>12.4f} {r2_test:>12.4f} {abs(r2_train - r2_test):>12.4f}")
    print(f"  {'RMSE':<8} {rmse_train:>11,.2f} {rmse_test:>11,.2f} {abs(rmse_train - rmse_test):>11,.2f}")
    print(f"  {'MAE':<8} {mae_train:>11,.2f} {mae_test:>11,.2f} {abs(mae_train - mae_test):>11,.2f}")
 
    # -----------------------------------------------------------------------
    # PASO 5: Diagnóstico automático de overfitting / underfitting
    # -----------------------------------------------------------------------
    diferencia_r2 = r2_train - r2_test
 
    print("\n  DIAGNÓSTICO DEL MODELO:")
    if r2_test < 0.3:
        diagnostico = "UNDERFITTING"
        print("  ✗ UNDERFITTING: El modelo es demasiado simple para capturar los patrones.")
        print("    Sugerencia: agregar más features o explorar modelos no lineales.")
    elif diferencia_r2 > 0.15:
        diagnostico = "OVERFITTING"
        print("  ⚠ OVERFITTING: El modelo memorizó el train pero no generaliza bien.")
        print("    Sugerencia: reducir features o aumentar datos de entrenamiento.")
    elif r2_test >= 0.7:
        diagnostico = "EXCELENTE"
        print("  ✓ EXCELENTE: Modelo con buen poder predictivo y buena generalización.")
    elif r2_test >= 0.5:
        diagnostico = "ACEPTABLE"
        print("  ✓ ACEPTABLE: Modelo funcional. Captura patrones principales del precio.")
    else:
        diagnostico = "MEJORABLE"
        print("  ⚠ MEJORABLE: El modelo captura algo, pero hay margen de mejora.")
        print("    Sugerencia: revisar feature engineering o limpiar outliers.")
 
    # Interpretación de negocio — cuantificar el error en términos concretos
    precio_medio = float(y_test.mean())
    error_relativo = (mae_test / precio_medio) * 100
    print(f"\n  INTERPRETACIÓN DE NEGOCIO:")
    print(f"  Precio promedio real    : ${precio_medio:,.2f}")
    print(f"  Error promedio (MAE)    : ${mae_test:,.2f}")
    print(f"  Error relativo          : {error_relativo:.1f}% del precio promedio")
    print(f"  → El modelo se equivoca en promedio ${mae_test:,.2f} por producto")

    # -----------------------------------------------------------------------
    # PASO 6: Empaquetar métricas en diccionario
    # -----------------------------------------------------------------------
    metricas = {
        'train': {
            'r2'  : round(r2_train, 4),
            'rmse': round(rmse_train, 2),
            'mae' : round(mae_train, 2)
        },
        'test': {
            'r2'  : round(r2_test, 4),
            'rmse': round(rmse_test, 2),
            'mae' : round(mae_test, 2)
        },
        'diagnostico'    : diagnostico,
        'diferencia_r2'  : round(diferencia_r2, 4),
        'error_relativo_pct': round(error_relativo, 2),
        'y_pred_test'    : y_pred_test   # Guardamos predicciones para las visualizaciones
    }
 
    return metricas

# =============================================================================
# 6. FUNCIONES DE VISUALIZACIÓN
# =============================================================================
 
def graficar_distribucion_datos(df):
    """
    Crear panel de 4 visualizaciones que describen la distribución del dataset.
 
    GRÁFICAS QUE GENERAMOS Y POR QUÉ:
    1. Histograma de precio       → ver si la distribución es normal, sesgada o bimodal
    2. Barras por categoría       → ver si las proporciones de categorías son balanceadas
    3. Boxplot precio/categoría   → detectar outliers y comparar rangos entre grupos
    4. Mapa de correlación (heatmap) → identificar qué features están relacionadas con precio
 
    INSIGHT ESPERADO:
    - Si el histograma es muy sesgado a la derecha → hay productos premium que jalan la media
    - Si el boxplot muestra muchos outliers → categorías con alta variabilidad de precios
    - Si el heatmap muestra correlación alta → esas features serán predictoras poderosas
 
    Args:
        df (pd.DataFrame): Dataset expandido con todas las variables
 
    Returns:
        matplotlib.figure.Figure: Figura con los 4 subplots
    """
 
    print("  Generando visualizaciones de distribución de datos...")
 
    # fig, axes: fig es la figura contenedora, axes es una matriz 2x2 de subplots
    # figsize=(16, 12): 16 pulgadas de ancho, 12 de alto — buena resolución para presentación
    # facecolor='white': fondo blanco explícito para exportar sin transparencias
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.suptitle('Análisis de Distribución del Dataset Expandido (500 registros)',
                 fontsize=16, fontweight='bold', y=1.01)
 
    # -----------------------------------------------------------------------
    # SUBPLOT 1 (axes[0,0]): Histograma de precio
    # -----------------------------------------------------------------------
    # El histograma divide el rango de valores en "bins" (cubetas) y cuenta
    # cuántos valores caen en cada bin. Nos muestra la FORMA de la distribución.
    #
    # kde=True: superpone una curva KDE (Kernel Density Estimation).
    # KDE es una versión suavizada del histograma — muestra la densidad
    # de probabilidad real sin depender del número de bins.
    ax1 = axes[0, 0]
    sns.histplot(
        data=df,
        x='precio',
        bins=30,          # 30 barras: suficiente detalle sin ruido excesivo
        kde=True,         # Curva de densidad superpuesta
        color='steelblue',
        edgecolor='white',
        linewidth=0.5,
        ax=ax1
    )
    # Líneas verticales para media y mediana — permiten ver el sesgo visualmente
    media_precio   = df['precio'].mean()
    mediana_precio = df['precio'].median()
    ax1.axvline(media_precio,   color='red',    linestyle='--', linewidth=1.8,
                label=f'Media: ${media_precio:,.0f}')
    ax1.axvline(mediana_precio, color='orange', linestyle='--', linewidth=1.8,
                label=f'Mediana: ${mediana_precio:,.0f}')
    ax1.set_title('Distribución de Precios', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Precio ($)', fontsize=11)
    ax1.set_ylabel('Frecuencia', fontsize=11)
    ax1.legend(fontsize=10)
 
    # -----------------------------------------------------------------------
    # SUBPLOT 2 (axes[0,1]): Barras por categoría
    # -----------------------------------------------------------------------
    # value_counts() ordena de mayor a menor por defecto — ideal para barras.
    # Un gráfico de barras para categóricas es más honesto que un pie chart:
    # es más fácil comparar alturas de barras que ángulos de sectores.
    ax2 = axes[0, 1]
    conteo_cat = df['categoria'].value_counts()
 
    # Paleta de colores cualitativos — colores distintos sin jerarquía implícita
    colores = sns.color_palette('Set2', n_colors=len(conteo_cat))
    barras  = ax2.bar(conteo_cat.index, conteo_cat.values, color=colores, edgecolor='white', linewidth=0.8)
 
    # Añadir etiquetas de valor encima de cada barra
    for barra, valor in zip(barras, conteo_cat.values):
        pct = valor / len(df) * 100
        ax2.text(
            barra.get_x() + barra.get_width() / 2,  # Centro horizontal de la barra
            barra.get_height() + 1,                  # Justo encima de la barra
            f'{valor}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
 
    ax2.set_title('Registros por Categoría', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Categoría', fontsize=11)
    ax2.set_ylabel('Número de Registros', fontsize=11)
    ax2.tick_params(axis='x', rotation=15)   # Rotar etiquetas para que no se solapen
 
    # -----------------------------------------------------------------------
    # SUBPLOT 3 (axes[1,0]): Boxplot precio por categoría
    # -----------------------------------------------------------------------
    # El boxplot resume 5 estadísticas en una figura:
    # ─ Línea central: mediana (Q2)
    # □ Caja: rango intercuartílico IQR (Q1 a Q3) — el 50% central
    # ─ Bigotes: Q1 - 1.5*IQR  y  Q3 + 1.5*IQR
    # • Puntos sueltos: outliers (fuera de los bigotes)
    ax3 = axes[1, 0]
    orden_cats = df.groupby('categoria')['precio'].median().sort_values(ascending=False).index
 
    sns.boxplot(
        data=df,
        x='categoria',
        y='precio',
        order=orden_cats,     # Ordenar categorías por mediana descendente
        palette='Set2',
        width=0.5,
        linewidth=1.2,
        flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5),
        ax=ax3
    )
    ax3.set_title('Distribución de Precio por Categoría', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Categoría', fontsize=11)
    ax3.set_ylabel('Precio ($)', fontsize=11)
    ax3.tick_params(axis='x', rotation=15)
 
    # -----------------------------------------------------------------------
    # SUBPLOT 4 (axes[1,1]): Mapa de correlación (Heatmap)
    # -----------------------------------------------------------------------
    # La correlación de Pearson mide la relación LINEAL entre dos variables.
    # Rango: -1 a +1
    # +1 → correlación positiva perfecta (suben juntas)
    #  0 → sin correlación lineal
    # -1 → correlación negativa perfecta (una sube, la otra baja)
    #
    # Para el modelo ML: features con correlación alta con 'precio' son las más útiles.
    # Features correlacionadas entre sí (multicolinealidad) pueden confundir al modelo.
    ax4 = axes[1, 1]
 
    # Seleccionar columnas numéricas relevantes para el heatmap
    cols_corr = ['precio', 'categoria_encoded', 'origen_encoded',
                 'mes_registro', 'dias_desde_registro',
                 'precio_relativo_categoria', 'es_precio_alto', 'es_nacional']
    cols_corr = [c for c in cols_corr if c in df.columns]
 
    # .corr(): calcula la matriz de correlación de Pearson — una tabla N×N
    # donde cada celda tiene la correlación entre dos columnas
    matriz_corr = df[cols_corr].corr()
 
    sns.heatmap(
        matriz_corr,
        annot=True,          # Mostrar el valor numérico en cada celda
        fmt='.2f',           # Formato: 2 decimales
        cmap='coolwarm',     # Azul=negativo, Blanco=cero, Rojo=positivo
        center=0,            # El blanco (neutro) en el valor 0
        square=True,         # Celdas cuadradas para mejor lectura
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax4
    )
    ax4.set_title('Mapa de Correlación entre Variables', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.tick_params(axis='y', rotation=0,  labelsize=8)
 
    # -----------------------------------------------------------------------
    # AJUSTE Y EXPORTACIÓN
    # -----------------------------------------------------------------------
    # tight_layout: ajusta automáticamente márgenes para que los subplots no se solapen
    plt.tight_layout()
 
    # Guardar en disco en alta resolución (dpi=150 es suficiente para presentaciones)
    ruta_guardado = 'grafico_distribucion_datos.png'
    plt.savefig(ruta_guardado, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Gráfico guardado: {ruta_guardado}")
    plt.show()
 
    return fig
 
 
def graficar_resultados_modelo(y_test, y_pred, metricas):
    """
    Crear panel de 4 gráficas que evalúan visualmente el modelo de regresión.
 
    GRÁFICAS Y SU PROPÓSITO DIAGNÓSTICO:
 
    1. Real vs Predicho (Scatter):
       Si el modelo es perfecto, todos los puntos caen sobre la línea diagonal y=x.
       Puntos lejos de esa diagonal = errores grandes del modelo.
       Patrones sistemáticos (curvas, grupos) = el modelo no captura algo importante.
 
    2. Residuos vs Predichos:
       Residuo = y_real - y_predicho (el error de cada predicción).
       Un buen modelo tiene residuos distribuidos aleatoriamente alrededor de 0.
       Si ves un patrón (embudo, curva) → el modelo viola supuestos de la regresión lineal.
 
    3. Distribución de residuos (Histograma + KDE):
       Los residuos deberían seguir una distribución normal centrada en 0.
       Si está sesgada → el modelo sobreestima o subestima sistemáticamente.
 
    4. Panel de métricas visuales:
       Tarjeta con R², RMSE, MAE y diagnóstico — lista para captura de pantalla
       y pegar en la presentación del proyecto.
 
    Args:
        y_test  (pd.Series):  Precios reales del conjunto de prueba
        y_pred  (np.ndarray): Precios predichos por el modelo
        metricas (dict):      Diccionario con R², RMSE, MAE y diagnóstico
 
    Returns:
        matplotlib.figure.Figure: Figura con los 4 subplots de evaluación
    """
 
    print("  Generando visualizaciones de resultados del modelo...")
 
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.suptitle('Evaluación del Modelo de Regresión Lineal',
                 fontsize=16, fontweight='bold', y=1.01)
 
    # Calcular residuos una vez — los usamos en dos subplots
    # Residuo = lo que el modelo NO pudo explicar
    residuos = np.array(y_test) - y_pred
 
    # -----------------------------------------------------------------------
    # SUBPLOT 1: Scatter — Valores Reales vs Predichos
    # -----------------------------------------------------------------------
    ax1 = axes[0, 0]
 
    # Scatter de predicciones
    ax1.scatter(
        y_test, y_pred,
        alpha=0.5,          # Transparencia: permite ver superposición de puntos
        color='steelblue',
        edgecolors='white',
        linewidth=0.3,
        s=50,               # Tamaño del punto en puntos²
        label='Predicciones'
    )
 
    # Línea diagonal perfecta — si el modelo fuera perfecto, todos los puntos estarían aquí
    # np.linspace: genera N puntos equidistantes entre min y max
    valor_min = min(float(y_test.min()), float(y_pred.min()))
    valor_max = max(float(y_test.max()), float(y_pred.max()))
    linea_x   = np.linspace(valor_min, valor_max, 100)
    ax1.plot(linea_x, linea_x, color='red', linewidth=2,
             linestyle='--', label='Predicción Perfecta (y=x)')
 
    ax1.set_title('Valores Reales vs Predichos', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Precio Real ($)', fontsize=11)
    ax1.set_ylabel('Precio Predicho ($)', fontsize=11)
    ax1.legend(fontsize=10)
 
    # Añadir R² dentro del gráfico — más claro que solo en el título
    r2_val = metricas['test']['r2']
    ax1.text(0.05, 0.92, f'R² = {r2_val:.4f}',
             transform=ax1.transAxes,   # Coordenadas relativas (0-1) dentro del subplot
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
 
    # -----------------------------------------------------------------------
    # SUBPLOT 2: Residuos vs Valores Predichos
    # -----------------------------------------------------------------------
    # Este gráfico es el diagnóstico MÁS IMPORTANTE de la regresión lineal.
    # Revela si los supuestos del modelo se cumplen.
    ax2 = axes[0, 1]
 
    ax2.scatter(
        y_pred, residuos,
        alpha=0.5,
        color='darkorange',
        edgecolors='white',
        linewidth=0.3,
        s=50
    )
 
    # Línea horizontal en y=0 — los residuos deberían oscilar aleatoriamente aquí
    ax2.axhline(y=0, color='red', linewidth=2, linestyle='--', label='Residuo = 0')
 
    # Bandas de ±1 desviación estándar de los residuos
    std_res = residuos.std()
    ax2.axhline(y= std_res, color='gray', linewidth=1, linestyle=':', alpha=0.7, label=f'±1σ (${std_res:,.0f})')
    ax2.axhline(y=-std_res, color='gray', linewidth=1, linestyle=':', alpha=0.7)
 
    ax2.set_title('Residuos vs Valores Predichos', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Precio Predicho ($)', fontsize=11)
    ax2.set_ylabel('Residuo ($)', fontsize=11)
    ax2.legend(fontsize=10)
 
    # -----------------------------------------------------------------------
    # SUBPLOT 3: Histograma de Residuos
    # -----------------------------------------------------------------------
    # Queremos que los residuos sigan una distribución normal centrada en 0.
    # Si no es así, el modelo tiene sesgo sistemático.
    ax3 = axes[1, 0]
 
    sns.histplot(
        residuos,
        bins=25,
        kde=True,           # Curva de densidad sobre el histograma
        color='mediumseagreen',
        edgecolor='white',
        linewidth=0.5,
        ax=ax3
    )
 
    # Línea en 0 para referencia
    ax3.axvline(0, color='red', linewidth=2, linestyle='--', label='Error = 0')
    ax3.axvline(residuos.mean(), color='orange', linewidth=1.5,
                linestyle='--', label=f'Media residuos: ${residuos.mean():,.1f}')
 
    ax3.set_title('Distribución de Residuos (Errores)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Residuo ($)', fontsize=11)
    ax3.set_ylabel('Frecuencia', fontsize=11)
    ax3.legend(fontsize=10)
 
    # -----------------------------------------------------------------------
    # SUBPLOT 4: Panel de Métricas — Tarjeta visual
    # -----------------------------------------------------------------------
    # ax.axis('off'): desactiva los ejes — vamos a dibujar texto puro
    ax4 = axes[1, 1]
    ax4.axis('off')
 
    # Determinar color del diagnóstico según resultado
    diagnostico = metricas.get('diagnostico', 'N/A')
    color_diag  = {'EXCELENTE': '#2ecc71', 'ACEPTABLE': '#f39c12',
                   'MEJORABLE': '#e67e22', 'OVERFITTING': '#e74c3c',
                   'UNDERFITTING': '#e74c3c'}.get(diagnostico, '#95a5a6')
 
    # Construir el texto del panel con formato tabular usando espacios
    texto_metricas = (
        "MÉTRICAS DEL MODELO\n"
        + "─" * 32 + "\n\n"
        + f"  R² (Test)   :  {metricas['test']['r2']:.4f}\n"
        + f"  R² (Train)  :  {metricas['train']['r2']:.4f}\n\n"
        + f"  RMSE (Test) :  ${metricas['test']['rmse']:>10,.2f}\n"
        + f"  RMSE (Train):  ${metricas['train']['rmse']:>10,.2f}\n\n"
        + f"  MAE (Test)  :  ${metricas['test']['mae']:>10,.2f}\n"
        + f"  MAE (Train) :  ${metricas['train']['mae']:>10,.2f}\n\n"
        + "─" * 32 + "\n"
        + f"  Error relativo: {metricas.get('error_relativo_pct', 0):.1f}%\n"
        + f"  Diagnóstico: {diagnostico}"
    )
 
    # ax.text con coordenadas (0.5, 0.5) = centro del subplot
    # transform=ax4.transAxes: coordenadas relativas (0 a 1)
    ax4.text(
        0.5, 0.5, texto_metricas,
        transform=ax4.transAxes,
        fontsize=12,
        fontfamily='monospace',   # Fuente monoespaciada para alinear columnas
        verticalalignment='center',
        horizontalalignment='center',
        bbox=dict(
            boxstyle='round,pad=1.0',
            facecolor='#f8f9fa',
            edgecolor=color_diag,
            linewidth=3
        )
    )
 
    # -----------------------------------------------------------------------
    # AJUSTE Y EXPORTACIÓN
    # -----------------------------------------------------------------------
    plt.tight_layout()
    ruta_guardado = 'grafico_resultados_modelo.png'
    plt.savefig(ruta_guardado, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Gráfico guardado: {ruta_guardado}")
    plt.show()
 
    return fig
 
 
def crear_dashboard_completo(df_original, df_expandido, modelo, metricas):
    """
    Dashboard ejecutivo de 6 paneles que resume todo el proyecto en una sola figura.
 
    DISEÑO DEL DASHBOARD (grid 3×2):
    ┌─────────────────────┬─────────────────────┐
    │ 1. Antes vs Después │ 2. Precio/Categoría  │
    │    (40 vs 500 reg.) │    (expandido)       │
    ├─────────────────────┼─────────────────────┤
    │ 3. Coeficientes     │ 4. Real vs Predicho  │
    │    del modelo       │    (scatter)         │
    ├─────────────────────┼─────────────────────┤
    │ 5. Precio promedio  │ 6. Tarjeta resumen   │
    │    por origen       │    del proyecto      │
    └─────────────────────┴─────────────────────┘
 
    PROPÓSITO: Este dashboard es la "diapositiva estrella" de la presentación.
    Muestra en una sola imagen: qué teníamos, qué generamos, qué aprendió el modelo
    y qué valor de negocio tiene. Diseñado para ser autoexplicativo en 10 segundos.
 
    Args:
        df_original  (pd.DataFrame)   : Dataset original de 40 registros
        df_expandido (pd.DataFrame)   : Dataset expandido de 500 registros
        modelo       (LinearRegression): Modelo entrenado
        metricas     (dict)            : Diccionario de métricas del modelo
 
    Returns:
        matplotlib.figure.Figure: Dashboard completo exportado como PNG
    """
 
    print("  Generando dashboard ejecutivo completo...")
 
    fig = plt.figure(figsize=(20, 15), facecolor='white')
    fig.suptitle(
        'Dashboard Ejecutivo — Pipeline Inteligente de Análisis de Datos',
        fontsize=18, fontweight='bold', y=0.98
    )
 
    # GridSpec: sistema de grillas flexible para posicionar subplots con precisión
    # 3 filas, 2 columnas, separación horizontal y vertical entre paneles
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.93, bottom=0.06)
 
    # -----------------------------------------------------------------------
    # PANEL 1: Comparación antes vs después de la expansión
    # -----------------------------------------------------------------------
    # Gráfico de barras agrupadas: una barra por dataset (original vs expandido)
    # para cada categoría. Muestra visualmente que las proporciones se mantuvieron.
    ax1 = fig.add_subplot(gs[0, 0])
 
    cats_orig = df_original['categoria'].value_counts().sort_index()
    cats_exp  = df_expandido['categoria'].value_counts().sort_index()
 
    # Alinear categorías — puede que el orden difiera
    todas_cats = sorted(set(cats_orig.index) | set(cats_exp.index))
    vals_orig  = [cats_orig.get(c, 0) for c in todas_cats]
    vals_exp   = [cats_exp.get(c, 0) for c in todas_cats]
 
    x      = np.arange(len(todas_cats))
    ancho  = 0.35   # Ancho de cada barra — las dos deben sumar menos de 1 para no solaparse
 
    barras1 = ax1.bar(x - ancho/2, vals_orig, ancho, label=f'Original ({len(df_original)} reg.)',
                      color='#3498db', edgecolor='white', linewidth=0.8)
    barras2 = ax1.bar(x + ancho/2, vals_exp,  ancho, label=f'Expandido ({len(df_expandido)} reg.)',
                      color='#2ecc71', edgecolor='white', linewidth=0.8)
 
    ax1.set_title('Expansión del Dataset por Categoría', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(todas_cats, rotation=15, fontsize=9)
    ax1.set_ylabel('Registros', fontsize=10)
    ax1.legend(fontsize=9)
 
    # Etiquetas encima de cada barra
    for barra in list(barras1) + list(barras2):
        h = barra.get_height()
        ax1.text(barra.get_x() + barra.get_width()/2, h + 0.5,
                 str(int(h)), ha='center', va='bottom', fontsize=8)
 
    # -----------------------------------------------------------------------
    # PANEL 2: Precio promedio por categoría — dataset expandido
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
 
    precio_cat = (df_expandido.groupby('categoria')['precio']
                  .agg(['mean', 'std'])
                  .sort_values('mean', ascending=False))
 
    colores_panel2 = sns.color_palette('Set2', n_colors=len(precio_cat))
    barras_p = ax2.barh(    # barh = barras HORIZONTALES — mejor para etiquetas largas
        precio_cat.index,
        precio_cat['mean'],
        color=colores_panel2,
        edgecolor='white',
        linewidth=0.8
    )
    # Barras de error: muestran la desviación estándar del precio por categoría
    ax2.errorbar(
        precio_cat['mean'], precio_cat.index,
        xerr=precio_cat['std'],
        fmt='none',         # Sin marcadores — solo las líneas de error
        color='gray',
        linewidth=1.5,
        capsize=4           # Pequeñas líneas horizontales en los extremos del error bar
    )
 
    # Etiquetas con el valor exacto al final de cada barra
    for barra, val in zip(barras_p, precio_cat['mean']):
        ax2.text(val + 5, barra.get_y() + barra.get_height()/2,
                 f'${val:,.0f}', va='center', fontsize=9, fontweight='bold')
 
    ax2.set_title('Precio Promedio por Categoría\n(±1 Desv. Estándar)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Precio Promedio ($)', fontsize=10)
 
    # -----------------------------------------------------------------------
    # PANEL 3: Coeficientes del modelo — cuánto aporta cada feature
    # -----------------------------------------------------------------------
    # Este gráfico es el más valioso para interpretar el modelo:
    # - Barra verde/positiva → esa feature SUBE el precio predicho
    # - Barra roja/negativa  → esa feature BAJA el precio predicho
    # - Longitud de barra    → magnitud del efecto
    ax3 = fig.add_subplot(gs[1, 0])
 
    # Crear Serie pandas con nombre de feature y su coeficiente
    # Ordenar por valor absoluto → las features más influyentes arriba
    features_nombres = list(df_expandido.select_dtypes(include=[np.number])
                            .drop(columns=['id', 'precio'], errors='ignore').columns)
 
    # Necesitamos los nombres exactos que usó el modelo al entrenar
    # Usamos los que están en modelo.feature_names_in_ si está disponible
    try:
        nombres_features = list(modelo.feature_names_in_)
    except AttributeError:
        # sklearn < 1.0 no tiene feature_names_in_
        nombres_features = [f'Feature_{i}' for i in range(len(modelo.coef_))]
 
    coefs = pd.Series(modelo.coef_, index=nombres_features).sort_values()
 
    colores_coef = ['#e74c3c' if v < 0 else '#2ecc71' for v in coefs.values]
    ax3.barh(coefs.index, coefs.values, color=colores_coef, edgecolor='white', linewidth=0.8)
    ax3.axvline(0, color='black', linewidth=1.5)
 
    ax3.set_title('Coeficientes del Modelo\n(impacto de cada feature en el precio)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Coeficiente β', fontsize=10)
    ax3.tick_params(axis='y', labelsize=9)
 
    # Leyenda explicativa
    ax3.text(0.98, 0.02, '█ Verde: sube precio\n█ Rojo: baja precio',
             transform=ax3.transAxes, fontsize=8,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
 
    # -----------------------------------------------------------------------
    # PANEL 4: Scatter Real vs Predicho
    # -----------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
 
    y_pred_vals = metricas.get('y_pred_test', np.array([]))
 
    if len(y_pred_vals) > 0:
        # Recuperar y_test desde los datos expandidos (aproximación usando precio)
        # En el pipeline principal, podemos pasar y_test directamente
        ax4.scatter([], [], alpha=0.5, color='steelblue', s=30, label='Predicciones')
 
        # Línea perfecta de referencia
        if hasattr(y_pred_vals, '__len__') and len(y_pred_vals) > 0:
            rng = [float(min(y_pred_vals))*0.9, float(max(y_pred_vals))*1.1]
            ax4.plot(rng, rng, 'r--', linewidth=2, label='Predicción perfecta')
 
        ax4.set_title(f'Real vs Predicho\nR² Test = {metricas["test"]["r2"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax4.set_xlabel('Precio Real ($)', fontsize=10)
        ax4.set_ylabel('Precio Predicho ($)', fontsize=10)
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Predicciones no disponibles\nen este contexto',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.set_title('Real vs Predicho', fontsize=12, fontweight='bold')
 
    # -----------------------------------------------------------------------
    # PANEL 5: Precio promedio por origen
    # -----------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
 
    precio_origen = (df_expandido.groupby('origen')['precio']
                     .mean()
                     .sort_values(ascending=True))
 
    colores_orig = sns.color_palette('Blues_d', n_colors=len(precio_origen))
    bars5 = ax5.barh(precio_origen.index, precio_origen.values,
                     color=colores_orig, edgecolor='white', linewidth=0.8)
 
    for barra, val in zip(bars5, precio_origen.values):
        ax5.text(val + 2, barra.get_y() + barra.get_height()/2,
                 f'${val:,.0f}', va='center', fontsize=8, fontweight='bold')
 
    ax5.set_title('Precio Promedio por País de Origen', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Precio Promedio ($)', fontsize=10)
    ax5.tick_params(axis='y', labelsize=9)
 
    # -----------------------------------------------------------------------
    # PANEL 6: Tarjeta resumen ejecutivo del proyecto
    # -----------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
 
    diagnostico  = metricas.get('diagnostico', 'N/A')
    color_estado = {'EXCELENTE': '#27ae60', 'ACEPTABLE': '#f39c12',
                    'MEJORABLE': '#e67e22', 'OVERFITTING': '#e74c3c',
                    'UNDERFITTING': '#e74c3c'}.get(diagnostico, '#7f8c8d')
 
    resumen = (
        "RESUMEN EJECUTIVO DEL PROYECTO\n"
        + "━" * 34 + "\n\n"
        + f" Dataset original  :   {len(df_original):>4} registros\n"
        + f" Dataset expandido :   {len(df_expandido):>4} registros\n"
        + f" Factor expansión  :   {len(df_expandido)/len(df_original):.1f}×\n\n"
        + f" R² del modelo     :   {metricas['test']['r2']:.4f}\n"
        + f" RMSE              :  ${metricas['test']['rmse']:>8,.2f}\n"
        + f" MAE               :  ${metricas['test']['mae']:>8,.2f}\n"
        + f" Error relativo    :   {metricas.get('error_relativo_pct',0):.1f}%\n\n"
        + "━" * 34 + "\n"
        + f" Diagnóstico: {diagnostico}\n"
        + f" Técnica: Gaussian Noise Augmentation\n"
        + f" Modelo : Regresión Lineal (OLS)"
    )
 
    ax6.text(
        0.5, 0.5, resumen,
        transform=ax6.transAxes,
        fontsize=10.5,
        fontfamily='monospace',
        verticalalignment='center',
        horizontalalignment='center',
        bbox=dict(boxstyle='round,pad=1.0', facecolor='#eaf4fb',
                  edgecolor=color_estado, linewidth=3)
    )
 
    # -----------------------------------------------------------------------
    # EXPORTACIÓN DEL DASHBOARD
    # -----------------------------------------------------------------------
    ruta_dashboard = 'dashboard_completo.png'
    plt.savefig(ruta_dashboard, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Dashboard guardado: {ruta_dashboard}")
    plt.show()
 
    return fig
