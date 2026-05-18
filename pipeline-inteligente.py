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