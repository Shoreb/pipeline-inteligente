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