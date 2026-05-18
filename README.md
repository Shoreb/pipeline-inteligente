# Pipeline Inteligente de Análisis de Datos

Este proyecto automatiza el flujo completo de datos: desde la carga de un archivo pequeño de 40 registros, su limpieza y expansión sintética a 500 filas, hasta el entrenamiento de un modelo de Inteligencia Artificial para predecir precios.

## ⚙️ Preparación

Instala las librerías necesarias con el siguiente comando:

```bash
pip install -r requirements.txt
```

## ▷ Cómo ejecutarlo

Asegúrate de que el archivo `datos_consolidados_40_registros.csv` esté dentro de la carpeta `datos/` y ejecuta:

```bash
python pipeline-inteligente.py
```

## 🗀 Archivos principales

- `pipeline-inteligente.py`: El corazón del proyecto (contiene todo el código).
- `datos/`: Carpeta que contiene la fuente de datos original.
- `requirements.txt`: Lista de librerías de Python requeridas.

## 📊 ¿Qué resultados obtendrás?

Al finalizar la ejecución, el script generará automáticamente estos archivos:

- `datos_500_registros.csv`: El dataset final procesado y ampliado.
- `reporte_metricas.txt`: Un resumen del rendimiento del modelo de IA.
- `dashboard_completo.png`: Una imagen que resume todo el análisis visualmente.
- `Gráficos de diagnóstico (.png)`: Imágenes adicionales con el análisis de datos y errores del modelo.