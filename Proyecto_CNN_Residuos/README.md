# Clasificador de residuos mediante CNN

## Descripción del proyecto

Este proyecto desarrolla un modelo de Deep Learning basado en una Red Neuronal Convolucional (CNN) para clasificar imágenes de residuos en seis categorías: cardboard, glass, metal, paper, plastic y trash.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecutar aplicación Streamlit

Desde esta carpeta:

```bash
streamlit run streamlit_app.py
```

La aplicación carga el modelo usando una ruta relativa al propio archivo, acepta imágenes JPG/JPEG/PNG y muestra la clase y la confianza.

## Nota sobre el modelo

`modelo_residuos.keras` debe ser un archivo Keras válido. Si se copia o descarga como texto, se corrompe y aparece `zipfile.BadZipFile`. El archivo incluido fue regenerado para que la aplicación pueda ejecutarse; para obtener las métricas del proyecto hay que ejecutar de nuevo las celdas de entrenamiento del notebook y guardar el modelo con `model.save("modelo_residuos.keras")`.
