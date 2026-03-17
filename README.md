# Análisis Exploratorio de Datos (EDA)

Este repositorio contiene un proyecto desarrollado como parte de la asignatura de Minería de Datos. El objetivo principal es aplicar técnicas de análisis exploratorio de datos (EDA) para comprender y preparar datasets, utilizando herramientas y bibliotecas de Python. A continuación, se detalla el contenido del proyecto y su propósito educativo.

## Estructura del Proyecto

- **Datasets/**: Contiene los archivos de datos utilizados para el análisis.
  - `BankChurners.csv`: Dataset relacionado con la retención de clientes en el sector bancario. Este dataset se utilizó para explorar patrones y características que influyen en la decisión de los clientes de abandonar el banco.
  - `Mall_Customers.csv`: Dataset utilizado para analizar el comportamiento de los clientes en un centro comercial, incluyendo segmentación y patrones de compra.

- **Notebooks/**: Incluye notebooks de Jupyter con el análisis exploratorio.
  - `paquete_eda.ipynb`: Notebook principal que utiliza una clase personalizada `AnalisisEDA` para realizar tareas de EDA. Este notebook incluye:
    - Importación del módulo `paquete_analisis_eda_ns`.
    - Carga de datasets.
    - Métodos para obtener vistas rápidas de los datos, como `head()` y `tail()`.
    - Análisis básicos como conteo de filas, detección de valores atípicos y generación de estadísticas descriptivas.

- **Projecto_Caso_Estudio/**: Contiene scripts Python relacionados con el análisis.
  - `AnalisisExploratorioEDA.py`: Script que implementa funciones reutilizables para el análisis exploratorio. Este archivo fue diseñado para automatizar tareas comunes y reducir la repetición de código.
  - `paquete_analisis_eda_ns.py`: Módulo que centraliza las funciones de análisis exploratorio, incluyendo carga de datos, limpieza y visualización.


## Detalles del Notebook `paquete_eda.ipynb`

El notebook `paquete_eda.ipynb` fue diseñado para estandarizar el flujo de trabajo del EDA, facilitando la comprensión y limpieza de los datos. Utiliza la clase `AnalisisEDA` para centralizar tareas comunes, como:

1. **Carga de Datos**: Permite cargar datasets desde archivos CSV con opciones de configuración como separador, decimales y columnas de índice.
2. **Vista Rápida**: Métodos como `head()` y `tail()` para validar la carga de datos y obtener una idea general del contenido.
3. **Análisis Básico**: Incluye métodos para contar filas y columnas, identificar valores nulos y generar estadísticas descriptivas.
4. **Visualización**: Generación de gráficos básicos para identificar patrones y tendencias en los datos.

## Propósito Educativo

Este proyecto fue desarrollado como parte de la clase de Minería de Datos para cumplir con los siguientes objetivos:

- **Aplicar Técnicas de EDA**: Familiarizarse con las técnicas de análisis exploratorio para comprender mejor los datos antes de aplicar modelos predictivos o de clustering.
- **Automatización de Tareas**: Diseñar herramientas reutilizables que permitan realizar análisis consistentes en diferentes datasets.
- **Visualización de Datos**: Utilizar gráficos para identificar patrones, tendencias y posibles problemas en los datos.
- **Preparación de Datos**: Identificar y manejar valores atípicos, datos faltantes y otras irregularidades.

## Cómo Usar Este Proyecto

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/AnalisisExploratorioDatos.git
   ```
2. Asegúrate de tener Python instalado y las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Abre los notebooks en Jupyter para explorar los análisis:
   ```bash
   jupyter notebook
   ```

## Requisitos

- Python 3.8 o superior.
- Bibliotecas necesarias: pandas, matplotlib, seaborn, entre otras.

## Contribuciones

Si deseas contribuir a este proyecto, por favor crea un fork del repositorio, realiza tus cambios y envía un pull request. ¡Las contribuciones son bienvenidas!

---

Este proyecto está diseñado para facilitar el análisis exploratorio de datos y estandarizar tareas comunes, reduciendo errores y mejorando la eficiencia en el análisis de datasets. Además, busca reforzar los conceptos aprendidos en la clase de Minería de Datos.