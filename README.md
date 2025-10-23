# Coffee Quality Prediction using CQI Database

Este proyecto tiene como objetivo desarrollar un sistema de predicción de la calidad del café utilizando técnicas de aprendizaje automático sobre la base de datos del Coffee Quality Institute (CQI). La base de datos contiene datos sensoriales, físicos y de origen de muestras de café arábica y robusta, evaluadas profesionalmente en términos de su puntaje total de catación (Total Cup Score).

La predicción de la calidad del café puede ayudar a democratizar la evaluación de calidad, reducir costos y sesgos asociados a catadores humanos certificados, y favorecer decisiones informadas en la cadena de valor del café, desde productores hasta compradores.

## Contenido del Repositorio

- `data/`
  - `raw/` : Datos originales sin modificar (no incluido por tamaño)
  - `processed/` : Datos limpios y preprocesados listos para análisis
- `notebooks/` : Jupyter notebooks para exploración, análisis y visualizaciones
- `src/` : Código fuente organizado en módulos
  - `preprocessing.py` : Funciones y clases para preprocesamiento de datos
  - `train.py` : Scripts para entrenamiento y validación de modelos
  - `evaluate.py` : Evaluación y generación de reportes de desempeño
- `models/` : Modelos entrenados guardados (pickle, joblib)
- `reports/` : Resultados, gráficos, y reporte final en formato PDF
- `requirements.txt` : Lista de dependencias y versiones para reproducibilidad
- `.gitignore` : Archivos y carpetas ignoradas en git (por ejemplo, datos pesados, modelos)
- `README.md` : Documento de descripción y guía del proyecto

## Requisitos e Instalación

Este proyecto fue desarrollado en Python 3.8+ con las siguientes principales librerías:

(por definir)

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- umap-learn

Para instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Uso y Ejecución

1. Descargar la base de datos original del CQI desde Kaggle:  
   https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi/data  
   y colocarla en `data/raw/`.

2. Ejecutar el preprocesamiento:  

```bash
python src/preprocessing.py
```

3. Entrenar modelos:  
```bash
python src/train.py
```

4. Evaluar modelos y generar reportes:  
```bash
python src/evaluate.py
```
5. Explorar notebooks para análisis visuales y exploratorios en `notebooks/`.

## Estructura del Proyecto

```bash
coffee-quality-prediction/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
├── src/
│ ├── preprocessing.py
│ ├── train.py
│ └── evaluate.py
├── models/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```
## Autores

- Ricardo Contreras Garzón
- GitHub: [RickContreras](https://github.com/RickContreras)

- Daniel Leon Danzo
- GitHub: [DainXOR](https://github.com/DainXOR)

- Santiago Graciano David
- GitHub: [santiagogracianod](https://github.com/santiagogracianod)

## Licencia

----

## Referencias

- Coffee Quality Institute - CQI database: https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi/data
- Artículos relacionados revisados y usados para guía metodológica (Por agregar)
