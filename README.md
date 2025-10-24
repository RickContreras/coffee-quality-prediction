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
- `scripts/` : Scripts utilitarios para configuración y descarga de datos
  - `download_data.py` : Script para descargar datos desde Kaggle
- `models/` : Modelos entrenados guardados (pickle, joblib)
- `reports/` : Resultados, gráficos, y reporte final en formato PDF
- `requirements.txt` : Lista de dependencias y versiones para reproducibilidad
- `.gitignore` : Archivos y carpetas ignoradas en git (por ejemplo, datos pesados, modelos)
- `README.md` : Documento de descripción y guía del proyecto

## Requisitos e Instalación

Este proyecto fue desarrollado en Python 3.12.3 con las siguientes principales librerías:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- umap-learn 
- kaggle
- plotly
- missingno
- jupyter
- ipykernel
- notebook
- joblib
- scipy  

Para instalar las dependencias:

Crea el entorno virtual

Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows (cmd)
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

Conda (opcional)
```bash
conda create -n coffee python=3.12.3 -y
conda activate coffee
```

Instala depedencias

```bash
pip install -r requirements.txt
```

### Configuración de Kaggle API

Para descargar los datos automáticamente, necesitas configurar tus credenciales de Kaggle:

1. Inicia sesión en [Kaggle](https://www.kaggle.com/)
2. Ve a tu perfil → **Settings** → **API** → **Create New API Token**
3. Se descargará un archivo `kaggle.json`
4. Pon `kaggle.json` en la raiz de este proyecto:

## Uso y Ejecución

1. **Descargar datos automáticamente**:

```bash
python scripts/download_data.py
```

   O manualmente desde: https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi/data

2. **Ejecutar el preprocesamiento**:  

```bash
python src/preprocessing.py
```

3. **Entrenar modelos**:  
```bash
python src/train.py
```

4. **Evaluar modelos y generar reportes**:  
```bash
python src/evaluate.py
```

5. **Explorar notebooks** para análisis visuales y exploratorios en `notebooks/`.

## Jupyter Notebook

```bash
# Opción 1: Jupyter Notebook clásico
jupyter notebook

# Opción 2: JupyterLab (más moderno)
jupyter lab
```

## Estructura del Proyecto

```bash
coffee-quality-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   └── download_data.py
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
