# ☕ Coffee Quality Prediction using CQI Database

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12.3-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

*Sistema de predicción de calidad del café usando Machine Learning*

[Características](#-características) • [Instalación](#-instalación) • [Uso](#-uso) • [Autores](#-autores) • [Licencia](#-licencia)

</div>


## 📖 Descripción

Este proyecto desarrolla un **sistema de predicción de la calidad del café** utilizando técnicas de aprendizaje automático sobre la base de datos del **Coffee Quality Institute (CQI)**. La base de datos contiene datos sensoriales, físicos y de origen de muestras de café arábica y robusta, evaluadas profesionalmente en términos de su puntaje total de catación (Total Cup Score).

### 🎯 Objetivos

- 🔍 Democratizar la evaluación de calidad del café
- 💰 Reducir costos asociados a catadores humanos certificados
- 📊 Reducir sesgos en evaluaciones de calidad
- 🤝 Favorecer decisiones informadas en la cadena de valor del café

## ✨ Características

- 📊 **Análisis Exploratorio completo** con visualizaciones interactivas
- 🧹 **Pipeline de preprocesamiento robusto**
- 🔧 **Feature engineering** avanzado con análisis de correlación
- 🤖 **5+ modelos de ML** (Linear, KNN, Random Forest, Neural Networks, SVM)
- 📉 **Reducción de dimensionalidad** (PCA y UMAP)
- 📈 **Evaluación completa** con métricas MAE, RMSE y R²

## 📂 Estructura del Proyecto


```bash
coffee-quality-prediction/
├── 📁 data/
│   ├── raw/              # Datos originales sin modificar
│   └── processed/        # Datos limpios listos para ML
├── 📓 notebooks/         # Jupyter notebooks (5 notebooks)
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_selection.ipynb
│   └── 05_dimensionality_reduction.ipynb
├── 📜 scripts/
│   └── download_data.py # Script automático para descargar datos
├── 🤖 models/           # Modelos entrenados (.pkl)
├── 📊 reports/          # Reportes y visualizaciones
│   ├── figures/         # Gráficos generados
│   └── tables/          # Tablas de resultados
├── 📋 requirements.txt
├── 📖 README.md
└── 🔒 .gitignore
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.12.3+
- pip o conda


### 📦 Dependencias Principales

| Librería | Propósito |
|----------|-----------|
| 🐼 `pandas` | Manipulación de datos |
| 🔢 `numpy` | Operaciones numéricas |
| 🤖 `scikit-learn` | Modelos de ML |
| 📊 `matplotlib` & `seaborn` | Visualizaciones |
| 🗺️ `umap-learn` | Reducción dimensional |
| ⚡ `xgboost` | Gradient Boosting |
| 📓 `jupyter` | Notebooks interactivos |
| 📥 `kaggle` | Descarga automática de datos | 

> 📋 Ver [`requirements.txt`](requirements.txt) para todas las dependencias

### 🔧 Pasos de Instalación

#### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/RickContreras/coffee-quality-prediction.git
cd coffee-quality-prediction
```

#### 2️⃣ Crear entorno virtual

<details>
<summary><b>🐧 Linux / macOS</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```
</details>

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
# Dependiendo de tu version, deberas usar una combinacion de
# comandos diferente
python -m venv .venv
python3 -m venv .venv

./.venv/Scripts/Activate.ps1
./.venv/bin/Activate.ps1
```
</details>

<details>
<summary><b>🪟 Windows (CMD)</b></summary>

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```
</details>

#### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

## 🔑 Configuración de Kaggle API

Para descargar los datos automáticamente, necesitas configurar tus credenciales de Kaggle:

1. 📝 Inicia sesión en [Kaggle](https://www.kaggle.com/)
2. ⚙️ Ve a **Settings** → **API** → **Create New API Token**
3. 💾 Descarga el archivo `kaggle.json`
4. 📁 Coloca `kaggle.json` en la **raíz del proyecto**

```bash
coffee-quality-prediction/
├── kaggle.json ✅  # Aquí
├── README.md
└── ...
```

## 💻 Uso

### 📥 1. Descargar datos

```bash
python scripts/download_data.py
```

> 💡 **Nota:** También puedes descargar manualmente desde [Kaggle](https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi/data)

### 📓 2. Ejecutar notebooks

```bash
# Opción 1: Jupyter Notebook clásico
jupyter notebook

# Opción 2: JupyterLab (interfaz moderna)
jupyter lab
```

### 🔢 3. Orden de ejecución de notebooks

| # | Notebook | Descripción |
|---|----------|-------------|
| 1️⃣ | [`01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb) | 🔍 EDA completo con visualizaciones |
| 2️⃣ | [`02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb) | 🧹 Limpieza e imputación de datos |
| 3️⃣ | [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb) | 🔧 Creación y selección de features |
| 4️⃣ | [`04_model_selection.ipynb`](notebooks/04_model_selection.ipynb) | 🤖 Entrenamiento de 5+ modelos ML |
| 5️⃣ | [`05_dimensionality_reduction.ipynb`](notebooks/05_dimensionality_reduction.ipynb) | 📉 PCA y UMAP |

## 👥 Autores

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/RickContreras">
        <img src="https://github.com/RickContreras.png" width="100px;" alt=""/>
        <br />
        <sub><b>Ricardo Contreras Garzón</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/DainXOR">
        <img src="https://github.com/DainXOR.png" width="100px;" alt=""/>
        <br />
        <sub><b>Daniel Leon Danzo</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/santiagogracianod">
        <img src="https://github.com/santiagogracianod.png" width="100px;" alt=""/>
        <br />
        <sub><b>Santiago Graciano David</b></sub>
      </a>
    </td>
  </tr>
</table>


----

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

----

## 🙏 Referencias

- 📊 **Dataset**: [Coffee Quality Institute - CQI Database](https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi/data)
- 🤖 **Scikit-learn**: [Documentation](https://scikit-learn.org/)
