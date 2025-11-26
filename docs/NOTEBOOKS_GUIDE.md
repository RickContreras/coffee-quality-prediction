# Guía de Notebooks para Proyecto de Predicción de Calidad del Café

Esta guía describe el propósito, contenido y entregables de cada notebook en el proyecto.

---

## 📊 01_exploratory_data_analysis.ipynb

### Objetivo
Explorar y entender profundamente el dataset de calidad del café del CQI antes de cualquier transformación.

### Tareas Principales

#### 1. Configuración Inicial
- Importar librerías necesarias (pandas, numpy, matplotlib, seaborn, missingno)
- Configurar opciones de visualización y estilo de gráficos
- Establecer semilla aleatoria para reproducibilidad

#### 2. Carga e Inspección de Datos
- Cargar datasets de Arabica y Robusta desde `data/raw/`
- Mostrar información básica: dimensiones, tipos de datos, primeras/últimas filas
- Listar todas las columnas y clasificarlas por tipo (numéricas, categóricas, target)

#### 3. Análisis de Calidad de Datos
- **Valores Faltantes:**
  - Calcular porcentaje de missing values por variable
  - Crear visualización con missingno (matriz y heatmap)
  - Identificar patrones de valores faltantes (MCAR, MAR, MNAR)
  - Clasificar variables: eliminar (>70% missing), imputar (<20%), analizar (20-70%)
- **Duplicados:**
  - Verificar registros duplicados completos
  - Mostrar ejemplos si existen
- **Consistencia:**
  - Verificar rangos válidos (variables sensoriales 0-10, Total Cup Score 0-100)
  - Identificar valores fuera de rango esperado

#### 4. Análisis Univariado

**Variables Numéricas:**
- Calcular estadísticas descriptivas completas (media, mediana, std, min, max, cuartiles)
- Calcular skewness y kurtosis para cada variable
- Interpretar distribuciones (simétrica, sesgada, colas pesadas)
- Crear histogramas + KDE para variables sensoriales
- Crear boxplots para detectar outliers
- Análisis detallado de variable objetivo (Total Cup Score):
  - Distribución completa con múltiples visualizaciones
  - Clasificación por categorías CQI (Exceptional ≥90, Excellent 85-89, Very Good 80-84)
  - Q-Q plot para test de normalidad
  - Función de distribución acumulada

**Variables Categóricas:**
- Calcular número de valores únicos
- Mostrar top categorías más frecuentes
- Crear gráficos de barras horizontales para top 10-15 categorías
- Analizar: Country of Origin, Processing Method, Variety, Region

#### 5. Análisis de Outliers
- Detectar outliers usando método IQR (rango intercuartílico)
- Detectar outliers usando Z-score (threshold=3)
- Calcular porcentaje de outliers por variable
- Crear tabla resumen de outliers
- Visualizar cantidad y porcentaje de outliers
- **Decisión preliminar:** Documentar qué hacer con outliers (mantener, transformar, eliminar)

#### 6. Conclusiones del EDA
- Resumen ejecutivo de hallazgos principales
- Insights clave sobre la calidad del café
- Decisiones documentadas para preprocesamiento
- Variables candidatas a eliminar
- Estrategias de imputación recomendadas

### Entregables
- Notebook completo con análisis y visualizaciones
- Archivo `reports/EDA_Summary_Report.txt` con conclusiones
- ~10-15 visualizaciones guardadas en `reports/figures/eda/`

---

## 🧹 02_data_preprocessing.ipynb

### Objetivo
Limpiar y preparar los datos aplicando las decisiones tomadas durante el EDA.

### Tareas Principales

#### 1. Carga de Datos
- Cargar dataset original desde `data/raw/`
- Verificar integridad de los datos

#### 2. Limpieza de Datos

**Valores Faltantes:**
- Implementar estrategia de imputación según tipo de variable:
  - Variables numéricas: mediana (robusto ante outliers)
  - Variables categóricas: moda (valor más frecuente)
  - Documentar cada decisión con justificación
- Eliminar variables con >70% missing values
- Crear indicadores de imputación si es necesario (flag columns)

**Duplicados:**
- Eliminar registros duplicados completos
- Documentar cuántos se eliminaron

**Inconsistencias:**
- Corregir formatos de texto (minúsculas, eliminar espacios)
- Estandarizar nombres de categorías (ej: "Washed" vs "washed" vs "WASHED")
- Convertir tipos de datos según corresponda

#### 3. Tratamiento de Outliers
- Aplicar decisión tomada en EDA para cada variable:
  - Mantener: documentar justificación (casos extremos legítimos)
  - Transformar: aplicar log, sqrt, o winsorizing
  - Eliminar: solo si son errores claros de datos
- Registrar cuántos outliers se trataron

#### 4. Eliminación de Variables Irrelevantes
- Eliminar columnas identificadas en EDA:
  - Variables con >70% missing
  - Variables con varianza cero
  - Variables ID o redundantes
- Documentar razón de eliminación

#### 5. Verificación de Datos Limpios
- Verificar que no quedan valores faltantes (excepto estratégicos)
- Verificar rangos de variables
- Calcular nuevas estadísticas descriptivas
- Comparar antes vs después de limpieza

#### 6. Guardar Dataset Limpio
- Guardar en `data/processed/coffee_cleaned.csv`
- Guardar versiones separadas si es necesario (arabica_cleaned, robusta_cleaned)
- Crear diccionario de datos actualizado

### Entregables
- Dataset limpio: `data/processed/coffee_cleaned.csv`
- Resumen de transformaciones aplicadas
- Comparación antes/después de limpieza

---

## 🔧 03_feature_engineering.ipynb

### Objetivo
Crear nuevas características, codificar variables categóricas y preparar datos para modelado.

### Tareas Principales

#### 1. Carga de Datos Limpios
- Cargar `data/processed/coffee_cleaned.csv`
- Verificar integridad

#### 2. Codificación de Variables Categóricas

**One-Hot Encoding:**
- Aplicar a variables con <20 categorías únicas
- Ejemplos: Processing Method, Color
- Verificar dimensionalidad resultante

**Label Encoding:**
- Considerar para variables ordinales si existen
- Documentar mapeo de etiquetas

**Frecuencia/Target Encoding (opcional):**
- Para variables con muchas categorías (Country, Region)
- Evitar data leakage (aplicar solo en train)

#### 3. Escalado y Normalización

**Estandarización (StandardScaler):**
- Aplicar a variables numéricas para modelos sensibles a escala
- Guardar parámetros (media, std) para aplicar en test

**Normalización Min-Max (opcional):**
- Considerar para redes neuronales
- Escalar entre [0,1] o [-1,1]

#### 4. Creación de Features Derivadas (opcional)
- Ratios entre variables sensoriales
- Interacciones entre features importantes
- Features polinómicas si mejoran correlación
- Agregaciones por país/región

#### 5. Análisis de Correlación Detallado

**Matriz de Correlación:**
- Calcular correlación de Pearson entre todas las variables numéricas
- Crear heatmap con máscara triangular
- Identificar top correlaciones con variable objetivo
- Visualizar top 10 positivas y negativas

**Análisis de Multicolinealidad:**
- Identificar pares con |r| > 0.85
- Decidir qué variables eliminar o combinar
- Calcular VIF (Variance Inflation Factor) si es necesario

#### 6. Análisis Bivariado

**Variables Numéricas vs Target:**
- Scatter plots de top 5-8 correlaciones con Total Cup Score
- Añadir líneas de regresión lineal
- Calcular R² para cada relación

**Variables Categóricas vs Target:**
- Boxplots/Violin plots por categoría
- Ejemplos: Total Cup Score por País, por Processing Method
- Test ANOVA para diferencias significativas
- Post-hoc tests si ANOVA es significativo

#### 7. Análisis Multivariado

**Pairplot:**
- Crear pairplot de top 5-6 variables más importantes
- Colorear por categoría de calidad

**Análisis por Segmentos:**
- Segmentar datos por Quality Category (Exceptional, Excellent, Very Good)
- Comparar estadísticas de features entre segmentos
- Identificar features que mejor discriminan calidad

#### 8. Feature Importance Preliminar

**Random Forest:**
- Entrenar Random Forest simple (no optimizado)
- Extraer importancia de Gini
- Visualizar top 15-20 features más importantes
- Usar como guía para selección de features

**Correlación Absoluta:**
- Rankear features por |correlación| con target
- Combinar con Random Forest importance

#### 9. Selección de Features Finales
- Eliminar features redundantes (alta multicolinealidad)
- Eliminar features con correlación muy baja (<0.05) con target
- Documentar features seleccionadas para modelado
- Crear lista final de features

#### 10. Guardar Datos Procesados
- Guardar dataset con features finales: `data/processed/coffee_features.csv`
- Guardar lista de features seleccionadas
- Guardar objetos de encoding/scaling (pickle/joblib)

### Entregables
- Dataset con features: `data/processed/coffee_features.csv`
- Lista de features seleccionadas
- Objetos de transformación guardados
- Matriz de correlación (figura)
- Top features por importancia (figura y tabla)

---

## 🤖 04_model_selection.ipynb

### Objetivo
Entrenar y comparar múltiples modelos de ML para identificar los mejores candidatos.

### Tareas Principales

#### 1. Preparación de Datos

**División de Datos:**
- Separar X (features) y y (target)
- División estratificada: 70% train, 15% validation, 15% test
- O usar train-test split (70-30) + cross-validation
- Verificar distribución de target en cada conjunto

**Pipeline de Preprocesamiento:**
- Crear ColumnTransformer para variables numéricas y categóricas
- Asegurar fit solo en train (evitar data leakage)

#### 2. Definición de Modelos Base

Entrenar al menos 5 modelos (requisito del proyecto):

**1. Modelo Paramétrico:**
- Linear Regression / Ridge / Lasso
- Explorar diferentes valores de alpha para regularización

**2. Modelo No Paramétrico:**
- K-Nearest Neighbors (KNN)
- Probar k = [3, 5, 7, 9, 11]
- Probar diferentes métricas de distancia

**3. Modelo de Ensemble (Árboles):**
- Random Forest Regressor
- Configuración base: n_estimators=100, max_depth=10

**4. Red Neuronal:**
- Multi-Layer Perceptron (MLPRegressor)
- Arquitecturas: (64,), (128,64), (256,128,64)
- Activation: relu, tanh

**5. Support Vector Machine:**
- SVR con diferentes kernels: linear, rbf, poly
- Explorar valores de C y gamma

**Opcionales (recomendados):**
- XGBoost / LightGBM
- Gradient Boosting
- Elastic Net

#### 3. Configuración Experimental

**Validación Cruzada:**
- K-Fold Cross-Validation (k=5 o 10)
- Calcular media y desviación estándar de métricas

**Métricas de Evaluación:**
- MAE (Mean Absolute Error): error promedio absoluto
- RMSE (Root Mean Squared Error): penaliza errores grandes
- R² (R-squared): bondad de ajuste
- MAPE (opcional): error porcentual

#### 4. Entrenamiento de Modelos Base

Para cada modelo:
- Entrenar con configuración base
- Evaluar en conjunto de validación
- Registrar métricas
- Calcular tiempo de entrenamiento
- Guardar predicciones

#### 5. Comparación de Modelos

**Tabla Comparativa:**
- Crear tabla con todos los modelos y sus métricas
- Incluir: MAE, RMSE, R², tiempo de entrenamiento
- Ordenar por mejor R² o MAE

**Visualizaciones:**
- Gráfico de barras: comparación de R² entre modelos
- Gráfico de barras: comparación de MAE entre modelos
- Scatter plot: MAE vs R² (trade-off)

#### 6. Curvas de Aprendizaje

Para top 2-3 modelos:
- Generar learning curves (train vs validation score)
- Analizar underfitting/overfitting
- Identificar si más datos ayudarían

#### 7. Análisis de Predicciones

**Predicción vs Real:**
- Scatter plot de valores predichos vs reales
- Añadir línea diagonal perfecta (y=x)
- Calcular R² en el gráfico

**Distribución de Errores:**
- Histograma de residuos (errores)
- Verificar normalidad de residuos
- Identificar patrones en errores

#### 8. Selección de Modelos para Optimización
- Identificar top 2-3 modelos con mejor desempeño
- Documentar justificación de selección
- Considerar: precisión, tiempo de entrenamiento, interpretabilidad

### Entregables
- Tabla comparativa de modelos
- Gráficos de comparación de métricas
- Learning curves de mejores modelos
- Identificación de top 2-3 modelos para optimizar

---

## 📉 05_dimensionality_reduction.ipynb

### Objetivo
Aplicar técnicas de reducción de dimensionalidad (PCA y UMAP) y evaluar impacto en modelos.

### Tareas Principales

#### 1. Preparación de Datos
- Cargar datos con todas las features
- Asegurar que datos estén escalados (crítico para PCA)
- Separar train/test

#### 2. Análisis de Componentes Principales (PCA)

**Varianza Explicada:**
- Aplicar PCA con todos los componentes
- Graficar varianza explicada por componente
- Graficar varianza explicada acumulada
- Identificar "codo" (elbow point)

**Criterios de Selección:**
- Mantener componentes que expliquen ≥85-95% varianza
- O criterio del codo
- Documentar justificación

**Aplicación de PCA:**
- Transformar datos con número seleccionado de componentes
- Calcular reducción de dimensionalidad (%)
- Ejemplo: 25 features → 10 componentes (60% reducción)

**Interpretación:**
- Analizar loadings de primeros 2-3 componentes
- Identificar qué features originales tienen mayor peso
- Visualizar en 2D/3D coloreando por Quality Category

#### 3. Aplicación de UMAP

**Exploración de Hiperparámetros:**
- n_neighbors: [5, 15, 30, 50]
- min_dist: [0.0, 0.1, 0.3, 0.5]
- n_components: [2, 3, 5, 10]
- metric: 'euclidean', 'manhattan', 'cosine'

**Selección de Configuración:**
- Probar diferentes combinaciones
- Visualizar embeddings 2D
- Seleccionar configuración que preserve estructura
- Balance entre estructura local y global

**Aplicación de UMAP:**
- Transformar con configuración óptima
- Visualizar en 2D/3D con colores por calidad
- Identificar clustering de categorías

#### 4. Re-entrenamiento de Modelos

**Modelos a Re-entrenar:**
- Top 2 mejores modelos de fase de optimización
- Ejemplos: Random Forest optimizado, MLP optimizado

**Con PCA:**
- Entrenar cada modelo con datos reducidos por PCA
- Evaluar en validation/test set
- Calcular métricas: MAE, RMSE, R²

**Con UMAP:**
- Entrenar cada modelo con datos reducidos por UMAP
- Evaluar en validation/test set
- Calcular mismas métricas

#### 5. Comparación de Resultados

**Tabla Comparativa:**
| Modelo | Datos Originales | PCA | UMAP |
|--------|-----------------|-----|------|
| Random Forest | R²=0.XX | R²=0.YY | R²=0.ZZ |
| MLP | R²=0.XX | R²=0.YY | R²=0.ZZ |

Incluir: MAE, RMSE, R², % reducción dimensión, tiempo entrenamiento

**Análisis de Trade-off:**
- Reducción de dimensión vs pérdida de desempeño
- ¿Vale la pena reducir dimensiones?
- ¿Qué método (PCA/UMAP) funciona mejor?

#### 6. Visualizaciones Finales

**Scatter Plots 2D:**
- Datos reducidos a 2D (PCA y UMAP)
- Colorear por categoría de calidad
- Identificar separabilidad de clases

**Importancia de Features (post-PCA):**
- Para Random Forest entrenado en componentes PCA
- Identificar componentes más importantes

#### 7. Conclusiones sobre Reducción

- ¿Se logró reducción significativa (>40%) sin pérdida de desempeño?
- ¿Qué técnica funciona mejor: PCA o UMAP?
- ¿Recomendación final: usar datos originales o reducidos?
- Documentar ventajas/desventajas de reducción

### Entregables
- Gráficos de varianza explicada (PCA)
- Visualizaciones 2D/3D de embeddings
- Modelos entrenados con datos reducidos
- Tabla comparativa completa
- Recomendación final documentada

