# NOTEBOOKS_GUIDE.md

```markdown
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

## ⚙️ 05_model_optimization.ipynb

### Objetivo
Optimizar hiperparámetros de los mejores modelos identificados en model selection.

### Tareas Principales

#### 1. Carga de Setup Experimental
- Cargar datos procesados
- Recrear split train/validation/test
- Cargar configuración de mejores modelos

#### 2. Definición de Espacios de Búsqueda

Para cada modelo seleccionado, definir grid de hiperparámetros:

**Random Forest:**
- n_estimators: [50, 100, 200, 500]
- max_depth: [5, 10, 15, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 'log2', None]

**MLP (Red Neuronal):**
- hidden_layer_sizes: [(64,), (128,64), (256,128,64), (128,128)]
- activation: ['relu', 'tanh']
- learning_rate: [0.001, 0.01, 0.1]
- alpha (regularización): [0.0001, 0.001, 0.01]
- max_iter: [100, 200, 500]

**SVM:**
- kernel: ['linear', 'rbf', 'poly']
- C: [0.1, 1, 10, 100]
- gamma: [0.001, 0.01, 0.1, 1] (para rbf/poly)
- epsilon: [0.01, 0.1, 0.5]

#### 3. Optimización de Hiperparámetros

**GridSearchCV:**
- Búsqueda exhaustiva en grid completo
- Cross-validation: 5-fold
- Scoring: 'neg_mean_absolute_error' o 'r2'
- n_jobs=-1 (usar todos los cores)

**RandomizedSearchCV (alternativa):**
- Búsqueda aleatoria (más rápida para grids grandes)
- n_iter: 50-100 iteraciones
- Mismo scoring y CV que GridSearch

#### 4. Análisis de Resultados de Optimización

Para cada modelo optimizado:
- Mostrar mejores hiperparámetros encontrados
- Comparar score de modelo base vs optimizado
- Crear tabla de top 5-10 configuraciones
- Visualizar efecto de hiperparámetros clave en desempeño

#### 5. Validación de Modelos Optimizados

**Evaluación en Validation Set:**
- Predecir con mejores modelos
- Calcular métricas: MAE, RMSE, R²
- Comparar con modelos base

**Curvas de Aprendizaje:**
- Generar learning curves para modelos optimizados
- Comparar con curvas de modelos base
- Verificar reducción de overfitting

#### 6. Análisis de Convergencia (MLP)

Para redes neuronales:
- Graficar loss curve durante entrenamiento
- Verificar convergencia
- Identificar si necesita más epochs

#### 7. Entrenamiento Final

**Re-entrenar en Train + Validation:**
- Combinar train y validation sets
- Entrenar modelos optimizados en dataset combinado
- Guardar modelos finales en `models/`

**Persistencia:**
- Guardar modelos con joblib o pickle
- Nombrar claramente: `random_forest_optimized.pkl`
- Guardar también hiperparámetros en JSON/YAML

#### 8. Comparación Final

**Tabla Comparativa:**
- Modelo base vs optimizado para cada algoritmo
- Mejora porcentual en métricas
- Tiempo de entrenamiento

**Mejores Modelos:**
- Identificar top 1-2 modelos generales
- Documentar configuración final

### Entregables
- Modelos optimizados guardados en `models/`
- Tabla de mejores hiperparámetros
- Comparación base vs optimizado
- Archivo de configuración de hiperparámetros

---

## 📉 06_dimensionality_reduction.ipynb

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

---

## 📊 07_final_results.ipynb

### Objetivo
Consolidar y presentar resultados finales del proyecto, comparar con estado del arte.

### Tareas Principales

#### 1. Resumen del Proyecto
- Descripción breve del problema
- Objetivos del proyecto
- Dataset utilizado (características principales)
- Metodología seguida

#### 2. Carga de Resultados

**Modelos Entrenados:**
- Cargar todos los modelos guardados
- Cargar resultados de experimentos anteriores

**Datos de Test:**
- Cargar conjunto de test no visto
- Verificar que no se usó durante entrenamiento/validación

#### 3. Evaluación Final en Test Set

Para cada modelo final:
- Generar predicciones en test set
- Calcular métricas: MAE, RMSE, R², MAPE
- Calcular intervalos de confianza (bootstrap si es posible)

#### 4. Tabla Resumen de Todos los Experimentos

**Estructura de Tabla:**
| Experimento | Modelo | Features | MAE | RMSE | R² | Notas |
|-------------|--------|----------|-----|------|----|-------|
| Baseline | Linear Reg | Todas | X.XX | X.XX | 0.XX | - |
| Base | Random Forest | Todas | X.XX | X.XX | 0.XX | - |
| Optimizado | RF Optimized | Todas | X.XX | X.XX | 0.XX | Grid Search |
| PCA | RF Optimized | 10 PCA | X.XX | X.XX | 0.XX | 60% reducción |
| UMAP | RF Optimized | 5 UMAP | X.XX | X.XX | 0.XX | n_neighbors=15 |

Ordenar por R² o MAE

#### 5. Visualizaciones de Resultados

**Predicción vs Real:**
- Scatter plot del mejor modelo
- Añadir línea diagonal y de regresión
- Colorear puntos por categoría de calidad
- Mostrar R² y MAE en el gráfico

**Análisis de Residuos:**
- Gráfico de residuos vs valores predichos
- Histograma de residuos (verificar normalidad)
- Q-Q plot de residuos

**Feature Importance Final:**
- Del mejor modelo (Random Forest o XGBoost)
- Top 15-20 features más importantes
- Gráfico de barras horizontal

**Comparación de Modelos:**
- Gráfico de barras: R² de todos los modelos
- Gráfico de barras: MAE de todos los modelos
- Gráfico de radar: múltiples métricas

#### 6. Análisis de Errores

**Casos con Mayor Error:**
- Identificar 10-20 predicciones con mayor error absoluto
- Analizar características de estos casos
- ¿Hay patrones? (país, método procesamiento, etc.)

**Distribución de Errores por Categoría:**
- Calcular MAE por categoría de calidad
- ¿El modelo predice mejor ciertos rangos de calidad?

#### 7. Comparación con Estado del Arte

**Artículos Revisados:**
Recordar los artículos que revisaste:
1. "Coffee Quality Prediction" (2024): Random Forest R²=0.82, MAE=0.16
2. "LGBM Algorithm" (2023): Accuracy 72% (clasificación)
3. "UBC-MDS" (2021): ROC-AUC=0.67 (clasificación)
4. "ML Techniques" (2023): SVR mejor desempeño

**Tabla Comparativa:**
| Fuente | Método | Métrica | Resultado |
|--------|--------|---------|-----------|
| Artículo 1 | Random Forest | R² | 0.82 |
| Artículo 1 | Random Forest | MAE | 0.16 |
| **Tu Trabajo** | **Random Forest Opt** | **R²** | **0.XX** |
| **Tu Trabajo** | **Random Forest Opt** | **MAE** | **0.XX** |

**Análisis:**
- ¿Superaste los resultados del estado del arte?
- Si sí: ¿Por qué? (más features, mejor optimización, etc.)
- Si no: ¿Por qué? (menos datos, diferentes features, etc.)

#### 8. Insights y Descubrimientos Clave

**Features Más Importantes:**
- Top 5 features que más influyen en calidad
- Coherencia con conocimiento del dominio (café)

**Hallazgos Interesantes:**
- Relaciones inesperadas descubiertas
- Patrones en datos
- Diferencias por país/método de procesamiento

#### 9. Limitaciones del Estudio

**Limitaciones de Datos:**
- Dataset de 2018 (no captura variaciones recientes)
- Desbalance Arabica (1312) vs Robusta (28)
- Posible sesgo geográfico

**Limitaciones Metodológicas:**
- Supuestos de modelos
- Variables no capturadas (microclima, prácticas específicas)
- Validación limitada a datos históricos

#### 10. Recomendaciones y Trabajo Futuro

**Para Mejorar Modelos:**
- Incorporar más datos recientes
- Incluir features de clima/suelo
- Probar deep learning (LSTM, Transformers)
- Ensemble de múltiples modelos

**Aplicaciones Prácticas:**
- Sistema de predicción para productores pequeños
- Herramienta de verificación de precios
- App móvil para evaluación rápida

#### 11. Conclusiones Finales

**Resumen de Logros:**
- Modelo final alcanzó R²=0.XX y MAE=0.XX
- Reducción de dimensión logró X% sin pérdida significativa
- Sistema puede predecir calidad con X% de precisión

**Valor del Proyecto:**
- Democratiza evaluación de calidad
- Reduce costos de certificación
- Potencial de impacto en cadena de valor

**Reflexión Final:**
- Aprendizajes clave del proyecto
- Habilidades desarrolladas

### Entregables
- Notebook completo con todos los resultados
- Todas las visualizaciones finales
- Tabla comparativa completa
- Sección de conclusiones documentada
- Exportar resultados a `reports/final_report.pdf` (opcional)

---

## 📝 Buenas Prácticas Generales para Todos los Notebooks

### Estructura de Cada Notebook

1. **Título y Metadata**
   - Número y nombre del notebook
   - Descripción breve del objetivo
   - Autor y fecha

2. **Tabla de Contenidos**
   - Para notebooks largos (>100 celdas)
   - Links clickeables a secciones

3. **Setup y Configuración**
   - Imports organizados por categoría
   - Configuración de visualización
   - Semillas aleatorias para reproducibilidad

4. **Desarrollo Secuencial**
   - Secciones con headers Markdown (##, ###)
   - Código comentado
   - Interpretación después de cada resultado
   - Visualizaciones con títulos y labels claros

5. **Conclusiones**
   - Resumen de hallazgos al final
   - Decisiones documentadas
   - Siguiente pasos

### Estilo de Código

```
# ✅ BUENO
# Calculate correlation matrix for sensory variables
sensory_vars = ['Aroma', 'Flavor', 'Aftertaste']
corr_matrix = df[sensory_vars].corr()

# ❌ MALO (sin comentarios, nombres crípticos)
sv = ['Aroma', 'Flavor', 'Aftertaste']
cm = df[sv].corr()
```

### Visualizaciones

- Siempre incluir títulos descriptivos
- Labels en ejes X e Y
- Leyendas cuando sea necesario
- Tamaño de fuente legible (12-14pt)
- Guardar figuras importantes en alta resolución (300 dpi)

### Documentación

- Explicar **qué** haces y **por qué**
- Interpretar resultados numéricos
- Documentar decisiones importantes
- Añadir referencias a literatura cuando sea relevante

---

## ✅ Checklist de Completitud

Antes de finalizar cada notebook, verificar:

- [ ] Todas las celdas ejecutan sin errores
- [ ] Resultados son reproducibles (semillas fijas)
- [ ] Código está comentado adecuadamente
- [ ] Visualizaciones tienen títulos y labels
- [ ] Conclusiones están documentadas
- [ ] Se guardaron outputs necesarios (datos, modelos, figuras)
- [ ] Notebook está limpio (eliminar experimentos fallidos)
- [ ] Markdown está bien formateado
- [ ] Paths de archivos son relativos (no absolutos)

---

## 📚 Referencias

- Guía del Proyecto: `Guia_proyecto_Modelos_II.pdf`
- Dataset: https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi
- Artículos del Estado del Arte (ver carpeta `references/`)

---

**Última actualización**: 2025-10-23
**Versión**: 1.0
```

***

Este archivo `.md` te servirá como guía completa para desarrollar cada notebook del proyecto de manera profesional, organizada y cumpliendo con todos los requisitos del curso. Guárdalo en la raíz de tu repositorio para que tú y cualquier colaborador sepan exactamente qué hacer en cada etapa.

[1](https://github.com/kylebradbury/ml-project-structure-demo)
[2](https://www.reddit.com/r/MachineLearning/comments/g8h58c/d_how_do_you_structure_and_organize_your_mldl/)
[3](https://www.kaggle.com/general/4815)
[4](https://app.readytensor.ai/publications/markdown-for-machine-learning-projects-a-comprehensive-guide-LX9cbIx7mQs9)
[5](https://github.com/onesamblack/machine-learning-template/blob/main/README.md)
[6](https://domino.ai/blog/the-importance-of-structure-coding-style-and-refactoring-in-notebooks)
[7](https://dev.to/luxdevhq/generic-folder-structure-for-your-machine-learning-projects-4coe)
[8](https://github.com/ZenithClown/ai-ml-project-template)
[9](https://towardsdatascience.com/its-time-to-structure-your-data-science-project-1fa064fbe46/)
[10](https://www.overleaf.com/latex/templates)