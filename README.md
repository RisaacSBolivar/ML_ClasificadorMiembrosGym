# 🏋️‍♂️ Gym Members Classification Project / Proyecto de Clasificación de Miembros de Gimnasio

## 📌 **Problem Description / Descripción del Problema**
In gyms, clients have different goals, experience levels, and exercise habits. Identifying patterns within this data allows segmenting clients into groups with similar characteristics, which can help personalize workouts, improve client retention, and optimize gym resources.

En los gimnasios, los clientes tienen diferentes objetivos, niveles de experiencia y hábitos de ejercicio. Identificar patrones dentro de estos datos permite segmentar a los clientes en grupos con características similares, lo que puede ayudar a personalizar entrenamientos, mejorar la retención de clientes y optimizar los recursos del gimnasio.

This project uses **Unsupervised Machine Learning** to classify gym members into different categories based on their training information and personal characteristics.

Este proyecto utiliza **Machine Learning No Supervisado** para clasificar a los miembros de un gimnasio en diferentes categorías en función de su información de entrenamiento y características personales.

## 📊 **Dataset**
- **Source / Fuente**: [Kaggle - Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)
- **Type / Tipo**: Public / Público
- **Description / Descripción**: Contains information about gym members' exercise habits, including:
   - Age, gender, height, and weight. / Edad, género, altura y peso.
   - Types of exercises performed. / Tipo de ejercicios realizados.
   - Training frequency and duration. / Frecuencia y duración de los entrenamientos.
   - Personal training goals. / Objetivos personales de entrenamiento.

## 🚀 **Adopted Solution / Solución Adoptada**
1. **Data Exploration and Analysis (EDA) / Exploración y análisis de datos (EDA)**: Evaluation of variable distributions, detection of outliers, and correlation analysis. / Evaluación de la distribución de variables, detección de valores atípicos y análisis de correlaciones.
2. **Preprocessing / Preprocesamiento**: Data normalization to improve model efficiency. / Normalización de datos para mejorar la eficiencia del modelo.
3. **Classification Model / Modelo de Clasificación**: **Clustering** is used to segment members into categories based on their habits and characteristics. / Se utiliza **clustering** para segmentar a los miembros en categorías según sus hábitos y características.
4. **Results Visualization / Visualización de Resultados**: Representation of detected groups for interpretation and analysis. / Representación de los grupos detectados para su interpretación y análisis.

## 📂 **Repository Structure / Estructura del Repositorio**
```
ML_Gym_Classification/
│── src/
│   ├── data_sample/         # Dataset sample / Muestra del dataset
│   ├── img/                 # Images generated in EDA / Imágenes generadas en el EDA
│   ├── notebooks/           # Test notebooks / Notebooks de pruebas
│   ├── results_notebook/    # Final notebook with analysis / Notebook final con el análisis
│   ├── models/              # Saved models / Modelos guardados
│   ├── utils/               # Auxiliary functions / Funciones auxiliares
│   │   ├── data_preprocessing.py  # Data preprocessing / Preprocesamiento de datos
│   │   ├── model.py                 # Model implementation / Implementación del modelo
│   │   ├── eda.py                   # EDA functions / Funciones del análisis exploratorio
│── README.md
│── requirements.txt
```

## 📜 **Run the Project / Ejecutar el Proyecto**
1. Clone this repository / Clonar este repositorio:
    ```bash
    git clone https://github.com/tu_usuario/ML_Gym_Classification.git
    cd ML_Gym_Classification
    ```
2. Install dependencies / Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the **final notebook** / Ejecutar el **notebook final**:
    ```bash
    jupyter notebook src/results_notebook/project_notebook.ipynb
    ```

## 📈 **Expected Results / Resultados Esperados**
- Identification of patterns in exercise habits. / Identificación de patrones en los hábitos de ejercicio.
- Classification of members into different categories. / Clasificación de miembros en diferentes categorías.
- Visualization and interpretation of generated clusters. / Visualización e interpretación de los clusters generados.


