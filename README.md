# 🏋️‍♂️ Gym Members Classification Project / Proyecto de Clasificación de Miembros de Gimnasio

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ScikitLearn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

## 📌 **Problem Description / Descripción del Problema**

**[EN]** In modern fitness centers, clients exhibit a variety of different goals, experience levels, and exercise habits. Identifying hidden patterns within this data allows gyms to segment clients into groups with similar characteristics. This **Client Segmentation** strategy is vital for personalizing workout regimes, improving client retention, and optimizing gym resources.
This project uses **Unsupervised Machine Learning** to classify gym members into distinct categories based on their training information and personal characteristics.

**[ES]** En los gimnasios modernos, los clientes exhiben diferentes objetivos, niveles de experiencia y hábitos de ejercicio. Identificar patrones ocultos en estos datos permite segmentar a los clientes en grupos con características similares. Esta estrategia de **Segmentación de Clientes** ayuda a personalizar entrenamientos, mejorar la retención de clientes y optimizar los recursos.
Este proyecto utiliza **Machine Learning No Supervisado** para clasificar a los miembros en diferentes categorías.

---

## 📊 **Dataset**
- **Source / Fuente**: [Kaggle - Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)
- **Type / Tipo**: Public / Público
- **Features / Características**:
    - Demographics (Age, gender, height, and weight). / Demografía (Edad, género, altura, peso).
    - Exercise types performed. / Tipo de ejercicios realizados.
    - Training frequency and duration. / Frecuencia y duración de entrenamientos.
    - Personal training goals. / Objetivos personales.

---

## 🚀 **Methodology & Solution / Metodología y Solución**
1. **Exploratory Data Analysis (EDA) / Análisis Exploratorio**:
   - Evaluation of variable distributions, detection of outliers, and correlation mapping.
2. **Preprocessing / Preprocesamiento**:
   - Handling categorical encodings (OneHotEncoder).
   - Data normalization using `StandardScaler` to ensure optimal model efficiency.
3. **Clustering Models / Modelos de Clustering**:
   - **K-Means**: Baseline approach using the Elbow Method for optimal K estimation.
   - **DBSCAN**: Density-based spatial clustering to handle non-linearly separable relationships and outliers.
   - **Spectral Clustering**: Graph-based approach utilizing Nearest Neighbors affinity matrices.
4. **Evaluation / Evaluación**:
   - Validation utilizing **Silhouette Score**, **Davies-Bouldin Score**, and **Calinski-Harabasz Score**.

---

## 📂 **Repository Structure / Estructura del Repositorio**

```text
ML_Gym_Classification/
│── src/
│   ├── data_sample/         # Dataset sample / Muestra del dataset
│   ├── notebooks/           # Test notebooks / Notebooks de pruebas
│   ├── results_notebook/    # Final notebook with analysis / Notebook final con el análisis (project_notebook.ipynb)
│   ├── models/              # Saved trained models / Modelos entrenados guardados (*.pkl)
│   ├── utils/               # Auxiliary scripts / Scripts auxiliares
│   │   ├── data_preprocessing.py # Modular Data Preprocessing pipeline
│   │   ├── eda.py                # Functions for Exploratory Data Analysis
│   │   ├── model.py              # Clustering and Evaluation logic
│   │   ├── visualization.py      # Plotting and cluster rendering
│── README.md                # Project documentation
│── requirements.txt         # Dependencies
```

---

## 📜 **How to Run / Cómo Ejecutar el Proyecto**

1. **Clone this repository / Clonar repositorio**:
    ```bash
    git clone https://github.com/tu_usuario/ML_Gym_Classification.git
    cd ML_Gym_Classification
    ```

2. **Create a virtual environment and install dependencies / Instalar dependencias**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Run the final notebook / Ejecutar el notebook**:
    ```bash
    cd src
    jupyter notebook results_notebook/project_notebook.ipynb
    ```

---

## 📈 **Expected Results & Value / Resultados Esperados y Valor**
- **Actionable Insights / Obtención de Insights**: Identification of strict habits corresponding to different fitness levels (e.g. bodybuilders vs casual cardio enthusiasts).
- **Targeted Segmentation / Segmentación precisa**: Members sorted autonomously into highly cohesive clusters.
- **Data-Driven Gym Management / Gestión con Datos**: Results that can empower gym managers to design targeted marketing campaigns, group class schedules, and personalized plans.


