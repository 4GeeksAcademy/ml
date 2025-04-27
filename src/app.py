from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Carga del conjunto de datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
df = pd.read_csv(url, sep=";")
raw_data_path = "./data/raw"
os.makedirs(raw_data_path, exist_ok=True)
df.to_csv(f"{raw_data_path}/bank-marketing-campaign-data.csv", index=False)

# Análisis exploratorio de datos
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())
print("Número de filas duplicadas:", df.duplicated().sum())

plt.figure(figsize=(10,5))
sns.histplot(df['age'], bins=50, kde=True)
plt.title("Distribución de edad de los clientes")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df['duration'], bins=50, kde=True)
plt.title("Distribución de duración de llamada")
plt.show()

# Preprocesamiento y división en Train/Test
columns_to_keep = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 
                   'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
                   'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 
                   'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
df_filtered = df[columns_to_keep]
df_filtered['y'] = df_filtered['y'].map({'yes': 1, 'no': 0})
train, test = train_test_split(df_filtered, test_size=0.2, random_state=42)
processed_data_path = "./data/processed"
os.makedirs(processed_data_path, exist_ok=True)
train.to_csv(f"{processed_data_path}/train.csv", index=False)
test.to_csv(f"{processed_data_path}/test.csv", index=False)

# Construcción del modelo de regresión logística
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                        'loan', 'contact', 'month', 'day_of_week', 'poutcome']
numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
numeric_transformer = SimpleImputer(strategy='mean')
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=500))
])

X_train = train.drop(columns=['y'])
y_train = train['y']
X_test = test.drop(columns=['y'])
y_test = test['y']
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.4f}")

# Optimización del modelo
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__solver': ['lbfgs', 'liblinear']
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Precisión del modelo optimizado: {accuracy_best:.4f}")

