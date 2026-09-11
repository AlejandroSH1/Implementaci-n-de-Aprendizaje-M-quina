"""
modelo_con_fw_mej.py: Versión mejorada del modelo con framework usando Random
                      Forest, se usan solo 2 features, prior_gpa y 
                      avg_sleep_hours, además se agrega el ajuste de 
                      hiperparámetros vía grid search (n_estimators,
                      max_depth, min_samples_leaf).
Autor: Alfredo Alejandro Soto Herrera
Matrícula: A01711368
Fecha: 10/09/2026
"""

# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargamos el Dataset
df = pd.read_csv('df_processed.csv')

target_col = 'term_gpa'
feature_cols = ['prior_gpa', 'avg_sleep_hours'] # Solo 2 features
x = df[feature_cols].values.astype(float)
y = df[target_col].values.astype(float)
feature_names = feature_cols

"""
60% train / 20% val / 20% test 
"""
x_train_val, x_test, y_train_val, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.25, random_state=42
)

# A diferencia del modelo sin framework, no hace falta estandarizar.

"""
Se probaron las combinaciones de n_estimators [100,300,500], max_depth
[2,3,5,8] y min_samples_leaf [5,10,20,30], seleccionando la combinación
con mejor R2 en el conjunto de Validación. Resultado: n_estimators=500,
max_depth=3, min_samples_leaf=10 
"""
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=10,
    random_state=42)
model.fit(x_train, y_train)


"""
Se tienen las mismas métricas (MSE, RMSE, MAE, R^2) en los tres conjuntos. Para 
poder compararlas y diagnosticar su sesgo y varianza.
"""
def evaluar(nombre, y_real, y_pred):
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    print(f"{nombre:<12} MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R^2: {r2:.4f}")
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

y_hat_train = model.predict(x_train)
y_hat_val = model.predict(x_val)
y_hat_test = model.predict(x_test)

metrics_train = evaluar("Train", y_train, y_hat_train)
metrics_val = evaluar("Validation", y_val, y_hat_val)
metrics_test = evaluar("Test", y_test, y_hat_test)


"""
Graficamos los valores reales contra los predichos, para el train y para test
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_train, y_hat_train, alpha=0.6, edgecolor='k')
lims_train = [min(y_train.min(), y_hat_train.min()), max(y_train.max(), y_hat_train.max())]
axes[0].plot(lims_train, lims_train, 'r--', label='Predicción perfecta')
axes[0].set_xlabel('GPA real')
axes[0].set_ylabel('GPA predicho')
axes[0].set_title(f"TRAIN — R²={metrics_train['r2']:.3f}")
axes[0].legend()
axes[1].scatter(y_test, y_hat_test, alpha=0.6, edgecolor='k')
lims_test = [min(y_test.min(), y_hat_test.min()), max(y_test.max(), y_hat_test.max())]
axes[1].plot(lims_test, lims_test, 'r--', label='Predicción perfecta')
axes[1].set_xlabel('GPA real')
axes[1].set_ylabel('GPA predicho')
axes[1].set_title(f"TEST — R²={metrics_test['r2']:.3f}")
axes[1].legend()
plt.suptitle('Random Forest (modelo mejorado) — Predicción vs. Real')
plt.tight_layout()
plt.savefig('static/rf_mejorado_pred_vs_real.png', dpi=150)
plt.close()