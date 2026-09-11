"""
modelo_mej.py: Versión mejorada del modelo sin framework. Se agregan 3 
               mejoras, solo se seleccionan 2 features, prior_gpa y 
               avg_sleep_hours ya que dan mejor R^2. Además de un early stopping
               si encuentra la epoch en la que mejor se desencuelve y después ya
               no mejora. Como segundo cambio se usan términos polinomiales para
               poder capturar relaciones no lineales. Finalmente se aplica L2
               ya que al implementar los términos polinomiales se tiene el 
               riesgo de multicolinealidad.
Autor: Alfredo Alejandro Soto Herrera
Matrícula: A01711368
Fecha: 10/09/2026
"""

# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  # Solo para dividir df

np.random.seed(42)  # Permite fijar la semilla de números pseudoaleatorios

df = pd.read_csv('df_processed.csv')  # Leemos el datset ya limpio

"""
El primer cambio es dejar solo la variable prior_gpa y avg_sleep_hours. Ya que
dejar las demás dan un R^2 menor.
"""
base_cols = ['prior_gpa', 'avg_sleep_hours']
target_col = 'term_gpa'

x_base = df[base_cols].values.astype(float)
y = df[target_col].values.astype(float).reshape(-1, 1)


"""
Agregamos términos polinomiales en las 2 features. Sigue siendo lineal, solo 
tiene más columnas de entrada para poder capturar relaciones no lineales 
entre prior_gpa/avg_sleep_hours y el GPA.
"""
def agregar_polinomiales(x):
    p1 = x[:, 0]  # prior_gpa
    p2 = x[:, 1]  # avg_sleep_hours
    return np.column_stack([p1, p2, p1 ** 2, p2 ** 2, p1 * p2])

x = agregar_polinomiales(x_base)
feature_names = ['prior_gpa', 'avg_sleep_hours', 'prior_gpa^2',
                  'avg_sleep_hours^2', 'prior_gpa*avg_sleep_hours']


"""
División 60% train / 20% val / 20% test
"""
x_train_val, x_test, y_train_val, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.25, random_state=42
)


"""
Estandarizamos valores para evitar que haya variables que dominen a otras
o que el gradiente se mueva de manera desigual.
"""
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)
x_std[x_std == 0] = 1.0  # evita división por cero
x_train = (x_train - x_mean) / x_std
x_val = (x_val - x_mean) / x_std
x_test = (x_test - x_mean) / x_std
n_features = x_train.shape[1]


"""
Hiperparámetros
"""
learning_rate = 0.05
epochs = 6000
lambda_reg = 150 # Regularización L2 para controlar la multicolinealidad de los 
                 # términos polinomiales.
patience = 800  # Epochs sin mejora antes de parar


"""
Inicializamos los parámetros de la regresión lineal, un peso (w) por cada
feature, y un sesgo (b). 
"""
w = np.zeros((n_features, 1))
b = 0.0


"""
Forward pass que es en base a la regresión lineal, por lo que la predicción es 
simplemente la combinación lineal de las features.
"""
def forward(x, w, b):
    return x @ w + b


"""
Utilizamos MSE como función de costo para luego calcular el gradiente y
actualizar los pesos y el sesgo.
"""
def mse_loss(y_hat, y):
    return np.mean((y_hat - y) ** 2)


"""
Calculamos el r^2 para observar que tal explica la varianza.
"""
def r2_score(y_hat, y):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


"""
Cálculo del gradiente y L2 de MSE respecto a w y b.
"""
def gradientes(x, y, y_hat, w, lambda_reg):
    n = y.shape[0]
    error = y_hat - y
    dw = (1 / n) * (x.T @ error) + (lambda_reg / n) * w
    db = (1 / n) * np.sum(error)
    return dw, db


"""
Esta función representa al Gradient Descent, el cual en este caso es batch
Gradient Descent ya que estamos pasando todas las muestras de una vez a 
diferencia de SGD que va una por una.
"""
def train_step(x, y, w, b, learning_rate, lambda_reg):
    y_hat = forward(x, w, b)
    dw, db = gradientes(x, y, y_hat, w, lambda_reg)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


"""
Entrenamiento
"""
losses_train = []
losses_val = []

best_val_loss = np.inf
best_w, best_b = None, None
best_epoch = 0
epochs_sin_mejora = 0

for epoch in range(epochs):
    w, b = train_step(x_train, y_train, w, b, learning_rate, lambda_reg)

    y_hat_train = forward(x_train, w, b)
    y_hat_val = forward(x_val, w, b)

    current_loss_train = mse_loss(y_hat_train, y_train)
    current_loss_val = mse_loss(y_hat_val, y_val)
    losses_train.append(current_loss_train)
    losses_val.append(current_loss_val)

    if current_loss_val < best_val_loss:
        best_val_loss = current_loss_val
        best_w, best_b = w.copy(), b
        best_epoch = epoch
        epochs_sin_mejora = 0
    else:
        epochs_sin_mejora += 1

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Training Loss (MSE): {current_loss_train:.4f}, '
              f'Validation Loss (MSE): {current_loss_val:.4f}')

    if epochs_sin_mejora > patience:
        print(f'\nEarly stopping en la época {epoch} ')
        break

# Usamos los pesos de la mejor época
w, b = best_w, best_b
print(f'\nMejor época: {best_epoch}\n')


"""
Gráfica para poder visualizar cómo se comporta el costo en el paso de los 
epochs, comparando Training y Validation en busca de Overfitting o
Underfitting.
"""
plt.figure(figsize=(10, 6))
plt.plot(losses_train, label='Training Loss (MSE)')
plt.plot(losses_val, label='Validation Loss (MSE)')
plt.axvline(best_epoch, color='gray', linestyle=':', label=f'Mejor época ({best_epoch})')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.title('Pérdida vs. Épocas (con términos polinomiales, L2 y early stopping)')
plt.legend()
plt.tight_layout()
plt.savefig('static/loss_vs_epochs_mejorado.png', dpi=150)
plt.close()


"""
Se calculan las mismas métricas (MSE, RMSE, MAE, R^2) en los tres conjuntos.
"""
def evaluar(nombre, y_real, y_pred):
    mse = mse_loss(y_pred, y_real)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_real))
    r2 = r2_score(y_pred, y_real)
    print(f"{nombre:<12} MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R^2: {r2:.4f}")
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


y_hat_train_final = forward(x_train, w, b)
y_hat_val_final = forward(x_val, w, b)
y_hat_test = forward(x_test, w, b)

metrics_train = evaluar("Train", y_train, y_hat_train_final)
metrics_val = evaluar("Validation", y_val, y_hat_val_final)
metrics_test = evaluar("Test", y_test, y_hat_test)


"""
Gráfica de predicción VS valores reales del sataset
"""
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_hat_test, alpha=0.6, edgecolor='k')
lims = [min(y_test.min(), y_hat_test.min()), max(y_test.max(), y_hat_test.max())]
plt.plot(lims, lims, 'r--', label='Predicción perfecta')
plt.xlabel('GPA real')
plt.ylabel('GPA predicho')
plt.title('Predicción vs. Real (modelo mejorado, conjunto de prueba)')
plt.legend()
plt.tight_layout()
plt.savefig('static/pred_vs_real_mejorado.png', dpi=150)
plt.close()