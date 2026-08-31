"""
Modelo.py: Archivo que presenta la construcción de un modelo de ML sin utilizar 
           ningún tipo de Framework. Tienendo como base una regresión lineal
           multivariable.
Autor: Alfredo Alejandro Soto Herrera
Matrícula: A01711368
Fecha: 30/08/2026
"""

# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42) # Permite fijar la semilla de números pseudoaleatorios

df = pd.read_csv('df_processed.csv') # Leemos el datset ya limpio

# Codificamos la única variable categórica (gender) como 0/1
df['gender'] = (df['gender'] == 'Male').astype(float)
target_col = 'term_gpa'
x = df.drop(columns=[target_col]).values.astype(float)
y = df[target_col].values.astype(float).reshape(-1, 1)
feature_names = df.drop(columns=[target_col]).columns.tolist()


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
n_hidden = 16 # neuronas en la capa oculta
learning_rate = 0.05
epochs = 2000


"""
Se utiliza la Inicialización de parámetros HE para poder evitar que haya 
neuronas muertas al inicio del entrenamiento.
"""
W1 = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / n_features)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
b2 = np.zeros((1, 1))


"""
Como función de activación utilizamos ReLU ya que es de las más utilizadas
por su eficiencia.
"""
def relu(z):
    return np.maximum(0, z)


"""
Creamos una función de derivada de ReLU porque se utilizará más adelante en 
la parte de back propagation.
"""
def relu_derivada(z):
    return (z > 0).astype(float)


"""
Forward pass, es una pasada de inicio a fin de nuestra red para obtener una
predicción en base a las x, pesos y biases.
"""
def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = z2
    cache = (x, z1, a1, z2)
    return y_hat, cache


"""
Utilizamos MSE como función de costo para luego retro propagar y actualizar los
pesos y biases.
"""
def mse_loss(y_hat, y):
    return np.mean((y_hat - y) ** 2)


"""
La siguiente función es Back propagation, en donde estamos obteniendo los 
gradientes de cada punto para ir actualizando los pesos y biases.
"""
def backward(y, y_hat, cache, W2):
    x, z1, a1, z2 = cache
    n = y.shape[0]

    # Gradiente de la pérdida respecto a la salida (dMSE/dy_hat)
    dz2 = (2.0 / n) * (y_hat - y)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = dz2 @ W2.T 
    dz1 = da1 * relu_derivada(z1) # regla de la cadena a través de ReLU
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


"""
Esta función representa al Gradient Descent, el cual en este caso es batch
Gradient Descent ya que estamos pasando todas las muestras de una vez a 
diferencia de SGD ue va una por una.
"""
def train_step(x, y, W1, b1, W2, b2, learning_rate):
    y_hat, cache = forward(x, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(y, y_hat, cache, W2)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2


"""
Entrenamiento
"""
losses_train = []
losses_val = []

for epoch in range(epochs):
    W1, b1, W2, b2 = train_step(x_train, y_train, W1, b1, W2, b2, learning_rate)

    y_hat_train, _ = forward(x_train, W1, b1, W2, b2)
    y_hat_val, _ = forward(x_val, W1, b1, W2, b2)

    current_loss_train = mse_loss(y_hat_train, y_train)
    current_loss_val = mse_loss(y_hat_val, y_val)
    losses_train.append(current_loss_train)
    losses_val.append(current_loss_val)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Training Loss (MSE): {current_loss_train:.4f}, '
              f'Validation Loss (MSE): {current_loss_val:.4f}')


"""
Gráfica para poder visualizar cómo se comporta el costo en el paso de los 
epochs, comparando Training y Validation en busca de Overfitting o
Underfitting.
"""
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses_train, label='Training Loss (MSE)')
plt.plot(range(epochs), losses_val, label='Validation Loss (MSE)')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.title('Pérdida vs. Épocas')
plt.legend()
plt.tight_layout()
plt.savefig('static/loss_vs_epochs.png', dpi=150)
plt.close()


"""
Se busca comprobar el funcionamiento del modelo
"""
y_hat_test, _ = forward(x_test, W1, b1, W2, b2)
test_mse = mse_loss(y_hat_test, y_test) # Error promedio al cuadrado
test_rmse = np.sqrt(test_mse) # En promedio cuantos puntos de GPA se equivoca
test_mae = np.mean(np.abs(y_hat_test - y_test)) # Promedio de errores absolutos

# R^2: qué tan bien el modelo explica la varianza del GPA real
ss_res = np.sum((y_test - y_hat_test) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\nMSE en prueba:  {test_mse:.4f}")
print(f"RMSE en prueba: {test_rmse:.4f}")
print(f"MAE en prueba:  {test_mae:.4f}")
print(f"R^2 en prueba:  {r2:.4f}")


"""
Gráfica de predicción VS valores reales del sataset
"""
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_hat_test, alpha=0.6, edgecolor='k')
lims = [min(y_test.min(), y_hat_test.min()), max(y_test.max(), y_hat_test.max())]
plt.plot(lims, lims, 'r--', label='Predicción perfecta')
plt.xlabel('GPA real')
plt.ylabel('GPA predicho')
plt.title('Predicción vs. Real (conjunto de prueba)')
plt.legend()
plt.tight_layout()
plt.savefig('static/pred_vs_real.png', dpi=150)
plt.close()