import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers, optimizers, initializers

# 1. PARAMETRELER VE SEED
OKUL_NO_SON_2_HANE = 21
np.random.seed(OKUL_NO_SON_2_HANE)
tf.random.set_seed(OKUL_NO_SON_2_HANE)

# 2. VERİ SETİ (Giriş: -2 <= p <= 2, Hedef: y = 1 + sin(pi * p / 4))
p = np.linspace(-2, 2, 200).reshape(-1, 1)
y = 1 + np.sin(np.pi * p / 4)

activations = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'silu']
act_display_names = ['Lineer', 'Sigmoid', 'Hiperbolik Tanjant', 'ReLU', 'LeakyReLU', 'ELU', 'SİLU']

results = []
histories = {}
predictions = {}

print("Eğitim başladı, modeller sırayla eğitiliyor...\n")

for act, name in zip(activations, act_display_names):
    print(f">> {name} eğitiliyor...", end=" ", flush=True)
    
    # Model Mimarisi: 1 Gizli Katman (2 Nöron), 1 Lineer Çıkış
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(
            2, 
            activation=layers.LeakyReLU(negative_slope=0.2) if act == 'leaky_relu' else act,
            kernel_initializer=initializers.RandomNormal(seed=OKUL_NO_SON_2_HANE)
        ),
        layers.Dense(1, activation='linear', kernel_initializer=initializers.RandomNormal(seed=OKUL_NO_SON_2_HANE))
    ])
    
    model.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss='mse')
    
    start_time = time.time()
    history = model.fit(p, y, epochs=300, verbose=0, batch_size=32)
    elapsed_time = time.time() - start_time
    
    y_pred = model(p, training=False).numpy()
    mse_val = mean_squared_error(y, y_pred)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y, y_pred)
    
    results.append({
        'name': name,
        'mse': mse_val,
        'rmse': rmse_val,
        'r2': r2_val,
        'time': elapsed_time
    })
    histories[name] = history.history['loss']
    predictions[name] = y_pred
    
    print(f"Tamamlandı! (MSE: {mse_val:.4f}, Süre: {elapsed_time:.2f} sn)")

# TABLO ÇIKTISI
print("\n" + "="*70)
print(f"{'Aktivasyon':<20} | {'MSE':<10} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("="*70)
for res in results:
    print(f"{res['name']:<20} | {res['mse']:<10.5f} | {res['rmse']:<10.5f} | {res['r2']:<10.5f} | {res['time']:<10.2f}")
print("="*70)

# EN İYİ MODEL VE GRAFİK
best_model = min(results, key=lambda x: x['mse'])
best_name = best_model['name']
print(f"\nEn Başarılı Model: {best_name} (MSE: {best_model['mse']:.5f})")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(histories[best_name], color='firebrick')
plt.title(f'Kayıp (Logaritmik Eksen) - {best_name}')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(p, y, label='Hedef Fonksiyon', color='navy', linewidth=2)
plt.plot(p, predictions[best_name], '--', label=f'Ağ Çıkışı ({best_name})', color='crimson', linewidth=2)
plt.title(f'Hedef vs Çıkış (R2: {best_model["r2"]:.4f})')
plt.xlabel('p (Giriş)')
plt.ylabel('Çıkış')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
