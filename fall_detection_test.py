import numpy as np
import tensorflow as tf
import time
from sklearn.preprocessing import StandardScaler
import joblib

# Carregar o modelo treinado
model = tf.keras.models.load_model('fall_detection_model.keras')

# Carregar o scaler (se tiver salvo)
try:
    scaler = joblib.load('scaler.save')
    usar_scaler = True
    print("Scaler carregado com sucesso.")
except:
    usar_scaler = False
    print("Scaler não encontrado. Usando dados sem normalização (apenas para demonstração).")

# Features usadas (ajustar conforme seu dataset)
features = ['acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z']

n_features = len(features)

def gerar_amostra_sintetica():
    acc = np.random.uniform(-20, 20, 3)        # acc_x, acc_y, acc_z
    gyro = np.random.uniform(-400, 400, 3)     # gyro_x, gyro_y, gyro_z
    mag = np.random.uniform(-50, 50, 3)        # mag_x, mag_y, mag_z
    return np.concatenate([acc, gyro, mag])

# Loop de simulação contínua
try:
    while True:
        # Gerar amostra
        amostra = gerar_amostra_sintetica().reshape(1, -1)

        # Escalar se possível
        if usar_scaler:
            amostra_escalada = scaler.transform(amostra)
        else:
            amostra_escalada = amostra

        # Fazer a predição
        predicao = model.predict(amostra_escalada, verbose=0)
        resultado = (predicao > 0.5).astype(int)

        # Exibir os dados e resultado
        print("\n📡 Nova leitura dos sensores:")
        for nome, valor in zip(features, amostra[0]):
            print(f"{nome}: {valor:.2f}")

        print("\n🔍 Resultado da previsão:")
        if resultado[0][0] == 1:
            print("🔴 Queda detectada!")
        else:
            print("🟢 Nenhuma queda detectada.")

        print(f"📊 Probabilidade de queda: {predicao[0][0]*100:.2f}%")

        # Esperar antes de gerar a próxima leitura
        time.sleep(2)  # Delay de 2 segundos (ajustável)

except KeyboardInterrupt:
    print("\n🛑 Simulação encerrada pelo usuário.")