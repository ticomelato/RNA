# fall_detection_rna.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Carregar os dados
train_data = pd.read_csv('data/Train.csv')
test_data = pd.read_csv('data/Test.csv')

print("Dados de treino carregados:", train_data.shape)
print("Dados de teste carregados:", test_data.shape)

print("\nPrimeiras linhas do treino:")
print(train_data.head())

# Verificar colunas
print("\nColunas do dataset:")
print(train_data.columns)

print("\nTipos dos dados:")
print(train_data.dtypes)

# Pré-processamento
# Remover as colunas irrelevantes
colunas_para_remover = ['Unnamed: 0', 'label']
train_data = train_data.drop(columns=colunas_para_remover)
test_data = test_data.drop(columns=colunas_para_remover)

# Verificar se há valores ausentes
print("\nValores nulos no treino:")
print(train_data.isnull().sum())

print("\nValores nulos no teste:")
print(test_data.isnull().sum())

# Separar features e labels
X_train = train_data.drop('fall', axis=1)
y_train = train_data['fall']

X_test = test_data.drop('fall', axis=1)
y_test = test_data['fall']

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir o modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Avaliar o modelo
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Salvar modelo se desejar
model.save('fall_detection_model.keras')
