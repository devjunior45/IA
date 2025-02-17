import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# Carregamento dos Dados
# Carregamos o dataset
df = pd.read_csv('dataset/dataset.csv')

# Separando caracteristicas e rotulos 
X = df.iloc[:, :-1].values  
y = df['label'].values 

# modificando os rótulos em números
codificador_rotulos = LabelEncoder()
y_codificado = codificador_rotulos.fit_transform(y)  # Converte rótulos para números (ex: 'A' -> 0, 'B' -> 1)

# Convertendo os  rótulos 
y_one_hot = to_categorical(y_codificado)

# Divisão em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

'''
caso queira visualizar os dados treino e teste
print("\nFormato dos dados:")
print(f"X_treino: {X_treino.shape}, y_treino: {y_treino.shape}")
print(f"X_teste: {X_teste.shape}, y_teste: {y_teste.shape}")'''


# Criando o modelo
modelo = Sequential([
    Dense(128, activation='relu', input_shape=(X_treino.shape[1],)),  # Camada de entrada
    Dropout(0.2),  # Regularização para evitar overfitting
    Dense(64, activation='relu'),  # Camada oculta
    Dropout(0.2),
    Dense(32, activation='relu'), 
    Dense(y_one_hot.shape[1], activation='softmax')  (softmax para classificação)
])

# Compilando o modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''# Resumo do modelo
print("\nResumo do modelo:")
modelo.summary()'''


# Treinando o modelo
historico = modelo.fit(
    X_treino, y_treino,
    epochs=50,  # Número de épocas
    batch_size=32,  # Tamanho do lote
    validation_data=(X_teste, y_teste),  # Dados de validação
    verbose=1  # progresso
)

'''# Salva o histórico de treinamento
with open('historico_treinamento.pkl', 'wb') as f:
    pickle.dump(historico.history, f)


# Avalia o modelo no conjunto de teste
perda, acuracia = modelo.evaluate(X_teste, y_teste, verbose=0)
print(f"\nAcurácia no conjunto de teste: {acuracia * 100:.2f}%")

# Visualiza as Métricas
# Carrega o histórico de treinamento
with open('historico_treinamento.pkl', 'rb') as f:
    historico = pickle.load(f)

# Plota a loss de treino e validação
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historico['loss'], label='Train Loss')
plt.plot(historico['val_loss'], label='Validation Loss')
plt.title('Loss durante as Épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

# Plotar a acurácia de treino e validação
plt.subplot(1, 2, 2)
plt.plot(historico['accuracy'], label='Train Accuracy')
plt.plot(historico['val_accuracy'], label='Validation Accuracy')
plt.title('Acurácia durante as Épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()'''

#Salvando o Modelo
modelo.save('modelo_libras.h5')
print("\nModelo libras salvo com sucesso!")
