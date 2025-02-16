import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# 2. Carregamento dos Dados
# Carregar o dataset
df = pd.read_csv('dataset.csv')

# Separar features 
X = df.iloc[:, :-1].values  
y = df['label'].values 

# modificar os rótulos em números
codificador_rotulos = LabelEncoder()
y_codificado = codificador_rotulos.fit_transform(y)  # Converte rótulos para números (ex: 'A' -> 0, 'B' -> 1)

# Converter rótulos 
y_one_hot = to_categorical(y_codificado)

# Divisão em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

'''
caso queira visualizar os dados treino e teste
print("\nFormato dos dados:")
print(f"X_treino: {X_treino.shape}, y_treino: {y_treino.shape}")
print(f"X_teste: {X_teste.shape}, y_teste: {y_teste.shape}")'''


# Criar o modelo
modelo = Sequential([
    Dense(128, activation='relu', input_shape=(X_treino.shape[1],)),  # Camada de entrada
    Dropout(0.2),  # Regularização para evitar overfitting
    Dense(64, activation='relu'),  # Camada oculta
    Dropout(0.2),
    Dense(32, activation='relu'), 
    Dense(y_one_hot.shape[1], activation='softmax')  (softmax para classificação)
])

# Compilar o modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''# Resumo do modelo
print("\nResumo do modelo:")
modelo.summary()'''


# Treinar o modelo
historico = modelo.fit(
    X_treino, y_treino,
    epochs=50,  # Número de épocas
    batch_size=32,  # Tamanho do lote
    validation_data=(X_teste, y_teste),  # Dados de validação
    verbose=1  # Mostrar progresso
)

'''# Salvar o histórico de treinamento
with open('historico_treinamento.pkl', 'wb') as f:
    pickle.dump(historico.history, f)


# Avaliar o modelo no conjunto de teste
perda, acuracia = modelo.evaluate(X_teste, y_teste, verbose=0)
print(f"\nAcurácia no conjunto de teste: {acuracia * 100:.2f}%")

# 6. Visualização das Métricas
# Carregar o histórico de treinamento
with open('historico_treinamento.pkl', 'rb') as f:
    historico = pickle.load(f)

# Plotar a loss de treino e validação
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historico['loss'], label='Train Loss')
plt.plot(historico['val_loss'], label='Validation Loss')
plt.title('Loss ao Longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

# Plotar a acurácia de treino e validação
plt.subplot(1, 2, 2)
plt.plot(historico['accuracy'], label='Train Accuracy')
plt.plot(historico['val_accuracy'], label='Validation Accuracy')
plt.title('Acurácia ao Longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()'''

# 7. Salvamento do Modelo
# Salvar o modelo treinado
modelo.save('modelo_treinado.h5')
print("\nModelo treinado salvo com sucesso!")
