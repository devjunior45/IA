# 🤖 Intérprete de Libras com IA 

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.9.3-red)

Este projeto usa **Python, OpenCV, mediapipe e TensorFlow** como bibliotecas principais, para reconhecer algumas letras do  alfabeto em Libras a partir de coordenadas geradas por imagens.  
Este projeto permite, capturar coordenadas das mãos utilizando mediapipe e reconhecer letras do alfabeto, utilizando tecnicas de DEEP LEARNING.

## 📂 Estrutura do Projeto
- `dataset/` → coordenadas das maos geradas por imagens, em arquivo .CSV 
- `models/` → Modelo treinado e salvo
- `src/` → Código-fonte utilizado para treinamento do modelo

- ## 📊 Resultados e Estatísticas

métricas obtidas durante o treinamento do modelo:

- **Acurácia no Treinamento**: 97.54%
- **Acurácia na Validação**: 98.18%
- **Loss no Treinamento**: 0.0819
- **Loss na Validação**: 0.0680

- ## 🎲 dataset
- o dataset deste modelo, se trata de um arquivo csv com as coordenadas  referente a cada letra, são 63 valores (21 landmarks × 3 coordenadas) retiradas de cada imagem, e rotuladas com seu respectivo  valor(letra).
- exemplo a seguir:

| Índice | x           | y             | z          |
|--------|------------|---------------|------------|
| 0      | 9.77e-07   | 9.1255557e-07 | 9.127e-07  |
| ...    | ...        | ...           | ...        |
| 21     | 9.77e-07   | 9.77e-07      | 9.77e-07   |


## Processando os Dados  

Aqui carregamos o dataset, e dividimos os dados em  treino e teste, obtendo a quantidade de dados em cada parte.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Carregar o dataset
df = pd.read_csv('dataset.csv')


# Separar features (características) e rótulos
X = df.iloc[:, :-1].values  # Todas as colunas, exceto a última
y = df['label'].values      # Última coluna (rótulos)

# Codificar os rótulos para números
codificador_rotulos = LabelEncoder()
y_codificado = codificador_rotulos.fit_transform(y)  # Converte rótulos para números (ex: 'A' -> 0, 'B' -> 1)

# Converter rótulos para one-hot encoding
y_one_hot = to_categorical(y_codificado)

# Dividir em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

print("Formato dos dados:")
print(f"X_treino: {X_treino.shape}, y_treino: {y_treino.shape}")
print(f"X_teste: {X_teste.shape}, y_teste: {y_teste.shape}")
```

## Criando um novo modelo
- aqui criamos um modelo de rede neural com sequential usando funçoes de ativação relu e softmax, e obtemos um resumo do modelo.
```from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, Dropout


modelo = Sequential([
    Dense(128, activation='relu', input_shape=(X_treino.shape[1],)),  # Camada de entrada
    Dropout(0.2),  # Regularização para evitar overfitting
    Dense(64, activation='relu'),  # Camada oculta
    Dropout(0.2),
    Dense(32, activation='relu'),  # Outra camada oculta
    Dense(y_one_hot.shape[1], activation='softmax')  # Camada de saída (softmax para classificação)
])

# Compilar o modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumo do modelo
modelo.summary()
```
## Treinamento
- agora treinamos o modelo utilizandos 50 épocas, com lotes de 32 por vez
```
historico = modelo.fit(
    X_treino, y_treino,
    epochs=50,  # Número de épocas
    batch_size=32,  # Tamanho do lote
    validation_data=(X_teste, y_teste),  # Dados de validação
    verbose=1  # Mostrar progresso
)
```











