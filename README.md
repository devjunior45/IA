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






