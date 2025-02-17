import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Dicionário para traduzir números em letras
dicionario_gestos = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'I', 8: 'L',
    9: 'M', 10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T',
    17: 'U', 18: 'V', 19: 'W', 20: 'Y'
}

# Carregando o modelo TensorFlow Lite
interpretador = tf.lite.Interpreter(model_path='modelolibraslite.tflite')
interpretador.allocate_tensors()

# detalhes de entrada/saída
detalhes_entrada = interpretador.get_input_details()
detalhes_saida = interpretador.get_output_details()

# carregar MediaPipe Hands
mp_maos = mp.solutions.hands
maos = mp_maos.Hands()

# Capturando um video ja salvo, 0 para camera
captura = cv2.VideoCapture("caminho video, 0 para a camera")

while captura.isOpened():
    sucesso, imagem = captura.read()
    if not sucesso:
        break

    imagem = cv2.flip(imagem, 0)
    # Converter a imagem para RGB
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    
    # Processando a imagem 
    resultados = maos.process(imagem_rgb)

    # Verifica se as mãos foram detectadas
    if resultados.multi_hand_landmarks:
        for marcos_mao in resultados.multi_hand_landmarks:
            # Extrair coordenadas dos landmarks
            coordenadas = []
            for marco in marcos_mao.landmark:
                coordenadas.extend([marco.x, marco.y, marco.z])
            
            # Preparando os dados 
            dados_entrada = np.array([coordenadas], dtype=np.float32)
            
        
            interpretador.set_tensor(detalhes_entrada[0]['index'], dados_entrada)
            interpretador.invoke()
            dados_saida = interpretador.get_tensor(detalhes_saida[0]['index'])
            
            # prevendo  a letra
            classe_prevista = np.argmax(dados_saida, axis=1)
            rotulo_previsto = dicionario_gestos.get(classe_prevista[0], 'Desconhecido')  # Traduzir número para letra
            
          
            cv2.putText(imagem, f"Letra: {rotulo_previsto}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  


            # Desenhando os landmarks 
            mp.solutions.drawing_utils.draw_landmarks(
                imagem, marcos_mao, mp_maos.HAND_CONNECTIONS)

    # imagem
    cv2.imshow('Reconhecimento de Gestos', imagem)
    if cv2.waitKey(5) & 0xFF == 27:  # Pressione ESC para sair
        break

captura.release()
cv2.destroyAllWindows()
