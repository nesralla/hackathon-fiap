import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from time import time
import requests
import os

# Carregar o modelo treinado
modelo_path = "modelo/my_model.keras"  # Substitua pelo caminho do seu modelo .keras
model = load_model(modelo_path)

# Parâmetros do modelo e vídeo
IMG_SIZE = 64
VIDEO_PATH = "video/video.mp4"  # Substitua pelo caminho do seu vídeo
ALERT_INTERVAL = 30  # Tempo mínimo entre notificações (segundos)

# Configuração da notificação (opcional)
PUSHBULLET_API_KEY = os.getenv("PUSHBULLET_API_KEY")  # Substitua pela sua chave API do Pushbullet
SEND_NOTIFICATIONS = True  # Defina como True para enviar notificações

# Variável para controlar tempo da última notificação
last_notification_time = 0

def send_notification(title, message):
    """Envia uma notificação via Pushbullet."""
    global last_notification_time
    current_time = time()
    
    # Verifica se já passou 30 segundos desde a última notificação
    if (current_time - last_notification_time) >= ALERT_INTERVAL:
        if SEND_NOTIFICATIONS and PUSHBULLET_API_KEY:
            url = "https://api.pushbullet.com/v2/pushes"
            headers = {
                "Access-Token": PUSHBULLET_API_KEY,
                "Content-Type": "application/json"
            }
            data = {
                "type": "note",
                "title": title,
                "body": message
            }
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                print("[NOTIFICAÇÃO ENVIADA] Objeto cortante detectado!")
                last_notification_time = current_time  # Atualiza o tempo da última notificação
            else:
                print("[ERRO] Falha ao enviar notificação.")
        else:
            print("[ALERTA] Objeto cortante detectado!")
    else:
        print("[IGNORADO] Objeto já detectado recentemente.")

# Captura de vídeo
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop quando o vídeo termina

    # Redimensiona para o tamanho do modelo
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Faz a predição
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = ''
    color = (0, 0, 0)

    # Verifica se é um objeto cortante (classe 1)
    if predicted_class == 1:
        label = "Cortante Detectado!"
        color = (0, 0, 255)  # Vermelho
        send_notification("Alerta!", "Objeto cortante detectado no vídeo.")
    else:
        label = "Nao Cortante"
        color = (0, 255, 0)  # Verde

    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Exibe o vídeo (opcional)
    cv2.imshow("Detecta objetos cortantes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
