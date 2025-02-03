import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import requests  # Para enviar notificações via API (exemplo: Pushbullet)

# Carregar o modelo treinado
modelo_path = "modelo/my_model.keras"  # Substitua pelo caminho do seu modelo .keras
model = load_model(modelo_path)

# Parâmetros do modelo
IMG_SIZE = 64  # Deve ser compatível com a entrada do modelo

# Configuração da notificação (opcional)
PUSHBULLET_API_KEY = "o.fTnOEQ5QXNUj7mkLCwreFLuEylREvqSO"  # Substitua pela sua chave API do Pushbullet
SEND_NOTIFICATIONS = True  # Defina como True para enviar notificações


def send_notification(title, message):
    """Envia uma notificação via Pushbullet."""
    if SEND_NOTIFICATIONS and PUSHBULLET_API_KEY:
        url = "https://api.pushbullet.com/v2/pushes"
        headers = {"Access-Token": PUSHBULLET_API_KEY, "Content-Type": "application/json"}
        data = {"type": "note", "title": title, "body": message}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print("Notificação enviada!")
        else:
            print("Falha ao enviar notificação:", response.text)


def preprocess_frame(frame):
    """Pré-processa o frame para ser compatível com o modelo."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converter para RGB
    image = Image.fromarray(image)  # Converter para formato PIL
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Redimensionar para o tamanho esperado
    image = img_to_array(image) / 255.0  # Normalizar para o intervalo [0, 1]
    return np.expand_dims(image, axis=0)  # Adicionar dimensão de batch


def detectar_objetos(video_path):
    """Processa um vídeo e detecta objetos cortantes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fim do vídeo

        frame_count += 1
        processed_frame = preprocess_frame(frame)

        # Fazer a previsão
        prediction = model.predict(processed_frame)[0]

        # Exibir a previsão no frame
        if prediction[1] > 0.5:  # Considera que [0] é "não cortante" e [1] é "cortante"
            label = "Cortante Detectado!"
            color = (0, 0, 255)  # Vermelho
            send_notification("Alerta de Detecção!", "Objeto cortante detectado no vídeo!")
        else:
            label = "Nao Cortante"
            color = (0, 255, 0)  # Verde

        # Desenhar rótulo no frame
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Exibir frame processado
        cv2.imshow("Detecção de Objetos Cortantes", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Caminho do vídeo
video_path = "video/video.mp4"  # Substitua pelo caminho do seu vídeo

# Iniciar a detecção
detectar_objetos(video_path)
