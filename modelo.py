from pathlib import Path
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from distutils.file_util import copy_file
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
import tensorflow as tf


# %%Obtendo o dataset

caminho_pasta = "data/"
classes = ['cortantes', 'nao_cortantes']

pasta_imgs_cortantes = Path(caminho_pasta + classes[0])
pasta_imgs_nao_cortantes = Path(caminho_pasta + classes[1])

extensao = ".jpg"

#Vou renomear todos os itens para que todas as imagens sejam .jpg
def renomear_arquivos(pasta_imgs, extensao):
    for arquivo in pasta_imgs.iterdir():
        if arquivo.is_file():
            novo_nome = arquivo.with_suffix(extensao)
            arquivo.rename(novo_nome)

    print(f"Arquivos na pasta {pasta_imgs} foram renomeados para a extensão {extensao}.\n\n")

renomear_arquivos(pasta_imgs_cortantes, extensao)
renomear_arquivos(pasta_imgs_nao_cortantes, extensao)


filepaths_cortantes = list(pasta_imgs_cortantes.glob(r'**/*.jpg'))
filepaths_nao_cortantes = list(pasta_imgs_nao_cortantes.glob(r'**/*.jpg'))

def exibir_numero_itens_filepath(filepaths_cortantes, filepaths_nao_cortantes):
    print("Pasta de objetos cortantes:")
    print(len(filepaths_cortantes))
    print("Pasta de objetos não cortantes:")
    print(f"{len(filepaths_nao_cortantes)} \n\n")

exibir_numero_itens_filepath(filepaths_cortantes, filepaths_nao_cortantes)

print("Realizando balanceamento manual...\n\n")

# %%Balanceando manualmente o dataset
#Vou pegar a quantidade de imagens da pasta com menos imagens e vou pegar a mesma quantidade da outra pasta
def manual_balance(filepaths_cortantes, filepaths_nao_cortantes):
    minimo = min(len(filepaths_cortantes), len(filepaths_nao_cortantes))
    filepaths_cortantes = filepaths_cortantes[:minimo]
    filepaths_nao_cortantes = filepaths_nao_cortantes[:minimo]
    return filepaths_cortantes, filepaths_nao_cortantes

filepaths_cortantes, filepaths_nao_cortantes = manual_balance(filepaths_cortantes, filepaths_nao_cortantes)

exibir_numero_itens_filepath(filepaths_cortantes, filepaths_nao_cortantes)

def get_labels_images(filepaths):
    labels = []
    images = []

    image_size = 64

    for filepath in filepaths:
        head = os.path.split(filepath)
        obj = os.path.split(head[0])

        labels.append(obj[1])#armazena rotulo cortante ou nao_cortante obtido pelo path

        img = cv2.imread(str(filepath))

        # Check if image was loaded correctly
        if img is not None:
            img = cv2.resize(img, (image_size, image_size)).astype('float32') / 255.0 #redimensiona e normaliza imagem
            images.append(img)
        else:
            print(f"Failed to load image: {filepath}") # Print the problematic filepath

    #Converte a imagem em lista de array
    images = np.array(images)
    #Converte as labels para uma lista de array
    labels = np.array(labels)
    
    return labels, images

print("Obtendo rótulo das imagens...\n\n")
labels_cortantes, images_cortantes = get_labels_images(filepaths_cortantes)
labels_nao_cortantes, images_nao_cortantes = get_labels_images(filepaths_nao_cortantes)
labels = np.concatenate((labels_cortantes, labels_nao_cortantes))
images = np.concatenate((images_cortantes, images_nao_cortantes))

print("Quantidade de imagens e rótulos obtidos:")
print(len(images))
print(len(labels))

print("Imagens e rótulos obtidos com sucesso!\n\n")

# %%Construindo o dataframe
print("Iniciando a construção do dataframe... \n\n")
filepaths = filepaths_cortantes + filepaths_nao_cortantes
pd_filepaths = pd.Series(filepaths, name='Filepath').astype(str)
pd_labels = pd.Series(labels, name='Label')

df = pd.concat([pd_filepaths, pd_labels], axis=1)
print(df['Label'].value_counts())

# %%Embaralhando o dataset
print("Embaralhando o dataset e exibindo as primeiras linhas... \n\n")
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())
print(df['Label'].value_counts())
print("Dataset embaralhado com sucesso!\n\n")

# %%Separando a base de treino e teste
print("Separando base de 80% treino e de 20% teste... \n\n")
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

print("Base de treino e teste separadas com sucesso!\n\n")

# %%Preparando os dados para a rede convolucional
print("Preparando dados para a rede convolucional... \n\n")
#Achatando os dados de teste e treino
x_train_flat = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_test_flat = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])

#Usar o labelEncoder para transformar nossas labels em numeros binarios
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#Converte um vetor de classe (inteiros) em uma matriz de classe categorica
y_train_tf = keras.utils.to_categorical(y_train_encoded, len(classes))
y_test_tf = keras.utils.to_categorical(y_test_encoded, len(classes))

"""Vamos configurar agora como próximo passo o ModelCheckpoint para usar os melhores pesos 
para este modelo. Um dos benefícios do ModelCheckpoint é salvar uma cópia do modelo em disco 
em intervalos regulares (como após cada época de processamento) para que você possa retomar 
o treinamento a partir do ponto em que parou, minimizando perdas de tempo e recursos computacionais"""
#Salva os melhores pesos para esse modelo
checkpointer = ModelCheckpoint(filepath='weights.best.hdf5.keras', verbose=0, save_best_only=True)

input_shape = (64, 64, 3) # imagens em dimensões 64x64 em RGB

learning_rate = 0.001

tf.random.set_seed(42) #Define uma semente aleatória para utilizar sempre os mesmo dados durante nossos testes

model = Sequential() #Abrindo uma sequencia de modelo

#1 camada convolucional com 128 neuronios
#filtro utilizando uma matriz 3x3 tendo o deslocamento de 2
#padding com bordas zero; função de ativação RELU; regularização L2 ativada (suavizar a penalidade dos coeficientes)
model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=input_shape))

#Camada MaxPolling ativada, com uma matriz 2x2. Padding ativado
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

#Regularizacao Dropout ativada
model.add(Dropout(0.2)) #remove 20%

#2 camada convolucional com 92 neuronios
#filtro utilizando uma matriz 3x3 tendo o deslocamento de 2
#padding com bordas zero; função de ativação RELU; regularização L2 ativada (suavizar a penalidade dos coeficientes)
model.add(Conv2D(92, (3,3), strides=(2,2), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))

#Camada MaxPolling ativada, com uma matriz 2x2. Padding ativado
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

#Regularizacao Dropout ativada
model.add(Dropout(0.2)) #remove 20%

#Camada que achata os dados da imagem
model.add(Flatten())

#Camada Dense da rede neural convolucional + função ativação ReLU
model.add(Dense(256, activation='relu'))

#Camada de saida da rede, utilizando a função de ativação sigmoid, classificação binaria
model.add(Dense(y_train_tf.shape[1], activation='sigmoid'))

#sigmoid -> classificação binária
#softmax -> classificação multiclasses

#y_train_tf = 2

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1, mode='auto')

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])

model.save('my_model.keras')

model.summary()

#Treinamento do model
history = model.fit(x_train, y_train_tf, validation_split=0.25, callbacks=[monitor,checkpointer],
                    verbose=1, epochs=45, batch_size=50, shuffle=True)

# %%Avaliando o modelo
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower left')
plt.grid(True)
plt.show()

#Validação Acuracia por acurácia por epocas
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower left')
plt.grid(True)
plt.show()

# %%Testando o modelo
cnn_predict = model.predict(x_test)

cnn_predict = np.argmax(cnn_predict, axis=1)

y_true = np.argmax(y_test_tf, axis=1)

cnn_cm = metrics.confusion_matrix(y_true, cnn_predict)

cnn_accuracy = metrics.classification_report(y_true, cnn_predict, target_names=classes)

print(metrics.classification_report(y_true, cnn_predict))

