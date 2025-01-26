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