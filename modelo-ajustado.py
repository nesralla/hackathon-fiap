from sklearn.model_selection import KFold
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from distutils.file_util import copy_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
import tensorflow as tf

# Configuração para validação cruzada
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ajustando o modelo para incluir BatchNormalization
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', 
                     kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(92, (3, 3), strides=(2, 2), padding='same', 
                     kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Aumentado dropout para regularização
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

input_shape = (64, 64, 3)
num_classes = y_train_tf.shape[1]

# Armazenar os resultados
fold_accuracies = []
fold_histories = []

# Validação cruzada
for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
    print(f"Treinando o fold {fold + 1}...")
    
    # Dados de treino e validação
    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
    y_train_fold, y_val_fold = y_train_tf[train_idx], y_train_tf[val_idx]
    
    # Data augmentation
    train_gen = datagen.flow(x_train_fold, y_train_fold, batch_size=50)
    
    # Modelo para o fold
    fold_model = build_model(input_shape, num_classes)
    
    # Callbacks
    checkpointer = ModelCheckpoint(filepath=f'fold_{fold+1}_weights.best.hdf5', verbose=0, save_best_only=True)
    monitor = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    # Treinamento
    history = fold_model.fit(
        train_gen,
        validation_data=(x_val_fold, y_val_fold),
        epochs=50,
        callbacks=[checkpointer, monitor],
        verbose=1
    )
    
    # Avaliação
    val_loss, val_accuracy = fold_model.evaluate(x_val_fold, y_val_fold, verbose=0)
    fold_accuracies.append(val_accuracy)
    fold_histories.append(history)

# Desempenho médio
print(f"Acurácia média da validação cruzada: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

# Modelo final usando todos os dados
final_model = build_model(input_shape, num_classes)
final_history = final_model.fit(
    datagen.flow(x_train, y_train_tf, batch_size=50),
    validation_split=0.25,
    epochs=50,
    callbacks=[checkpointer, monitor],
    verbose=1
)
