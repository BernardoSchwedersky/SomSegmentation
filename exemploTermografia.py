# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:23:02 2025

@author: obern
"""


import somSeg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import plotly.express as px
from plotly.offline import plot

    
# Carregando uma imagem local
img_name=r"C:\Users\obern\Documents\GitHub\SomSegmentation\ExemploImagens\termo.bmp"
im = Image.open(img_name)


ROI_sample=np.asarray(im)
# Se a imagem for em escala de cinza, cria uma terceira dimensão
if ROI_sample.ndim<3:
    ROI_sample=np.expand_dims(ROI_sample, axis=2)



# Criando a img de entrada com os canais desejados (r+g+b)
img_shape=ROI_sample.shape
numero_canais=ROI_sample.shape[2]

# Se quiser adicionar mais canais, adicionar aqui - numero_canais+1 e ROI_sample_conv[:,:,4]=novo_canal
ROI_sample_conv=np.zeros((img_shape[0],img_shape[1], numero_canais))
ROI_sample_conv[:,:,0:3]=ROI_sample

# Vizualizando os canais de entrada
plt.figure(figsize=(15, 5))
for i in range(0,numero_canais):
    plt.subplot(1, numero_canais, i+1)
    plt.imshow(ROI_sample_conv[:,:,i])
    
# Parâmetros de ajuste do método 
patch_size=2 # Representa o tamanho de cada porção quadrada da imagem que será usada como cada exemplo
som_size=5 # Número de unidades do SOM
epochs=1 # Quantidade de épocas do treinamento
dataset_percentage=1 # Porcentagem dos exemplos do dataset que serão usados

# Criando o dataset para treinar o SOM
train_data,imgInputSOM=somSeg.prepare_dataset_multichanel(ROI_sample_conv,patch_size)
print("Samples")
print(train_data.shape[0])

# Criando o dataset para treinar o SOM
train_data,imgInputSOM=somSeg.prepare_dataset_multichanel2(ROI_sample_conv,patch_size,percentage=dataset_percentage)
print("Samples")
print(train_data.shape[0])

# Vizualizando os canais de entrada
numero_canais=imgInputSOM.shape[2]

plt.figure(figsize=(15, 5))
for i in range(0,numero_canais):
    plt.subplot(1, numero_canais, i+1)
    plt.imshow(imgInputSOM[:,:,i])


# Inicializando o SOM

#SOM = somSeg.create_map(som_size,1,3) # Usar quando o patch size for 0
SOM = somSeg.create_map(som_size,1,(2*patch_size+1)*(2*patch_size+1)*numero_canais)

# Realizando o treinamento sem plotar
SOM = somSeg.train_SOM(SOM, train_data, epochs=epochs)

# Realizando o treinamento plotando (fixo em 5 épocas)
#SOM = somSeg.train_SOM_plot(SOM, train_data, ROI_sample_conv, patch_size)
   
    
# Plota imagens do resultado da segmentaçao
result=somSeg.evaluate_SOM(SOM,ROI_sample_conv, patch_size)
#result=np.transpose(result, (1, 0))  # Transpor a imagem se necessário


somSeg.plot_result_multiclass_matplotlib(result, som_size, ROI_sample_conv)
 
somSeg.plot_result_multiclass_plotly(result, som_size) 

# Plotando a distância entre as unidades - pode ser usada para decidir o limiar
nd_x, nd_y=somSeg.find_ND(SOM['weights'])
hits=somSeg.find_Hits(SOM,ROI_sample_conv, patch_size)

fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(12, 3.5),
    subplot_kw=dict(xticks=[], yticks=[]))
ax[0].plot(nd_x, label='Neighbour Distance')
ax[1].plot(hits, label='Hits')