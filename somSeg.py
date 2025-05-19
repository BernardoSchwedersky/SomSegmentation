# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:41:48 2025

@author: obern
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


from progress.bar import Bar

# Cria um novo SOM
def create_map(x,y,nInputs):
    map={'xSize':x, 'ySize': y}
    r=0.5774*3/2
    coord=np.zeros(shape=(x*y,2),dtype=float)
    for i in range(1,x+1):
        for j in range(1,y+1):
            if i % 2 == 0:
                cx=j
            else:
                cx=j+0.5

            cy=i*r
            coord[(i-1)*y+j-1,0]=cx
            coord[(i-1)*y+j-1,1]=cy
    map['centers']=coord
    map['weights']=rand.randint(0, 255, (x, y, nInputs)).astype(float)
    return map

# Fixa a seed aleatória
rand = np.random.RandomState(0)

# Calcula a BMU do SOM e retorna a posição (x, y)
def find_BMU(SOM_weights,sample):
    distSq = (np.square(SOM_weights - sample)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

# Calcula a BMU do SOM e retorna a posição (x, y)
def find_BMU_Dist(SOM_weights,sample):
    distSq = (np.square(SOM_weights - sample)).sum(axis=2)
    return np.sqrt(distSq)

# Calcula a distância para os vizinhos
def find_ND(SOM_weights):
    dist_x=np.zeros((SOM_weights.shape[0]-1,SOM_weights.shape[1]))
    dist_y=np.zeros((SOM_weights.shape[0],SOM_weights.shape[1]-1))
    for i in range(0,SOM_weights.shape[0]-1):
        for j in range(0,SOM_weights.shape[1]):
            dist_x[i,j]=(np.square(SOM_weights[i,j,:] - SOM_weights[i+1,j,:])).sum(axis=0)

    dist_y=np.zeros((SOM_weights.shape[0],SOM_weights.shape[1]-1))
    for i in range(0,SOM_weights.shape[0]):
        for j in range(0,SOM_weights.shape[1]-1):
            dist_y[i,j]=(np.square(SOM_weights[i,j,:] - SOM_weights[i,j+1,:])).sum(axis=0)

    return np.sqrt(dist_x), np.sqrt(dist_y)

# Calcula a posição de um exemplo em relação ao SOM
def find_Sample_Position(SOM_weights,sample):
    bmu_x, bmu_y=find_BMU(SOM_weights,sample)
    dist=find_BMU_Dist(SOM_weights,sample)
    d_x, d_y=find_ND(SOM_weights)
    p_x=np.cumsum(d_x)
    
    if (bmu_x>0):    
        sp_x=np.square(dist[bmu_x-1])-np.square(dist[bmu_x])-np.square(d_x[bmu_x-1])
        sp_x=sp_x/(2*d_x[bmu_x-1])
        sp_x+p_x[bmu_x-1]
    else:
        sp_x=np.square(dist[1])-np.square(dist[0])-np.square(d_x[0])
        sp_x=sp_x/(2*d_x[0])
    
    sp_y=np.sqrt(np.square(dist[0])-np.square(sp_x))
    
    return sp_x, sp_y

# Retorna a quantidade de pixels assoiada a cada unidade do SOM treinado para a imagem fornecida
def find_Hits(SOM,im,patch_size):
    ROI_sample=np.asarray(im)
    test_data,img=prepare_dataset_multichanel(ROI_sample,patch_size)
    som_quantized=np.arange(test_data.shape[0])

    i=0
    for sample in test_data:
        x, y=find_BMU(SOM['weights'],sample)
        som_quantized[i]=x
        i=i+1

    hits=np.zeros(SOM['xSize']*SOM['ySize'])
    for un in range(0,som_quantized.shape[0]-1):
        hits[som_quantized[un]]+=1
    
    return hits

# Atualiza os pesos do SOM para um exemplo de treinamento
def update_weigths(SOM, sample, learn_rate, radius_sq, step=3):
    SOM_weights=SOM['weights']
    g, h = find_BMU(SOM_weights, sample)
    #if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM_weights[g,h,:] += learn_rate * (sample - SOM_weights[g,h,:])
        SOM['weights']=SOM_weights
        return SOM
    # Change all cells in a small neighborhood of BMU
    for i in range(max(0, g-step), min(SOM_weights.shape[0], g+step)):
        for j in range(max(0, h-step), min(SOM_weights.shape[1], h+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM_weights[i,j,:] += learn_rate * dist_func * (sample - SOM_weights[i,j,:])
    SOM['weights']=SOM_weights
    return SOM

# Treina o SOM já inicializado usando o dataset
def train_SOM(SOM, data, learn_rate=0.05, radius_sq=2,
              lr_decay=0.1, radius_decay=0.1, epochs=5):
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    
    # Configura a barra de progresso única
    bar = Bar('Processing', max=epochs)
    
    for epoch in range(epochs):
        rand.shuffle(data)
        for sample in data:
            SOM = update_weigths(SOM, sample, learn_rate, radius_sq)
            # Atualiza a barra de progresso
             
        bar.next() 
        # Atualiza learning rate e raio
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        
    
    bar.finish()
    return SOM

# Treina o SOM plotando o resultado de cada época, com um número fixo de 5 épocas
def train_SOM_plot(SOM, data, img, patch_size, learn_rate = .1, radius_sq = 1,
             lr_decay = .1, radius_decay = .1, epochs = 5):
    fig, ax = plt.subplots(
        nrows=1, ncols=5, figsize=(15, 3.5),
        subplot_kw=dict(xticks=[], yticks=[]))
    
    total_epochs = 0
    for epochs, i in zip([1, 1, 1, 1, 1], range(0,5)):
        total_epochs += epochs
        SOM = train_SOM(SOM, data, epochs=epochs, radius_sq=1)
        result=evaluate_SOM(SOM,img, patch_size)
        ax[i].imshow(result, cmap='Spectral', vmin=0, vmax=SOM['xSize'])
        ax[i].title.set_text('Epochs = ' + str(total_epochs))
         
    return SOM

# Avalia cada pixel da imagem para o SOM treinado
def evaluate_SOM(SOM,im,patch_size):
    ROI_sample=np.asarray(im)
    sample_shape=ROI_sample.shape
    test_data,img=prepare_dataset_multichanel(ROI_sample,patch_size)
    som_quantized=np.arange(test_data.shape[0])

    i=0
    for sample in test_data:
        x, y=find_BMU(SOM['weights'],sample)
        som_quantized[i]=x
        i=i+1

    result=som_quantized.reshape(sample_shape[0], sample_shape[1])
    return result

# Transforma uma imagem em um dataset para treinar o SOM considerando uma imagem com mais de um canal
def prepare_dataset_multichanel(img, border_size, percentage=1) :
    img_shape=img.shape
    if border_size==0:
        dataset=img.reshape(img_shape[0]*img_shape[1], 3)
    else:
        img2 = img[:,:,:]
        leftB = img2[:,0:1,:]
        expanded_leftB = np.tile(leftB, (1, border_size,1))
        rigthB = img[:,img_shape[1]-1:img_shape[1],:]
        expanded_rigthB = np.tile(rigthB, (1, border_size,1))
        img3=np.concatenate((expanded_leftB, img2, expanded_rigthB), axis=1)
        upB=img3[0:1,:,:]
        expanded_upB = np.tile(upB, (border_size, 1, 1))
        downB=img3[img_shape[0]-1:img_shape[0],:,:]
        expanded_downB = np.tile(downB, (border_size, 1, 1))
        img4=np.concatenate((expanded_upB, img3, expanded_downB), axis=0)

        
        numero_canais=img4.shape[2]

        dataset=np.zeros((img_shape[0]*img_shape[1], numero_canais*(2*border_size+1)*(2*border_size+1)))

        for i in range(0,numero_canais):
            channel=img4[:,:,i]
            counter=0
            datasetAux=np.zeros((img_shape[0]*img_shape[1], (2*border_size+1)*(2*border_size+1)))
            for j in range(border_size,border_size+img_shape[0]):
                for k in range(border_size,border_size+img_shape[1]):
                    patch=channel[j-border_size:j+border_size+1,k-border_size:k+border_size+1]
                    patch_line=np.reshape(patch,(1,-1))
                    datasetAux[counter,:]=patch_line
                    counter=counter+1
            dataset[:, i*(2*border_size+1)*(2*border_size+1):(i+1)*(2*border_size+1)*(2*border_size+1)]=datasetAux[:,:]
    return dataset, img4

def prepare_dataset_multichanel2(img, border_size, percentage=1):
    img_shape = img.shape
    if border_size == 0:
        # Caso sem borda: seleciona subconjunto aleatório de pixels diretamente
        total_pixels = img_shape[0] * img_shape[1]
        num_selected = int(total_pixels * percentage)
        indices = np.random.choice(total_pixels, num_selected, replace=False)
        dataset = img.reshape(-1, img_shape[2])[indices, :]
        return dataset, img  # Retorna a imagem original (sem padding)
    
    else:
        # Adiciona bordas à imagem (código existente)
        img2 = img[:, :, :]
        leftB = img2[:, 0:1, :]
        expanded_leftB = np.tile(leftB, (1, border_size, 1))
        rightB = img2[:, img_shape[1]-1:img_shape[1], :]
        expanded_rightB = np.tile(rightB, (1, border_size, 1))
        img3 = np.concatenate((expanded_leftB, img2, expanded_rightB), axis=1)
        upB = img3[0:1, :, :]
        expanded_upB = np.tile(upB, (border_size, 1, 1))
        downB = img3[img_shape[0]-1:img_shape[0], :, :]
        expanded_downB = np.tile(downB, (border_size, 1, 1))
        img4 = np.concatenate((expanded_upB, img3, expanded_downB), axis=0)

        # Configurações para extração de patches
        numero_canais = img4.shape[2]
        total_pixels = img_shape[0] * img_shape[1]
        num_selected = int(total_pixels * percentage)
        indices = np.random.choice(total_pixels, num_selected, replace=False)  # Índices aleatórios
        
        # Converte índices para coordenadas (j, k) na imagem com bordas
        selected_j = border_size + (indices // img_shape[1])
        selected_k = border_size + (indices % img_shape[1])

        # Inicializa o dataset com as dimensões corretas
        patch_size = (2 * border_size + 1) ** 2
        dataset = np.zeros((num_selected, numero_canais * patch_size))

        # Processa cada canal
        for i in range(numero_canais):
            channel = img4[:, :, i]
            dataset_aux = np.zeros((num_selected, patch_size))
            
            for counter, (j, k) in enumerate(zip(selected_j, selected_k)):
                # Extrai o patch do canal atual
                patch = channel[j - border_size:j + border_size + 1, k - border_size:k + border_size + 1]
                patch_line = patch.reshape(1, -1)
                dataset_aux[counter, :] = patch_line
            
            # Adiciona ao dataset final
            start_col = i * patch_size
            end_col = (i + 1) * patch_size
            dataset[:, start_col:end_col] = dataset_aux

        return dataset, img4

# Funções básicas para manipular imagens
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

def plot_result_multiclass_plotly(result, som_size):
    # Plotando usando plotly - gráfico interativo
    fig = px.imshow(result, color_continuous_scale='spectral')
    plot(fig)
    
    # Plotando as possíveis segmentações, dependendo do threshold escolhido
    fig = go.Figure()
    
    output_up=np.zeros(np.array([som_size-1, result.shape[0], result.shape[1]]))
    
    for i in range(0,som_size-1):
        output_up[i,:,:]=result<=i
    
    fig = px.imshow(
        output_up,
        animation_frame=0,
        color_continuous_scale='inferno'
    )
    plot(fig)

def plot_result_multiclass_matplotlib(result,som_size,img):
    # Plotando com matplotlib
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 3.5),
        subplot_kw=dict(xticks=[], yticks=[]))

    #ROI_sample=np.transpose(ROI_sample, (1, 0, 2)) # Transpor se necessário
    ax[0].imshow(img)
    ax[1].imshow(result, cmap='Spectral', vmin=0, vmax=som_size)
    
    # Plotando o resultado para os vários possíveis limiares para definir em duas classes
    fig, ax = plt.subplots(
        nrows=1, ncols=som_size, figsize=(12, 3.5),
        subplot_kw=dict(xticks=[], yticks=[]))
    
    for i in range(0,som_size):
      output_up=result<=i
      ax[i].imshow(output_up, cmap='Spectral', vmin=0, vmax=1)
  

def plot_som_distance(SOM, im, patch_size):
    d_x, d_y=find_ND(SOM['weights'])
    p_x=np.cumsum(d_x)
    p_y=np.zeros(p_x.shape[0])
    
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(12, 3.5),
        subplot_kw=dict(xticks=[], yticks=[]))
    ax.scatter(0, 0, color = 'black')
    ax.scatter(p_x, p_y, color = 'black')
    
    ROI_sample=np.asarray(im)
    sample_shape=ROI_sample.shape
    test_data,img=prepare_dataset_multichanel(ROI_sample,patch_size)
    pos_x=np.arange(test_data.shape[0])
    pos_y=np.arange(test_data.shape[0])

    i=0
    for sample in test_data:
        x, y=find_Sample_Position(SOM['weights'],sample)
        pos_x[i]=x[0]
        pos_y[i]=y[0]
        
        i=i+1
    
    ax.scatter(pos_x, pos_y, color = 'red')
    