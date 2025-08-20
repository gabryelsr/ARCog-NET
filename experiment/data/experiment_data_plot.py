import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import random
import numpy as np

# Caminho para o arquivo de pontos
caminho_arquivo = "/ARCog-NET/experiment/data/nuvem_pontos4.csv"  # ou .csv
# Carregar os dados (sem cabeçalho)
df = pd.read_csv(caminho_arquivo, header=None, names=["x", "y", "z"])
amostra = df #df.sample(frac=1, random_state=42)
# Lista de cores possíveis
cores_possiveis = ['violet', 'springgreen'] #['red', 'yellow', 'blue', 'green']
# Gera uma cor aleatória para cada ponto
cores = [random.choice(cores_possiveis) for _ in range(len(amostra))]

# Plotar as trajetórias
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#label_scat = ['Cluster 1', 'Cluster 2']
ax.scatter(amostra["x"], amostra["y"], amostra["z"]*0.55, alpha=0.5, s=5, c=cores)

# Lista de arquivos de trajetória (ajuste os caminhos conforme necessário)
arquivos = [
    "/ARCog-NET/experiment/data/trajetoria_uav1.json",
    "/ARCog-NET/experiment/data/trajetoria_uav2.json",
    "/ARCog-NET/experiment/data/trajetoria_uav3.json",
    "/ARCog-NET/experiment/data/trajetoria_uav4.json",
    "/ARCog-NET/experiment/data/trajetoria_uav5.json",
    "/ARCog-NET/experiment/data/trajetoria_uav6.json",
]

# Deslocamentos para UAVs 2 a 6
deslocamentos_x = [2.0, 2.5, 2.6, 3.1, 3.6, 4.1]
deslocamento_y = 1.5  # Todos os UAVs 2–6 são deslocados no Y
deslocamentos_z = [-0.5, -0.2, -0.1, -0.1, -0.5, -0.7]

# Cores e rótulos
cores = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
labels = [f'UAV{i+1}' for i in range(6)]

# Lista para armazenar trajetórias
trajetorias = []

# Ler, deslocar e armazenar
for i, caminho in enumerate(arquivos):
    with open(caminho, 'r') as f:
        dados = json.load(f)
        df = pd.DataFrame(dados, columns=['x', 'y', 'z'])

        # Aplicar deslocamento (exceto UAV1)
        df['x'] += deslocamentos_x[i]
        df['y'] += deslocamento_y
        df['z'] += deslocamentos_z[i]

        trajetorias.append(df)

xt = df['x']
yt = df['y']
zt = df['z']


for df, cor, label in zip(trajetorias, cores, labels):
    ax.plot(df['x'], df['y'], df['z'], color=cor, label=label)

#ax.view_init(elev=90, azim=270)
ax.view_init(elev=90, azim=225)
# Estética do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D UAV trajectory in Point Cloud')
patches_legenda = [
    mpatches.Patch(color='violet', label='Cluster 1'),
    mpatches.Patch(color='springgreen', label='Cluster 2'),
    mpatches.Patch(color='red', label='UAV 1'),
    mpatches.Patch(color='blue', label='UAV 2'),
    mpatches.Patch(color='green', label='UAV 3'),
    mpatches.Patch(color='yellow', label='UAV 4'),
    mpatches.Patch(color='purple', label='UAV 5'),
    mpatches.Patch(color='orange', label='UAV 6')
]
ax.legend(handles=patches_legenda, loc="lower left")
plt.tight_layout()
plt.show()
