import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import EMNIST

# 1. Baixar e carregar o dataset EMNIST (usaremos o split 'balanced' como exemplo)
# O download=True garante que ele baixe os dados na primeira vez que rodar
dataset = EMNIST(root='./data', split='balanced', train=True, download=True)

# Pegar a primeira imagem e seu rótulo para teste
img_original, label = dataset[0]

# 2. Definir a transformação para redimensionar
# O EMNIST nativo tem o tamanho 28x28 pixels
redimensionar = T.Resize((40, 40))

# 3. Aplicar o redimensionamento na imagem
img_redimensionada = redimensionar(img_original)

# 4. Visualizar as imagens lado a lado para conferência
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot da imagem original
axes[0].imshow(img_original, cmap='gray')
axes[0].set_title(f"Original\nTamanho: {img_original.size}") # Mostra (28, 28)
axes[0].axis('off') # Esconde os eixos para ficar mais limpo

# Plot da imagem redimensionada
axes[1].imshow(img_redimensionada, cmap='gray')
axes[1].set_title(f"Redimensionada\nTamanho: {img_redimensionada.size}") # Mostra (40, 40)
axes[1].axis('off')

plt.tight_layout()
plt.show()