===================================================================
                       EMNIST Classifier Core                      
===================================================================

Um classificador de caracteres (letras) baseado em uma Rede Neural 
Convolucional (CNN) utilizando PyTorch e o dataset EMNIST.

-------------------------------------------------------------------
1. ESTRUTURA DO PROJETO
-------------------------------------------------------------------

* model.py             - Define a arquitetura da CNN e o pipeline de 
                         pré-processamento dos dados (Transforms).
* train.py             - Executa o loop de treinamento por 15 épocas
                         e salva os pesos do modelo.
* testeRedimensionar.py - Script auxiliar para conferência visual do
                         redimensionamento das matrizes via Matplotlib.

-------------------------------------------------------------------
2. REQUISITOS E DEPENDÊNCIAS
-------------------------------------------------------------------

Certifique-se de ter o Python instalado com as seguintes bibliotecas:

* torch (PyTorch)
* torchvision
* matplotlib

Para instalar tudo via terminal:
$ pip install torch torchvision matplotlib

-------------------------------------------------------------------
3. COMO EXECUTAR
-------------------------------------------------------------------

1. Para treinar o modelo e salvar os pesos:
   $ python train.py

2. O script fará o download do dataset EMNIST na primeira execução,
   processará as imagens em lotes (batches) e, ao fim das 15 épocas,
   gerará o arquivo binário:
   -> 'character_recognizer.pth'

3. Para testar e visualizar o redimensionamento de imagem (28x28 para 40x40):
   $ python testeRedimensionar.py

-------------------------------------------------------------------
4. ARQUITETURA DA REDE
-------------------------------------------------------------------
* 2 Blocos Convolucionais (com Batch Normalization e ativação ReLU).
* Camada de Achatamento (Flatten).
* Camadas Densas (Fully Connected) com Dropout de 10%.
* Função de Perda: CrossEntropyLoss.
* Otimizador: Adam (Learning Rate: 0.001).

===================================================================
