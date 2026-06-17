import torch
import torch.nn as nn
import torch.optim as optm
from torch.utils.data import DataLoader as DtL
from torchvision import datasets as DtSts
from torchvision import transforms as Trns

#hiperparametros e config

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #pode is para train
print(f"iniciando treinamento")
#dropout e arch

#

# tratamento dos dados

trns = Trns.Compose([
  Trns.Resize((40,40)),
  Trns.ToTensor(),
  Trns.Normalize((0.1736,), (0.3317,))#redundante
])

trainData = DtSts.EMNIST(
    root='./data',
    split='letters',
    train=True, 
    download=True, 
    transform=Trns
)
testData = DtSts.EMNIST(
    root='./data', 
    split='letters', 
    train=False, 
    download=True, 
    transform=Trns
)

trainLoader = DtL(trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DtL(testData, batch_size=BATCH_SIZE, shuffle=False)


#rede

class Chars(nn.Module):
    def __init__(self, num_classes):
        super(Chars, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, num_classes)

    def foward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    


#treino:

"""

def trainModel():
    model.train()
    for EPOCH in range(EPOCHS):
        runLoss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(trainLoader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 500 == 0:
                print(f"Época [{EPOCH+1}/{EPOCHS}] | Progresso [{batch_idx}/{len(trainLoader)}] "
                      f"| Loss: {loss.item():.4f}")
                
        epoch_acc = 100. * correct / total
        print(f"--> Fim da Época {EPOCH+1} | Loss Média: {running_loss/len(trainLoader):.4f} | Precisão: {epoch_acc:.2f}%")


if __name__ == "__main__":
    
    trainModel()
    
    
    torch.save(model.state_dict(), "character_recognizer.pth")
    print("Modelo treinado e salvo com sucesso como 'character_recognizer.pth'!")

        
"""

