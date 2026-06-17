import torch
import torch.nn as nn
import torch.optim as optm
from torch.utils.data import DataLoader as DtL
from torchvision import datasets as DtSts
from torchvision import transforms as Trns

from model import Chars, trainData, trainLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
LEARNING_RATE = 0.001

nClasses = len(trainData.classes)
model = Chars(nClasses)

criterion = nn.CrossEntropyLoss()
optimizer = optm.Adam(model.parameters(), LEARNING_RATE)


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
