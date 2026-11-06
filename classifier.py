import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def oneHotEncoding(value, labelList):
    array = np.zeros(len(labelList))
    array[labelList.index(value)] = 1
    return array

# TODO: create separate .py files for each category of symbols (greek and math)
## greek chars array
#greekChars = [
#    r"\pi", r"\alpha", r"\beta", r"\sigma", r"\Sigma",
#    r"\gamma", r"\Gamma", r"\delta", r"\Delta", r"\zeta",
#    r"\eta", r"\theta", r"\Theta", r"\epsilon", r"\varepsilon",
#    r"\iota", r"\kappa", r"\varkappa", r"\lambda", r"\Lambda",
#    r"\mu", r"\nu", r"\xi", r"\Xi", r"\Pi",
#    r"\rho", r"\varrho", r"\tau", r"\phi", r"\Phi",
#    r"\varphi", r"\chi", r"\psi", r"\Psi", r"\omega",
#    r"\Omega", r"\vartheta", r"\varpi",
#]

# math chars array
mathOperators = [
    r"\sum", r"\partial", r"\int", r"\cdot", "-", "+",
    r"\amalg", r"\cup", r"\oplus", r"\times", r"\ast",
    r"\triangleleft", r"\otimes", r"\triangleright", r"\diamond", r"\pm",
    r"\div", r"\bullet", r"\setminus", r"\uplus", r"\cap",
    r"\mp", r"\sqcap", r"\vee", r"\odot", r"\sqcup",
    r"\wedge", r"\circ", r"\ominus", r"\star", r"\wr",
    r"\barwedge", r"\circledcirc", r"\boxdot", r"\ltimes", r"\boxplus",
    r"\boxtimes", r"\rtimes", r"\circledast", r"\lhd", r"\prod",
    r"\coprod", r"\oint", r"\parr", r"\with", r"\fint",
    r"\varoiint", r"\oiint", r"\nabla", r"\neg", r"\sqrt{}",
]

#
#. INITIALIZE DATASET
# 

# TODO: once model is complete, iterate over all folds into one dataset, not just fold 1 as is currently
df = pd.read_csv("data/classification-task/fold-1/train.csv")

dataset = df[df['latex'].isin(mathOperators)]
dataset = dataset.drop('user_id',axis=1)

# this is our x
mathInput = np.zeros((len(dataset),32,32)) # 3D array

# this is our y from the NN
mathPrediction = ''

# this is the label y
mathActual = np.zeros((len(dataset),len(mathOperators)))


#
#  BUILD THE TRAIN DATASET
#

for index, imgPath in enumerate(dataset['path']):
    
    # read image
    image = cv2.imread(f"data/{imgPath[6:]}", cv2.IMREAD_GRAYSCALE)
    
    # new image row
    newRow = image.reshape((1,32,32))
    
    # add to 3D input array
    mathInput[index] = newRow


for index, latex in enumerate(dataset['latex']):
    
    # one hot encode
    oheOutput = oneHotEncoding(latex,mathOperators)

    # add to label array
    mathActual[index] = oheOutput 


#
#  INITIALIZE MODEL 
#

class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # convolution with 3x3 kernel
            nn.BatchNorm2d(32),                             # number of feature maps
            nn.ReLU(),
            nn.MaxPool2d(2),                                # reduce dimensions by half
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # convolution with 3x3 kernel
            nn.BatchNorm2d(64),                             # new number of feature maps
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
       
        self.classify = nn.Sequential(
                nn.Flatten(),                               # flatten image to 1 dimension
                nn.Linear(64 * 8 * 8, 512),                 # fully connected layer with 512 neurons
                nn.ReLU(),
                nn.Linear(512, 512),                        # learns a non-linear combination of features
                nn.ReLU(),
                nn.Linear(512,len(mathOperators))
        )
                
    def forward(self, x):
        return self.classify(self.convolutions(x))

# Initialize the model
mathModel = ClassifierModel().to(device)

X = torch.from_numpy(mathInput).float().unsqueeze(1) / 255.0 # not rgb
y = torch.from_numpy(mathActual.argmax(axis=1)).long()

print("Input shape:", X.shape)

# ~~~ debugging print lines, may not be needed anymore ~~~~
# fit = mathModel(X)
# pred_prob = nn.Softmax(dim=1)(fit)
# y_pred = pred_prob.argmax(1)
# # print("Fit:", fit[:10])
# # print("Probabilities:", pred_prob[:10])
# # print("Predicted labels:", y_pred[:10])
# # print("Actual labels:", torch.tensor(mathActual[:10], dtype=torch.float32).argmax(1))

#
#  TRAIN THE MODEL 
#

trainDataset = TensorDataset(X, y) # pairing input images X with their symbol_id labels y
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True) # split into batches for training

error = nn.CrossEntropyLoss()                               # loss function
optimizer = torch.optim.Adam(mathModel.parameters(), lr=1e-3)   # hyperparam. optimizer

def trainEpoch(modele):
    modele.train()
    totalLoss, correct, n = 0, 0, 0

    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # used in training to reset gradients before backpropagation
        outputs = modele(images)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return totalLoss / n, correct / n

#
#  BUILD THE TEST DATASET
#

test_df = pd.read_csv("data/classification-task/fold-1/test.csv")

# building the test dataset in the same process as the training dataset
testDataset = test_df[test_df['latex'].isin(mathOperators)]
mathTestInput = np.zeros((len(testDataset),32,32)) # 3D array
mathTestActual = np.zeros((len(testDataset),len(mathOperators)))

for index, imgPath in enumerate(testDataset['path']):
    # read image
    image = cv2.imread(f"data/{imgPath[6:]}", cv2.IMREAD_GRAYSCALE)
    
    # new image row
    newRow = image.reshape((1,32,32))
    
    # add to 3D input array
    mathTestInput[index] = newRow

for index, latex in enumerate(testDataset['latex']):
    # one hot encode
    oheOutput = oneHotEncoding(latex,mathOperators)

    # add to label array
    mathTestActual[index] = oheOutput 

X_test = torch.from_numpy(mathTestInput).float().unsqueeze(1) / 255.0 # not rgb
y_test = torch.from_numpy(mathTestActual.argmax(axis=1)).long()

#                   
#  EVALUATE MODEL ON TEST DATA
#    
               
testDataset = TensorDataset(X_test, y_test) # pairing input images X with their symbol_id labels y
testLoader = DataLoader(testDataset, batch_size=64, shuffle=False) # split into batches for testing

@torch.no_grad() # used in testing because we want to disable gradient tracking during evaluation
def evaluate(modele):
    modele.eval()
    correct, n = 0, 0
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = modele(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return correct / n


# training & testing
for epoch in range(1,16):
    loss, train_acc = trainEpoch(mathModel)
    test_acc = evaluate(mathModel)
    print(f"Epoch {epoch:2d} - loss {loss:.3f} - train {train_acc:.2%} - test {test_acc:.2%}")

