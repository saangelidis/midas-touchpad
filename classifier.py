import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def oneHotEncoding(value, labelList):
    array = np.zeros(len(labelList))
    array[labelList.index(value)] = 1
    return array

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

# initialize dataset
df = pd.read_csv("data/classification-task/fold-1/train.csv")

dataset = df[df['latex'].isin(mathOperators)]
dataset = dataset.drop('user_id',axis=1)

# this is our x
mathInput = np.zeros((len(dataset),32,32)) # 3D array

# this is our y from the NN
mathPrediction = ''

# this is the label y
mathActual = np.zeros((len(dataset),len(mathOperators)))

#####################
#                   #
# BUILD THE DATASET #
#                   #
#####################

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


#####################
#                   #
# INITIALIZE MODEL  #
#                   #
#####################


# flatten image input





# 

class FuckingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(32*32, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512,len(mathOperators))
        )
                
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = FuckingModel().to("cpu")

X = mathInput

logits = model(X)

pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
