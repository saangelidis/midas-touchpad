import numpy as np
import pandas as pd
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

  
# TODO: 
# decide on: create separate .py files for each category of symbols (greek and math)
# or: have one file that loads two separate, previously trained, pickled models (one for greek, one for math) and then have a function that takes in an image and returns the predicted symbol

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

def oneHotEncoding(value, labelList):
    array = np.zeros(len(labelList))
    array[labelList.index(value)] = 1
    return array


# TODO: add parameter to accept using either mathOperators or greekChars depending on the model being used
def buildDataset(csvPath):
    '''
    Helper function that builds a train or test dataset given a csvPath of train or test data.
    Called over all ten folds of trian/test data in model initialization and training.
    csvPath: string
    '''
    df = pd.read_csv(csvPath)
    dataset = df[df['latex'].isin(mathOperators)]
    inputs = np.zeros((len(dataset),32,32)) # 3D array
    labels = np.zeros((len(dataset),len(mathOperators)))
    for index, imgPath in enumerate(dataset['path']):
        # read image
        image = cv2.imread(f"data/{imgPath[6:]}", cv2.IMREAD_GRAYSCALE)
        inputs[index] = image
    for index, latex in enumerate(dataset['latex']):
        labels[index] = oneHotEncoding(latex, mathOperators)

    X = torch.from_numpy(inputs).float().unsqueeze(1) / 255.0 # not rgb
    y = torch.from_numpy(labels.argmax(axis=1)).long()

    return TensorDataset(X, y)


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


error = nn.CrossEntropyLoss() # loss function

def trainEpoch(model, trainLoader, optimizer):
    '''
    Trains a given classifier model.
    model: ClassifierModel()
    trainLoader: DataLoader, batch size 64
    optimizer: torch.optim.Adam
    '''
    model.train()
    totalLoss, correct, n = 0, 0, 0

    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # used in training to reset gradients before backpropagation
        outputs = model(images)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return totalLoss / n, correct / n



@torch.no_grad() # used in testing because we want to disable gradient tracking during evaluation
def evaluate(model, testLoader, detailed=False):
    '''
    Evaluates a prior trained classifier model on accuracy, precision, recall, and F1 score.
    model: ClassifierModel()
    testLoader: DataLoader, batch size 64
    '''
    model.eval()
    correct, n = 0, 0

    # TODO: handle updating from math to greek: add parameter to adjust
    numClasses = len(mathOperators)

    precision = MulticlassPrecision(num_classes=numClasses, average='macro').to(device)
    recall = MulticlassRecall(num_classes=numClasses, average='macro').to(device)
    f1 = MulticlassF1Score(num_classes=numClasses, average='macro').to(device)

    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(1)

        correct += (predictions == labels).sum().item()
        n += labels.size(0)

        # Update detailed evaluation metrics outside of just accuracy
        if detailed:
            precision.update(predictions, labels)
            recall.update(predictions, labels)
            f1.update(predictions, labels)

    accuracy = correct / n
    if detailed: 
        return (accuracy, precision.compute().item(), recall.compute().item(), f1.compute().item())
    else:
        return accuracy


# Training & testing metrics 
foldAccuracies = []
foldPrecisions = []
foldRecalls = []
foldF1s = []

for fold in range(1,11): # Ten folds of train/test data in the HASYv2 dataset
    print(f"\nFold {fold}:")

    # Call helper function pairing input images X with their symbol_id labels y
    trainDataset = buildDataset(f"data/classification-task/fold-{fold}/train.csv")
    testDataset = buildDataset(f"data/classification-task/fold-{fold}/test.csv")

    # split into batches for training/testing
    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True) 
    testLoader = DataLoader(testDataset, batch_size=64, shuffle=False) 

    # TODO: make it a general model, not mathModel, corresponding to user's model selection, math vs greek
    mathModel = ClassifierModel().to(device)

    optimizer = torch.optim.Adam(mathModel.parameters(), lr=1e-3)   # hyperparam. optimizer

    for epoch in range(1,11): # Ten epochs per fold, our chosen optimal amount through trial and error of maximizing accuracy and reducing overfitting
        loss, train_acc = trainEpoch(mathModel, trainLoader, optimizer)
        test_acc = evaluate(mathModel, testLoader)
        print(f"Epoch {epoch:2d} - loss {loss:.3f} - train {train_acc:.2%} - test {test_acc:.2%}")
    
    # Compute and store fold metrics before iterating to the next one
    test_acc, precision, recall, f1 = evaluate(mathModel, testLoader, detailed=True)
    foldAccuracies.append(test_acc)
    foldPrecisions.append(precision)
    foldRecalls.append(recall)
    foldF1s.append(f1)

    # 
    # PRINT FOLD RESULTS
    #

    print(f"\nFold {fold} Final Results:")
    print(f"Accuracy:  {test_acc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")


# save model
# TODO: currently this is pickling only fold 10, we need to build a dataset over all ten folds at once and train a model on that and pickle it for the expected result
# TODO: adjust for math vs. greek
with open("math-classifier.pkl",'wb') as mc:
    pkl.dump(mathModel, mc)