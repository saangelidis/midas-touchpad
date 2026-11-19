import pickle as pkl
import os
import sys # to check OS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

from classifierModel import ClassifierModel # Our created ClassifierModel class


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

symbols_arr = []
symbols_arr_length = 0

# Array of LaTeX Greek characters supported in the dataset
greekChars = [
   r"\pi", r"\alpha", r"\beta", r"\sigma", r"\Sigma",
   r"\gamma", r"\Gamma", r"\delta", r"\Delta", r"\zeta",
   r"\eta", r"\theta", r"\Theta", r"\epsilon", r"\varepsilon",
   r"\iota", r"\kappa", r"\varkappa", r"\lambda", r"\Lambda",
   r"\mu", r"\nu", r"\xi", r"\Xi", r"\Pi",
   r"\rho", r"\varrho", r"\tau", r"\phi", r"\Phi",
   r"\varphi", r"\chi", r"\psi", r"\Psi", r"\omega",
   r"\Omega", r"\vartheta", r"\varpi",
]

# Array of LaTeX Math operators supported in the dataset
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


def buildDataset(csvPath):
    '''
    Helper function that builds a train or test dataset given a csvPath of train or test data.
    Called over all ten folds of trian/test data in model initialization and training.
    csvPath: string
    '''
    df = pd.read_csv(csvPath)
    dataset = df[df['latex'].isin(symbols_arr)] # Filters the dataset to include only the subset of characters inputted by the user (math or greek)

    inputs = np.zeros((len(dataset),32,32)) # 3D array
    labels = np.zeros((len(dataset),symbols_arr_length))
    for index, imgPath in enumerate(dataset['path']):
        # read image
        image = cv2.imread(f"data/{imgPath[6:]}", cv2.IMREAD_GRAYSCALE)
        inputs[index] = image
    for index, latex in enumerate(dataset['latex']):
        labels[index] = oneHotEncoding(latex, symbols_arr)

    X = torch.from_numpy(inputs).float().unsqueeze(1) / 255.0 # not rgb
    y = torch.from_numpy(labels.argmax(axis=1)).long()

    return TensorDataset(X, y)


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

    # numClasses is the length of the symbols array (math or greek) specified by user input
    numClasses = symbols_arr_length

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


def trainAndSaveModel(pickle_model_path):
    '''
    Trains a ClassifierModel and saves it in binary pickled form.
    Only reached if the pickled model file does not exist.
    pickle_model_path: str
    '''
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

        # ClassifierModel class imported from classifierModel.py, created on math-or-greek-only dataset specified by user input
        model = ClassifierModel(symbols_arr_length).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # hyperparam. optimizer

        for epoch in range(1,11): # Ten epochs per fold, our chosen optimal amount through trial and error of maximizing accuracy and reducing overfitting
            loss, train_acc = trainEpoch(model, trainLoader, optimizer)
            test_acc = evaluate(model, testLoader)
            print(f"Epoch {epoch:2d} - loss {loss:.3f} - train {train_acc:.2%} - test {test_acc:.2%}")
        
        # Compute and store fold metrics before iterating to the next one
        test_acc, precision, recall, f1 = evaluate(model, testLoader, detailed=True)
        foldAccuracies.append(test_acc)
        foldPrecisions.append(precision)
        foldRecalls.append(recall)
        foldF1s.append(f1)

        # Print fold results
        print(f"\nFold {fold} Final Results:")
        print(f"Accuracy:  {test_acc:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 Score:  {f1:.2%}")

    # Pickle the Fold 10 model to the already-selected file path
    model = model.cpu()
    model.eval()

    with open(pickle_model_path, 'wb') as mc:
        pkl.dump(model, mc)

    print("Model saved to", pickle_model_path)


def loadOrTrainModel(pickle_model_path):
    '''
    Loads a pickled model if it exists.
    If it does not exist, calls trainAndSaveModel.
    pickle_model_path: str
    '''
    if not os.path.exists(pickle_model_path):
        print("Previously saved/pickled model does not exist. Beginning ten-fold training and testing.")
        trainAndSaveModel(pickle_model_path)

    with open(pickle_model_path, "rb") as mc:
        model = pkl.load(mc)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def selectTestImage():
    """
    User selects an image as a row number from the filtered database.
    Required for milestone submission proof of concept; final submission will delete this and instead accept user input from a handdrawn character on the touchpad.
    """
    # Load Fold 10 test data to match the saved model
    df = pd.read_csv("data/classification-task/fold-10/test.csv")

    # Filter dataset based on selected category (math or greek)
    dataset = df[df["latex"].isin(symbols_arr)].reset_index(drop=True)

    print(f"\nAvailable test images: {len(dataset)}")

    while True:
        try:
            rowNumber = int(input(
                f"Select a row number from 0-{len(dataset) - 1} as an input image: "
            ))

            if 0 <= rowNumber < len(dataset):
                break

            print("Row number out of range. Please try again.")

        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Retrieve selected row
    selectedRow = dataset.loc[rowNumber]

    imgPath = selectedRow["path"]
    actualSymbol = selectedRow["latex"]
    return imgPath, actualSymbol


@torch.no_grad()
def predictSymbol(model, imgPath):
    '''
    Predicts the mathematical or Greek symbol from a given input image.
    For the milestone submission, this input is the path to the image in a user-selected row from the filtered dataset.
    In the final submission, user input will be the new handwritted character image from touchpad saved as a file, not a path to a dataset's existing image.

    model: Previously trained and unpickled ClassifierModel()
    imgPath: Path to the selected HASYv2 PNG image
    '''

    # Read the selected handwritten image
    image = cv2.imread(f"data/{imgPath[6:]}", cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(imgPath)

    # Convert image to PyTorch tensor and normalize
    X = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    X = X.to(device)

    # Predict using the loaded CNN
    model.eval()
    outputs = model(X)

    # Get the index of the predicted symbol
    predictedIndex = outputs.argmax(1).item()

    # Convert index back into the corresponding LaTeX symbol
    predictedSymbol = symbols_arr[predictedIndex]

    return predictedSymbol


def main():
    '''
    Main function of the Milestone Submission.
    Our method of obtaining user inputs will be changed for the final submission.
    In this milestone submission version, the user selects Math vs Greek classification models to build/load, and a dataset handwritten image, all within the terminal.
    In the final version of the project, the GUI will support predicting on users' handdrawn touchpad images and Math vs Greek model selection.
    '''
    global symbols_arr, symbols_arr_length

    while True:
        print("\nMidas Touchpad Neural Network -- Milestone Submission")
        print("Select the category of handwritten symbols to classify.")
        print("1) Math Operators")
        print("2) Greek Characters")
        print("3) Quit")

        model_input = input("\nChoose a model (1, 2, or 3): ").strip()

        while model_input not in ["1", "2", "3"]:
            model_input = input("Invalid selection. Enter 1, 2, or 3: ").strip()

        if model_input == "3":
            print("\nExiting Midas Touchpad. Goodbye.")
            break

        if model_input == "1":
            print("\nModel selected: Math Operators")
            symbols_arr = mathOperators
            pickle_model_path = "math-classifier.pkl"
        else:
            print("\nModel selected: Greek Characters")
            symbols_arr = greekChars
            pickle_model_path = "greek-classifier.pkl"

        symbols_arr_length = len(symbols_arr)

        # Load an existing pickle, or train and save the selected model
        model = loadOrTrainModel(pickle_model_path)

        # Let the user choose a row from the filtered dataset
        imgPath, actualSymbol = selectTestImage()

        # Predict the selected image
        predictedSymbol = predictSymbol(model, imgPath)

        print("\n=====CLASSIFICATION RESULTS=====")
        print(f"Actual Symbol:    {actualSymbol}")
        print(f"Predicted Symbol: {predictedSymbol}")
        print("Classification:", "Correct" if predictedSymbol == actualSymbol else "Incorrect")
        print("================================")


        # Display the selected handwritten PNG for the user
        image = cv2.imread(f"data/{imgPath[6:]}", cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(imgPath)

        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap="gray")
        plt.title(f"Actual: {actualSymbol}  |  Predicted: {predictedSymbol}")
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)

        print("\nSending predicted character to virtual keyboard...")

        # Supports virtual keyboard character printing on Linux machines only
        print(sys.platform)
        if sys.platform == "linux":
            from keyOut import keyOut

            # Pass the CNN prediction and selected category
            keyOut(predictedSymbol, model_input)
            print("Character sent to virtual keyboard.")
        else:
            print("Virtual keyboard output is only supported on Linux; skipping.")
        

main()
