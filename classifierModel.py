import torch.nn as nn

#
# INITIALIZE MODEL
# This classification model will be loaded into classifier.py and use the same architecture for classifying handwritten images of both math operators and greek characters.
#

class ClassifierModel(nn.Module):
    def __init__(self, numClasses):
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
            nn.Flatten(),                                   # flatten image to 1 dimension
            nn.Linear(64 * 8 * 8, 512),                     # fully connected layer with 512 neurons
            nn.ReLU(),
            nn.Linear(512, 512),                            # learns a non-linear combination of features
            nn.ReLU(),
            nn.Linear(512, numClasses)                      # output neurons correspond to symbol category
        )

    def forward(self, x):
        return self.classify(self.convolutions(x))
