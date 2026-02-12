## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.
### STEP 3: 
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.
### STEP 5: 
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.
### STEP 6: 
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.

## PROGRAM
### Name:muthurevulasahithi
### Register Number:212224040208

```

   class CNNClassifier(nn.Module):
    def __init__(self):
       super(CNNClassifier, self).__init__()
       self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
       self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
       self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
       self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
       self.fc1=nn.Linear(128*3*3,128)
       self.fc2=nn.Linear(128,64)
       self.fc3=nn.Linear(64,10)
    def forward(self,x):
       x=self.pool(torch.relu(self.conv1(x)))
       x=self.pool(torch.relu(self.conv2(x)))
       x=self.pool(torch.relu(self.conv3(x)))
       x=x.view(x.size(0),-1)
       x=torch.relu(self.fc1(x))
       x=torch.relu(self.fc2(x))
       x=self.fc3(x)
       return x


```

### OUTPUT

## Training Loss per Epoch
<img width="323" height="286" alt="image" src="https://github.com/user-attachments/assets/644da72f-b75b-4e56-bb92-d904ef0c291f" />


## Confusion Matrix

<img width="1060" height="753" alt="image" src="https://github.com/user-attachments/assets/687d715f-2411-4616-a550-b9a9bb8338e0" />


## Classification Report
<img width="595" height="437" alt="image" src="https://github.com/user-attachments/assets/008e1fbc-6a95-4243-9e45-889dbd7afc3c" />

### New Sample Data Prediction
<img width="571" height="638" alt="image" src="https://github.com/user-attachments/assets/d62c662f-1db2-4867-bb06-512e2123efff" />

## RESULT
The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
