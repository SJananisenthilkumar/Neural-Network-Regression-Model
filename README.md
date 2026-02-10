# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: JANANI S
### Register Number: 212223230086
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
        self.fc1=nn.Linear(1,4)
        self.fc2=nn.Linear(4,6)
        self.fc3=nn.Linear(6,8)
        self.fc4=nn.Linear(8,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.relu(self.fc3(x))
    x=self.fc4(x)
    return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()
   # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

<img width="350" height="463" alt="image" src="https://github.com/user-attachments/assets/b8a67076-449a-4b26-bed9-8599059a9ae0" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="666" height="462" alt="image" src="https://github.com/user-attachments/assets/c970857f-ba08-410b-bcce-d7fc040f1641" />


### New Sample Data Prediction

<img width="849" height="121" alt="image" src="https://github.com/user-attachments/assets/a4224d04-381b-4297-aa74-7240f9929289" />


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
