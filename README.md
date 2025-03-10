# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/30b7017a-6e4c-4dcf-8547-a1e594d821a5)


## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: G.TEJASWINI
### Register Number: 212222230157

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):  # Corrected: __init__ instead of _init
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
        

```
```python
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

![image](https://github.com/user-attachments/assets/28514589-4307-4616-bc5a-3107da94ace1)


## OUTPUT

### Confusion Matrix

![image](https://github.com/user-attachments/assets/69effeb6-d219-412c-b74c-cca0d0056595)

### Classification Report

![image](https://github.com/user-attachments/assets/b6e64068-c069-4589-96cf-b4df55579eba)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/252a9090-d4ca-476e-8b41-f1f2197441b1)


## RESULT
Thus the neural network classification model for the given dataset is developed successfully.
