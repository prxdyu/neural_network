## **Neural Network Implementation**
<br>
This repository contains a simple implementation of a neural network with one hidden layer, designed for classification tasks. The neural network is written in Python and utilizes the NumPy library for numerical operations.

### **Overview**
The implemented neural network, named NN, is a feedforward neural network capable of handling classification tasks. The key features of this implementation include:

### **Architecture**:
The neural network architecture is specified by the number of input features (n_features), the number of output classes (n_classes), and the number of neurons in the hidden layer (n_hidden).

### **Activation Functions**: 
ReLU (Rectified Linear Unit) activation is applied to the hidden layer, and softmax activation is applied to the output layer.

### **Loss Function**: 
The cross-entropy loss function is used for training the neural network.

### **Regularization**: 
<br>
L2 regularization is incorporated to prevent overfitting during training.

### **Usage**
Initialization
python
Copy code
-         Example Initialization
          model = NN(n_features, n_classes, n_hidden)
Training

          model.fit(X_train, y_train, reg=0.01, max_iters=10000, eta=0.001)
Prediction

      
       predictions = model.predict(X_test)
   
#### **Parameters**
- **n_features**: Number of input features.
- **n_classes**: Number of output classes.
- **n_hidden**: Number of neurons in the hidden layer.
- **reg**: Regularization strength.
- **max_iters**: Maximum number of training iterations.
- **eta**: Learning rate.

**Example**
An example script demonstrating the usage of the neural network for a classification task is provided in the evaluation.ipynb notebook.


