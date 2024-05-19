import numpy

# creating a class for Neural Network
class NN:

   """
   n_features: no of features
   n_classes: no of classes (no of output neurons)
   n_hidden: no of neurons in the hidden layers

   """

   def __init__(self,n_features,n_classes,n_hidden):

      self.d=n_features
      self.n=n_classes
      self.h=n_hidden

      #creating the weight matrices W1 (collection of weight values from input layer to hidden layer) of dimension (dxh) each column is a weight vectors for each neuron
      self.W1=0.01*np.random.randn(self.d,self.h)

      #creating a bias matrix b1 (collection of bias values from input layer to hidden layer) of dimension (1xh)
      self.b1 = np.zeros((1,self.h))

      #creating the weight matrices W2 (collection of weight values from hidden layer to output layer) of dimension (hxn) each column is a weight vectors for each neuron

      self.W2=0.01*np.random.randn(self.h,self.n)

      #creating a bias matrix b2 (collection of bias values from hidden layer to output layer) of dimension (1xn)
      self.b2 = np.zeros((1,self.n))


   def frwd_prop(self,x):

      # multiplying the weight with the values(datapoint) and adding the bias term b1
      z1=np.dot(x,self.W1)+self.b1

      # applying the relu function to z1
      A1=np.maximum(0,z1)

      # multiplying the weight with the values (r1) and adding the bias term b2
      z2=np.dot(A1,self.W2)+self.b2

      # applying the softmax to the z2
      A2=np.exp(z2)
      A2=A2/np.sum(A2,axis=1,keepdims=True)

      return A1,A2
   

   def ce_loss(self,y_true,y_pred_proba):

          # computing the cross entropy loss
          num_examples=y.shape[0]
          yij_pij=-np.log(y_pred_proba[range(num_examples),y])
          loss=np.sum(yij_pij)/num_examples
          return loss
   


   def backward_prop(self,x,y,A1,A2):

      # capturing the no of datapoints
      num_examples=y.shape[0]

      # computing the derivatives of CE loss wrt to z(inputs to sfmx layer)
      """ derivative of CE loss wrt to zj  dL/dzj= Pij-Yij """
      dZ2 =A2
      dZ2[range(num_examples),y] -= 1

      # normalizing the gradients
      dZ2 /= num_examples
      # computing the derivative of loss wrt W2)
      dW2=np.dot(A1.T,dZ2)
      # computing the derivative of loss wrt b2
      db2=np.sum(dZ2,axis=0,keepdims=True)

      # computing the derivative of loss wrt A1
      dA1=np.dot(dZ2,self.W2.T)

      # computing the gradient for ReLu (gradient is 0 for the negative points)
      dA1[dA1<0]==0

      # computing the gradient for z1
      dZ1=dA1

      # computing the gradient for W1
      dW1=np.dot(x.T,dZ1)

      # computing the gradient for b2
      db1=np.sum(dZ1,axis=0,keepdims=True)

      return dW1, db1, dW2, db2



   def fit(self,x,y,reg,max_iters,eta):

      num_examples=x.shape[0]

      # doing forward and backward prop max_iter times
      for i in range(max_iters):

          #forward prop
          A1,A2=self.frwd_prop(x)

          #calculating the loss
          loss=self.ce_loss(y,A2)
          # calculating the regularization loss
          reg_loss = 0.5*reg*np.sum(self.W1*self.W1) + 0.5*reg*np.sum(self.W2*self.W2)
          # computing the total loss
          total_loss=loss+reg_loss

          if i % 1000 == 0:
                print("iteration %d: loss %f" % (i, total_loss))

          # backprop
          dW1, db1, dW2, db2  = self.backward_prop(x,y,A1,A2)

          # during the backprop we have computed the gradients only with respect to loss, not regularization.
          # add regularization gradient contribution
          dW2 += reg * self.W2
          dW1 += reg * self.W1

          # updating the parameters
          self.W1+= -eta*dW1
          self.W2+= -eta*dW2
          self.b1+= -eta*db1
          self.b2+= -eta*db2


   def predict(self,x):

      # doing foward prop
      _,y_pred=self.frwd_prop(x)

      # converting the  class probabilities into class labels
      y_pred=np.argmax(y_pred,axis=1)

      return y_pred
