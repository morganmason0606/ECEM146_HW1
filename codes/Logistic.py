import numpy as np

class Logistic(object):
    def __init__(self, d=784, reg_param=0):
        """"
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg  = reg_param
        self.dim = [d+1, 1]
        self.w = np.zeros(self.dim)
    def gen_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N,d = X.shape
        X_out= np.zeros((N,d+1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        X_out[:,0] = 1
        X_out[:,1:] = X
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out  
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        N,d = X.shape 
        
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #
        X_ = self.gen_features(X)
        y_ = (y.squeeze() + 1) / 2 #y is given as -1,1, this transforms it to 0,1 
        h  = np.dot(X_, self.w)        
        loss = (1.0/N) * (np.sum(np.log(1 + np.exp(h))) - np.sum(h[(y_== 1).squeeze()])) 

        s = (1.0 /(1 + np.exp(-h))).squeeze()
        diff = np.subtract(s, y_)
        diff = diff.reshape((-1,1)) #want this to be in a (n, 1) vector
        mult = np.multiply(diff, X_)
        grad = (1.0 / N) * np.sum(mult,axis=0)
        grad = grad.reshape(-1,1) #want this to be a (n,1) vector, like w
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
                X_batch = None
                y_batch = None
                # ================================================================ #
                # YOUR CODE HERE:
                # Sample batch_size elements from the training data for use in gradient descent.  
                # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
                # The indices should be randomly generated to reduce correlations in the dataset.  
                # Use np.random.choice.  It is better to user WITHOUT replacement.
                # ================================================================ #
                N, d = X.shape
                choices = np.random.choice(a=N, size = batch_size, replace = False)
                X_batch = X[choices, :]
                y_batch = y[choices]

                #print(f"batches of size : {X_batch.shape}, {y_batch.shape}")
                
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss = 0.0
                grad = np.zeros_like(self.w)
                # ================================================================ #
                # YOUR CODE HERE: 
                # evaluate loss and gradient for batch data
                # save loss as loss and gradient as grad
                # update the weights self.w
                # ================================================================ #
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                #print(f"gradients of :{grad.shape} ; w shape : {self.w.shape}")
                self.w = np.subtract(self.w , eta*grad)
                #print(f"shape of w {self.w.shape}")
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        return loss_history, self.w
    
    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0]+1)
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        X = self.gen_features(X)
        h_arr = np.dot(X, self.w)
        y_pred = 1. / (1 + np.exp(-h_arr))
        y_pred = y_pred.squeeze()
        y_pred = np.vectorize((lambda x: 1 if x > 0.5 else -1))(y_pred)
        y_pred = y_pred.reshape((-1,1))
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred