import numpy as np


class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)
    
    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out= np.zeros((N,m+1))
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
                X_out[:,0] = 1
                X_out[:,1:] = X
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_ = X.squeeze()
            for i in range(m+1): 
                X_out[:,i] = np.power(X_,i)    
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        m = self.m
        N,d = X.shape 
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the linear regression
            # save loss function in loss
            # Calculate the gradient and save it as grad
            #
            # ================================================================ #
            X_ = self.gen_poly_features(X)
            h = np.matmul(X_,self.w)
            y_= y.reshape((-1,1))
            diff = h-y_
            #reg_t = self.reg * np.sqrt((np.sum(self.w**2)))/N
            reg_t = self.reg * np.linalg.norm(self.w/(N**2))
            costs = np.sum(diff**2)/N
            loss = costs + reg_t

            grad = 2/N * np.multiply(diff, X_)
            reg_grad = (2/N*self.reg*self.w).squeeze()
            grad = grad.sum(axis = 0)
            grad = grad + reg_grad
            grad = grad.reshape((-1,1))

            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the polynomial regression with order m
            # ================================================================ #
            X_ = self.gen_poly_features(X)
            h = np.matmul(X_,self.w)
            y_= y.reshape((-1,1))
            diff = h-y_
            reg_t = self.reg * np.linalg.norm(self.w)  / N

            costs = np.linalg.norm(diff)**2 / N
            loss = costs + reg_t

            grad = 2/N * np.multiply(diff, X_)
            reg_grad = (2/N*self.reg*self.w).squeeze()
            grad = grad.sum(axis = 0)
            grad = grad + reg_grad
            grad = grad.reshape((-1,1))
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=30, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
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
                choices = np.random.choice(a=N, size = batch_size, replace = False)
                X_batch = X[choices, :]
                y_batch = y[choices]
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
                step = eta * grad
                new_w = np.subtract(self.w,step)
                """
                if(
                    (not np.any(np.isinf(self.w))) 
                    and 
                    np.any(np.isinf(new_w)
                )):
                    print(self.w.squeeze())
                """
                self.w = new_w
                #self.w = self.w - 2*eta*grad
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        return loss_history, self.w
    
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        N,d = X.shape
        loss = 0
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # obtain the optimal weights from the closed form solution 
            # ================================================================ #
            X_ = self.gen_poly_features(X)
            _, nd = X_.shape
            xx = (X_.T @ X_) 
            xx-= self.reg * np.identity(nd)
            inv = np.linalg.inv(xx)

            self.w = (inv @ X_.T @ y).reshape((-1,1))

            yt_pred = self.predict(X)
            loss = np.sum(np.power(yt_pred - y, 2)) / N
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_ = self.gen_poly_features(X)
            _, nd = X_.shape
            xx = (X_.T @ X_) 
            reg = self.reg * np.identity(nd)
            xx += reg
            inv = np.linalg.inv(xx)

            self.w = (inv @ X_.T @ y).reshape((-1,1))

            yt_pred = self.predict(X)
            loss = np.sum(np.power(yt_pred - y, 2)) / N

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            X_ = self.gen_poly_features(X)
            y_pred = np.matmul(X_,self.w).reshape(-1)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_ = self.gen_poly_features(X)
            y_pred = np.matmul(X_,self.w).reshape(-1)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return y_pred