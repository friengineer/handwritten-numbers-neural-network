import numpy as np


class MLP:
    " Multi-layer perceptron " 
    def __init__(self, sizes, beta=1, momentum=0.9):

        """
        sizes is a list of length four. The first element is the number of features 
                in each samples. In the MNIST dataset, this is 784 (28*28). The second 
                and the third  elements are the number of neurons in the first 
                and the second hidden layers, respectively. The fourth element is the 
                number of neurons in the output layer which is determined by the number 
                of classes. For example, if the sizes list is [784, 5, 7, 10], this means 
                the first hidden layer has 5 neurons and the second layer has 7 neurons. 
        
        beta is a scalar used in the sigmoid function
        momentum is a scalar used for the gradient descent with momentum 
        """
        self.beta = beta
        self.momentum = momentum

        # number of features in each sample
        self.nin = sizes[0]
        # number of neurons in the first hidden layer
        self.nhidden1 = sizes[1]
        # number of neurons in the second hidden layer
        self.nhidden2 = sizes[2]
        # number of classes / the number of neurons in the output layer
        self.nout = sizes[3]


        # Initialise the network of two hidden layers
        # hidden layer 1
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden1)-0.5)*2/np.sqrt(self.nin)
        # hidden layer 2
        self.weights2 = (np.random.rand(self.nhidden1+1,self.nhidden2)-0.5)*2/np.sqrt(self.nhidden1)
        # output layer
        self.weights3 = (np.random.rand(self.nhidden2+1,self.nout)-0.5)*2/np.sqrt(self.nhidden2)


    def train(self, inputs, targets, eta, niterations):
        """
        inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.

        targets is a numpy array of shape (num_train, D) containing the training labels
                    consisting of num_train samples each of dimension D.

        eta is the learning rate for optimization 
        niterations is the number of iterations for updating the weights 

        """
        # number of data samples
        ndata = np.shape(inputs)[0]
        # adding the bias
        inputs = np.concatenate((inputs,-np.ones((ndata,1))),axis=1)

        # numpy array to store the update weights 
        updatew1 = np.zeros((np.shape(self.weights1))) 
        updatew2 = np.zeros((np.shape(self.weights2)))
        updatew3 = np.zeros((np.shape(self.weights3)))


        for n in range(niterations):            
            # forward phase 
            self.outputs = self.forwardPass(inputs)

            # Error using the sum-of-squares error function
            error = 0.5*np.sum((self.outputs-targets)**2)

            if (np.mod(n,100)==0):
                print("Iteration: ",n, " Error: ",error)

            # backward phase 
            # Compute the derivative of the output layer using the derivative of the softmax function
            deltao = np.zeros((np.shape(self.outputs)))
            
            # compute the derivative of each row in the outputs matrix and apply the resulting jacobian
            # matrix to the difference between the output and the target
            for i in range(ndata):                
                jacobian = np.zeros((self.nout, self.nout))
                
                for j in range(self.nout):
                    for k in range(self.nout):
                    
                        if j == k:
                            jacobian[j][k] = self.outputs[i][j] * (1 - self.outputs[i][j])
                        else:
                            jacobian[j][k] = -self.outputs[i][j] * self.outputs[i][k]
                            
                deltao[i] = np.dot((self.outputs[i]-targets[i]), jacobian)

            # compute the derivative of the second hidden layer 
            deltah2 = (self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao,np.transpose(self.weights3))))

            # compute the derivative of the first hidden layer 
            deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2[:,:-1],np.transpose(self.weights2)))            

            # update the weights of the three layers using gradient descent and add momentum
            updatew1 = (eta*(np.dot(np.transpose(inputs),deltah1[:,:-1]))) + (self.momentum*updatew1)
            updatew2 = (eta*(np.dot(np.transpose(self.hidden1),deltah2[:,:-1]))) + (self.momentum*updatew2)
            updatew3 = (eta*(np.dot(np.transpose(self.hidden2),deltao))) + (self.momentum*updatew3)

            self.weights1 -= updatew1
            self.weights2 -= updatew2
            self.weights3 -= updatew3


    def forwardPass(self, inputs):
        """
            inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.  
        """
        # layer 1 
        # compute the forward pass on the first hidden layer with the sigmoid function
        hidden1 = np.dot(inputs,self.weights1)
        hidden1 = 1.0/(1.0+np.exp(-self.beta*hidden1))
        hidden1 = np.concatenate((hidden1,-np.ones((np.shape(inputs)[0],1))),axis=1)
        
        self.hidden1 = hidden1

        # layer 2
        # compute the forward pass on the second hidden layer with the sigmoid function
        hidden2 = np.dot(hidden1,self.weights2)
        hidden2 = 1.0/(1.0+np.exp(-self.beta*hidden2))
        hidden2 = np.concatenate((hidden2,-np.ones((np.shape(hidden1)[0],1))),axis=1)
        
        self.hidden2 = hidden2

        # output layer 
        # compute the forward pass on the output layer with softmax function
        outputs = np.dot(hidden2, self.weights3)
        
        # apply softmax to each row in the outputs matrix
        for row in range(np.shape(outputs)[0]):
            outputs[row] = np.exp(outputs[row])/np.sum(np.exp(outputs[row]))
        
        return outputs


    def evaluate(self, X, y):
        """ 
            this method is to evaluate our model on unseen samples 
            it computes the confusion matrix and the accuracy 
    
            X is a numpy array of shape (num_train, D) containing the testing images
                    consisting of num_train samples each of dimension D. 
            y is  a numpy array of shape (num_train, D) containing the testing labels
                    consisting of num_train samples each of dimension D.
        """

        inputs = np.concatenate((X,-np.ones((np.shape(X)[0],1))),axis=1)
        outputs = self.forwardPass(inputs)
        nclasses = np.shape(y)[1]

        # 1-of-N encoding
        outputs = np.argmax(outputs,1)
        targets = np.argmax(y,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print("The confusion matrix is:")
        print(cm)
        print("The accuracy is ",np.trace(cm)/np.sum(cm)*100)
