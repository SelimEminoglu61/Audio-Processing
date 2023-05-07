import numpy as np
from random import random


#save activations and derivatives
#implement backpropagation
#implement gradient descent
#implement train
#train our net with some dummy dataset
#make some predictions

class MLP:

    def __init__(self,num_inputs,num_hidden,num_outputs):
        self.num_inputs=num_inputs
        self.num_hidden=num_hidden
        self.num_outputs=num_outputs

        layers=[self.num_inputs]+num_hidden+[self.num_outputs]

        #initiate random weights
        self.weights=[]
        for i in range(len(layers)-1):
            w=np.random.rand(layers[i],layers[i+1])
            self.weights.append(w)


        #save act. and der.
        activations=[]
        for i in range(len(layers)):
            a=np.zeros(layers[i])
            activations.append(a)
        self.activations=activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i],layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propogate(self,inputs):
        activations=inputs
        self.activations[0]=inputs

        for i,w in enumerate(self.weights):
            #calculate net inputs
            net_inputs=np.dot(activations,w)

            #calculate activations
            activations=self.sigmoid(net_inputs)
            self.activations[i+1]=activations

        return activations

    def backpropagation(self,error,verbose=False):

        # dE/dW_1=(y-a_(i+1)) s'(h_(i+1))a_1      first derivatives
        # s'(h_(i+1))=s(h_(i+1))(1-s(h_(i+1)))
        # s(h_(i+1))=a_(i+1)

        # dE/dW_1=(y-a_(i+1)) s'(h_(i+1)) W_i s'(h_i) a_(i-1)  second derivatives

        for i in reversed(range(len(self.derivatives))):
            activations=self.activations[i+1]
            delta =error*self.sigma_derivatives(activations) #ndarray([0.1,0.2])-->ndarray(([0.1,0.2]))
            delta_reshaped=delta.reshape(delta.shape[0],-1).T
            current_activations=self.activations[i] #ndarray([0.1,0.2])-->ndarray(([0.1],[0.2]))
            current_activations_reshaped=current_activations.reshape(current_activations.shape[0],-1)

            self.derivatives[i]=np.dot(current_activations_reshaped,delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{} : {}".format(i,self.derivatives[i]))

        return error   #last error on input layer

    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            weights=self.weights[i]
            #print("original W{} {}".format(i,weights))

            derivatives=self.derivatives[i]

            weights += derivatives * learning_rate
            #print("original W{} {}".format(i, weights))

    def train(self,inputs,targets,epochs,learning_rate):
        for i in range(epochs):
            sum_error=0
            for input,target in zip(inputs,targets):
                # perform forward propagation
                output = self.forward_propogate(input)

                # calculate error
                error = target - output

                # perform back propagation
                self.backpropagation(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error+=self.mse(target,output)

            #report error
            print("Error: {} at epoch {}".format(sum_error/len(inputs),i))

    def mse(self,target,output):
        return np.average((target-output)**2)

    def sigma_derivatives(self,x):
        return x*(1.0-x)


    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))

if __name__ == "__main__":
    #create an mlp
    mlp=MLP(2,[5],1)

    #create a dataset to train a network for sum operation( create random data)
    inputs=np.array([[random()/2 for _ in range(2)] for _ in range(1000)]) # array([[0.1 0.2],[0.3 0.4]])
    targets=np.array([[i[0]+i[1]] for i in inputs]) # array([[0.3],[0.7]])

    #train our mlp
    mlp.train(inputs,targets,50,0.1)

    # make prediction
    input = np.array([0.1, 0.2])
    target=np.array([0.3])

    output = mlp.forward_propogate(input)
    print("\n\nOur network believes {}+{} equal to {}".format(input[0],input[1],output[0]))

    #create some inputs (dummy data --- inputs and targets)--- in this part, we can give what we want
    #input=np.array([0.1,0.2])
    #target=np.array([0.3])
    #inputs=np.random.rand(mlp.num_inputs) #<---just for mlp

    #perform forward propagation
   # output=mlp.forward_propogate(input)

    #calculate error
    #error=target-output

    #perform back propagation
    #mlp.backpropagation(error,verbose=True)  #if ı dont want to type, ı can delete verbose parameter

    #apply gradient descent
    #mlp.gradient_descent(learning_rate=0.1)

    #print the results
    #print("the network inputs :{}".format(inputs))
    #print("the network outputs :{}".format(outputs))
