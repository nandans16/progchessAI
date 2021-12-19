#This code is from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#Several changes have been made for the specific problem I have been working on 

#PyTorch library in https://github.com/pytorch/pytorch
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#The pickle package is used to store the training and test data
import pickle 

import numpy as np

#Matplotlib library in https://github.com/matplotlib/matplotlib
import matplotlib.pyplot as plt


#Defines the neural network
#This code is from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 3 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 3, 3)
        # The output of this conv layer is 10 channels each of dimension 4x4
        self.conv2 = nn.Conv2d(3, 10, 3)

        # This layer gets the output of conv2 and flattens it; 4 neurons are added for the other 4 elements of state 
        self.fc1 = nn.Linear(10 * 4 * 4 + 4, 80) 

        # The output of this conv layer is a flattened layer of 80 neurons
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x_b, x):
        # Max pooling over a (2, 2) window


        #Only the board passes through the convolutional layers
        x1 = F.max_pool2d(F.relu(self.conv1(x_b)), (1,1))
        # If the size is a square, you can specify with a single number
        x1 = F.max_pool2d(F.relu(self.conv2(x1)), (1,1))
        
        x1 = tc.flatten(x1, 1) # flatten all dimensions except the batch dimension
        x = tc.flatten(x,1)
        
       #print(x1.size())
        #print(x.size())
        
        x1 = x1.detach().numpy()
        x = x.detach().numpy()
        


        #The other elemets of state (player, # moves, total # moves in the turn and game type) are fed to the neural network
        xc = np.concatenate((x1, x), axis = 1)

        xc = tc.Tensor(xc)
        
        xc = tc.tanh(self.fc1(xc))
        
        xc = tc.tanh(self.fc2(xc))
        
        xc = tc.sigmoid(self.fc3(xc))

        return xc


def unpick(filename):
    data_f = []
    data = []
    with open(filename, 'rb') as f:
        while True:
            try:
                data_f = pickle.load(f)
            except EOFError:
                break
            else:
               data = data + data_f
                
    return data

#This part is based on the code in the course notes of CIS 667, EECS, Syracuse University, 2021
def batch_error(net, batch):
    boards, states, utilities = batch
    u = utilities.reshape(-1,1).float()
    y = net(boards, states)
    e = tc.sum((y - u)**2) / utilities.shape[0]
    return e

#This part is based on the code in the course notes of CIS 667, EECS, Syracuse University, 2021
if __name__ == "__main__":
    net = Net()
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    #Load pickle files
    inp = []
    
    targ = []

    
    #To prepare training data
    filein = 'train_data_input.pkl'
    fileout = 'train_data_target.pkl'
    
    inp = unpick(filein)
    
    targ = unpick(fileout)
    
    train_data_board = []
    train_data_state = []
    train_targ = []
    
    for i in range(len(inp)):
        in_c_board = []
        in_c_state = []
        out_c = []
        
        in_c_board.append(inp[i][2])
        in_c_state.append([inp[i][0], inp[i][1], inp[i][3], inp[i][4]])
        in_c_board = np.array(in_c_board)
        in_c_state = np.array(in_c_state)        
        
        out_c.append(targ[i])
        
        train_data_board.append(in_c_board)
        train_data_state.append(in_c_state)
        train_targ.append(out_c)
    
    #To prepare test data
    filein = 'test_data_input.pkl'
    fileout = 'test_data_target.pkl'
    
    inp = unpick(filein)
    
    targ = unpick(fileout)
    
    test_data_board = []
    test_data_state = []
    test_targ = []
    
    for i in range(len(inp)):
        in_c_board = []
        in_c_state = []
        out_c = []
        
        in_c_board.append(inp[i][2])
        in_c_state.append([inp[i][0], inp[i][1], inp[i][3], inp[i][4]])
        in_c_board = np.array(in_c_board)
        in_c_state = np.array(in_c_state)
        
        out_c.append(targ[i])
        
        test_data_board.append(in_c_board)
        test_data_state.append(in_c_state)
        test_targ.append(out_c)
        
    train_targ = F.softmax(tc.Tensor(train_targ), 0)
    test_targ = F.softmax(tc.Tensor(test_targ), 0)
    #Make the training and test tensor dataset batches
    training_batch = tc.Tensor(train_data_board), tc.Tensor(train_data_state), train_targ
    testing_batch = tc.Tensor(test_data_board), tc.Tensor(test_data_state), test_targ
    
    curves = [], []
    
    t = []
    
    for epoch in range(1000):
        optimizer.zero_grad()
        training_error, testing_error = 0, 0
        
        
        e = batch_error(net, training_batch)
        e.backward()
        training_error = e.item()

        with tc.no_grad():
            e = batch_error(net, testing_batch)
            testing_error = e.item()

        # take the next optimization step
        optimizer.step()    
        
        # print/save training progress
        if epoch % 10 == 0:
            print("%d: %f, %f" % (epoch, training_error, testing_error))
        curves[0].append(training_error)
        curves[1].append(testing_error)
        t.append(epoch)
    
    plt.plot(t, curves[0]) 
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Training error v/s time")
    plt.savefig(r"NN_training_error")
    
    plt.figure()

    plt.plot(t, curves[1]) 
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Testing error v/s time")
    plt.savefig(r"NN_testing_error")
    plt.figure()
       
