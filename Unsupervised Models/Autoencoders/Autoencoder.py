# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:42:56 2021

@author: clovi
"""


# IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# IMPOPRTING THE DATASET
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# PREPARING THE TRAINING SET AND THE TEST SET
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


# GETTING THE NUMBERS OF 
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# CONVERTING THE DATA INTO AN ARRAY WITH USERS IN LINES AND MOVIES IN COLUMNS
def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1):
    id_movies = data[:, 1] [data[:, 0] == id_users]
    id_ratings = data[:, 2] [data[:, 0] == id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies - 1] = id_ratings
    new_data.append(list(ratings))
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)


# CONVERTING THE DATA INTO TORCH TENSORS
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# CREATING THE ARCHITECTURE OF THE NEURAL NETWOR
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)  # fc1 is the first full connected layer (length of the input, number of neurons in the first hidden layer
        self.fc2 = nn.Linear(20, 10) #(number of newrons in the first layer, number of neurons in the second layer)    
        self.fc3 = nn.Linear(10, 20) #(number of newrons in the second layer, number of neurons in the third layer)  
        self.fc4 = nn.Linear(20, nb_movies) #(number of newrons in the third layer, number of neurons in the final layer = input)  
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x)) #the last x is the not encoded features and the first x receuves is the encoded data
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) #the dacay is used to reduce the learning rate at some few epochs

# TRAINING THE SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
        print('epoch: '+str(epoch)+' loss: '+ str(train_loss/s))


# TESTING THE SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))