from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn
import torch.nn.functional as F
import torch.utils.data as Data

# import AbstractClassifier

# a=torch.tensor(np.load('train_data_grp.npy')).float()
# b=torch.tensor(np.load('train_labels_grp.npy')).long()-1

#n_data = torch.ones(100, 2)
#x0 = torch.normal(2*n_data, 1)      # type0 x data (tensor), shape=(100, 2)
#y0 = torch.zeros(100)               # type0 y data (tensor), shape=(100, )
#x1 = torch.normal(-2*n_data, 1)     # type1 x data (tensor), shape=(100, 1)
#y1 = torch.ones(100)                # type1 y data (tensor), shape=(100, )

#x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
#y = torch.cat((y0, y1), ).type(torch.LongTensor)

class KaiyangClassifier(torch.nn.Module):
    def __init__(self, n_feature, neurons):  # n_hidden1, n_hidden2, n_output):
        super(KaiyangClassifier, self).__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(n_feature, neurons[0]))
        for i in range(1, len(neurons)):
            self.layers.append(torch.nn.Linear(neurons[i - 1], neurons[i]))
        self.layers = torch.nn.ModuleList(self.layers)
        # self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        # self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        # self.out = torch.nn.Linear(n_hidden2, n_output)       # output layer
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = torch.sigmoid(self.layers[i](x))
        # h_sigmoid = torch.sigmoid(self.hidden1(x))      #sigmoid func
        # h_sigmoid2 = torch.sigmoid(self.hidden2(h_sigmoid))      #sigmoid func
        # y_out = self.out(h_sigmoid2)                 # output data but not prediction data
        return x
    
    def datasplit(self,x,y,batch_size):
        '''
            Split the data based on the batchsize
        '''
        torch_dataset = Data.TensorDataset(x, y)
        # Batchsize=int(0.8*y.shape[0])
        loader = Data.DataLoader(
                dataset=torch_dataset,      # torch TensorDataset format
                batch_size=batch_size,      # mini batch size
                shuffle=True,               
                num_workers=2,
                )
        for step, (batch_x, batch_y) in enumerate(loader):
            if step==1:
                data_tr=batch_x
                labels_tr=batch_y
            else:
                data_te=batch_x
                labels_te=batch_y
        return data_tr,labels_tr,data_te,labels_te
    
    def train(self, x, y, epochs=10, batch_size=16, lr_rate=0.001, val_x=None, val_y=None, debug_idx=0):
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr_rate)  
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_rate)
        # import all prarmeters(weight,bais) and learning rate
        loss_func = torch.nn.CrossEntropyLoss()

        accu = list()
        accu_te = list()
        loss_tr = list()
        loss_te = list()
        para = list()

        # Reformat as tensors for input into NN module
        # x = torch.tensor(x).float()
        # y = torch.tensor(y).long() - 1

        # if val_x is not None and val_y is not None:
        #     val_x = torch.tensor(val_x).float()
        #     val_y = torch.tensor(val_y).long() - 1

        train_dataset = Data.TensorDataset(x, y)
        train_data = Data.DataLoader(train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0)

        for epoch in tqdm(range(epochs)):   # training times
            # print("Epoch: {}".format(epoch))
            # Run batches of training data
            accum = 0
            avg_loss = 0
            scores = torch.tensor(0)
            for step, (train_x, train_y) in enumerate(train_data):
                # print(step+1)
                out = self(train_x)
                loss = loss_func(out, train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prediction = torch.max(F.softmax(out), dim=1)[1]
                # pred_y = prediction.data.numpy().squeeze()
                # print(sum(prediction == train_y))
                # print(train_y)
                # print(pred_y)
                scores += sum(prediction == train_y)
                avg_loss += loss
                accum += 1

            accuracy = scores.numpy() / y.shape[0]
            avg_loss /= accum

            # Add to the accuracy and loss lists
            accu.append(accuracy)
            loss_tr.append(avg_loss)

            # If we have validation data, we need our stats per epoch
            if val_x is not None and val_y is not None:
                out_te = self(val_x)
                prediction_te = torch.max(F.softmax(out_te), dim=1)[1]
                accuracy_te = sum(prediction_te == val_y).numpy() / val_y.shape[0]
                # pred_te = prediction_te.data.numpy().squeeze()

                loss_te.append(loss_func(out_te, val_y))
                accu_te.append(accuracy_te)

            # Print output after every 100th cycle
            # if debug_idx > 0 and epoch % debug_idx == 0:
            #     if val_x is not None and val_y is not None:
            #         print("Epoch {}: {}, {}".format(epoch, accuracy, accuracy_te))
            #     else:
            #         print("Epoch {}: {}".format(epoch, accuracy))

        return [accu, loss_tr], [accu_te, loss_te]
    
    def predict(self, x):
        '''
            Implementation of a 'predict_proba'
        '''
        out = self(x)
        pred_proba = F.softmax(out)

        return pred_proba

    def evaluate(self, x, y):
        return []
