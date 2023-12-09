import torch
import pandas as pd
import numpy as np
import random

# Normalization of data. Input is arrays
def normalize(data):
    means = data.mean(dim=0, keepdim=True)
    stds = data.std(dim=0, keepdim=True)
    normalized_data = (data - means) / stds
    return normalized_data

# Given the old Gamma (projections), return the new Gamma
# Input:
# data: a list of tensors. Each tensor represent one view
# group_label: a list of numeric values indicating the belonging of each sample
# Gamma_old: a list of tensors. Each tensor is composed of eigen vectors for one view, before the current iteration
# rho: a parameter between 0 and 1 controlling the relative importance of seperation and association. Larger rho weighs more on seperation

# Output:
# Gamma_new: a list of tensors. Each tensor is composed of eigen vectors for one view, after the current iteration
# Lambda_new: a list of tensors. Each tensor is composed of eigen values for one view, after the current iteration
def IDA_solver(data,group_label,Gamma_old,rho):
    
    # rank: number of dimensions on the final projected space
    rank = len(torch.unique(group_label)) - 1
    num_of_views = len(data)
    num_of_groups = torch.unique(group_label)
    num_of_subjects = len(group_label)
    c1 = rho
    c2 = 2*(1-rho)/num_of_views/(num_of_views-1)
    
    # list of root inverse of total covaraince
    StRootInv = []
    # list of M
    Md_list = []
    
    # Compute the covariance matrix for each view
    for k in range(num_of_views):
        
        H1 = data[k]

        # Calculate between class scatter matrix
        S1b = torch.zeros_like(torch.matmul(H1.t(),H1))
        mean_total = torch.mean(H1,0)
        for label in num_of_groups:
            idx = (group_label == label).nonzero(as_tuple=True)[0]
            mean = torch.unsqueeze(torch.mean(H1[idx,],0),0)
            S1b += len(idx) * torch.matmul((mean-mean_total).t(),(mean-mean_total))
        S1b = S1b/(num_of_subjects-1)

        # Calculate total covariance matrix
        S1t = torch.matmul((H1-torch.mean(H1,0,True)).t(),(H1-torch.mean(H1,0,True)))/(num_of_subjects-1)

        # Calculate root inverse of total covaraince
        # [D1, V1] = torch.symeig(S1t, eigenvectors=True)
        S1t = S1t + 0.00001*torch.diag(torch.randn(S1t.shape[0]))

        [D1, V1] = torch.linalg.eigh(S1t)

        #Added to increase stability 
        D1 = D1 + 0.0001   ##This was initially commented
        S1tRootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        
        StRootInv.append(S1tRootInv)

        Md = torch.matmul(torch.matmul(S1tRootInv,S1b),S1tRootInv)
        Md_list.append(Md)
        
    Gamma_new = []
    Lambda_new = []

    # Compute the cross covariance matrix for different views
    for d in range(num_of_views):


        need = c1*Md_list[d]

        for j in range(num_of_views):

            if d == j :
                continue

            H1 = data[d]
            H2 = data[j]
            
            # Numerical issues lead Sjd not necessarily equal to Sjd.t()
            Sdj = torch.matmul((H1-torch.mean(H1,0,True)).t(),(H2-torch.mean(H2,0,True)))/(num_of_subjects-1)
    

            Ndj = torch.matmul(torch.matmul(StRootInv[d],Sdj),StRootInv[j])
           

            need += c2*torch.matmul(torch.matmul(torch.matmul(Ndj,Gamma_old[j]),Gamma_old[j].t()),Ndj.t())
          
            
        
        need = need + 0.00001*torch.diag(torch.randn(need.shape[0]))
        # print(need)
        # [Lambda, Gamma] = torch.symeig(need, eigenvectors=True)

        [Lambda,Gamma] = torch.linalg.eigh(need)


        # Choose top rank(k-1) out of o features to create new Gamma
        Gamma = Gamma[:,Lambda.topk(rank)[1]]
        
        Gamma_new.append(Gamma)
        Lambda_new.append(Lambda)
    
    return Gamma_new,Lambda_new

# Calculate the DeepIDA Loss for the current epoch

# Input:
# data: a list of tensors. Each tensor represent one view
# Y: a list of numeric values indicating the belonging of each sample
# rho: a parameter between 0 and 1 controlling the relative importance of seperation and association. Larger rho weighs more on seperation

# Output:
# loss: the DeepIDA optimized loss for the current epoch

def DeepIDALoss(data,Y,rho):
    
    # Gamma_old/new: a list of eigen vectors
    # Lambda_old/new: a list of eigen values corresponding to Gamma_old/new
    # diff_Gamma: a list of relative Frobenius norm difference between the current and previous Gamma
    # Length of the list represents number of views
    Gamma_old = []
    Lambda_old = []
    diff_Gamma = []
    # rank: number of dimensions on the final projected space, at most (# of groups - 1)
    rank = len(torch.unique(Y)) - 1
    # Initialize values
    for i in data:
        Gamma_old.append(torch.ones(i.shape[1],rank))
        Lambda_old.append(0)
        diff_Gamma.append(1)
   
    
    # Iteratively solving for Projections (Gamma) 
    iteration = 0
    max_iter = 100
    
    for i in range(0,max_iter,1):
        
        iteration += 1
        
        # One-step update for Gamma
        Gamma_new,Lambda_new = IDA_solver(data,Y,Gamma_old,rho)
        
        for j in range(len(diff_Gamma)):
            diff_Gamma[j] = (torch.norm(Gamma_new[j] - Gamma_old[j], p='fro')**2 / torch.norm(Gamma_old[j], p='fro')**2).item()
        
        Gamma_old = Gamma_new
        Lambda_old = Lambda_new
        
        if max(diff_Gamma)<0.00000001 :
            break
    
    
    loss = 0
    for i in Lambda_old:
        loss += sum(i.topk(rank)[0])
        
    return -loss




#/////////////////////////////////////////////////
def DeepIDAProj(data, Y, rho):
    # Gamma_old/new: a list of eigen vectors
    # Lambda_old/new: a list of eigen values corresponding to Gamma_old/new
    # diff_Gamma: a list of relative Frobenius norm difference between the current and previous Gamma
    # Length of the list represents number of views
    Gamma_old = []
    Lambda_old = []
    diff_Gamma = []
    # rank: number of dimensions on the final projected space, at most (# of groups - 1)
    rank = len(torch.unique(Y)) - 1
    # Initialize values
    for i in data:
        Gamma_old.append(torch.ones(i.shape[1], rank))
        Lambda_old.append(0)
        diff_Gamma.append(1)

    # Iteratively solving for Projections (Gamma)
    iteration = 0
    max_iter = 100

    for i in range(0, max_iter, 1):

        iteration += 1

        # One-step update for Gamma
        Gamma_new, Lambda_new = IDA_solver(data, Y, Gamma_old, rho)

        for j in range(len(diff_Gamma)):
            diff_Gamma[j] = (torch.norm(Gamma_new[j] - Gamma_old[j], p='fro') ** 2 / torch.norm(Gamma_old[j],
                                                                                                p='fro') ** 2).item()

        Gamma_old = Gamma_new
        Lambda_old = Lambda_new

        if max(diff_Gamma) < 0.00000001:
            break

    loss = 0
    for i in Lambda_old:
        loss += sum(i.topk(rank)[0])

    return Gamma_new

#/////////////////////////////////////////////////


# Nearest_Centroid_classifier

# Input:
# X_train/test: n*p tensor 
# Y_train/test: n tensor
# Assume that the Y_train and Y_test have the same number of groups

# Output:
# acc_train/acc_test: classification accuracy for training and test data respectively

def Nearest_Centroid_classifier(X_train,X_test,Y_train,Y_test):

    
    # Build the k centroids
    group_index = Y_train.unique()
    centroids = []
    for i in group_index:
        # Calculate the centroids for each group based on train data
        centroids.append(torch.mean(X_train[Y_train==i],dim=0,keepdim=False))
    
    
    # Assign groups for X_test
    labels_test = []
    for j in X_test:
        distance = []
        for k in centroids:
            # The distance (2-norm) of a test point to group k
            distance.append(torch.norm(j-k))
        # Assign the group label for point j
        labels_test.append(group_index[distance.index(min(distance))])
    labels_test = torch.tensor(labels_test)    
    
    # Calculate accuracy for test data
    acc_test = (torch.sum(Y_test==labels_test)).numpy()/len(Y_test)
    
    # Assign groups for X_train
    labels_train = []
    for j in X_train:
        distance = []
        for k in centroids:
            # The distance (2-norm) of a test point to group k
            distance.append(torch.norm(j-k))
        # Assign the group label for point j
        labels_train.append(group_index[distance.index(min(distance))])
    labels_train = torch.tensor(labels_train)    
    
    # Calculate accuracy for train data
    acc_train = (torch.sum(Y_train==labels_train)).numpy()/len(Y_train)   
    
    return acc_train,acc_test,labels_train,labels_test


from sklearn import svm
from sklearn.metrics import accuracy_score,balanced_accuracy_score


# Linear SVM on the data

# Input:
# train_data/test_data: n_train*p/n_test*p Numpy Arrays
# train_label/test_label: n_train/n_test Numpy Arrays 
# C: the penalty factor of SVM

# Output:
# train_acc/test_acc: classification accuracy for training and test data respectively
# p_train/test: predicted labels for training/test data
 
def svm_classify(train_data,test_data,train_label,test_label, C):

    
    print('training SVM...')
    clf = svm.SVC(C=C,class_weight="balanced",kernel='linear')
    clf.fit(train_data, train_label.ravel())

    p_test = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p_test)

    p_train = clf.predict(train_data)
    train_acc = accuracy_score(train_label, p_train)

    return [train_acc,test_acc,p_train,p_test]

# Return the bootstrap and out-of-bag index given the proportion of sampling

def index_sampler(old_index,prop,seed):
    import random
    random.seed(seed)
    selected_index = np.random.choice(old_index,size=int(len(old_index)*prop),replace=True)
    oob_index = np.delete(old_index,np.unique(selected_index))
    return selected_index,oob_index

# Do one bootstrap sampling and corresponding feature selection
# Input:
# data_all: a list of tensors. Each tensor represent one view
# Y: a list of numeric values indicating the belonging of each sample
# bt_seed: random seed for bootstrap
# model_seed: random seed for initialization of model parameters
# structure: a list of arrays. Each array specifies the network structure (number of nodes in each layer) beyond the input layer
# e.g. Construct "input-256-64-20" network structure for three views: [[256,64,20],[256,64,20],[256,64,20]]
# n_epoch: number of epochs to train the model

# Output:
# baseline_acc: the classification accuracy for the out-of-bag data on the model trained by bootstrap data
# p_eff: a list of arrays. Each array is the index of parameters that permuting leads to a decrease in accuracy, for a specific view
# p_select: a list of arrays. Each array is the bootstrap parameter index of a specific view

def variable_select_bs(data_all,Y,bt_seed,model_seed,structure,n_epoch, islong, long_structure):
    
    # Number of views 
    n_views = len(data_all)
    # Number of training samples (or the number of subjects)
    n_sample = Y.shape[0]
    # Bootstrap the sample index to get sample bootstrap index and oob index, where oob = "out of box"
    n_select, n_oob = index_sampler(list(range(0, n_sample)), 1, bt_seed)

    # Bootstrap the variable index of each view
    p_number = []
    p_select = []

    for i in range(0,n_views):
        
        # Number of dimensions of view i
        if islong[i]==0:
            p_number.append(data_all[i].shape[1])
        else:
            p_number.append(data_all[i].shape[2])
        
        # Bootstrap parameter index of view i
        p_select.append(index_sampler(list(range(0,p_number[i])),0.8,bt_seed)[0])

    # Fiexed pairing, pair bootstrap sample and variables at each iteration
    data_select = []
    data_oob = []
    for i in range(0,n_views):
        
        # Bootstraped data of view i to train the process
        # data_select.append(data_all[i][n_select,:][:,p_select[i]])
        if islong[i] == 0:
            data_select.append(data_all[i][n_select,:][:,p_select[i]])
        if islong[i] == 1:
            data_select.append(data_all[i][n_select,:,:][:,:,p_select[i]])

        # Out-of-bag data of view i
        # data_oob.append(data_all[i][n_oob,:][:,p_select[i]])
        if islong[i] == 0:
            data_oob.append(data_all[i][n_oob,:][:,p_select[i]])
        if islong[i] == 1:
            data_oob.append(data_all[i][n_oob,:,:][:,:,p_select[i]])
        
    # Bootstraped labels
    Y_select = Y[n_select]
    # Out-of-bag labels
    Y_oob = Y[n_oob]

    import torch.nn.functional as F
    from torch import nn

    # Set up a deep network structure for one view
    
    class SingleNet(nn.Module):
        
        def __init__(self, layer_sizes):
            super(SingleNet, self).__init__()
            layers = []
            for layer_id in range(len(layer_sizes) - 1):
                if layer_id == len(layer_sizes) - 2:
                    layers.append(nn.Sequential(
                        nn.BatchNorm1d(num_features=layer_sizes[layer_id], affine=False),
                        nn.Linear(layer_sizes[layer_id], layer_sizes[layer_id + 1]),
                    ))
                else:
                    layers.append(nn.Sequential(
                        nn.Linear(layer_sizes[layer_id], layer_sizes[layer_id + 1]),
                        nn.LeakyReLU(0.1),
                        nn.BatchNorm1d(num_features=layer_sizes[layer_id + 1], affine=False),
                    ))
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        # class RNN(nn.module):
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(RNN, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            # x -> (batch_size, sequence-length, input_size)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            # x.size(0): batch_size
            out, _ = self.gru(x, h0)
            # batch_size, seq_length, hidden_size
            out = out[:, -1, :]
            out = self.fc(out)
            return (out)
    
    # Fix the initialization points
    torch.manual_seed(model_seed)

    num_of_views = len(data_select)
    # Initialize the models
    models=[]

    if len(islong) == 0:
        islong = [0] * num_of_views
    #One can self set the network structure
    input_dims = []
    for i in range(len(data_select)):
        d = data_select[i]
        if islong[i] == 0:
            input_dims.append(d.shape[1])
        else:
            input_dims.append(d.shape[2])

    # input_dims=[]
    # for d in data_select:
    #
    #     input_dims.append(d.shape[1])


    for i in range(len(input_dims)):

        # Each single network is of structure [input_layer,hidden_1,hidden_2,...,final_layer]
        if islong[i] == 0:
            structure[i].insert(0,input_dims[i])
            models.append(SingleNet(structure[i]))
        else:
            models.append(RNN(input_dims[i], long_structure[1], long_structure[0], long_structure[2]))

    # for i in range(len(input_dims)):
    #
    #     # Each single network is of structure [input_layer,hidden_1,hidden_2,...,final_layer]
    #     structure[i].insert(0,input_dims[i])
    #     models.append(SingleNet(structure[i]))

    params = list()

    for i in range(n_views):

        params += list(models[i].parameters())

    # Alternative: optimizer = torch.optim.RMSprop(params, lr=0.001, weight_decay=1e-3)
    optimizer = torch.optim.Adam(params,lr=0.001)
    losses_train = []
    relative_losses_train = []

    # Train the model based on bootstrap data: loop over the dataset multiple times
    for epoch in range(n_epoch):  

        optimizer.zero_grad()
        output_train = []

        for i in range(n_views):
            models[i].train()
            output_train.append(models[i](data_select[i].float()))


        loss_train = DeepIDALoss(output_train,Y_select,0.5)

        losses_train.append(loss_train)

        loss_train.backward()
        optimizer.step()

        print("epoch:",epoch)
        print("train_loss:",loss_train.item())

    print('Finished Training')

    output_test = []
    for i in range(n_views):
        
        models[i].eval()
        output_test.append(models[i](data_oob[i].float()))

    # Baseline test accuracy of NCC on all views of data
    print('Calculating Baseline Accuracy')
    baseline_acc = Nearest_Centroid_classifier(X_train=torch.cat(output_train,1), 
                            X_test = torch.cat(output_test,1), Y_train = Y_select, Y_test = Y_oob)[1]
    print('Done!')
    # Returns a random permutation of integers from 0 to n - 1.
    rand_index = torch.randperm(len(data_oob[0]))

    # Store the variables that lead to decrease in test acc in each view
    p_eff = []


    for k in range(n_views):
        test_permuted_view_all_acc = []

        if islong[k] == 0:
            ll = data_oob[k].shape[1]
        else:
            ll = data_oob[k].shape[2]
        # for j in range(data_oob[k].shape[1]):
        for j in range(ll):
            if j%100==0:
                print('View ', k, '-- variable: ', j, ' out of ', ll)
            data_oob_permuted = []
            for i in data_oob:
                
                # Create a cloned data_oob
                # This is important, to avoid becoming a pointer
                data_oob_permuted.append(i.detach().clone())

            # Permute j th variable in k th view
            # data_oob_permuted[k][:,j] = data_oob[k][:,j][rand_index]
            if islong[k]==0:
                data_oob_permuted[k][:, j] = data_oob[k][:, j][rand_index]
            else:
                data_oob_permuted[k][:, :, j] = data_oob[k][:, :, j][rand_index]
            # Batch normalization with last layer of network
            output_test_permuted = []
            for i in range(n_views):
                models[i].eval()
                output_test_permuted.append(models[i](data_oob_permuted[i].float()))

            test_permuted_view_all_acc.append(Nearest_Centroid_classifier(X_test=torch.cat(output_test_permuted,1),
                        X_train=torch.cat(output_train,1), Y_train=Y_select,Y_test=Y_oob)[1])

        p_eff.append(p_select[k][[x-baseline_acc<0 for x in test_permuted_view_all_acc]])

    return baseline_acc,p_eff,p_select