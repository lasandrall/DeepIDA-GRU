
# Contains the following functions:
#   DeepIDA_nonBootstrap
#   DeepIDA_Bootstrap
#   DeepIDA_VS
# Please see helper_functions.py for additional functions that the above three functions make use of
# Original Authors: Jiuzhou Wang and Professor Sandra Safo (https://arxiv.org/abs/2111.09964) --- (https://github.com/lasandrall/DeepIDA.git)
# Date: Nov 20, 2021

# Modified by: Sarthak Jain and Professor Sandra Safo
# Imports and Set-up
#----------------------------------------------------------------------------------

import torch
import numpy as np
import pandas as pd
from helper_functions import IDA_solver,DeepIDALoss,Nearest_Centroid_classifier,svm_classify
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler


# Non-Bootstrap version of DeepIDA method

# Input:
# data_train: a list of tensors. Each tensor is the training data for one view 
# data_valid: a list of tensors. Each tensor is the validation data for one view
# data_test: a list of tensors. Each tensor is the test data for one view
# Y_train/valid/test: a tensor representing the group belongings for training/validation/test data
# structure: a list of arrays. Each array specifies the network structure (number of nodes in each layer) beyond the input layer 
# e.g. Construct "input-256-64-20" network structure for three views: [[256,64,20],[256,64,20],[256,64,20]]
# TS_num_features: an array. Each value represents number of features to select by Teacher Student Method on one view
# lr_rate: learning rate for each optimization step (Adam is used)
# n_epoch: number of epochs to train the model

# Output:
# DeepIDA_train/test_acc: classification accuracy based on DeepIDA model trained by each view
# DeepIDA_train/test_labels: training/test data labels predicted by DeepIDA classification model trained by individual view
# DeepIDA_train/test_acc_com: DeepIDA classification accuracy based on combined last layers from all views
# DeepIDA_train/test_labels_com: 
#   training/test data labels predicted by DeepIDA classification model trained by combined last layers from all views
# SVM_train/test_acc: classification accuracy based on SVM trained by each view
# SVM_train/test_labels: training/test data labels predicted by SVM trained by individual view
# SVM_train/test_acc_com: classification accuracy based on SVM trained by stacked all views 
# SVM_train/test_labels_com: training/test data labels predicted by SVM trained by stacked all views
# TS_selected: features selected by Teacher Student Method
# islong: list of 0s and 1s of length equal to the number of views where, 0 indicates cross-sectional input, while 1 indicates longitudinal data input
# long_structure: Structure for RNN network: num_layers, hidden_size, output_size

def DeepIDA_nonBootstrap(data_train, data_valid, data_test, Y_train, Y_valid, Y_test, 
                         structure, TS_num_features, islong = [], long_structure = [1, 128, 128], lr_rate = 0.01, n_epoch = 50, svmTS_yes = 0):

    import torch.nn.functional as F
    from torch import nn


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
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
            # x -> (batch_size, sequence-length, input_size)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            # x.size(0): batch_size
            out, _ = self.gru(x, h0)
            # batch_size, seq_length, hidden_size
            out = out[:, -1, :]
            out = self.fc(out)
            return(out)

    num_of_views = len(data_train)
    
    def combine(all_data):
        data_combined = all_data[0]
        if len(all_data[0].shape) == 3:
            data_combined = torch.mean(all_data[0], dim=1)
            for i in range(num_of_views - 1):
                data_combined = torch.cat((data_combined, torch.mean(all_data[i + 1],dim = 1)), 1)
            return data_combined


        for i in range(num_of_views-1):
            # if all_data[0].shape == all_data[i + 1].shape:
            if len(all_data[i + 1].shape) == 2:
                data_combined = torch.cat((data_combined,all_data[i+1]),1)
            else:
                data_combined = torch.cat((data_combined, torch.mean(all_data[i + 1],dim = 1)), 1)
        return data_combined
    
    import os
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    import tensorflow as tf
    from tensorflow.python.keras.layers import Layer
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import Model
    from keras.layers import Dense,Input
    from keras import backend as K
    import numpy as np

    # define the keras model

    def layer1_reg(weight_matrix):
        return 0.1 * K.sum(K.sqrt(K.square(K.sum(weight_matrix))))
    def loss_mse(y, y_true):
        return K.mean(K.square(y-y_true))

    # Train the student network based on original data and the last layer of the teacher network
    # Note this is used in the simulation
    # The first "num_top_features" features are simulated to be important variables
    # If simulation setup changes, this function needs change also

    # Input:
    # X: an array of the original data
    # y: an array of the last layer of the teacher network
    # num_features: number of features to select by TS method
    # num_top_features: the number of features which defined as important features

    # Output:
    # correct: number of correct features selected by TS method

    def student_network(X,y,num_features):

        hidden = 20
        y1 = Input((X.shape[1],))
        y2 = Dense(10*hidden, activation='relu', kernel_regularizer = layer1_reg)(y1)
        y3 = Dense(hidden)(y2)

        fs_model = Model(y1, y3)

        fs_model.compile(optimizer='adam', loss='mse', metrics=[loss_mse])

        fs_model.fit(X, y, epochs=30, batch_size=int(X.shape[0]/5))

        fs_model.summary()

        w = fs_model.layers[1].get_weights()[0]
        w = np.sum(np.square(w),1)

        dim = X.shape[1]
        features = np.argsort(w)[-num_features:]

        #the first "num_features" features are important
        #correct = sum(features<num_top_features)
        #return correct

        return features    
    
    
    models=[]
    #Each single network is of structure [input_layer,hidden_1,hidden_2,...,final_layer]

    #The following is a default way to set each view network of the same size
    #for d in data_train:
        #models.append(MyModel([d.shape[1],512,256,64,10]))    

    if len(islong) == 0:
        islong = [0] * num_of_views
    #One can self set the network structure
    input_dims = []
    for i in range(len(data_train)):
        d = data_train[i]
        if islong[i] == 0:
            input_dims.append(d.shape[1])
        else:
            input_dims.append(d.shape[2])


    for i in range(len(input_dims)):

        # Each single network is of structure [input_layer,hidden_1,hidden_2,...,final_layer]
        if islong[i] == 0:
            structure[i].insert(0,input_dims[i])
            models.append(SingleNet(structure[i]))
        else:
            models.append(RNN(input_dims[i], long_structure[1], long_structure[0], long_structure[2]))



    params = list()

    for i in range(num_of_views):

        params += list(models[i].parameters())

    optimizer = torch.optim.Adam(params, lr = lr_rate)
    losses_train = []
    losses_valid = []
    relative_losses_train = []
    relative_losses_valid = []
    best_val_loss = 100
    best_train_loss = 100


    for epoch in range(n_epoch):  # loop over the dataset multiple times

        optimizer.zero_grad()
        output_train = []

        for i in range(num_of_views):
            models[i].train()
            output_train.append(models[i](data_train[i]))

        #print(output_train)
        loss_train = DeepIDALoss(output_train,Y_train,0.5)
        losses_train.append(loss_train)

        output_valid = []
        with torch.no_grad():
            for i in range(num_of_views):
                models[i].eval()
                output_valid.append(models[i](data_valid[i]))
            loss_valid = DeepIDALoss(output_valid,Y_valid,0.5)
            losses_valid.append(loss_valid)

            if loss_valid<best_val_loss:                                          # This is original
                best_val_loss = loss_valid
                for i in range(num_of_views):
                    torch.save(models[i].state_dict(),
                               "best_model_view"+str(i)+".pt")


            # if loss_train<best_train_loss:                                          # This is only for n-fold
            #     best_train_loss = loss_train
            #     for i in range(num_of_views):
            #         torch.save(models[i].state_dict(),
            #                    "best_model_view"+str(i)+".pt")


            if(epoch>1):
                pass
                # relative_loss_train = (loss_train - losses_train[epoch-1])/losses_train[epoch-1]
                # relative_losses_train.append(relative_loss_train)
                # relative_loss_valid = (loss_valid - losses_valid[epoch-1])/losses_valid[epoch-1]
                # relative_losses_valid.append(relative_loss_valid)

        loss_train.backward()
        optimizer.step()#do not need with torch.no_grad()

        print("epoch:",epoch)
        print("train_loss:",loss_train.item())
        # print("val_loss:",loss_valid.item())
       

    print('Finished Training')

    new_models=[]

    for i in range(len(input_dims)):

        # Each single network is of structure [input_layer,hidden_1,hidden_2,...,final_layer]
        # new_models.append(SingleNet(structure[i]))
        if islong[i] == 0:
            new_models.append(SingleNet(structure[i]))
        else:
            new_models.append(RNN(input_dims[i], long_structure[1], long_structure[0], long_structure[2]))

    # for name, param in new_models[i].named_parameters():
    #     print(name, param)

    for i in range(num_of_views):
        state = torch.load("best_model_view"+str(i)+".pt")    
        new_models[i].load_state_dict(state)


    # batch normalization with last layer of network
    output_test = []
    with torch.no_grad():
        for i in range(num_of_views):
            new_models[i].eval()
            output_test.append(new_models[i](data_test[i]))
        # print("test_loss:",DeepIDALoss(output_test,Y_test,0.5))

    # Nearest_centroid for each view
    DeepIDA_train_acc = []
    DeepIDA_test_acc = []
    DeepIDA_train_labels = []
    DeepIDA_test_labels = []

    for i in range(num_of_views):
        train_acc,test_acc,train_labels,test_labels = Nearest_Centroid_classifier(X_test=output_test[i],X_train=output_train[i], 
                                                         Y_train=Y_train,Y_test=Y_test)
        DeepIDA_train_acc.append(train_acc)
        DeepIDA_test_acc.append(test_acc)
        DeepIDA_train_labels.append(train_labels)
        DeepIDA_test_labels.append(test_labels)

    # Nearest_centroid for all combined views
    combine(output_test)
    DeepIDA_train_acc_com,DeepIDA_test_acc_com,DeepIDA_train_labels_com,DeepIDA_test_labels_com = Nearest_Centroid_classifier(
        X_test=combine(output_test),X_train=combine(output_train),Y_train=Y_train,Y_test=Y_test)

    if svmTS_yes == 1:
        # SVM for original data
        # SVM for each view
        SVM_train_acc = []
        SVM_test_acc = []
        SVM_train_labels = []
        SVM_test_labels = []

        # for i in range(num_of_views):
        for i in range(sum(1-np.array(islong))):
            [train_acc,test_acc,train_labels,test_labels] = svm_classify(data_train[i].detach().numpy(),
                                            data_test[i].detach().numpy(),
                                        Y_train.numpy().reshape(1,-1)[0],Y_test.numpy().reshape(1,-1)[0],0.01)
            SVM_train_acc.append(train_acc)
            SVM_test_acc.append(test_acc)
            SVM_train_labels.append(train_labels)
            SVM_test_labels.append(test_labels)

        # SVM for all combined views
        [SVM_train_acc_com,SVM_test_acc_com,SVM_train_labels_com,SVM_test_labels_com] = svm_classify(
            combine(data_train).detach().numpy(),combine(data_test).detach().numpy(),
                                        Y_train.numpy().reshape(1,-1)[0],Y_test.numpy().reshape(1,-1)[0],0.01)


        # Feature selection via Teacher Student
        TS_selected = list()
        for i in range(sum(1-np.array(islong))):
            TS_selected.append(student_network(data_train[i].detach().numpy(),output_train[i].detach().numpy(),TS_num_features[i]))

    if svmTS_yes == 1:
        return (DeepIDA_train_acc_com,DeepIDA_test_acc_com,DeepIDA_train_labels_com,DeepIDA_test_labels_com,DeepIDA_train_acc,DeepIDA_test_acc,DeepIDA_train_labels,DeepIDA_test_labels,
                    SVM_train_acc_com,SVM_test_acc_com,SVM_train_labels_com,SVM_test_labels_com,
                    SVM_train_acc,SVM_test_acc,SVM_train_labels,SVM_test_labels,TS_selected)
    else:
        return (DeepIDA_train_acc_com, DeepIDA_test_acc_com, DeepIDA_train_labels_com, DeepIDA_test_labels_com, DeepIDA_train_acc,
                    DeepIDA_test_acc, DeepIDA_train_labels, DeepIDA_test_labels)


# DeepIDA Bootstrap Variable Selection
# By default we select top 10% of the variables for each view

# Input:
# data_all: a list of tensors. Each tensor represent one view
# Y: a tensor representing the group labels
# n_boot: number of bootstrap 
# structure: a list of arrays. Each array specifies the network structure (number of nodes in each layer) beyond the input layer
# e.g. Construct "input-256-64-20" network structure for three views: [[256,64,20],[256,64,20],[256,64,20]]
# n_epoch: number of epochs to train the model
# variables_name_list: a list of arrays. Each array contains the variables's names for each view

# Output:
# p_VS: a list of arrays. Each array represents index of selected features for one view
# num_features_selected: an array. Each value indicated number of features selected in each view
# Generate for each view:
# a csv file which containing the relative feature importance
# a pdf of graph of all feature importance
# a pdf of graph of top 10 percent of variables' feature importance

def DeepIDA_VS(data_all, Y, n_boot, structure, n_epoch, variables_name_list, islong, long_structure, sub_id):
    from multiprocessing import Pool
    import multiprocessing
    from timeit import default_timer as timer
    from helper_functions import variable_select_bs
    import os
    import os.path
    import pickle

    all_to_run = []
    #i serves as the random seed index
    for i in range(n_boot):
       all_to_run.append((data_all,Y,i,i,structure,n_epoch, islong, long_structure))

    
    num_processors = multiprocessing.cpu_count()//2
    print('Number of processors are: ', num_processors)
    print('Number of bootstraps are: ', n_boot)
    tic = timer()
    p = Pool(processes =4)

    # if os.path.isfile('./DeepIDA-BootStrap-Outputs/outputs_'+str(sub_id)+'.npy'):
    #     print('File exists!!')
    #     outputs = np.load('./DeepIDA-BootStrap-Outputs/outputs_'+str(sub_id)+'.npy', allow_pickle=True)
    #     print(len(outputs))
    #     pass
    # else:
    #     outputs = p.starmap(variable_select_bs,all_to_run)
    outputs = p.starmap(variable_select_bs, all_to_run)  #  Remove this line when working with saved data
    # print(outputs)
    # outputs = []
    # for nb in range(n_boot):
    #     print('Bootstrap Number: ', nb)
    #     outputs.append(variable_select_bs(data_all, Y, nb, nb, structure, n_epoch, islong,
    #                                     long_structure))  # For debugging!
    #
    #
    tac = timer()
    print("time for parallel sorting: ",tac-tic)
    #print(outputs)
    #calculate the accuracy of 100 bootstrap
    n_boot = len(outputs)
    accuracy_list = []
    for i in range(n_boot):
        accuracy_list.append(outputs[i][0])

    p_eff_list = []
    p_select_list = []


    for j in range(len(data_all)):
        one_view_eff_list = []
        one_view_select_list = []
        for i in range(n_boot):

            # one_view_eff_list.append(outputs[i][0][1][j])     #Remove [0] later (fix this)
            # one_view_select_list.append(outputs[i][0][2][j])   #Remove [0] later (fix this)
            one_view_eff_list.append(outputs[i][1][j])  # Remove [0] later (fix this)
            one_view_select_list.append(outputs[i][2][j])  # Remove [0] later (fix this)
        p_eff_list.append(one_view_eff_list)
        p_select_list.append(one_view_select_list)


    # Generate for each view:
    # a csv file which containing the relative feature importance
    # a pdf of graph of all feature importance
    # a pdf of graph of top 10 percent of variables' feature importance

    num_features_selected = []
    p_VS = []

    for i in range(len(data_all)):
        p1_eff_list_merged = []
        for ii in p_eff_list[i]:
            p1_eff_list_merged += ii.tolist()
        from collections import Counter
        p1_eff_count = Counter(p1_eff_list_merged)
        p1_eff_count_sorted = sorted(p1_eff_count.items(), key=lambda item: item[1])

        print('Length of p1_eff_count', len(p1_eff_count_sorted))
        if len(p1_eff_count_sorted)==0:
            p1_eff_count_sorted.append((1,1))

        p1_select_list_merged = []
        for ii in p_select_list[i]:
            p1_select_list_merged += ii.tolist()
        from collections import Counter
        p1_select_count = Counter(p1_select_list_merged)
        p1_select_count_sorted = sorted(p1_select_count.items(), key=lambda item: item[1])
        print('Length of p1_select_count', len(p1_select_count_sorted))
        import pandas as pd

        df = pd.DataFrame(p1_select_count_sorted)
        df.columns = ["index","bootstrap_times"]
        # if this does not work, means there are few selected variables
        other = pd.DataFrame(p1_eff_count_sorted)
        other.columns = ["index", "eff_times"]
        df = df.join(other.set_index('index'),on='index')
        df['eff_prop'] = df['eff_times']/df['bootstrap_times']


        df = df.dropna()
        df['rank'] = df['eff_prop'].rank(ascending=False)


        # variables names of selected variables
        df['var_name'] = variables_name_list[i][df['index']]

        # Save a csv file which containing the relative feature importance
        df.sort_values(by=['eff_prop'],ascending=False).to_csv("sub"+str(sub_id)+"view_"+str(i+1)+".csv",index=False)

        # decide numbers of parameters to select
        if islong[i]==0:
            num_selected = round(data_all[i].shape[1]*0.1)
            num_features_selected.append(num_selected)
        else:
            num_selected = round(data_all[i].shape[2] * 0.1)
            num_features_selected.append(num_selected)

        if(num_selected<100):
            df_selected = df.sort_values(by=['eff_prop'],ascending=False).head(num_selected)
        else:
            df_selected = df.sort_values(by=['eff_prop'],ascending=False).head(100)

        # select the top 10% variables
        print('Length of df rank',len(df['rank']))
        p_VS.append(df[df['rank']<=num_selected]['index'])

        # plot for all variables
        import matplotlib.pyplot as plt
        from textwrap import wrap

        features = df['var_name'].values
        #wrap the names into multiple lines if it is too long
        features = [ '\n'.join(wrap(l, 35)) for l in features ]
        importances = df['eff_prop'].values
        #get the rank of every element , rank 0 means the lowest
        indices = np.argsort(importances)
        #every # number of y_labels to show
        ticks = df.shape[0]//100

        f = plt.figure()
        plt.rcParams['figure.figsize'] = [15, 30]
        plt.title('Feature Importances of All Variables in View '+str(i+1))
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        if ticks>0: 
            plt.yticks(range(len(indices))[::ticks], [features[i] for i in indices[::ticks]])
        else:
            plt.yticks(range(len(indices)), [features[i] for i in indices])
        #the height of y equal to numbers of variables
        #show where to plot the top 100 variables
        if len(indices)>100:
            plt.axhline(y = len(indices)-100, color = 'r', linestyle = ':',label = "Top 100 variables")
        #show where to choose parameters
        plt.axhline(y = len(indices)-num_selected, color = 'y', linestyle = '--',label = "Top 10% varaibles (to select)")
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature Name')
        plt.text(0,len(indices)+1,"Total number of variables is "+str(data_all[0].shape[1])+ " and top "+str(num_selected)+" variables are selected." )
        plt.legend(bbox_to_anchor = (0.5, 1), loc = 'upper center')
        plt.show()
        address = "Feature_score_view_"+str(i+1)+"_all.pdf"
        # Save a pdf of graph of all feature importance
        f.savefig(address, bbox_inches='tight')    

        #plot for top 10% variables

        import matplotlib.pyplot as plt
        features = df_selected['var_name'].values
        features = [ '\n'.join(wrap(l, 30)) for l in features ]
        importances = df_selected['eff_prop'].values
        indices = np.argsort(importances)

        f = plt.figure()
        plt.rcParams['figure.figsize'] = [15, 30]
        plt.title('Feature Importances of Top 10% Variables (Max = 100) in View '+str(i+1))
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()
        address = "Feature_score_view_"+str(i+1)+"_top10percent.pdf"
        # Save a pdf of graph of top 10 percent of variables' feature importance
        f.savefig(address, bbox_inches='tight')

    return(p_VS,num_features_selected)

# Bootstrap version of DeepIDA method

# Input:
# data_train: a list of tensors. Each tensor is the training data for one view 
# data_valid: a list of tensors. Each tensor is the validation data for one view
# data_test: a list of tensors. Each tensor is the test data for one view
# Y_train/valid/test: a tensor representing the group belongings for training/validation/test data
# structure: a list of arrays. Each array specifies the network structure (number of nodes in each layer) beyond the input layer 
# e.g. Construct "input-256-64-20" network structure for three views: [[256,64,20],[256,64,20],[256,64,20]]
# TS_num_features: an array. Each value represents number of features to select by Teacher Student Method on one view
# lr_rate: learning rate for each optimization step (Adam is used)
# n_epoch_boot: number of epochs to train the model
# n_boot: number of bootstrap 
# n_epoch_nonboot: number of epochs to train the model
# variables_name_list: a list of arrays. Each array contains the variables's names for each view

# Output:
# DeepIDA_train/test_acc: classification accuracy based on DeepIDA model trained by each view
# DeepIDA_train/test_labels: training/test data labels predicted by DeepIDA classification model trained by individual view
# DeepIDA_train/test_acc_com: DeepIDA classification accuracy based on combined last layers from all views
# DeepIDA_train/test_labels_com: 
#   training/test data labels predicted by DeepIDA classification model trained by combined last layers from all views
# SVM_train/test_acc: classification accuracy based on SVM trained by each view
# SVM_train/test_labels: training/test data labels predicted by SVM trained by individual view
# SVM_train/test_acc_com: classification accuracy based on SVM trained by stacked all views 
# SVM_train/test_labels_com: training/test data labels predicted by SVM trained by stacked all views
# TS_selected: features selected by Teacher Student Method

def DeepIDA_Bootstrap(data_train, data_valid, data_test, Y_train, Y_valid, Y_test, 
                         structure,n_boot, n_epoch_boot, variables_name_list, TS_num_features, islong = [], long_structure = [1, 128, 128], lr_rate = 0.01, n_epoch_nonboot = 50, sub_id=0 ):
    
    data_VS = []

    Y_VS = torch.cat((Y_train,Y_valid),0)
    for i in range(len(data_train)):
        data_VS.append(torch.cat((data_train[i],data_valid[i]),0))

    VS_result = DeepIDA_VS(data_VS, Y_VS, n_boot, structure, n_epoch_boot, variables_name_list, islong, long_structure, sub_id)
    data_train_s = []
    data_valid_s = []
    data_test_s = []
    for i in range(len(data_train)):
        print(i)
        print(VS_result[0][i].values)
        if islong[i]==0:
            data_train_s.append(data_train[i][:,VS_result[0][i].values])
            data_valid_s.append(data_valid[i][:,VS_result[0][i].values])
            data_test_s.append(data_test[i][:,VS_result[0][i].values])
        else:
            data_train_s.append(data_train[i][:,:, VS_result[0][i].values])
            data_valid_s.append(data_valid[i][:,:, VS_result[0][i].values])
            data_test_s.append(data_test[i][:,:, VS_result[0][i].values])
    # print(data_train_s[0].shape)
    # print(data_train_s[1].shape)
    # print(data_test_s[2])
    # print(data_train_s[0].shape)
    # print(data_train_s[1].shape)
    # print(data_train_s[2].shape)
    # print(structure)
    # print(islong)
    # print(long_structure)
    final = DeepIDA_nonBootstrap(data_train_s, data_valid_s, data_test_s, Y_train, Y_valid, Y_test, structure, [5, 5], islong, long_structure, lr_rate, n_epoch_nonboot)

    return (VS_result,final)











