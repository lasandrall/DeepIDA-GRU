# Author: Sarthak Jain (Supervisor: Professor Sandra Safo)
import numpy as np
import numpy
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import skfda
from skfda import FDataGrid
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.representation.basis import BSpline, Fourier, Monomial
from mpl_toolkits import mplot3d
from main_functions import DeepIDA_nonBootstrap, DeepIDA_Bootstrap
from scipy.stats import random_correlation
import math
import pickle
import pandas as pd
# from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from numpy.random import normal
import time
import os
import sys
import subprocess



def ECurve(M, npoints = -10):
    '''
    Computes Euler Characteristic Curve for a given symmetric matrix -M- and number of points -npoints-
    '''

    thr_list = np.linspace(0,1,npoints)
    rat = euler_char(M,thr_list)
    arr = numpy.array(rat)
    return arr, thr_list


def euler_char(A, thresh):
    '''
    Computes Euler Characteristic Curve for a given symmetric matrix -A- using threshold values -thresh-
    (Code sourced from https://github.com/zavalab/ML/tree/master/ECpaper with minor modifications)
    '''

    ECs = []

    for t in thresh:
        M = np.array((A <= t) * 1)

        # Number Edges
        Edges = np.sum(M) / 2

        # Number Vertices
        Vertices = np.shape(A)[0]

        # Euler Characteristic
        EC = Vertices - Edges

        ECs.append(EC)

    return ECs


def multiviewCovMat(n1, n2):
    '''
    Generates block covariance matrix for multiple views where every element is uniformly distributed
    -n1-: Number of variables in first view
    -n2-: Number of variables in second view
    '''

    C = np.random.uniform(-1,1,(n1+n2,n1+n2))
    C = (C+C.T)/2
    for i in range(C.shape[0]):
        C[i,i] = np.random.uniform(0,1)
    return np.dot(C,C.transpose())


def timeseries(C, T, ar_v1, ma_v1,ar_v2, ma_v2, nfeat1, nfeat2, ar_common_v1, ma_common_v1, ar_common_v2, ma_common_v2, thres=0):
    '''
    Simulate a time series of length -T- with covariance matrix -C-
    -nfeatd- is the number of variables in view d
    -ar_vd- and -ma_vd- are vectors providing AR and MA parameters, respectively, of view d for generating variables 1 to -thres-
    -ar_common_v1-, -ar_common_v2-, -ma_common_v1-, -ma_common_v2- - AR and MA parameters of the two views for generating variables thres+1 to nfeatd

    Example: For view d, the data is generated as:
            -> ar_vd[0] x(t) = ar_vd[1] x(t-1) + ma_vd[0] w(t) + ma_vd[1] w(t-1) for the first -thres- variables (1 to -thres-)
            and
            -> ar_common_vd[0] x(t) = ar_common_vd[1] x(t-1) + ma_common_vd[0] w(t) + ma_common_vd[1] w(t-1) for the rest of the variables (thres+1 to nfeatd).
            The noise vector w is generated according to the covariance matrix -C-.

    ----Outputs----
    data - multiview data in list form where (d-1)-th element of the list is (N, p_d, t_d) if view d is longitudinal and (N, p_d) if cross-sectional
    y - label
    '''

    x = np.zeros((C.shape[0], T))
    mu = np.zeros(C.shape[0])
    ww = np.random.multivariate_normal(mu,C,T)
    w = ww.T
    x[:,0] = w[:,0]

    for t in range(1,T):
        ar_n = 0
        x[0:thres, t] = (ar_v1[1]+ar_n*np.random.randn()) * x[0:thres, t - 1] + (ma_v1[0]+ar_n*np.random.randn()) * w[0:thres, t] + (ma_v1[1]+ar_n*np.random.randn()) * w[0:thres, t - 1]
        x[nfeat1:nfeat1+thres, t] = (ar_v2[1]+ar_n*np.random.randn()) * x[nfeat1:nfeat1+thres, t - 1] + (ma_v2[0]+ar_n*np.random.randn()) * w[nfeat1:nfeat1+thres, t] + (ma_v2[1]+ar_n*np.random.randn()) * w[nfeat1:nfeat1+thres, t - 1]
        x[thres:nfeat1, t] = (ar_common_v1[1]+ar_n*np.random.randn()) * x[thres:nfeat1, t-1] + (ma_common_v1[0]+ar_n*np.random.randn()) * w[thres:nfeat1, t] + (ma_common_v1[1]+ar_n*np.random.randn()) * w[thres:nfeat1,t-1]
        x[nfeat1+thres:nfeat1+nfeat2,t] = (ar_common_v2[1]+ar_n*np.random.randn())*x[nfeat1+thres:nfeat1+nfeat2,t-1]+(ma_common_v2[0]+ar_n*np.random.randn())*w[nfeat1+thres:nfeat1+nfeat2,t]+(ma_common_v2[1]+ar_n*np.random.randn())*w[nfeat1+thres:nfeat1+nfeat2,t-1]

    return x


def plotin3D(X,zz,ax,col, lab=''):
    '''
    -X- : (p x T) multivariate time-series with rows corresponding to variables and columns corresponding to times
    -zz- : The z-axis shift at which the multivariate time series is plotted
    This function plots a 3D plot of the multivariable time series, where the multivariate time series is plotted in the
    x-y plane and the z-coordinate given by -zz- offsets (or separates) the data of the two classes.
    '''

    cc = 1
    for i in range(X.shape[0]):
        z = zz * np.ones(X.shape[1])
        x = X[i,:]
        y = np.linspace(1, X.shape[1], X.shape[1])
        if cc>0:
            ax.plot3D(x, y, z,col, label = lab)
            cc = cc-1
        else:
            ax.plot3D(x, y, z, col)

def generate_data(eps, eta, time_length_v1 = 20, time_length_v2 = 20, nfeat_v1 = 25, nfeat_v2 = 25, total_subjects = 200,
                  ar_c1_v1 = [1, 0.5], ar_c1_v2 = [1, 0.5], ma_c1_v1 = [1, 0.6], ma_c1_v2 = [1, 0.6], ar_common_v1 = [1, 0.5],
                  ar_common_v2 = [1, 0.5], ma_common_v1 = [1, 0.5], ma_common_v2 = [1, 0.5], threshold = 50):
    '''
    Refer to the **synthetic_data** function below for more details about this function's arguments
    '''

    time_length_v1 = time_length_v1   # This is the number of time samples in view v1 for both classes
    time_length_v2 = time_length_v2   # This is the number of time samples in view v2 for both classes
    nfeat_v1 = nfeat_v1          # Number of features in view v1 for both classes
    nfeat_v2 = nfeat_v2         # Number of features in view v2 for both classes
    total_subjects = total_subjects   # Total number of subjects

    ar_c1_v1 = ar_c1_v1      # ar parameters for view 1 in class c1
    ar_c1_v2 = ar_c1_v2      # ar parameters for view 2 in class c1
    ar_c2_v1 = [1, 0.5-eta]     # ar parameters for view 1 in class c2
    ar_c2_v2 = [1, 0.5-eta]     # ar parameters for view 2 in class c2

    ma_c1_v1 = ma_c1_v1      # ma parameters for view 1 of class c1
    ma_c1_v2 = ma_c1_v2      # ma parameters for view 2 of class c1
    ma_c2_v1 = [1, 0.6-eta]     # ma parameters for view 1 of class c2
    ma_c2_v2 = [1, 0.6-eta]  # ma parameters for view 1 of class c2
    ar_common_v1 = ar_common_v1
    ar_common_v2 = ar_common_v2
    ma_common_v1 = ma_common_v1
    ma_common_v2 = ma_common_v2


    # Joint covariance matrix for class c1 (for both views combined)
    C_c1 = np.random.random((nfeat_v1 + nfeat_v2, nfeat_v1 + nfeat_v2))
    C_c1 = np.dot(C_c1, C_c1.transpose())

    # Joint covariance matrix for class c2 (for both views combined)

    C_c2 = np.random.power(10,(nfeat_v1 + nfeat_v2, nfeat_v1 + nfeat_v2))
    C_c2 = np.dot(C_c2, C_c2.transpose())
    C_c2 = (1-eps)*C_c1+eps*C_c2

    # tensor for data of both classes and all views and all subjects
    X = torch.zeros(nfeat_v1+nfeat_v2, max(time_length_v1,time_length_v2), total_subjects)
    y = torch.zeros(total_subjects)
    thr = math.ceil(threshold*nfeat_v1/100)
    for i in range(total_subjects):
        print(i)
        if np.random.rand()>=0.5:
            X[:, :, i] = torch.tensor(timeseries(C_c1, max(time_length_v1, time_length_v2), ar_c1_v1, ma_c1_v1,ar_c1_v2, ma_c1_v2, nfeat_v1, nfeat_v2, ar_common_v1, ma_common_v1, ar_common_v2, ma_common_v2,thr))
            y[i] = 0
        else:
            X[:, :, i] = torch.tensor(timeseries(C_c2, max(time_length_v1, time_length_v2), ar_c2_v1, ma_c2_v1, ar_c2_v2, ma_c2_v2,nfeat_v1, nfeat_v2, ar_common_v1, ma_common_v1, ar_common_v2, ma_common_v2,thr))
            y[i] = 1

    # Separate whole data X into data for each views
    X_v1 = X[0:nfeat_v1,0:time_length_v1,:]
    X_v2 = X[nfeat_v1:nfeat_v1+nfeat_v2,0:time_length_v2,:]

    return X, y, X_v1, X_v2


def synthetic_data(eps, eta, time_length_v1 = 20, time_length_v2 = 20, nfeat_v1 = 25, nfeat_v2 = 25, total_subjects = 200,
                  ar_c1_v1 = [1, 0.5], ar_c1_v2 = [1, 0.5], ma_c1_v1 = [1, 0.6], ma_c1_v2 = [1, 0.6], ar_common_v1 = [1, 0.5],
                  ar_common_v2 = [1, 0.5], ma_common_v1 = [1, 0.5], ma_common_v2 = [1, 0.5], threshold = 50, plots = True, npoints_EC = 1000):
    '''
        Function to generate synthetic dataset consisting of two longitudinal views as described in the paper.
        ----Inputs----
        -eps- : The epsilon parameter used to control the difference between the covariance structure of the two classes
                (see section - 'Synthetic Analysis of EC and FPCA' of the paper)
        -eta- : The eta parameter used to control the difference between the ARMA parameters of the two classes
                (see section - 'Synthetic Analysis of EC and FPCA' of the paper)
        -time_length_v1-, -time_length_v2- : Number of time points in views 1 and 2 respectively
        -nfeat_v1-, -nfeat_v2- : Number of variables in views 1 and 2 respectively
        -total_subjects- : Total number of subjects
        -ar_c1_v1-, -ar_c1_v2-, -ar_c2_v1-, -ar_c2_v2- : AR parameters of classes 1 and 2 and views 1 and 2
        -ar_common_v1-, -ar_common_v2-, -ma_common_v1-, -ma_common_v2- : AR and MA parameters of the two views that are common for the two classes
        -threshold- : This parameter was set to 100 while creating the synthetic dataset for the paper. In general, it works as follows:
                    The first -threshold- percent variables of each view d (= 1 or 2) and class k (= 1 or 2) are generated using
                        ARMA parameters: -ar_ck_vd-, -ma_ck_vd-.
                    For rest of the variables of view d, the ARMA parameters for the two classes are the
                        same and are given by -ar_common_vd-, -ma_common_vd-.
                    This ensures that the first 'threshold' variables of each view have different ARMA parameters across
                        the classes, whereas the rest of the variables are similar.
        Example of data generation:
            For view d, class k, the data is generated as:
            ar_ck_vd[0] x(t) = ar_ck_vd[1] x(t-1) + ma_ck_vd[0] w(t) + ma_ck_vd[1] w(t-1),
            for the first -threshold- percent variables and
            ar_common_vd[0] x(t) = ar_common_vd[1] x(t-1) + ma_common_vd[0] w(t) + ma_common_vd[1] w(t-1),
            for the rest of the variables.
            The noise vector w is generated according to uniform or power distribution
            depending on the value of eps (as described in the paper).
        plots - if True, the function plots the following figures:
               (i) longitudinal data of one of the subjects from each class for both the views
               (ii) EC curves for both the view (different color for each class)
               (iii) Functional Principal Components and FPC scores of all the subjects of first variable of both views
        -npoints_EC- : Number of threshold values used while generating EC curves (used if plots == True).
        -n_components_FPCA- : Number of FPCs used to compute the FPC scores (used if plots == True).

        ----Outputs----
        -data - multiview data in list form where the (d-1)-th element of the list is a tensor of shape (N, p_d, t_d) if
                view d is longitudinal and a tensor of shape (N, p_d) if view d cross-sectional
        -y- : labels
    '''


    X, y, X_v1, X_v2 = generate_data(eps, eta, time_length_v1 = 20, time_length_v2 = 20, nfeat_v1 = 25, nfeat_v2 = 25, total_subjects = 200,
                  ar_c1_v1 = [1, 0.5], ar_c1_v2 = [1, 0.5], ma_c1_v1 = [1, 0.6], ma_c1_v2 = [1, 0.6], ar_common_v1 = [1, 0.5],
                  ar_common_v2 = [1, 0.5], ma_common_v1 = [1, 0.5], ma_common_v2 = [1, 0.5], threshold = threshold)  # X_v1 = P1xT1xN, X_v2 = P2xT2xN
    X1 = X_v1.permute(2, 0, 1)
    X2 = X_v2.permute(2, 0, 1)
    data = [X1, X2]  # List of tensors for D views, each of shape (N, p_d, t_d) if longitudinal and (N, p_d) if cross-sectional

    if plots == True:



        for v in range(2):
            '''The following line plots data of one of the subjects from each class'''
            plot_OneFromEachClass(data, y, view=v)
            ''' The following line plots the EC curves of both the view'''
            plotECCurves(data, y, view = v, npoints = npoints_EC)
            '''The following line plots Functional Principal Components and FPC scores of all the subjects of variable 'variable' of both views'''
            plotFPCs(data, y, view = v, variable = 0, n_components=3)
    return data, y


def plotECCurves(data, y, view = 1, npoints = 1000):
    '''
        Inputs:
        -data- : list containing longitudinal/cross-sectional data of all views
        -y- : labels
        -npoints- number of threshold values for EC curves
        -view- : view for which the plots are created

        Output:
        Figure showing the EC curves of the two classes for all the subjects from view 'view'
    '''
    d = view
    curves = torch.zeros((data[d - 1].shape[0], npoints))  # This tensor will store the EC curves of the N subjects
    plt.figure()
    classes = y.unique()  # Unique classes
    num_sub_in_classes = torch.bincount(y.int())  # Number of subjects in each class
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Different colors for upto 7 classes
    c = torch.ones(len(classes))
    for i in range(data[d - 1].shape[0]):  # looping over the N subjects
        print(i)
        from nilearn.connectome import ConnectivityMeasure
        tangent_measure = ConnectivityMeasure(kind="correlation",
                                              vectorize=False)  # Other options: "covariance", "precision" etc.
        X_v = data[d - 1].permute(1, 2, 0)
        x = X_v[:, :, i]  # (P,T) data of i-th subject
        z = x.permute(1, 0).numpy()
        z = z.reshape(1, z.shape[0], z.shape[1])

        M = tangent_measure.fit_transform(z)  # Computing correlation matrix
        h, thr_list = ECurve(M, npoints)  # EC curve of i-th subject

        curves[i, :] = torch.from_numpy(h)  # Storing EC curve of i-th subject in curves

        if c[y.int()[i]] == 1:
            plt.plot(thr_list, h, colors[y.int()[i]], label=f'Class {y.int()[i]}')
            c[y.int()[i]] = 0
        else:
            plt.plot(thr_list, h, colors[y.int()[i]])

    plt.ylabel('Euler Characteristic', fontsize=15)
    plt.xlabel('Threshold', fontsize=15)
    plt.title(f'Euler Curves for different classes for view {view+1}', fontsize=15)
    print(num_sub_in_classes)
    plt.legend()
    plt.show()

def plot_OneFromEachClass(data, y, view = 1):
    '''
    Inputs:
    -data- : list containing longitudinal/cross-sectional data of all views
    -y- : labels
    -view- : view for which the plots are created

    Output:
    Figure showing the p_d by t_d longitudinal data of one subject from each class for view -view-.
    '''
    d = view-1
    classes = y.unique()  # Unique classes
    c = torch.ones(len(classes))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Different colors for upto 7 classes
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    i = 0
    while torch.sum(c) > 0:
        if c[y.int()[i]] == 1 or i >= data[d - 1].shape[0]:
            plotin3D(data[d-1][i, :, :], i, ax, colors[y.int()[i]], f'Class {y.int()[i] + 1}')
            c[y.int()[i]] = 0
            # plotin3D(X_v2[:, :, i], i, ax, 'g', 'Class 1')
        i += 1
    plt.ylabel('Time', fontsize=15)
    plt.title(f'Longitudinal data of one subject from each class for view {view+1}')
    plt.legend()
    plt.show()


def plotFPCs(data, y, view = 1, n_components = 3, variable = 0):
    '''
        Inputs:
        -data- : list containing longitudinal/cross-sectional data of all views
        -y- : labels
        -view- : view for which the plots are created
        -n_components- : number of Functional principal components to use while computing FPC scores (must be set to 3)
        -variable- : which variable's FPC and FPC scores to plot

        Output:
        Figure showing the 3 dimensional FPC scores of all the subjects of both classes for variable -variable- and view -view-
    '''
    # FPCA
    nc = n_components
    d = view
    X_v = data[d-1].permute(1,2,0)
    data_v = torch.zeros((X_v.shape[2], nc * X_v.shape[0]))

    for i in range(X_v.shape[0]):
        x = X_v[i, :, :]
        x = x.T
        fd = FDataGrid(x, range(x.shape[1]),
                       dataset_name='Time Series',
                       argument_names=['t'],
                       coordinate_names=['x(t)'])
        fpca_discretized = FPCA(n_components=nc)
        fpca_discretized.fit(fd)
        h = fpca_discretized.components_

        print(np.mean(h.data_matrix[1]))
        if i == variable:
            plt.show()
            plt.figure()
            plt.plot(h.data_matrix[0], 'k', linewidth=3, label='FPC-1')
            plt.plot(h.data_matrix[1], 'b', linewidth=3, label='FPC-2')
            plt.plot(h.data_matrix[2], 'c', linewidth=3, label='FPC-3')
            plt.title(f'{n_components} FPCs of variable indexed {variable} for view {view + 1}')
            plt.legend()
            plt.show()

        h = fpca_discretized.transform(fd)

        for j in range(nc):
            data_v[:, i * nc + j] = torch.tensor(h[:, j])

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    classes = y.unique()  # Unique classes
    c = torch.ones(len(classes))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Different colors for upto 7 classes
    for i in range(data_v.shape[0]):
        if c[y.int()[i]] == 1:
            ax.scatter3D(data_v[i, 0], data_v[i, 1], data_v[i, 2], color=colors[y.int()[i]], label = f'Class {y.int()[i]}')
            c[y.int()[i]] -= 1
        else:
            ax.scatter3D(data_v[i, 0], data_v[i, 1], data_v[i, 2], color=colors[y.int()[i]])

    # # show plot
    plt.title(f'{n_components}-dimensional FPC scores (of all the subjects) of variable indexed {variable} for view {view+1}')
    plt.legend()
    plt.show()

def convertData(data, y, conversion, npoints = 1000, n_components = 3):
    '''
            Inputs:
            -data- : list containing longitudinal/cross-sectional data of all views
            -y- : labels
            -conversion- : List containing what conversion to perform for each view
            -n_components- : number of Functional principal components
            -npoints- : number of threshold values for EC curves

            Output:
            Converted data (where each view d is converted according to the function argument -conversion[d]-)
    '''
    if conversion == 'EC':
        curves = torch.zeros(
            (data.shape[0], npoints))  # This tensor will store the EC curves of the N subjects
        classes = y.unique()  # Unique classes
        num_sub_in_classes = torch.bincount(y.int())  # Number of subjects in each class

        for i in range(data.shape[0]):  # looping over the N subjects
            from nilearn.connectome import ConnectivityMeasure
            tangent_measure = ConnectivityMeasure(kind="correlation",
                                                  vectorize=False)  # Other options: "covariance", "precision" etc.
            X_v = data.permute(1, 2, 0)
            x = X_v[:, :, i]  # (P,T) data of i-th subject
            z = x.permute(1, 0).numpy()
            z = z.reshape(1, z.shape[0], z.shape[1])

            M = tangent_measure.fit_transform(z)  # Computing correlation matrix
            h, thr_list = ECurve(M, npoints)  # EC curve of i-th subject

            curves[i, :] = torch.from_numpy(h)  # Storing EC curve of i-th subject in curves

        return curves

    elif conversion == 'FPC':
        nc = n_components
        X_v = data.permute(1, 2, 0)
        data_v = torch.zeros((X_v.shape[2], nc * X_v.shape[0]))

        for i in range(X_v.shape[0]):
            x = X_v[i, :, :]
            x = x.T
            fd = FDataGrid(x, range(x.shape[1]),
                           dataset_name='Time Series',
                           argument_names=['t'],
                           coordinate_names=['x(t)'])
            fpca_discretized = FPCA(n_components=nc)
            fpca_discretized.fit(fd)
            h = fpca_discretized.components_
            h = fpca_discretized.transform(fd)

            for j in range(nc):
                data_v[:, i * nc + j] = torch.tensor(h[:, j])
        return data_v
    elif conversion == 'mean':
        return torch.mean(data, dim = 2)
    elif conversion == 'nothing':
        return data
    else:
        print('Invalid conversion given. Please check again. Exiting code.')
        exit()


def featureExtraction_DeepIDA_GRU(data, y, structuresNN = None, structureGRU = None, LR=0.001, n_epochs = 50, list_of_conversions = None, npoints_EC=1000, n_components_FPCA=3, split_seed = 1234):
    '''
    -data- : List (of length D) containing inputs for each of the D views:
                                           Shape: (N,p_d,t_d) for longitudinal and (N,p_d) for cross-sectional.
    -y- : Output labels
    -structuresNN- : Structures of neural network for each view  ( Example: structuresNN = [[200, 20, 20], [200, 20, 20]] )
    -structureGRU- : Structure of GRU (if one or more converted data is/are longitudinal) ( Example: structureGRU = [3, 100, 20] )
    -LR- : Learning Rate
    -n_epochs- : Number of epochs
    -list_of_conversions- : List of length D (the number of views), which tells which feature extraction method to use for each view.
                          The data of a given view is unchanged or converted to cross-sectional form using FPCA, EC or mean.
                          Conversion is represented by a string that can take the following options: 'FPC', 'EC', 'mean', 'nothing'.
                          For example, for a two view data, if list_of_conversions == ['EC','FPCA'], then the first view
                          uses EC for feature extraction and the second view uses FPCA for feature extraction.
    -n_components- : number of Functional principal components used if conversion was FPCA
    -npoints- : number of threshold values used for EC curves if conversion was EC
    -split_seed- : seed used for splitting data into train, test and valid
    '''

    if list_of_conversions == None:
        list_of_conversions = ['nothing'] * len(data)
    else:
        list_of_conversions = list_of_conversions

    data_converted = []
    for i in range(len(data)):
        output = convertData(data[i], y, conversion=list_of_conversions[i], npoints=npoints_EC,
                             n_components=n_components_FPCA)
        if len(output.shape) == 3:
            output = output.permute(0, 2, 1)  # Converting (N,P,T) to (N,T,P) for GRUs
        data_converted.append(output)

    '''Split data into train, test and valid'''
    data_converted_train = []
    data_converted_test = []
    data_converted_valid = []
    for i in range(len(data_converted)):
        X_train, X_test, y_train, y_test = train_test_split(data_converted[i], y, stratify=y,
                                                            test_size=0.20, random_state=split_seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train,
                                                              test_size=0.25, random_state=split_seed)
        data_converted_train.append(X_train)
        data_converted_test.append(X_test)
        data_converted_valid.append(X_valid)


    '''DeepIDA-GRU Parameters'''
    long_cross = []  # This vector tells DeepIDA-GRU if the converted data of the d-th view is longitudinal or cross-sectional
    for i in range(len(data_converted)):
        if len(data_converted[i].shape) == 2:
            long_cross.append(0)
        else:
            long_cross.append(1)
    if structuresNN == None:
        structuresNN = [[200, 20, 20]]*len(long_cross)  # Structure of neural network for each view
    if structureGRU == None:
        structureGRU = [3, 100, 20]  # Structure of GRU (if one or more converted data is/are longitudinal)
    LR = LR  # Learning Rate
    n_epochs = n_epochs  # Number of epochs


    results = DeepIDA_nonBootstrap(data_converted_train, data_converted_test, data_converted_valid,
                                   y_train, y_test, y_valid,
                                   structuresNN,
                                   [], long_cross, structureGRU, LR, n_epochs)


    return results


def variable_selection_DGB(data, y, top_features = 'default', nboots=50, n_epoch_boot=50, structuresNN = None, structureGRU = None, LR=0.001, n_epochs=50, split_seed = 1234):
    '''
    -data- : Input multiview data in a list where the (d-1)-th item of the list corresponds to the d-th view and has that
          view's data in the form (N,p_d,t_d) if longitudinal and (N, p_d) if cross-sectional.
    -y- : labels
    -top_features- : list (of length equal to the number of views) where each element tells how many variables have to be
                  selected from a given view. If top_features == 'default', top 25% variables are selected from each
                  view.
    -split_seed- : Seed used for splitting the data into train, test, valid splits.
    -structuresNN- : Structure of neural networks for each cross-sectional view
                    (Example: structuresNN = [[200, 20, 20], [200, 20, 20]] means for both the views, the number of
                    neurons in the first, second and third layers are 200, 20, 20, respectively)
    -structureGRU- : Structure of GRU (if one or more views are longitudinal) ( Example: structureGRU = [3, 100, 20] )
    LR - Learning Rate
    nboots - Number of bootstraps
    n_epoch_boot - Number of epochs for baseline training of each epoch
    n_epochs - Number of bootstraps for final training using 10% of the selected variables.


    ---Output---
    data_selected: Variable-selected multiview data in a list where the (d-1)-th item of the list corresponds to the
                   d-th view and has that view's data in the form (N,top_features[d-1],t_d) if longitudinal and
                   (N, top_features[d-1]) if cross-sectional, where top_features[d-1] are the number of selected
                   variables from view d.
    '''

    list_of_conversions = ['nothing']*len(data)
    data_converted = []
    for i in range(len(data)):
        output = convertData(data[i], y, conversion=list_of_conversions[i])
        if len(output.shape) == 3:
            output = output.permute(0, 2, 1)  # Changing shape from (N,P,T) to (N,T,P) for GRUs of the DGB network
        data_converted.append(output)

    '''Split data into train, test and valid'''
    data_converted_train = []
    data_converted_test = []
    data_converted_valid = []
    for i in range(len(data_converted)):
        X_train, X_test, y_train, y_test = train_test_split(data_converted[i], y, stratify=y,
                                                            test_size=0.20, random_state=split_seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train,
                                                              test_size=0.25, random_state=split_seed)
        data_converted_train.append(X_train)
        data_converted_test.append(X_test)
        data_converted_valid.append(X_valid)

    '''Variable Selection only happens on training data. Therefore further split the training data into train, test, valid'''
    data1 = []
    data2 = []
    data3 = []
    for i in range(len(data_converted_train)):
        X_1, X_2, y_1, y_2 = train_test_split(data_converted_train[i], y_train, stratify=y_train,
                                                            test_size=0.20, random_state=split_seed)
        X_1, X_3, y_1, y_3 = train_test_split(X_1, y_1, stratify=y_1,
                                                              test_size=0.25, random_state=split_seed)
        data1.append(X_1)
        data2.append(X_2)
        data3.append(X_3)

    '''DeepIDA-GRU Parameters'''
    long_cross = []  # This vector tells DeepIDA-GRU if the converted data of the d-th view is longitudinal or cross-sectional
    for i in range(len(data_converted)):
        if len(data_converted[i].shape) == 2:
            long_cross.append(0)
        else:
            long_cross.append(1)
    if structuresNN == None:
        structuresNN = [[200, 20, 20]]*len(long_cross)  # Structure of neural network for each view
    if structureGRU == None:
        structureGRU = [3, 100, 20]  # Structure of GRU (if one or more converted data is/are longitudinal)
    LR = LR  # Learning Rate
    n_epochs = n_epochs  # Number of epochs
    nboots = nboots
    n_epoch_boot = n_epoch_boot
    variables_name_list = []

    for d in range(len(data_converted)):
        variables_name_list.append(pd.Index([f'View{d + 1}Var%d' % i for i in range(data_converted[d].shape[-1])]))

    results = DeepIDA_Bootstrap(data1, data2, data3,
                                y_1, y_2, y_3,
                                structuresNN, nboots, n_epoch_boot, variables_name_list,
                                [], long_cross, structureGRU, LR, n_epochs)


    data_selected = []
    for view in range(len(data)):
        if top_features == 'default':
            top = math.ceil(25*data[view].shape[1]/100)
        else:
            top = top_features[view]
        indices = pd.read_csv(f'sub0view_{view+1}.csv')['index'].tolist()[0:top]
        data_selected.append(torch.index_select(data[view], 1, torch.LongTensor(indices)))
    for i in range(len(data_selected)):
        print(data_selected[i].shape)
    return data_selected

def variable_selection_JPTA(data, y, top_features = 'default', nboots='NA', n_epoch_boot='NA', structuresNN = 'NA', structureGRU = 'NA', LR='NA', n_epochs='NA', split_seed = 'NA'):
    '''
    Only works with two views data where both the views are longitudinal.
    -data- : Input multiview data in a list where the (d-1)-th item of the list corresponds to the d-th view and has that
          view's data in the form (N,p_d,t_d) if longitudinal and (N, p_d) if cross-sectional.
    -y- : labels
    -top_features- : list of length equal to the number of views where each element tells how many variables have to be
                  selected from a given view. If top_features == 'default', top 25% variables are selected from each
                  view.
    Other variables: Not relevant for this method.

    ---Output---
    -data_selected- : Variable-selected multiview data in a list where the (d-1)-th item of the list corresponds to the
                   d-th view and has that view's data in the form (N,top_features[d-1],t_d) if longitudinal and
                   (N, top_features[d-1]) if cross-sectional, where top_features[d-1] are the number of selected
                   variables from view d.
    '''

    if len(data) != 2 or len(data[0].shape) != 3 or len(data[1].shape) != 3:
        print('Input not suitable for JPTA method. Please check again. Exiting code!')
        exit()
    if top_features == 'default':
        top_features = []  # How many variables to select from each view
        for view in range(len(data)):
            top = math.ceil(25 * data[view].shape[1] / 100)
            top_features.append(top)

    import rpy2.robjects as robjects
    from rpy2.robjects import StrVector
    from rpy2.robjects import numpy2ri

    # Pass Python variables to R using the robjects.r() function
    numpy2ri.activate()

    robjects.r(f'topfeau <- {top_features[0]}')
    robjects.r(f'topfeav <- {top_features[1]}')

    # Convert the PyTorch tensor to a NumPy array
    data_v1 = data[0].numpy()
    data_v2 = data[1].numpy()

    numpy2ri.activate()
    # mat1 = numpy2ri(data_v1)
    # mat2 = numpy2ri(data_v2)
    # Pass the NumPy array to R
    robjects.r.assign("mat1", data_v1)
    robjects.r.assign("mat2", data_v2)


    # Load and execute the R script
    with open("variable_selection_JPTA.R", "r") as file:
        r_code = file.read()
    robjects.r(r_code)

    data_selected = []
    for view in range(len(data)):
        top = top_features[view]
        indices = pd.read_csv(f'sub0view_{view + 1}.csv')['index'].tolist()[0:top]
        data_selected.append(torch.index_select(data[view], 1, torch.LongTensor(indices)))
    for i in range(len(data_selected)):
        print(data_selected[i].shape)
    return data_selected


def variable_selection_LMM(data, y, top_features = 'default', nboots='NA', n_epoch_boot='NA', structuresNN = 'NA', structureGRU = 'NA', LR='NA', n_epochs='NA', split_seed = 1234):
    '''
    -data- : Input multiview data in a list where the (d-1)-th item of the list corresponds to the d-th view and has that
          view's data in the form (N,p_d,t_d) if longitudinal and (N, p_d, t_d) if cross-sectional.
    -y- : labels
    -top_features- : list of length equal to the number of views where each element tells how many variables have to be
                  selected from a given view. If top_features == 'default', top 25% variables are selected from each view.
    -split_seed- : Seed used for splitting the data into train, test, valid splits.

    Other variables: Not relevant for this method.

    ---Output---
    -data_selected-: Variable-selected multiview data in a list where the (d-1)-th item of the list corresponds to the
                   d-th view and has that view's data in the form (N,top_features[d-1],t_d) if longitudinal and
                   (N, top_features[d-1]) if cross-sectional, where top_features[d-1] are the number of selected
                   variables from view d.
    '''

    '''Split data into train, test and valid because LMM can only be applied on train data
       because it considers class labels while selecting variables.'''
    data_train = []
    data_test = []
    data_valid = []
    for i in range(len(data)):
        X_train, X_test, y_train, y_test = train_test_split(data[i], y, stratify=y,
                                                            test_size=0.20, random_state=split_seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train,
                                                              test_size=0.25, random_state=split_seed)
        data_train.append(X_train)
        data_test.append(X_test)
        data_valid.append(X_valid)

    if top_features == 'default':
        top_features = []  # How many variables to select from each view
        for view in range(len(data)):
            top = math.ceil(25 * data[view].shape[1] / 100)
            top_features.append(top)

    import rpy2.robjects as robjects
    from rpy2.robjects import StrVector
    from rpy2.robjects import numpy2ri

    # Pass Python variables to R using the robjects.r() function
    numpy2ri.activate()
    data_selected = []
    for d in range(len(data)):

        # Convert the PyTorch tensor to a NumPy array
        data_v = data[d].numpy()
        y_v = y.numpy()

        # Pass information from python to R
        robjects.r(f'topfeat <- {top_features[d]}')
        robjects.r(f'd <- {d+1}')
        robjects.r.assign("mat1", data_v)
        robjects.r.assign("y", y_v)

        # Load and execute the R script
        with open("variable_selection_LMM.R", "r") as file:
            r_code = file.read()
        robjects.r(r_code)


        top = top_features[d]
        indices = pd.read_csv(f'sub0view_{d + 1}.csv').iloc[:,0].tolist()[0:top]
        data_selected.append(torch.index_select(data[d], 1, torch.LongTensor(indices)))
    for i in range(len(data_selected)):
        print(data_selected[i].shape)
    return data_selected


def variable_selection(data, y, method = 'nothing', top_features = 'default', nboots=50, n_epoch_boot=50, structuresNN = None, structureGRU = None, LR=0.001, n_epochs=50, split_seed = 1234):
    '''
    ---Returns variables selected for any choice of the following methods: LMM, DGB, JPTA and nothing---
    -data- : Input multiview data in a list where the (d-1)-th item of the list corresponds to the d-th view and has that
          view's data in the form (N,p_d,t_d) if longitudinal and (N, p_d) if cross-sectional.
    -y- : labels
    -method- : Which method to use for variable selection (the four options are: 'nothing', 'LMM', 'DGB', 'JPTA')
    -top_features- : list (of length equal to the number of views) where each element tells how many variables have to be
                  selected from a given view. If top_features == 'default', top 25% variables are selected from each
                  view.

    -----Variables relevant for LMM and DGB only (and not for JPTA)-----
    -split_seed- : Seed used for splitting the data into train, test, valid splits.

    -----Variable relevant for DGB only (not LMM or JPTA)-----
    -structuresNN- : Structure of neural network for each view
                     (Example: structuresNN = [[200, 20, 20], [200, 20, 20]] means for both the views, the number of
                    neurons in the first, second and third layers are 200, 20, 20, respectively)
    -structureGRU- : Structure of GRU (if one or more converted data is/are longitudinal)
                    (Example: structureGRU = [3, 100, 20] for 3 layers, 100 dimensional hidden unit and 20 dimensional output)
    -LR- : Learning Rate
    -nboots- : Number of bootstraps
    -n_epoch_boot- : Number of epochs for baseline training of each epoch
    -n_epochs- : Number of bootstraps for final training using 10% of the selected variables.


    -----Output-----
    -data_selected- : Variable-selected multiview data in a list where the (d-1)-th item of the list corresponds to the
                   d-th view and has that view's data in the form (N,top_features[d-1],t_d) if longitudinal and
                   (N, top_features[d-1]) if cross-sectional, where top_features[d-1] are the number of selected
                   variables from view d.
    -sub0view_d.csv- : If method == 'LMM', 'DGB', 'JPTA', this function creates a csv file (in the current directory) for
                    all views d in [1:D], which contains the indices of variables selected from view d.
    '''

    if method == 'nothing':
        return data
    elif method == 'DGB':
        return variable_selection_DGB(data, y, top_features=top_features, nboots=nboots, n_epoch_boot=n_epoch_boot,
                                      structuresNN=structuresNN, structureGRU=structureGRU, LR=LR, n_epochs=n_epochs,
                                      split_seed=split_seed)
    elif method == 'JPTA':
        return variable_selection_JPTA(data, y, top_features=top_features, nboots=nboots, n_epoch_boot=n_epoch_boot,
                                       structuresNN=structuresNN, structureGRU=structureGRU, LR=LR, n_epochs=n_epochs,
                                       split_seed=split_seed)
    elif method == 'LMM':
        return variable_selection_LMM(data, y, top_features=top_features, nboots=nboots, n_epoch_boot=n_epoch_boot,
                               structuresNN=structuresNN, structureGRU=structureGRU, LR=LR, n_epochs=n_epochs,
                               split_seed=split_seed)
    else:
        print('Invalid method for variable selection. Please check again. Exiting code!')
        exit()



