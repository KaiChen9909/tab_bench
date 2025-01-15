### this is for training a single generator for all labels ###
""" with the analysis of """
### weights = weights + N(0, sigma**2*(sqrt(2)/N)**2)
### columns of mean embedding = raw + N(0, sigma**2*(2/N)**2)
import sys
target_path="./"
sys.path.append(target_path)

import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import method.DP_MERF.util as util
import random
import socket
import argparse
import sys
import os
import pandas as pd
import seaborn as sns
import json
import math
# %matplotlib inline

from preprocess_common.preprocess import * 
from collections import Counter
from sklearn.model_selection import ParameterGrid
from autodp import privacy_calibrator
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
from method.DP_MERF.sample import *


import warnings
warnings.filterwarnings('ignore')

def rdp_rho(epsilon, delta):
    return (np.sqrt(math.log(1/delta) + epsilon) - np.sqrt(math.log(1/delta)))**2


############################### kernels to use ###############################
""" we use the random fourier feature representation for Gaussian kernel """

def RFF_Gauss(n_features, X, W, device='cuda:0'):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""

    W = torch.Tensor(W).to(device)
    X = X.to(device)

    XWT = torch.mm(X, torch.t(W)).to(device)
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features])).to(device)
    return Z


""" we use a weighted polynomial kernel for labels """

def Feature_labels(labels, weights, device):

    weights = torch.Tensor(weights)
    weights = weights.to(device)

    labels = labels.to(device)

    weighted_labels_feature = labels/weights

    return weighted_labels_feature


############################### end of kernels ###############################

############################### generative models to use ###############################
""" two types of generative models depending on the type of features in a given dataset """

class Generative_Model_homogeneous_data(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dataset):
        super(Generative_Model_homogeneous_data, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)

        self.dataset = dataset


    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)

        # if self.dataset=='credit':
        #     all_pos = self.relu(output[:,-1])
        #     output = torch.cat((output[:,:-1], all_pos[:,None]),1)

        return output


class Generative_Model_heterogeneous_data(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, num_categorical_inputs, num_numerical_inputs):
        super(Generative_Model_heterogeneous_data, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.num_numerical_inputs = num_numerical_inputs
        self.num_categorical_inputs = num_categorical_inputs

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)

        output_numerical = self.relu(output[:, 0:self.num_numerical_inputs])  # these numerical values are non-negative
        output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
        output_combined = torch.cat((output_numerical, output_categorical), 1)

        return output_combined

############################### end of generative models ###############################



def undersample(raw_input_features, raw_labels, undersampled_rate):
    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 0
    idx_positive_label = raw_labels == 1

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 10 percent of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = undersampled_rate  # 0.4
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    return feature_selected, label_selected
    

def add_default_params(args):
    args.n_features_arg = 2000
    args.mini_batch_size_arg = 0.05 
    args.how_many_epochs_arg = 1000
    return args


######################################## beginning of main script ############################################################################

def merf_main(
        args,
        df, 
        domain,
        rho, 
        parent_dir = None,
        seed_number = 0,
        is_priv_arg = True,
        **kwargs
    ):
    np.random.seed(seed_number)

    args = add_default_params(args)

    device = args.device
    n_features_arg = args.n_features_arg
    mini_batch_size_arg = args.mini_batch_size_arg
    how_many_epochs_arg = args.how_many_epochs_arg
    total_rho = rho

    dataset = args.dataset
    homogeneous_datasets = []
    heterogeneous_datasets = [f'{dataset}'] # by default dataset is heterogeneour
    is_private = is_priv_arg 

    ############################### data loading ##################################
    if len(domain['X_num']) == 0: 
        numerical_X_train = None 
        categorical_X_train = df['X_cat']
        num_encoder = None
        y_train = df['y']
    elif len(domain['X_cat']) == 0:
        numerical_X_train = df['X_num']
        categorical_X_train = None
        num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,np.pi))
        numerical_X_train = num_encoder.fit_transform(numerical_X_train)
        y_train = df['y']
    else:
        numerical_X_train = df['X_num']
        categorical_X_train = df['X_cat']
        num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,np.pi))
        numerical_X_train = num_encoder.fit_transform(numerical_X_train)
        y_train = df['y']

    if numerical_X_train is None:
        X_train = categorical_X_train
    elif categorical_X_train is None: 
        X_train = numerical_X_train 
    else:
        X_train = np.concatenate((numerical_X_train, categorical_X_train), axis=1)
    
    # with open(f'data/{dataset}/info.json', 'r') as file:
    #     data_info = json.load(file)
    # n_classes = data_info['n_classes']

    num_numerical_inputs = numerical_X_train.shape[1] if numerical_X_train is not None else 0 
    num_categorical_inputs = categorical_X_train.shape[1] if categorical_X_train is not None else 0 

    sample_num = y_train.shape[0]

    # specify heterogeneous dataset or not
    dataset = 'your dataset'
    heterogeneous_datasets = ['your dataset']


    ###########################################################################
    # PREPARING GENERATOR

    # one-hot encoding of labels.
    n, input_dim = X_train.shape
    y_onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train = np.expand_dims(y_train, 1)
    true_labels = y_onehot_encoder.fit_transform(y_train)
    n_classes = true_labels.shape[1]


    ######################################
    # MODEL

    # model specifics
    mini_batch_size = int(np.round(mini_batch_size_arg * n))
    print("minibatch: ", mini_batch_size)
    input_size = 10 + 1
    hidden_size_1 = 4 * input_dim
    hidden_size_2 = 2 * input_dim
    output_size = input_dim

    if dataset in homogeneous_datasets:
        model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                        hidden_size_2=hidden_size_2,
                                                        output_size=output_size, dataset=dataset).to(device)

    elif dataset in heterogeneous_datasets:
        model_init = {
                    'input_size': input_size, 'hidden_size_1': hidden_size_1,
                    'hidden_size_2': hidden_size_2,
                    'output_size': output_size,
                    'num_categorical_inputs': num_categorical_inputs,
                    'num_numerical_inputs': num_numerical_inputs
            }
        model = Generative_Model_heterogeneous_data(**model_init).to(device)
    else:
        print('sorry, please enter the name of your dataset either in homogeneous_dataset or heterogeneous_dataset list ')

    # define details for training
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    how_many_epochs = how_many_epochs_arg
    how_many_iter = 1 #np.int(n / mini_batch_size)
    training_loss_per_epoch = np.zeros(how_many_epochs)


    ##########################################################################


    """ specifying random fourier features """

    idx_rp = np.random.permutation(n)

    if dataset=='census': # some columns of census data have many zeros, so we need more datapoints to get meaningful length scales
        num_data_pt_to_discard = 100
    else:
        num_data_pt_to_discard = 10

    idx_to_discard = idx_rp[0:num_data_pt_to_discard]
    idx_to_keep = idx_rp[num_data_pt_to_discard:]

    if dataset=='census':

        sigma_array = np.zeros(num_numerical_inputs)
        for i in np.arange(0, num_numerical_inputs):
            med = util.meddistance(np.expand_dims(X_train[idx_to_discard, i], 1))
            sigma_array[i] = med


        print('we will use separate frequencies for each column of numerical features')
        sigma2 = sigma_array**2
        sigma2[sigma2==0] = 1.0
        # sigma2[sigma2>500] = 500
        #print('sigma values are ', sigma2)
        # sigma2 = np.mean(sigma2)

    elif dataset=='credit':

        # large value at the last column

        med = util.meddistance(X_train[idx_to_discard, 0:-1])
        med_last = util.meddistance(np.expand_dims(X_train[idx_to_discard, -1],1))
        sigma_array = np.concatenate((med*np.ones(input_dim-1), [med_last]))

        sigma2 = sigma_array**2
        sigma2[sigma2==0] = 1.0

        #print('sigma values are ', sigma2)

    else:

        if dataset in heterogeneous_datasets:
            med = util.meddistance(X_train[idx_to_discard, 0:num_numerical_inputs])
            # med = util.meddistance(X_train[idx_to_discard, ])
        else:
            med = util.meddistance(X_train[idx_to_discard, ])

        sigma2 = med ** 2

    X_train = X_train[idx_to_keep,:]
    true_labels = true_labels[idx_to_keep,:]
    n = X_train.shape[0]
    print('total number of datapoints in the training data is', n)

    # random Fourier features
    n_features = n_features_arg
    draws = n_features // 2

    # random fourier features for numerical inputs only
    if dataset in heterogeneous_datasets:
        W_freq = np.random.randn(draws, num_numerical_inputs) / np.sqrt(sigma2)
        # W_freq = np.random.randn(draws, num_numerical_inputs + num_categorical_inputs) / np.sqrt(sigma2)
    else:
        W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)

    """ specifying ratios of data to generate depending on the class lables """
    unnormalized_weights = np.sum(true_labels,0)
    weights = unnormalized_weights/np.sum(unnormalized_weights)
    print('\nweights with no privatization are', weights, '\n'  )

    ####################################################
    # Privatising quantities if necessary

    """ privatizing weights """
    if is_private:
        print("private")
        # k = n_classes + 1   # this dp analysis has been updated
        privacy_param_label = np.sqrt(1/(2 * 0.1 * total_rho))
        print(f'eps,delta = ({0.1 * total_rho}) ==> Noise level sigma=', privacy_param_label)

        sensitivity_for_weights = 1/n 
        noise_std_for_weights = privacy_param_label * sensitivity_for_weights
        weights = weights + np.random.randn(weights.shape[0])*noise_std_for_weights
        weights[weights < 0] = 1e-3 # post-processing so that we don't have negative weights.
        print('weights after privatization are', weights)

    """ computing mean embedding of subsampled true data """
    if dataset in homogeneous_datasets:

        emb1_input_features = RFF_Gauss(n_features, torch.Tensor(X_train), W_freq, device)
        emb1_labels = Feature_labels(torch.Tensor(true_labels), weights, device)
        outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
        mean_emb1 = torch.mean(outer_emb1, 0)

    else:  # heterogeneous data

        numerical_input_data = X_train[:, 0:num_numerical_inputs]
        emb1_numerical = (RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq, device)).to(device)

        categorical_input_data = X_train[:, num_numerical_inputs:]

        emb1_categorical = (torch.Tensor(categorical_input_data) / np.sqrt(num_categorical_inputs)).to(device)

        emb1_input_features = torch.cat((emb1_numerical, emb1_categorical), 1)

        # emb1_input_features = (RFF_Gauss(n_features, torch.Tensor(X_train), W_freq, device)).to(device)

        emb1_labels = Feature_labels(torch.Tensor(true_labels), weights, device)
        outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
        mean_emb1 = torch.mean(outer_emb1, 0)


    """ privatizing each column of mean embedding """
    if is_private:
        if dataset in heterogeneous_datasets:
            sensitivity = 2 / n
        else:
            sensitivity = 1 / n
        privacy_param_embedding = np.sqrt(1/(2 * 0.9 * total_rho))
        noise_std_for_privacy = privacy_param_embedding * sensitivity

        # make sure add noise after rescaling
        weights_torch = torch.Tensor(weights)
        weights_torch = weights_torch.to(device)

        rescaled_mean_emb = weights_torch*mean_emb1
        noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
        noise = noise.to(device)

        rescaled_mean_emb = rescaled_mean_emb + noise

        mean_emb1 = rescaled_mean_emb/weights_torch # rescaling back\

    # End of Privatising quantities if necessary
    ####################################################

    ##################################################################################################################
    # TRAINING THE GENERATOR
    if args.test:
        tvd_tracker = pd.DataFrame(columns=['epoch', 'num-num tvd', 'cat-cat tvd', 'num-cat tvd'])

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):

            """ computing mean embedding of generated data """
            # zero the parameter gradients
            optimizer.zero_grad()

            if dataset in homogeneous_datasets: # In our case, if a dataset is homogeneous, then it is a binary dataset.

                label_input = (1 * (torch.rand((mini_batch_size)) < weights[1])).type(torch.FloatTensor)
                label_input = label_input.to(device)
                feature_input = torch.randn((mini_batch_size, input_size-1)).to(device)
                input_to_model = torch.cat((feature_input, label_input[:,None]), 1)

                #we feed noise + label (like in cond-gan) as input
                outputs = model(input_to_model)

                """ computing mean embedding of generated samples """
                emb2_input_features = RFF_Gauss(n_features, outputs, W_freq, device)

                label_input_t = torch.zeros((mini_batch_size, n_classes))
                idx_1 = (label_input == 1.).nonzero()[:,0]
                idx_0 = (label_input == 0.).nonzero()[:,0]
                label_input_t[idx_1, 1] = 1.
                label_input_t[idx_0, 0] = 1.

                emb2_labels = Feature_labels(label_input_t, weights, device)
                outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
                mean_emb2 = torch.mean(outer_emb2, 0)

            else:  # heterogeneous data

                # (1) generate labels
                label_input = torch.multinomial(torch.Tensor([weights]), mini_batch_size, replacement=True).type(torch.FloatTensor)
                label_input=torch.cat((label_input, torch.arange(len(weights), out=torch.FloatTensor()).unsqueeze(0)),1) #to avoid no labels
                label_input = label_input.transpose_(0,1)
                label_input = label_input.to(device)

                # (2) generate corresponding features
                feature_input = torch.randn((mini_batch_size+len(weights), input_size-1)).to(device)
                input_to_model = torch.cat((feature_input, label_input), 1)
                outputs = model(input_to_model)

                # (3) compute the embeddings of those
                numerical_samps = outputs[:, 0:num_numerical_inputs] #[4553,6]
                emb2_numerical = RFF_Gauss(n_features, numerical_samps, W_freq, device) #W_freq [n_features/2,6], n_features=10000

                categorical_samps = outputs[:, num_numerical_inputs:] #[4553,8]

                emb2_categorical = categorical_samps /(torch.sqrt(torch.Tensor([num_categorical_inputs]))).to(device) # 8

                emb2_input_features = torch.cat((emb2_numerical, emb2_categorical), 1)

                # emb2_input_features = RFF_Gauss(n_features, outputs, W_freq, device)

                generated_labels = y_onehot_encoder.fit_transform(label_input.cpu().detach().numpy()) #[1008]
                emb2_labels = Feature_labels(torch.Tensor(generated_labels), weights, device)
                outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
                mean_emb2 = torch.mean(outer_emb2, 0)

            loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 100 == 0:
            print('epoch # and running loss are ', [epoch, running_loss])
            training_loss_per_epoch[epoch] = running_loss 
        if args.test:
            if (epoch % 50 == 0) or (epoch == how_many_epochs - 1):
                tvd1, tvd2,tvd3 = tvd_summary_simple(args, numerical_samps, categorical_samps, label_input, num_encoder, kwargs.get('preprocesser', None))
                tvd_tracker.loc[len(tvd_tracker)] = [epoch+1,tvd1,tvd2,tvd3]

    
    torch.save(model.state_dict(), os.path.join(parent_dir, 'merf_model.pt')) 
    
    with open(os.path.join(parent_dir, 'model_init.json'), 'w', encoding = 'utf-8') as file: 
        json.dump(model_init, file)

    if args.test:
        tvd_tracker.to_csv(os.path.join(parent_dir, 'tvd_track.csv'), header=True)

    generator = merf_generator(model, model_init, weights, num_encoder, num_numerical_inputs)
    return {"merf_generator": generator}
