import torch 
import os
import numpy as np 
import pandas as pd
import torch.nn as nn
from evaluator.eval_tvd import tvd_divide


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


class merf_generator():
    def __init__(self, model, model_init, weights, num_encoder, num_numerical_inputs):
        self.model = model 
        self.model_init = model_init
        self.weights = weights
        self.num_encoder = num_encoder
        self.num_numerical_inputs = num_numerical_inputs


    def sample(self, n, preprocessor, parent_dir, device):
        """ draw final data samples """
        input_size = self.model_init['input_size']
        label_input = torch.multinomial(torch.Tensor([self.weights]), n, replacement=True).type(torch.FloatTensor)
        label_input = label_input.transpose_(0, 1)
        label_input = label_input.to(device)

        feature_input = torch.randn((n, input_size - 1)).to(device)
        input_to_model = torch.cat((feature_input, label_input), 1)
        outputs = self.model(input_to_model).cpu().detach().numpy() 

        output_numerical = outputs[:, 0:self.num_numerical_inputs]
        output_categorical = outputs[:, self.num_numerical_inputs:]
        if output_numerical.shape[1] > 0: 
            output_numerical = self.num_encoder.inverse_transform(output_numerical)

        label_input = label_input.cpu().detach().numpy() 

        data = np.hstack([output_numerical, output_categorical, label_input])
        preprocessor.reverse_data(data, parent_dir)



        # output_numerical = outputs[:, 0:num_numerical_inputs].cpu().detach().numpy() 
        # if output_numerical.shape[1] > 0:
        #     if num_encoder is not None: 
        #         output_numerical = num_encoder.inverse_transform(output_numerical)
        #     if save: 
        #         np.save(os.path.join(parent_dir, 'X_num_train.npy'), output_numerical)

        # output_categorical = outputs[:, num_numerical_inputs:].cpu().detach().numpy()
        # if output_categorical.shape[1] > 0: 
        #     output_categorical = cat_encoder.inverse_transform(output_categorical)

        #     for col in range(output_categorical.shape[1]):
        #         idx = (output_categorical[:,col] == None)
        #         idx_true = (output_categorical[:,col] != None)
        #         output_categorical[idx, col] = np.random.choice(output_categorical[idx_true, col], size=sum(idx), replace=True)

        #     if save:
        #         np.save(os.path.join(parent_dir, 'X_cat_train.npy'), output_categorical)
        
        # label_input = label_input.cpu().detach().numpy().reshape(-1)

        # if save:
        #     np.save(os.path.join(parent_dir, 'y_train.npy'), label_input)
        #     print('Sample finished, data store path:', parent_dir)
        #     return None
        # else:
        #     return output_numerical, output_categorical, label_input

# def reverse_cat_rare(X_cat, cat_rare_dict) -> np.ndarray:
#     if (cat_rare_dict is None) or (all(len(sublist) == 0 for sublist in cat_rare_dict)):
#         print('No rare categorical value')
#         return X_cat 
#     else: 
#         for column_idx in range(len(cat_rare_dict)):
#             idx = (X_cat[:, column_idx] == 'cat_rare')
#             if (len(cat_rare_dict[column_idx]) > 0) & (idx.any()):
#                 X_cat[idx, column_idx] = np.random.choice(cat_rare_dict[column_idx], size=sum(idx), replace = True)
#         return X_cat

def tvd_summary(args, numerical_samps, categorical_samps, label_input, num_encoder, preprocessor):
    with torch.no_grad():
        numerical_samps = numerical_samps.cpu().detach().numpy()
        categorical_samps = categorical_samps.cpu().detach().numpy()
        label_input = label_input.cpu().detach().numpy().reshape(-1,1)

        if numerical_samps.shape[1] > 0: 
            numerical_samps = num_encoder.inverse_transform(numerical_samps)
        
        data = np.hstack([numerical_samps, categorical_samps, label_input])
        x_num_fake, x_cat_fake, y_fake = preprocessor.reverse_data(data)
        
        x_num_real = None 
        x_cat_real = None
        if os.path.exists(f'data/{args.dataset}/X_num_test.npy'):
            x_num_real = np.load(f'data/{args.dataset}/X_num_test.npy', allow_pickle=True)
        if os.path.exists(f'data/{args.dataset}/X_cat_test.npy'):
            x_cat_real = np.load(f'data/{args.dataset}/X_cat_test.npy', allow_pickle=True)
        y_real = np.load(f'data/{args.dataset}/y_test.npy', allow_pickle=True)

        num_id = x_num_real.shape[1] if x_num_real is not None else 0
        cat_id = x_cat_real.shape[1] if x_cat_real is not None else 0

        tvd1 = tvd_divide(
            x_num_real, x_cat_real, y_real,
            x_num_fake, x_cat_fake, y_fake,
            dim=2, part = ['num-num']
        )['2way margin'][0]

        tvd2 = tvd_divide(
            x_num_real, x_cat_real, y_real,
            x_num_fake, x_cat_fake, y_fake,
            dim=2, part = ['cat-cat']
        )['2way margin'][0]

        tvd3 = tvd_divide(
            x_num_real, x_cat_real, y_real,
            x_num_fake, x_cat_fake, y_fake,
            dim=2, part = ['num-cat']
        )['2way margin'][0]

        tvd4 = tvd_divide(
            x_num_real, x_cat_real, y_real,
            x_num_fake, x_cat_fake, y_fake,
            dim=2, part = ['num-y']
        )['2way margin'][0]

        tvd5 = tvd_divide(
            x_num_real, x_cat_real, y_real,
            x_num_fake, x_cat_fake, y_fake,
            dim=2, part = ['cat-y']
        )['2way margin'][0]

        tvd = (num_id * tvd4 + cat_id * tvd5 + (num_id-1)*num_id/2 * tvd1 + num_id*cat_id * tvd3 + (cat_id-1)*cat_id/2 * tvd2) / ((num_id + cat_id) * (num_id + cat_id + 1)/2)

        return tvd1, tvd2, tvd3, tvd4, tvd5, tvd 
    

def tvd_summary_simple(args, numerical_samps, categorical_samps, label_input, num_encoder, preprocessor):
    with torch.no_grad():
        numerical_samps = numerical_samps.cpu().detach().numpy()
        categorical_samps = categorical_samps.cpu().detach().numpy()
        label_input = label_input.cpu().detach().numpy().reshape(-1,1)

        if numerical_samps.shape[1] > 0: 
            numerical_samps = num_encoder.inverse_transform(numerical_samps)
        
        data = np.hstack([numerical_samps, categorical_samps, label_input])
        x_num_fake, x_cat_fake, y_fake = preprocessor.reverse_data(data)
        
        x_num_real = None 
        x_cat_real = None
        if os.path.exists(f'data/{args.dataset}/X_num_test.npy'):
            x_num_real = np.load(f'data/{args.dataset}/X_num_test.npy', allow_pickle=True)
        if os.path.exists(f'data/{args.dataset}/X_cat_test.npy'):
            x_cat_real = np.load(f'data/{args.dataset}/X_cat_test.npy', allow_pickle=True)
        y_real = np.load(f'data/{args.dataset}/y_test.npy', allow_pickle=True)

        num_id = x_num_real.shape[1] if x_num_real is not None else 0
        cat_id = x_cat_real.shape[1] if x_cat_real is not None else 0

        tvd1 = tvd_divide(
            x_num_real, x_cat_real, y_real,
            x_num_fake, x_cat_fake, y_fake,
            dim=2, part = ['num-num']
        )['2way margin'][0]

        tvd2 = tvd_divide(
            x_num_real, x_cat_real, y_real,
            x_num_fake, x_cat_fake, y_fake,
            dim=2, part = ['caty-caty']
        )['2way margin'][0]

        tvd3 = tvd_divide(
            x_num_real, x_cat_real, y_real,
            x_num_fake, x_cat_fake, y_fake,
            dim=2, part = ['num-caty']
        )['2way margin'][0]

        return tvd1, tvd2, tvd3
