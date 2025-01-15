import pickle
import os

import method.privsyn.config as config


class DataStore:
    def __init__(self, args):
        self.args = args
        self.parent_dir = 'exp/'+ str(self.args['dataset'])+ '/' + str(self.args['method'])+'/'+str(self.args['epsilon']) + '_' + str(self.args['num_preprocess']) + '_' + str(self.args['rare_threshold'])
        self.determine_data_path()
        self.generate_folder()
    
    def determine_data_path(self):
        synthesized_records_name = '_'.join((self.args['dataset_name'], 'records'))
        marginal_name = '_'.join((self.args['dataset_name'], 'marginal'))
        
        self.synthesized_records_file = os.path.join(self.parent_dir, synthesized_records_name)
        self.marginal_file = os.path.join(self.parent_dir, marginal_name)
        
    def generate_folder(self):
        for path in config.ALL_PATH:
            if not os.path.exists(path):
                os.makedirs(path)
    
    def load_processed_data(self):
        return pickle.load(open(config.PROCESSED_DATA_PATH + self.args['dataset_name'], 'rb'))
    
    def save_synthesized_records(self, records, save_path = None):
        if save_path is None:
            pickle.dump(records, open(self.synthesized_records_file, 'wb'))
        else:
            pickle.dump(records, open(os.path.join(save_path, '_'.join( (self.args['dataset_name'], str(self.args['epsilon'])) )), 'wb'))
        
    def save_marginal(self, marginals):
        pickle.dump(marginals, open(self.marginal_file, 'wb'))
    
    def load_marginal(self):
        return pickle.load(open(self.marginal_file, 'rb'))
