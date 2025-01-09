
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import config
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from parameter_parser import parameter_parser

# Renaming the class to FlowClassifier
class FlowClassifier:
    def __init__(self, data_path, model):
        self.logger = logging.getLogger('FlowClassifier')
        self.data_path = data_path
        self.model = model
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.logger.info("loading data from " + self.data_path + " and model is " + self.model.__class__.__name__)

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.dropna()

    def preprocess_data(self):
        features = ["srcport", "dstport", "proto", "td", "pkt", "byt"]
        target = "type"

        # Encoding the 'proto' categorical field
        label_encoder = LabelEncoder()
        self.data['proto'] = label_encoder.fit_transform(self.data['proto'])

        #ZL: split randomly
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.X_train = train_data[features]
        self.y_train = train_data[target]
        self.X_test = test_data[features]
        self.y_test = test_data[target]

        '''
        #ZL: split by time, netshare approach, for some unknown reason, this split has low accuracy on raw data
        self.data = self.data.sort_values(by="ts")
        X = self.data[features]
        y = self.data[target]


        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        '''

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self):
        self.logger.info("train model")
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.logger.info("evaluate model")
        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)    

# Test driver function
def main(args):
    os.chdir("../../")
    
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    file_prefix = args['dataset_name']

    datasets = {
        file_prefix + 'Raw': config.RAW_DATA_PATH + file_prefix + '.csv',
        file_prefix + 'Syn': config.SYNTHESIZED_RECORDS_PATH + ('_'.join((args['dataset_name'], str(args['epsilon']))) + '.csv')
    }

    models = [
        DecisionTreeClassifier(random_state=42),
        LogisticRegression(max_iter=1000, random_state=42),
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
        MLPClassifier(max_iter=1000, random_state=42)
    ]

    results = []

    for dataset_name, data_path in datasets.items():
        for model in models:
            classifier = FlowClassifier(data_path, model)
            classifier.load_data()
            classifier.preprocess_data()
            classifier.train_model()
            accuracy = classifier.evaluate_model()
            results.append({
                'Model': model.__class__.__name__,
                'Dataset': dataset_name,
                'Accuracy': accuracy
            })
            print(f'Model: {model.__class__.__name__}, Dataset: {dataset_name}, Accuracy: {accuracy}')
    
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    args = parameter_parser()
    
    main(args)

