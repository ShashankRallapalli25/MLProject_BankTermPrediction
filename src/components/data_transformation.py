import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os
from src.utils import save_object
from imblearn import FunctionSampler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
        '''
            
        #numerical_features = ["reading_score","writing_score"]
        #idx = [0,5,9,11,12,13,14]
        #numerical_features  = data[data.columns[idx]]
        numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        categorical_features = ['job', 'marital', 'education', 'default', 
            'housing', 'loan', 'contact', 'month', 'poutcome', 'Public Holiday']
        outlier_column = ['balance']

        num_pipeline = Pipeline (
                steps = [
                    ("imputer", SimpleImputer( strategy= "median" )),
                    ("scaler", MinMaxScaler(with_mean=False))
                ]
            )
        cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer (strategy= "most_frequent")),
                    ("encoder", OneHotEncoder())
                ]
            )

        preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline,categorical_features)
                ]
            )

        return preprocessor
def initiate_data_transformation(self, train_path, test_path):

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformer_object()

            target_column = "math_score"

            #idx = [0,5,9,11,12,13,14]
            #numerical_features  = train_df.columns[idx]
            numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
            categorical_features = ['job', 'marital', 'education', 'default', 
            'housing', 'loan', 'contact', 'month', 'poutcome', 'Public Holiday']
            input_train_df = train_df.drop(columns= [target_column],axis=1)
            target_train_df = train_df[target_column]
            input_test_df = test_df.drop(columns= [target_column],axis=1)
            target_test_df = test_df[target_column]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_test_df) 

            train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]       

            save_object(
                filepath = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )