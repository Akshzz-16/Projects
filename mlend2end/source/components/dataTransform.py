'''
THIS FILE IS USED TO: 

feature engineering 
data cleaning 
convert my categorical features to numerical 
'''
import sys 
import os 
from dataclasses import dataclass

import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object

@dataclass
class dataTransformConfig:
    preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")

class dataTransform:
    def __init__(self):
        self.dataTransformConfig= dataTransformConfig()

    def get_data_tranformer_obj(self):
        try:
            numericalFeatures = ['reading_score', 'writing_score']
            categoricalFeatures = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                 steps=[
                     ("imputer",SimpleImputer(strategy="median")),
                     ("scaler",StandardScaler(with_mean=False))
                 ]
            )

            logging.info('standard scaling completed')
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info('categorical columns encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numericalFeatures),
                    ('cat_pipeline', cat_pipeline, categoricalFeatures)

                ]
            )

            logging.info(print(preprocessor))

            logging.info('preprocessor was created successfully')

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            trainDf= pd.read_csv(train_path)
            testDf = pd.read_csv(test_path)

            logging.info(' read the train and test ')
            logging.info('obtaining preprocessing object')
            
            preprocessorObj = self.get_data_tranformer_obj()

            targetColumn = 'math_score'
            # numericalFeatures = ['reading_score', 'writing_score']

            input_feature_train_df = trainDf.drop(columns=[targetColumn],axis=1)
            input_feature_test_df = testDf.drop(columns=[targetColumn],axis=1)

            target_feature_train_df = trainDf[(targetColumn)]
            target_feature_test_df = testDf[(targetColumn)]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr= preprocessorObj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessorObj.fit_transform(input_feature_test_df)
            

            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]

            logging.info(train_arr)

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info(f"Saving preprocessing object.")

            save_object(
                file_path = self.dataTransformConfig.preprocessor_obj_file_path,obj = preprocessorObj
            )

            return(
                train_arr,
                test_arr,
                self.dataTransformConfig.preprocessor_obj_file_path
            )

            
        except Exception as e :
            raise CustomException(e,sys)

