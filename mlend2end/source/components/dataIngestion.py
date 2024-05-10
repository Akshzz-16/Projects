import os 
import sys
from source.exception import CustomException
from source.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from source.components.dataTransform import dataTransform
from source.components.dataTransform import dataTransformConfig
from source.components.modelTrainer import ModelTrainerConfig
from source.components.modelTrainer import ModelTrainer


#this class will handle way to segregate and store data

'''
to declare class variables you use __init__ constructor but by declaring DECORATORS i.e. @dataclass you can directly define the class variable for your class.
'''
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"raw.csv")

class dataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def initiate_Data_Ingestion(self):
        logging.info("data initiate method was called")
        try:
            rawdf = pd.read_csv('source/notebook/data/stud.csv')
            logging.info('dataset was imported')

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path),exist_ok=True)

            rawdf.to_csv(self.ingestionConfig.raw_data_path, index=False, header=True)

            logging.info('train test split initiated')
            train_Set, test_Set =  train_test_split(rawdf, test_size=0.2, random_state=1)

            train_Set.to_csv(self.ingestionConfig.train_data_path,index=False, header=True)
            
            test_Set.to_csv(self.ingestionConfig.test_data_path,index=False, header=True)

            logging.info('all files created successfully ingestion completed')

            return (
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == '__main__':
    obj = dataIngestion()
    
    train_data,test_data = obj.initiate_Data_Ingestion()

    dataTransformation=dataTransform()
    train_arr,test_arr,_=dataTransformation.initiate_data_transformation(train_data,test_data)

    ModelTrainer = ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))