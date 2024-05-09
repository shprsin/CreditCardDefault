import os
import pandas as pd
import sys
from src.pipelines.logger import logging
from src.pipelines.exception import CustomException
from sklearn.model_selection import train_test_split

from src.components.data_ingestion import DataIngestion 
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer 


if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    print(train_path,test_path)
    data_transformation=DataTransformation()
    train_arr,test_arr,obj_path=data_transformation.initiate_data_transformation(train_path,test_path)
    model_trainer=ModelTrainer()


    model_trainer.initiate_model_training(train_arr,test_arr)