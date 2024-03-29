from src.CampusPlacement.components.data_ingestion import DataIngestion
from src.CampusPlacement.components.data_transformation import DataTransformation
from src.CampusPlacement.components.model_trainer import ModelTrainer
import os
import sys
from src.CampusPlacement.logger import logging
from src.CampusPlacement.exception import customexception
import pandas as pd

obj=DataIngestion()
train_data_path,test_data_path= obj.initiate_data_ingestion()

data_transform= DataTransformation()
train_arr, test_arr= data_transform.initiate_data_transformation(train_data_path,test_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)