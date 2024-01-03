import os
import sys
import pandas as pd
import numpy as np

from src.CampusPlacement.logger import logging
from src.CampusPlacement.exception import customexception
from dataclasses import dataclass
from src.CampusPlacement.utils.utils import save_obj
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from src.CampusPlacement.utils.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path= os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
    def initiate_model_training(self,train_arr,test_arr):
        try:
            # Spliting Train and test data
            x_train, y_train, x_test, y_test= (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models= {
                'Logistic Regression':LogisticRegression(),
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'KNeighbors Classifier': KNeighborsClassifier()
            }
            model_report:dict= evaluate_model(x_train,y_train,x_test,y_test,models)
            print("Model Report: ", model_report)
            print("="* 148,"\n")
            logging.info(f"Model Reports {model_report}")

            # Findong out the best model
            best_model_score= max(sorted(model_report.values()))
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]
            best_model_score= round(best_model_score*100,2)
            
            print(f"Best model found, Model Name: {best_model_name} \nAccuracy score: {best_model_score}")
            logging.info(f"Best model found Model Name: {best_model_name} and Accuracy score is {best_model_score}")

            # Saving pkl file
            save_obj(
                file_path= self.model_trainer_config.trained_model_path,
                obj= best_model
            )


        except Exception as e:
            logging.info("Error occured in Model Training")
            raise customexception(e,sys)