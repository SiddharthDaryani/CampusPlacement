import os
from src.CampusPlacement.exception import customexception
import pickle
import sys
from sklearn.metrics import accuracy_score
from src.CampusPlacement.logger import logging



def save_obj(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok= True)

        with open (file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models:dict):
    try:
        report= {}
        for i in range(len(models)):
            model= list(models.values())[i]

            # Transforming data
            model.fit(x_train,y_train)

            # Predict the model
            y_pred= model.predict(x_test)

            # Get r2 scores of the model
            test_model_score= accuracy_score(y_test,y_pred)

            report[list(models.keys())[i]]= test_model_score
        
        return report
    except Exception as e:
        raise customexception(e,sys)

def load_obj(file_path):
    try:
        with open (file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Error occured while load_obj")
        raise customexception(e,sys)