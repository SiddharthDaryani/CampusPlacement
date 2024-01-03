import pandas as pd
import os
import sys
import numpy as np
from dataclasses import dataclass
from src.CampusPlacement.exception import customexception
from src.CampusPlacement.logger import logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.CampusPlacement.utils.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation(self):
        logging.info("Data Transformation initiated!!")
        try:
            categorical_columns= ['gender', 'ssc_b', 'hsc_b','hsc_s','degree_t','workex','specialisation']
            numerical_columns= ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']

            gender_category= ['M','F']
            ssc_b_category= ['Others', 'Central']
            hsc_b_category= ['Others', 'Central']
            hsc_s_category= ['Commerce', 'Science', 'Arts']
            degree_t_category= ['Sci&Tech', 'Comm&Mgmt', 'Others']
            workex_category= ['No', 'Yes']
            specialisation_category= ['Mkt&HR', 'Mkt&Fin']

            # numerical column pipeline
            num_pipeline= Pipeline(
                steps= [
                ('scaler', StandardScaler(with_mean= False))
                ]
            )

            # categorical pipeline
            cat_pipeline= Pipeline(
                steps=[
                ('labelencoder', OneHotEncoder(categories= [gender_category,ssc_b_category,hsc_b_category,hsc_s_category,degree_t_category,
                                                           workex_category,specialisation_category])),
                ('scaler',StandardScaler(with_mean= False))
                ]
            )

            preprocessor= ColumnTransformer([
                ('num_pipeline',num_pipeline, numerical_columns),
                ('cat_pipeline',cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor


            



        except Exception as e:
            logging.info("Error occured in initiate_data_transformation")
            raise customexception(e,sys)

    def initiate_data_transformation(self,train_path, test_path):
        
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info(f'Train Dataframe Head: \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n {test_df.head().to_string()}')

            preprocessing_obj= self.get_data_transformation()

            target_column_name= 'status'
            drop_columns= [target_column_name, 'status']

            input_feature_train_df=train_df.drop(columns= drop_columns,axis= 1)
            target_feature_train_df= train_df[target_column_name]
            input_feature_test_df= test_df.drop(columns=drop_columns,axis= 1)
            target_feature_test_df= test_df[target_column_name]

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            logging.info ("Applying preprocessing on training and testing datasets")

            train_arr= np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            logging.info("preprocessor pickle file saved")
            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Error occured in initiate_data_transformation")
            raise customexception(e,sys)