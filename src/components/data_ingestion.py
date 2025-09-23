import os
import sys
import pandas as pd
from dataclasses import dataclass
from pymongo import MongoClient
from src.exception import CustomException
from src.logger import logging
from datetime import datetime, timedelta, date

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')
    train_data_path:str = os.path.join('artifacts','train_data.csv')
    test_data_path:str = os.path.join('artifacts','test_data.csv')
    new_data_path:str = os.path.join('artifacts','new_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_config = DataIngestionConfig()

    def fetch_data(self):
        try:
            logging.info("Creating instace to fetch data from mongodb atlas server....")
            client = MongoClient("mongodb+srv://himayudhoke:so34JLowMFfHWGb5@cluster0.lufwdkw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
            database = client["thane_traffic"]
            collection = database["traffic_flow_data"]

            logging.info("Fetching data from atlas cloud server....")
            document = list(collection.find().limit(29446))
            df = pd.DataFrame(document)
            df.to_csv(self.data_config.raw_data_path,index=False,header=True)
        
        except Exception as e:
            raise CustomException(sys,e)    

    def split_into_train(self):
        try:
            logging.info("Reading Raw Data....")
            df = pd.read_csv(r"artifacts\raw_data.csv")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            tranning_time = pd.to_datetime('2025-09-09 17:52:17.764')

            logging.info("splitting into Train data and saving it into artifacts folder...")
            tranning_phase = df[df['timestamp']<=tranning_time]
            return tranning_phase.to_csv(self.data_config.train_data_path,index=False,header=True)
        
        except Exception as e:
            raise CustomException(sys,e)    


    
    def split_into_test(self):
        try:
            logging.info("Reading Raw Data....")
            df = pd.read_csv(r"artifacts\raw_data.csv")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            testing_time = pd.to_datetime('2025-09-09 17:52:17.764')

            logging.info("splitting into Test data and saving it into artifacts folder...")
            testing_phase = df[df['timestamp']>testing_time]
            return testing_phase.to_csv(self.data_config.test_data_path,index=False,header=True)
        except Exception as e:
            raise CustomException(sys,e)
        
    def fetch_new_data(self):
        try:
            logging.info("Fetching new Data ...")
            client = MongoClient("mongodb+srv://himayudhoke:1X5idC51cKy8EntW@cluster0.lufwdkw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
            database = client["thane_traffic"]
            collection = database["traffic_flow_data"]

            start_date = datetime.combine(datetime.today(),datetime.min.time())
            end_date = start_date + timedelta(days=2)

            query = {"timestamp":{"$gte":start_date,"$lt":end_date}}
            document = list(collection.find(query))
            logging.info("Fetch data sucessfully...")

            df = pd.DataFrame(document)
            df.to_csv(self.data_config.new_data_path,index=False,header=True)


        except Exception as e:
            raise CustomException(e,sys)