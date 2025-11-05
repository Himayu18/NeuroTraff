import os
import sys
import pandas as pd
from dataclasses import dataclass
from pymongo import MongoClient
from src.exception import CustomException
from src.logger import logging
from datetime import datetime, timedelta

@dataclass
class DataIngestionConfig:
    data_dir = os.path.join('artifacts', 'RawData')
    raw_data_path: str = os.path.join(data_dir, 'raw_data.csv')
    train_data_path: str = os.path.join(data_dir, 'train_data.csv')
    test_data_path: str = os.path.join(data_dir, 'test_data.csv')
    new_data_path: str = os.path.join(data_dir, 'new_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_config = DataIngestionConfig()

    def fetch_data(self):
        try:
            logging.info("Creating instance to fetch data from MongoDB Atlas...")
            client = MongoClient("mongodb+srv://<your_connection_string>")
            database = client["thane_traffic"]
            collection = database["traffic_flow_data"]

            logging.info("Fetching data from Atlas...")
            document = list(collection.find().limit(123621))
            df = pd.DataFrame(document)
            os.makedirs(self.data_config.data_dir, exist_ok=True)
            df.to_csv(self.data_config.raw_data_path, index=False, header=True)
        
        except Exception as e:
            raise CustomException(sys, e)

    def split_into_train(self):
        try:
            logging.info("Reading Raw Data...")
            df = pd.read_csv(self.data_config.raw_data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            training_time = pd.to_datetime('2025-10-18 17:52:04.288')

            logging.info("Splitting into Train data...")
            training_phase = df[df['timestamp'] <= training_time]
            os.makedirs(self.data_config.data_dir, exist_ok=True)
            training_phase.to_csv(self.data_config.train_data_path, index=False, header=True)
        
        except Exception as e:
            raise CustomException(sys, e)

    def split_into_test(self):
        try:
            logging.info("Reading Raw Data...")
            df = pd.read_csv(self.data_config.raw_data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            testing_time = pd.to_datetime('2025-10-18 17:52:04.288')

            logging.info("Splitting into Test data...")
            testing_phase = df[df['timestamp'] > testing_time]
            os.makedirs(self.data_config.data_dir, exist_ok=True)
            testing_phase.to_csv(self.data_config.test_data_path, index=False, header=True)
        
        except Exception as e:
            raise CustomException(sys, e)

    def fetch_new_data(self):
        try:
            logging.info("Fetching new data...")
            client = MongoClient("mongodb+srv://<your_connection_string>")
            database = client["thane_traffic"]
            collection = database["traffic_flow_data"]

            end_date = datetime.combine(datetime.today(), datetime.min.time())
            start_date = end_date - timedelta(days=3)

            query = {"timestamp": {"$gte": start_date, "$lt": end_date}}
            document = list(collection.find(query))
            logging.info("Fetched data successfully from last 3 days")

            df = pd.DataFrame(document)
            os.makedirs(self.data_config.data_dir, exist_ok=True)
            df.to_csv(self.data_config.new_data_path, index=False, header=True)
        
        except Exception as e:
            raise CustomException(sys, e)
