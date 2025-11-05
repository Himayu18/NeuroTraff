import pandas as pd
import warnings
import joblib
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer,TransformerConfig
from src.logger import logging
from src.exception import CustomException

try:
#Data Ingestion ----
    logging.info("Intilizing instance for Data Ingestion...")
    data = DataIngestion()

    logging.info("Fetching Raw Data From MongoDB Atlas Server...")
    data.fetch_data()

    logging.info("Spliting raw data into Trainning Data...")
    data.split_into_train()

    logging.info("Splitting raw data into Test Data...")
    data.split_into_test()


#Data Transformation ---
    logging.info("Intilizing instanceses for Data Transformation...")
    transform_config = TransformerConfig()
    data_transformer = DataTransformer()

    logging.info("Loading Raw data for transformation...")
    df_train = pd.read_csv(r'artifacts\RawData\train_data.csv')
    df_test = pd.read_csv(r'artifacts\RawData\test_data.csv')

    logging.info("Fit pipeline...")
    data_transformer.pipeline.fit(df_train)

    logging.info("Transforming data...")
    df_train_transformed = data_transformer.pipeline.transform(df_train)
    df_test_transformed = data_transformer.pipeline.transform(df_test)

    logging.info("Saving transformed data...")
    df_train_transformed.to_csv(transform_config.transformed_train_data_path, index=False)
    df_test_transformed.to_csv(transform_config.transformed_test_data_path, index=False)

    logging.info("Debuging each pipeline step on a small sample to catch errors early...")
    df_step = df_train.head(1) 
    for name, step in data_transformer.pipeline.named_steps.items():
        try:
            df_step = step.transform(df_step)
            print(f"Transformer '{name}' passed.")
        except Exception as e:
            print(f"Transformer '{name}' error: {e}")
        break

    logging.info("Saving the pipeline without future warnings...")
    data_dir = os.path.join('artifacts', 'Piplines')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        joblib.dump(data_transformer.pipeline, os.path.join(data_dir, 'traffic_pipeline.pkl'))

    print("✅ Pipeline trained and saved. ")
except Exception as e:
    raise CustomException(sys,e)


