from src.components.data_ingestion import DataIngestion

data = DataIngestion()
print(data.split_into_train())
print(data.split_into_test())
