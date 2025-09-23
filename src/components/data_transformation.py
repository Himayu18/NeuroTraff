import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    transformed_train_data_path: str = os.path.join('artifacts', 'train_data_transformed.csv')
    transformed_test_data_path: str = os.path.join('artifacts', 'test_data_transformed.csv')

class ColumnRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Drop only columns that exist to avoid errors
        cols_to_drop = [col for col in self.columns if col in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        return df

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        self.le.fit(X[self.column])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.column] = self.le.transform(df[self.column])
        return df

class TimestampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.column] = pd.to_datetime(df[self.column])

        if df[self.column].dt.tz is None:
            df[self.column] = df[self.column].dt.tz_localize('UTC')

        df[self.column] = df[self.column].dt.tz_convert('Asia/Kolkata')
        df[self.column] = df[self.column].dt.tz_localize(None)

        df['Day'] = df[self.column].dt.day
        df['Hour'] = df[self.column].dt.hour
        df['minute'] = df[self.column].dt.minute
        df.drop(columns=self.column, inplace=True)
        return df

class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = OrdinalEncoder()
        self.encoder.fit(X[[self.column]])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.column] = self.encoder.transform(df[[self.column]])
        return df

class DelayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, currentTravelTime, freeFlowTravelTime):
        self.currentTravelTime = currentTravelTime
        self.freeFlowTravelTime = freeFlowTravelTime

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Check required columns presence
        if self.currentTravelTime not in df.columns or self.freeFlowTravelTime not in df.columns:
            raise ValueError(f"Missing columns: {self.currentTravelTime} or {self.freeFlowTravelTime} not found in input data")
        df['Delay'] = df[self.currentTravelTime] - df[self.freeFlowTravelTime]
        df['delay ratio'] = np.where(
            df[self.freeFlowTravelTime] > 0,
            df['Delay'] / df[self.freeFlowTravelTime],
            np.nan
        )
        return df

class CoordinatesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in input data")
        df[['latitude', 'longitude']] = df[self.column].str.split(',', expand=True)
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        df.drop(columns=self.column, inplace=True)
        return df

class TrafficLevelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, delay_ratio_column='delay ratio', new_column='Traffic level'):
        self.delay_ratio_column = delay_ratio_column
        self.new_column = new_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if self.delay_ratio_column not in df.columns:
            raise ValueError(f"Column '{self.delay_ratio_column}' not found in input data")
        
        def categorize_traffic(delay_ratio):
            if pd.isna(delay_ratio):
                return np.nan
            if delay_ratio <= 0.15:
                return 'Low'
            elif delay_ratio <= 0.5:
                return 'Medium'
            else:
                return 'High'

        df[self.new_column] = df[self.delay_ratio_column].apply(categorize_traffic)
        return df

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        self.le.fit(X[self.column])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.column] = self.le.transform(df[self.column])
        return df
    
class DataTransformer:
    def __init__(self):
        self.removing_columns = ['_id', 'roadName', 'roadClosure']
        self.currentTravelTime = 'currentTravelTime'
        self.freeFlowTravelTime = 'freeFlowTravelTime'
        self.delay_column = 'delay ratio'
        self.timestamp = 'timestamp'
        self.point = 'point'

        self.pipeline = Pipeline([
            ('remove_columns', ColumnRemover(columns=self.removing_columns)),
            ('delay', DelayTransformer(currentTravelTime=self.currentTravelTime, freeFlowTravelTime=self.freeFlowTravelTime)),
            ('traffic_level', TrafficLevelTransformer(delay_ratio_column=self.delay_column, new_column='Traffic level')),
            ('timestamp', TimestampTransformer(column=self.timestamp)),
            ('ordinal_encode_road', OrdinalEncoderTransformer(column='road')),
            ('ordinal_encode_frc', OrdinalEncoderTransformer(column='frc')),
            ('label_encode_traffic', LabelEncoderTransformer(column='Traffic level')),  # <-- Added inside pipeline
            ('coordinates', CoordinatesTransformer(column=self.point))
        ])

transform_config = TransformerConfig()
data_transformer = DataTransformer()

# Load data
df_train = pd.read_csv(r'C:\Users\himay\OneDrive\Desktop\NeuroTraff\artifacts\train_data.csv')
df_test = pd.read_csv(r'C:\Users\himay\OneDrive\Desktop\NeuroTraff\artifacts\test_data.csv')

# Fit pipeline
data_transformer.pipeline.fit(df_train)

# Transform data
df_train_transformed = data_transformer.pipeline.transform(df_train)
df_test_transformed = data_transformer.pipeline.transform(df_test)


# Save transformed data
df_train_transformed.to_csv(transform_config.transformed_train_data_path, index=False)
df_test_transformed.to_csv(transform_config.transformed_test_data_path, index=False)

# Debug each pipeline step on a small sample to catch errors early
df_step = df_train.head(1)  # start with raw input
for name, step in data_transformer.pipeline.named_steps.items():
    try:
        df_step = step.transform(df_step)
        print(f"Transformer '{name}' passed.")
    except Exception as e:
        print(f"Transformer '{name}' error: {e}")
        break


# Save the pipeline without future warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    joblib.dump(data_transformer.pipeline, os.path.join('artifacts', 'traffic_pipeline.pkl'))

print("✅ Pipeline trained and saved.")
