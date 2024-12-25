import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        """Load train and test datasets"""
        train_df = pd.read_csv(self.config.TRAIN_DATA_PATH)
        test_df = pd.read_csv(self.config.TEST_DATA_PATH)
        return train_df, test_df
    
    def split_data(self, X, y):
        """Split data into train and validation sets"""
        return train_test_split(
            X, y,
            test_size=self.config.VAL_SIZE,
            random_state=self.config.RANDOM_STATE
        )