class FeatureProcessor:
    def __init__(self, config):
        self.config = config
        
    def prepare_features(self, df):
        """Prepare features for model training"""
        X = df[self.config.CAT_FEATURES]
        return X
    
    def prepare_target(self, df):
        """Prepare target variable"""
        return df['ACTION'] if 'ACTION' in df.columns else None