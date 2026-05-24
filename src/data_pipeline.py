# src/data_pipeline.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    """Custom PyTorch Dataset for loading emotion text and labels."""
    def __init__(self, dataframe):
        self.texts = dataframe['text'].values
        self.labels = dataframe['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        return text, torch.tensor(label, dtype=torch.long)


class DataPipelineManager:
    """Manages the entire data engineering workflow for Task 1."""
    def __init__(self, raw_data_path, mock_data_output_path):
        self.raw_data_path = raw_data_path
        self.mock_data_output_path = mock_data_output_path
        self.df_raw = None

    def load_raw_data(self):
        """Loads the heavy raw CSV file into memory."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")
        self.df_raw = pd.read_csv(self.raw_data_path)
        return self.df_raw

    def generate_mock_data(self, num_samples=5000, random_state=42):
        """Extracts and saves a smaller split for modeling team prototyping."""
        if self.df_raw == None:
            self.load_raw_data()
        
        df_mock = self.df_raw.sample(n=num_samples, random_state=random_state)
        os.makedirs(os.path.dirname(self.mock_data_output_path), exist_ok=True)
        df_mock.to_csv(self.mock_data_output_path, index=False)
        return df_mock

    def get_pytorch_loaders(self, df, batch_size=32, shuffle=True):
        """Wraps a dataframe into an executable PyTorch DataLoader."""
        dataset = EmotionDataset(df)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader 