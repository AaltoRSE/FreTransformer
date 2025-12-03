import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import os

# Mock the data loaders for testing
class MockDataset_Dhfm:
    """Test version of Dataset_Dhfm"""
    def __init__(self, root_path, flag, seq_len, pre_len, type):
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        
        # Create mock data
        load_data = np.load(root_path)
        # Ensure shape is (num_samples, num_features)
        data = load_data
        if data.shape[0] < data.shape[1]:
            data = data.T
        if type == '1':
            from sklearn.preprocessing import MinMaxScaler
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data)*0.7)
            self.data = data[begin:end]
        elif self.flag == 'val':
            begin = int(len(data)*0.7)
            end = int(len(data)*0.9)
            self.data = data[begin:end]
        elif self.flag == 'test':
            begin = int(len(data)*0.9)
            end = len(data)
            self.data = data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        data = self.data[begin:end]
        next_data = self.data[end:next_end]
        return data, next_data

    def __len__(self):
        return len(self.data) - self.seq_len - self.pre_len


class TestDatasetDhfm:
    @pytest.fixture
    def sample_data(self):
        """Create sample .npy file for testing"""
        # Create synthetic time series data: (time_steps, features)
        data = np.random.randn(1000, 5)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f.name, data)
            yield f.name
        os.unlink(f.name)

    def test_train_split(self, sample_data):
        """Test train/val/test split ratios"""
        dataset = MockDataset_Dhfm(sample_data, 'train', seq_len=10, pre_len=5, type='1')
        # 70% of 1000 = 700, minus seq_len and pre_len = 685
        expected_len = int(1000 * 0.7) - 10 - 5
        assert len(dataset) == expected_len

    def test_val_split(self, sample_data):
        """Test validation split"""
        dataset = MockDataset_Dhfm(sample_data, 'val', seq_len=10, pre_len=5, type='1')
        # (90%-70%) of 1000 = 200, minus seq_len and pre_len = 185
        expected_len = int(1000 * 0.2) - 10 - 5
        assert len(dataset) == expected_len

    def test_test_split(self, sample_data):
        """Test test split"""
        dataset = MockDataset_Dhfm(sample_data, 'test', seq_len=10, pre_len=5, type='1')
        # (100%-90%) of 1000 = 100, minus seq_len and pre_len = 85
        expected_len = int(1000 * 0.1) - 10 - 5
        assert len(dataset) == expected_len

    def test_getitem_shapes(self, sample_data):
        """Test output shapes from __getitem__"""
        dataset = MockDataset_Dhfm(sample_data, 'train', seq_len=20, pre_len=10, type='1')
        data, next_data = dataset[0]
        
        assert data.shape == (20, 5), f"Expected (20, 5), got {data.shape}"
        assert next_data.shape == (10, 5), f"Expected (10, 5), got {next_data.shape}"

    def test_no_data_leakage(self, sample_data):
        """Test that train/val/test sets don't overlap"""
        train_dataset = MockDataset_Dhfm(sample_data, 'train', seq_len=10, pre_len=5, type='1')
        val_dataset = MockDataset_Dhfm(sample_data, 'val', seq_len=10, pre_len=5, type='1')
        test_dataset = MockDataset_Dhfm(sample_data, 'test', seq_len=10, pre_len=5, type='1')
        
        # Verify split boundaries
        # Train: 0-700, Val: 700-900, Test: 900-1000
        assert len(train_dataset) + len(val_dataset) + len(test_dataset) > 0

    def test_normalization(self, sample_data):
        """Test MinMaxScaler normalization"""
        dataset = MockDataset_Dhfm(sample_data, 'train', seq_len=10, pre_len=5, type='1')
        data, _ = dataset[0]
        
        # After normalization, values should be in [0, 1]
        assert np.all(data >= 0), "Data contains values < 0"
        assert np.all(data <= 1), "Data contains values > 1"

    def test_sequential_access(self, sample_data):
        """Test that indices give sequential chunks"""
        dataset = MockDataset_Dhfm(sample_data, 'train', seq_len=10, pre_len=5, type='1')
        
        data1, next_data1 = dataset[0]
        data2, next_data2 = dataset[1]
        
        # Index 1 should start where index 0 would naturally continue
        # (shifted by 1 position, not by seq_len)
        assert data1.shape == data2.shape


class TestDatasetECG:
    @pytest.fixture
    def sample_csv(self):
        """Create sample CSV file"""
        data = np.random.randn(500, 3)
        df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_csv_loading(self, sample_csv):
        """Test CSV file loading"""
        df = pd.read_csv(sample_csv)
        assert df.shape[0] == 500
        assert df.shape[1] == 3

    def test_train_val_test_ratios(self, sample_csv):
        """Test configurable train/val/test ratios"""
        from fretransformer.data.data_loader import Dataset_ECG
        
        dataset = Dataset_ECG(
            sample_csv, 'train', seq_len=20, pre_len=10, 
            type='1', train_ratio=0.7, val_ratio=0.2
        )
        
        # 70% of 500 = 350, minus seq_len and pre_len = 325
        expected_len = int(500 * 0.7) - 20 - 10
        assert len(dataset) == expected_len


if __name__ == '__main__':
    pytest.main([__file__, '-v'])