"""
数据预处理模块测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from utils.data_process import DataProcessor, load_and_process


class TestDataProcessor:
    """DataProcessor 类测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'Weight': np.random.normal(70, 15, 100),
            'Height': np.random.normal(175, 10, 100),
            'BodyFat': np.random.gamma(2, 2, 100),
            'Age': np.random.randint(20, 80, 100),
            'Gender': np.random.choice(['M', 'F'], 100)
        })
        # 引入缺失值
        data.loc[0:5, 'Weight'] = np.nan
        data.loc[10:12, 'Height'] = np.nan
        # 引入异常值
        data.loc[20, 'Weight'] = 500  # 异常高
        data.loc[21, 'Height'] = 10   # 异常低
        return data
    
    def test_init(self, sample_data):
        """测试初始化"""
        processor = DataProcessor(sample_data)
        assert processor.df.shape == sample_data.shape
        assert len(processor.processing_log) == 0
    
    def test_get_missing_info(self, sample_data):
        """测试缺失值统计"""
        processor = DataProcessor(sample_data)
        missing_info = processor.get_missing_info()
        
        # 检查列名存在
        assert 'Column' in missing_info.columns
        assert 'Missing' in missing_info.columns
        
        # 检查 Weight 和 Height 在结果中
        columns = missing_info['Column'].tolist()
        assert 'Weight' in columns
        assert 'Height' in columns
        
        # 检查缺失数量
        weight_row = missing_info[missing_info['Column'] == 'Weight']
        height_row = missing_info[missing_info['Column'] == 'Height']
        assert weight_row['Missing'].values[0] == 6
        assert height_row['Missing'].values[0] == 3
    
    def test_fill_missing_auto(self, sample_data):
        """测试自动填充缺失值"""
        processor = DataProcessor(sample_data)
        original_missing = processor.df.isnull().sum().sum()
        
        processor.fill_missing(strategy='auto')
        
        new_missing = processor.df.isnull().sum().sum()
        assert new_missing < original_missing
        assert len(processor.processing_log) > 0
    
    def test_fill_missing_median(self, sample_data):
        """测试中位数填充"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='median', columns=['Weight'])
        
        assert processor.df['Weight'].isnull().sum() == 0
    
    def test_fill_missing_drop(self, sample_data):
        """测试删除缺失行"""
        processor = DataProcessor(sample_data)
        original_rows = len(processor.df)
        
        processor.fill_missing(strategy='drop', columns=['Weight'])
        
        assert len(processor.df) < original_rows
        assert processor.df['Weight'].isnull().sum() == 0
    
    def test_detect_outliers_iqr(self, sample_data):
        """测试 IQR 异常值检测"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='auto')
        
        result = processor.detect_outliers_iqr('Weight')
        
        assert 'Q1' in result
        assert 'Q3' in result
        assert 'IQR' in result
        assert 'outlier_count' in result
        assert result['outlier_count'] >= 1  # 至少有一个异常值
    
    def test_detect_outliers_zscore(self, sample_data):
        """测试 Z-Score 异常值检测"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='auto')
        
        result = processor.detect_outliers_zscore('Weight')
        
        assert 'threshold' in result
        assert 'outlier_count' in result
    
    def test_handle_outliers_cap(self, sample_data):
        """测试盖帽法处理异常值"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='auto')
        
        # 获取原始异常值数量
        original_outliers = processor.detect_outliers_iqr('Weight')['outlier_count']
        
        processor.handle_outliers(columns=['Weight'], method='cap')
        
        # 盖帽后应无异常值
        new_outliers = processor.detect_outliers_iqr('Weight')['outlier_count']
        assert new_outliers < original_outliers
    
    def test_handle_outliers_drop(self, sample_data):
        """测试删除异常值"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='auto')
        original_rows = len(processor.df)
        
        processor.handle_outliers(columns=['Weight'], method='drop')
        
        assert len(processor.df) <= original_rows
    
    def test_transform_log(self, sample_data):
        """测试对数变换"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='auto')
        
        processor.transform_log(['BodyFat'])
        
        assert 'BodyFat_log' in processor.df.columns
        # 对数变换后偏度应减小
        original_skew = abs(processor.df['BodyFat'].skew())
        transformed_skew = abs(processor.df['BodyFat_log'].skew())
        assert transformed_skew < original_skew
    
    def test_get_high_correlation_pairs(self, sample_data):
        """测试高相关性特征对检测"""
        # 创建有高相关性的数据
        data = sample_data.copy()
        data['Weight2'] = data['Weight'] * 1.1 + np.random.normal(0, 1, len(data))
        
        processor = DataProcessor(data)
        processor.fill_missing(strategy='auto')
        
        high_corr = processor.get_high_correlation_pairs(threshold=0.8)
        
        assert isinstance(high_corr, pd.DataFrame)
    
    def test_plot_missing_matrix(self, sample_data):
        """测试缺失值矩阵图"""
        processor = DataProcessor(sample_data)
        fig = processor.plot_missing_matrix()
        
        assert fig is not None
    
    def test_plot_boxplot(self, sample_data):
        """测试箱线图"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='auto')
        
        fig = processor.plot_boxplot(columns=['Weight', 'Height'])
        
        assert fig is not None
    
    def test_plot_distribution(self, sample_data):
        """测试分布图"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='auto')
        
        fig = processor.plot_distribution(columns=['Weight', 'BodyFat'])
        
        assert fig is not None
    
    def test_plot_correlation_heatmap(self, sample_data):
        """测试相关性热力图"""
        processor = DataProcessor(sample_data)
        processor.fill_missing(strategy='auto')
        
        fig = processor.plot_correlation_heatmap()
        
        assert fig is not None
    
    def test_reset(self, sample_data):
        """测试重置功能"""
        processor = DataProcessor(sample_data)
        original_shape = processor.df.shape
        
        processor.fill_missing(strategy='drop')
        assert processor.df.shape != original_shape
        
        processor.reset()
        assert processor.df.shape == original_shape
    
    def test_get_summary(self, sample_data):
        """测试数据摘要"""
        processor = DataProcessor(sample_data)
        summary = processor.get_summary()
        
        assert '缺失值' in summary.columns
        assert '缺失率(%)' in summary.columns
    
    def test_chain_operations(self, sample_data):
        """测试链式操作"""
        processor = DataProcessor(sample_data)
        
        result = (processor
                  .fill_missing(strategy='auto')
                  .handle_outliers(method='cap')
                  .transform_log(['BodyFat']))
        
        assert result is processor
        assert 'BodyFat_log' in processor.df.columns
        assert len(processor.processing_log) >= 3


class TestLoadAndProcess:
    """load_and_process 便捷函数测试"""
    
    @pytest.fixture
    def temp_csv(self, tmp_path):
        """创建临时 CSV 文件"""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.normal(0, 1, 50),
            'B': np.random.normal(0, 1, 50),
            'C': np.random.randint(0, 2, 50)
        })
        data.loc[0:2, 'A'] = np.nan
        
        filepath = tmp_path / "test_data.csv"
        data.to_csv(filepath, index=False)
        return str(filepath)
    
    def test_load_and_process(self, temp_csv):
        """测试一键加载处理"""
        df, processor = load_and_process(temp_csv)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(processor, DataProcessor)
        assert df.isnull().sum().sum() == 0  # 缺失值已处理


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

