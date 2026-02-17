"""
Feedback Data Processor
Extracts feedback data, analyzes column types, and generates synthetic data using SDV
"""

import pandas as pd
import numpy as np
import sweetviz as sv
from sdv.tabular import GaussianCopula, CTGAN
from sdv.constraints import FixedCombinations
import warnings
warnings.filterwarnings('ignore')

class FeedbackDataProcessor:
    def __init__(self):
        self.data = None
        self.column_types = {}
        self.synthetic_model = None
        self.report = None
    
    def load_data(self, file_path, **kwargs):
        """
        Load feedback data from various file formats
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError("Unsupported file format")
            
            print(f"Data loaded successfully: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def analyze_columns(self):
        """
        Analyze and categorize columns using sweetviz and custom logic
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Generate sweetviz report
        self.report = sv.analyze(self.data)
        
        # Categorize columns
        categorical_cols = []
        numerical_cols = []
        text_cols = []
        
        for col in self.data.columns:
            col_data = self.data[col]
            
            # Skip if mostly null
            if col_data.isnull().sum() / len(col_data) > 0.8:
                continue
            
            # Check for text columns (free-text feedback)
            if col_data.dtype == 'object':
                # Calculate average string length for non-null values
                non_null_strings = col_data.dropna().astype(str)
                if len(non_null_strings) > 0:
                    avg_length = non_null_strings.str.len().mean()
                    unique_ratio = len(non_null_strings.unique()) / len(non_null_strings)
                    
                    # If average length > 50 or high uniqueness, likely text
                    if avg_length > 50 or unique_ratio > 0.8:
                        text_cols.append(col)
                    else:
                        categorical_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            # Numerical columns
            elif pd.api.types.is_numeric_dtype(col_data):
                # Check if it's actually categorical (few unique values)
                unique_count = col_data.nunique()
                if unique_count <= 10 and unique_count < len(col_data) * 0.1:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            
            # DateTime columns
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                numerical_cols.append(col)  # Treat as numerical for SDV
        
        self.column_types = {
            'categorical': categorical_cols,
            'numerical': numerical_cols,
            'text': text_cols
        }
        
        print("Column Analysis Complete:")
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        print(f"Text columns: {text_cols}")
        
        return self.column_types
    
    def preprocess_data(self):
        """
        Preprocess data for synthetic generation
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        processed_data = self.data.copy()
        
        # Handle text columns - convert to categorical or remove
        for col in self.column_types.get('text', []):
            # For feedback text, we might want to extract sentiment or keywords
            # For now, we'll create a simplified categorical version
            if col in processed_data.columns:
                # Extract basic sentiment categories or length categories
                text_series = processed_data[col].fillna('')
                
                # Create length-based categories
                lengths = text_series.str.len()
                processed_data[f'{col}_length_category'] = pd.cut(
                    lengths, 
                    bins=[0, 10, 50, 200, float('inf')], 
                    labels=['very_short', 'short', 'medium', 'long']
                )
                
                # Remove original text column for SDV (can't synthesize free text well)
                processed_data = processed_data.drop(columns=[col])
                self.column_types['categorical'].append(f'{col}_length_category')
        
        # Handle missing values
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].fillna('Unknown')
            else:
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        return processed_data
    
    def train_synthetic_model(self, model_type='gaussian_copula'):
        """
        Train synthetic data generation model
        """
        processed_data = self.preprocess_data()
        
        if processed_data is None or processed_data.empty:
            print("No processed data available.")
            return None
        
        try:
            if model_type == 'gaussian_copula':
                self.synthetic_model = GaussianCopula()
            elif model_type == 'ctgan':
                self.synthetic_model = CTGAN(epochs=100)
            else:
                raise ValueError("Unsupported model type")
            
            print(f"Training {model_type} model...")
            self.synthetic_model.fit(processed_data)
            print("Model training completed!")
            
            return self.synthetic_model
        
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def generate_synthetic_data(self, num_samples=1000):
        """
        Generate synthetic feedback data
        """
        if self.synthetic_model is None:
            print("No trained model available. Please train a model first.")
            return None
        
        try:
            print(f"Generating {num_samples} synthetic samples...")
            synthetic_data = self.synthetic_model.sample(num_samples)
            print("Synthetic data generation completed!")
            
            return synthetic_data
        
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return None
    
    def save_report(self, filename='feedback_analysis_report.html'):
        """
        Save sweetviz analysis report
        """
        if self.report is not None:
            self.report.show_html(filename)
            print(f"Analysis report saved as {filename}")
        else:
            print("No report available. Please run analyze_columns() first.")
    
    def get_data_summary(self):
        """
        Get summary of the loaded data
        """
        if self.data is None:
            return "No data loaded."
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'column_types': self.column_types
        }
        
        return summary

# Example usage and utility functions
def create_sample_feedback_data():
    """
    Create sample feedback data for testing
    """
    np.random.seed(42)
    
    # Sample feedback data
    data = {
        'customer_id': range(1, 501),
        'rating': np.random.choice([1, 2, 3, 4, 5], 500, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'satisfaction_score': np.random.normal(7.5, 1.5, 500).clip(1, 10),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 500),
        'feedback_text': [
            f"This is feedback text number {i} with varying lengths and content" + 
            (" " + "additional content " * np.random.randint(0, 10))
            for i in range(500)
        ],
        'recommend': np.random.choice(['Yes', 'No', 'Maybe'], 500, p=[0.6, 0.2, 0.2]),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], 500),
        'purchase_amount': np.random.exponential(100, 500)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_feedback_data.csv', index=False)
    print("Sample feedback data created: sample_feedback_data.csv")
    return df

if __name__ == "__main__":
    # Example workflow
    processor = FeedbackDataProcessor()
    
    # Create sample data for demonstration
    sample_data = create_sample_feedback_data()
    
    # Load data
    processor.load_data('sample_feedback_data.csv')
    
    # Analyze columns
    processor.analyze_columns()
    
    # Save analysis report
    processor.save_report()
    
    # Train synthetic model
    processor.train_synthetic_model('gaussian_copula')
    
    # Generate synthetic data
    synthetic_data = processor.generate_synthetic_data(200)
    
    if synthetic_data is not None:
        synthetic_data.to_csv('synthetic_feedback_data.csv', index=False)
        print("Synthetic data saved as synthetic_feedback_data.csv")
    
    # Print summary
    print("\nData Summary:")
    print(processor.get_data_summary())