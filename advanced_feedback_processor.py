"""
Advanced Feedback Data Processor with Text Analysis
Enhanced version with better text processing and validation
"""

import pandas as pd
import numpy as np
import sweetviz as sv
from sdv.tabular import GaussianCopula, CTGAN
from sdv.evaluation import evaluate
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeedbackProcessor:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.column_types = {}
        self.synthetic_model = None
        self.report = None
        self.text_features = {}
    
    def load_data(self, file_path, **kwargs):
        """Load feedback data with enhanced error handling"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError("Unsupported file format")
            
            print(f"✓ Data loaded successfully: {self.data.shape}")
            print(f"✓ Columns: {list(self.data.columns)}")
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def analyze_text_columns(self, text_columns):
        """Advanced text analysis for feedback columns"""
        text_features = {}
        
        for col in text_columns:
            if col not in self.data.columns:
                continue
                
            text_data = self.data[col].dropna().astype(str)
            
            # Basic text statistics
            text_features[col] = {
                'avg_length': text_data.str.len().mean(),
                'max_length': text_data.str.len().max(),
                'word_count_avg': text_data.str.split().str.len().mean(),
                'unique_ratio': len(text_data.unique()) / len(text_data)
            }
            
            # Sentiment categories based on common feedback words
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'satisfied']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointed', 'poor', 'worst']
            
            def categorize_sentiment(text):
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    return 'positive'
                elif neg_count > pos_count:
                    return 'negative'
                else:
                    return 'neutral'
            
            # Add sentiment column
            sentiment_col = f'{col}_sentiment'
            self.data[sentiment_col] = text_data.apply(categorize_sentiment)
            
            # Add length category
            length_col = f'{col}_length_cat'
            lengths = text_data.str.len()
            self.data[length_col] = pd.cut(
                lengths,
                bins=[0, 20, 100, 300, float('inf')],
                labels=['very_short', 'short', 'medium', 'long']
            )
            
            # Topic clustering (simplified)
            if len(text_data) > 10:
                try:
                    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(text_data)
                    
                    n_clusters = min(5, len(text_data.unique()) // 10 + 1)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(tfidf_matrix)
                    
                    topic_col = f'{col}_topic'
                    self.data[topic_col] = [f'topic_{i}' for i in clusters]
                    
                except Exception as e:
                    print(f"Warning: Could not perform topic clustering for {col}: {e}")
        
        self.text_features = text_features
        return text_features
    
    def analyze_columns(self):
        """Enhanced column analysis with better categorization"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Generate sweetviz report
        print("Generating sweetviz analysis report...")
        self.report = sv.analyze(self.data)
        
        categorical_cols = []
        numerical_cols = []
        text_cols = []
        
        for col in self.data.columns:
            col_data = self.data[col]
            
            # Skip if mostly null
            null_ratio = col_data.isnull().sum() / len(col_data)
            if null_ratio > 0.9:
                print(f"Skipping {col}: too many null values ({null_ratio:.2%})")
                continue
            
            if col_data.dtype == 'object':
                non_null_data = col_data.dropna().astype(str)
                if len(non_null_data) == 0:
                    continue
                
                avg_length = non_null_data.str.len().mean()
                unique_ratio = len(non_null_data.unique()) / len(non_null_data)
                max_length = non_null_data.str.len().max()
                
                # Enhanced text detection
                if (avg_length > 30 or max_length > 100) and unique_ratio > 0.7:
                    text_cols.append(col)
                elif unique_ratio < 0.5 or len(non_null_data.unique()) <= 20:
                    categorical_cols.append(col)
                else:
                    # Check if it looks like categorical data
                    sample_values = non_null_data.head(100).str.lower()
                    if any(len(val.split()) <= 3 for val in sample_values):
                        categorical_cols.append(col)
                    else:
                        text_cols.append(col)
            
            elif pd.api.types.is_numeric_dtype(col_data):
                unique_count = col_data.nunique()
                total_count = len(col_data.dropna())
                
                # Check if it's ordinal/categorical
                if unique_count <= 10 and unique_count < total_count * 0.1:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                numerical_cols.append(col)
        
        # Analyze text columns
        if text_cols:
            print(f"Analyzing {len(text_cols)} text columns...")
            self.analyze_text_columns(text_cols)
            
            # Update categorical columns with new text-derived features
            for col in text_cols:
                potential_cats = [f'{col}_sentiment', f'{col}_length_cat', f'{col}_topic']
                for cat_col in potential_cats:
                    if cat_col in self.data.columns:
                        categorical_cols.append(cat_col)
        
        self.column_types = {
            'categorical': categorical_cols,
            'numerical': numerical_cols,
            'text': text_cols
        }
        
        print("\n" + "="*50)
        print("COLUMN ANALYSIS RESULTS")
        print("="*50)
        print(f"📊 Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        print(f"🔢 Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        print(f"📝 Text columns ({len(text_cols)}): {text_cols}")
        
        if self.text_features:
            print(f"\n📝 Text Analysis Summary:")
            for col, features in self.text_features.items():
                print(f"  {col}:")
                print(f"    - Avg length: {features['avg_length']:.1f} chars")
                print(f"    - Avg words: {features['word_count_avg']:.1f}")
                print(f"    - Uniqueness: {features['unique_ratio']:.2%}")
        
        return self.column_types
    
    def preprocess_for_synthesis(self):
        """Prepare data for synthetic generation"""
        if self.data is None:
            print("No data loaded.")
            return None
        
        processed_data = self.data.copy()
        
        # Remove original text columns (keep derived features)
        text_cols_to_remove = self.column_types.get('text', [])
        processed_data = processed_data.drop(columns=text_cols_to_remove, errors='ignore')
        
        # Handle missing values intelligently
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                # For categorical, use mode or 'Unknown'
                mode_val = processed_data[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                processed_data[col] = processed_data[col].fillna(fill_val)
            else:
                # For numerical, use median
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        # Ensure categorical columns are properly typed
        for col in self.column_types.get('categorical', []):
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].astype('category')
        
        self.processed_data = processed_data
        print(f"✓ Data preprocessed for synthesis: {processed_data.shape}")
        return processed_data
    
    def train_synthetic_model(self, model_type='gaussian_copula', **model_kwargs):
        """Train synthetic data model with validation"""
        if self.processed_data is None:
            self.preprocess_for_synthesis()
        
        if self.processed_data is None or self.processed_data.empty:
            print("No processed data available.")
            return None
        
        try:
            print(f"🚀 Training {model_type} model...")
            
            if model_type == 'gaussian_copula':
                self.synthetic_model = GaussianCopula(**model_kwargs)
            elif model_type == 'ctgan':
                default_kwargs = {'epochs': 100, 'batch_size': 500}
                default_kwargs.update(model_kwargs)
                self.synthetic_model = CTGAN(**default_kwargs)
            else:
                raise ValueError("Unsupported model type. Use 'gaussian_copula' or 'ctgan'")
            
            self.synthetic_model.fit(self.processed_data)
            print("✓ Model training completed!")
            
            return self.synthetic_model
        
        except Exception as e:
            print(f"✗ Error training model: {e}")
            return None
    
    def generate_synthetic_data(self, num_samples=1000, validate=True):
        """Generate and optionally validate synthetic data"""
        if self.synthetic_model is None:
            print("No trained model available. Please train a model first.")
            return None
        
        try:
            print(f"🎲 Generating {num_samples} synthetic samples...")
            synthetic_data = self.synthetic_model.sample(num_samples)
            
            if validate and self.processed_data is not None:
                print("🔍 Validating synthetic data quality...")
                evaluation_results = evaluate(synthetic_data, self.processed_data)
                print(f"✓ Validation completed. Overall score: {evaluation_results:.3f}")
            
            print("✓ Synthetic data generation completed!")
            return synthetic_data
        
        except Exception as e:
            print(f"✗ Error generating synthetic data: {e}")
            return None
    
    def save_results(self, synthetic_data=None, prefix='feedback'):
        """Save all results and reports"""
        # Save analysis report
        if self.report is not None:
            report_file = f'{prefix}_analysis_report.html'
            self.report.show_html(report_file)
            print(f"📊 Analysis report saved: {report_file}")
        
        # Save processed data
        if self.processed_data is not None:
            processed_file = f'{prefix}_processed_data.csv'
            self.processed_data.to_csv(processed_file, index=False)
            print(f"🔧 Processed data saved: {processed_file}")
        
        # Save synthetic data
        if synthetic_data is not None:
            synthetic_file = f'{prefix}_synthetic_data.csv'
            synthetic_data.to_csv(synthetic_file, index=False)
            print(f"🎲 Synthetic data saved: {synthetic_file}")
        
        # Save analysis summary
        summary_file = f'{prefix}_analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("FEEDBACK DATA ANALYSIS SUMMARY\n")
            f.write("="*40 + "\n\n")
            f.write(f"Original data shape: {self.data.shape if self.data is not None else 'N/A'}\n")
            f.write(f"Processed data shape: {self.processed_data.shape if self.processed_data is not None else 'N/A'}\n")
            f.write(f"Synthetic data shape: {synthetic_data.shape if synthetic_data is not None else 'N/A'}\n\n")
            
            if self.column_types:
                f.write("Column Types:\n")
                for col_type, cols in self.column_types.items():
                    f.write(f"  {col_type}: {len(cols)} columns\n")
                    for col in cols[:5]:  # Show first 5
                        f.write(f"    - {col}\n")
                    if len(cols) > 5:
                        f.write(f"    ... and {len(cols)-5} more\n")
                f.write("\n")
            
            if self.text_features:
                f.write("Text Analysis:\n")
                for col, features in self.text_features.items():
                    f.write(f"  {col}:\n")
                    f.write(f"    - Average length: {features['avg_length']:.1f} characters\n")
                    f.write(f"    - Average words: {features['word_count_avg']:.1f}\n")
                    f.write(f"    - Uniqueness ratio: {features['unique_ratio']:.2%}\n")
        
        print(f"📋 Analysis summary saved: {summary_file}")

def main():
    """Example usage of the advanced processor"""
    processor = AdvancedFeedbackProcessor()
    
    # You can replace this with your actual data file
    print("Creating sample data for demonstration...")
    sample_data = create_comprehensive_sample_data()
    
    # Load and process data
    processor.load_data('comprehensive_feedback_data.csv')
    processor.analyze_columns()
    
    # Train model and generate synthetic data
    processor.train_synthetic_model('gaussian_copula')
    synthetic_data = processor.generate_synthetic_data(500, validate=True)
    
    # Save all results
    processor.save_results(synthetic_data, prefix='comprehensive_feedback')
    
    print("\n🎉 Processing completed! Check the generated files.")

def create_comprehensive_sample_data():
    """Create a more comprehensive sample dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic feedback data
    ratings = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.15, 0.35, 0.35])
    
    # Create feedback text based on ratings
    feedback_templates = {
        1: ["Terrible experience", "Very disappointed", "Would not recommend", "Poor quality"],
        2: ["Not satisfied", "Below expectations", "Could be better", "Had issues"],
        3: ["Average experience", "It's okay", "Neither good nor bad", "Acceptable"],
        4: ["Good experience", "Satisfied with purchase", "Would recommend", "Quality product"],
        5: ["Excellent!", "Amazing quality", "Exceeded expectations", "Perfect!"]
    }
    
    feedback_texts = []
    for rating in ratings:
        base_text = np.random.choice(feedback_templates[rating])
        # Add some variation
        if np.random.random() > 0.5:
            additional = [
                " The delivery was fast.", " Customer service was helpful.", 
                " Good value for money.", " Will buy again.", " Easy to use.",
                " High quality materials.", " Exactly as described."
            ]
            base_text += np.random.choice(additional)
        feedback_texts.append(base_text)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'rating': ratings,
        'satisfaction_score': np.random.normal(7, 2, n_samples).clip(1, 10),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_samples),
        'feedback_text': feedback_texts,
        'recommend': np.random.choice(['Yes', 'No', 'Maybe'], n_samples, p=[0.6, 0.25, 0.15]),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_samples),
        'purchase_amount': np.random.exponential(75, n_samples),
        'delivery_rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.03, 0.07, 0.15, 0.35, 0.4]),
        'ease_of_use': np.random.choice(['Very Easy', 'Easy', 'Moderate', 'Difficult', 'Very Difficult'], n_samples),
        'purchase_channel': np.random.choice(['Online', 'Store', 'Phone', 'Mobile App'], n_samples),
        'customer_type': np.random.choice(['New', 'Returning', 'VIP'], n_samples, p=[0.4, 0.5, 0.1])
    }
    
    df = pd.DataFrame(data)
    df.to_csv('comprehensive_feedback_data.csv', index=False)
    print("✓ Comprehensive sample data created: comprehensive_feedback_data.csv")
    return df

if __name__ == "__main__":
    main()