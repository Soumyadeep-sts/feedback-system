"""
Hybrid Feedback Data Processor
Uses SDV for categorical/numerical data and llama-cpp-python for text generation
"""

import pandas as pd
import numpy as np
import sweetviz as sv
from sdv.tabular import GaussianCopula, CTGAN
from sdv.evaluation import evaluate
import re
import json
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not available. Text generation will be disabled.")

class HybridFeedbackProcessor:
    def __init__(self, llama_model_path: Optional[str] = None):
        self.data = None
        self.processed_data = None
        self.column_types = {}
        self.sdv_model = None
        self.llama_model = None
        self.report = None
        self.text_patterns = {}
        self.llama_model_path = llama_model_path
        
        # Initialize Llama model if path provided
        if llama_model_path and LLAMA_AVAILABLE:
            self._load_llama_model(llama_model_path)
    
    def _load_llama_model(self, model_path: str):
        """Load Llama model for text generation"""
        try:
            print(f"🦙 Loading Llama model from {model_path}...")
            self.llama_model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # Number of CPU threads
                verbose=False
            )
            print("✓ Llama model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading Llama model: {e}")
            self.llama_model = None
    
    def load_data(self, file_path: str, **kwargs):
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
    
    def analyze_columns(self):
        """Analyze and categorize columns using sweetviz and custom logic"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Generate sweetviz report
        print("📊 Generating sweetviz analysis report...")
        self.report = sv.analyze(self.data)
        
        categorical_cols = []
        numerical_cols = []
        text_cols = []
        
        for col in self.data.columns:
            col_data = self.data[col]
            
            # Skip if mostly null
            null_ratio = col_data.isnull().sum() / len(col_data)
            if null_ratio > 0.9:
                print(f"⚠️ Skipping {col}: too many null values ({null_ratio:.2%})")
                continue
            
            if col_data.dtype == 'object':
                non_null_data = col_data.dropna().astype(str)
                if len(non_null_data) == 0:
                    continue
                
                avg_length = non_null_data.str.len().mean()
                unique_ratio = len(non_null_data.unique()) / len(non_null_data)
                max_length = non_null_data.str.len().max()
                
                # Enhanced text detection for free-text feedback
                if (avg_length > 30 or max_length > 100) and unique_ratio > 0.7:
                    text_cols.append(col)
                    # Analyze text patterns for generation
                    self._analyze_text_patterns(col, non_null_data)
                elif unique_ratio < 0.5 or len(non_null_data.unique()) <= 20:
                    categorical_cols.append(col)
                else:
                    # Check if it looks like categorical data
                    sample_values = non_null_data.head(100).str.lower()
                    if any(len(val.split()) <= 3 for val in sample_values):
                        categorical_cols.append(col)
                    else:
                        text_cols.append(col)
                        self._analyze_text_patterns(col, non_null_data)
            
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
        
        self.column_types = {
            'categorical': categorical_cols,
            'numerical': numerical_cols,
            'text': text_cols
        }
        
        print("\n" + "="*60)
        print("COLUMN ANALYSIS RESULTS")
        print("="*60)
        print(f"📊 Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        print(f"🔢 Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        print(f"📝 Text columns ({len(text_cols)}): {text_cols}")
        print(f"📝 Text patterns analyzed for: {list(self.text_patterns.keys())}")
        
        return self.column_types
    
    def _analyze_text_patterns(self, column: str, text_data: pd.Series):
        """Analyze text patterns for better generation"""
        patterns = {
            'avg_length': text_data.str.len().mean(),
            'length_std': text_data.str.len().std(),
            'word_count_avg': text_data.str.split().str.len().mean(),
            'common_starts': [],
            'common_words': [],
            'sentiment_words': {'positive': [], 'negative': []},
            'sample_texts': text_data.sample(min(10, len(text_data))).tolist()
        }
        
        # Find common sentence starters
        starts = text_data.str.split().str[0].value_counts().head(5)
        patterns['common_starts'] = starts.index.tolist()
        
        # Find common words (simple approach)
        all_words = ' '.join(text_data.astype(str)).lower().split()
        word_freq = pd.Series(all_words).value_counts()
        patterns['common_words'] = word_freq.head(20).index.tolist()
        
        # Basic sentiment words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'satisfied', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointed', 'poor', 'worst', 'horrible']
        
        patterns['sentiment_words']['positive'] = [w for w in positive_words if w in word_freq.index]
        patterns['sentiment_words']['negative'] = [w for w in negative_words if w in word_freq.index]
        
        self.text_patterns[column] = patterns
        
        print(f"📝 Text analysis for '{column}':")
        print(f"   - Avg length: {patterns['avg_length']:.1f} chars")
        print(f"   - Avg words: {patterns['word_count_avg']:.1f}")
        print(f"   - Common starts: {patterns['common_starts'][:3]}")
    
    def preprocess_for_sdv(self):
        """Prepare data for SDV (excluding text columns)"""
        if self.data is None:
            print("No data loaded.")
            return None
        
        # Create dataset without text columns for SDV
        sdv_data = self.data.copy()
        text_cols = self.column_types.get('text', [])
        
        # Remove text columns from SDV data
        sdv_data = sdv_data.drop(columns=text_cols, errors='ignore')
        
        # Handle missing values
        for col in sdv_data.columns:
            if sdv_data[col].dtype == 'object':
                mode_val = sdv_data[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                sdv_data[col] = sdv_data[col].fillna(fill_val)
            else:
                sdv_data[col] = sdv_data[col].fillna(sdv_data[col].median())
        
        # Ensure categorical columns are properly typed
        for col in self.column_types.get('categorical', []):
            if col in sdv_data.columns:
                sdv_data[col] = sdv_data[col].astype('category')
        
        self.processed_data = sdv_data
        print(f"✓ Data preprocessed for SDV: {sdv_data.shape}")
        print(f"✓ Excluded text columns: {text_cols}")
        return sdv_data
    
    def train_sdv_model(self, model_type='gaussian_copula', **model_kwargs):
        """Train SDV model for categorical and numerical data"""
        if self.processed_data is None:
            self.preprocess_for_sdv()
        
        if self.processed_data is None or self.processed_data.empty:
            print("No processed data available for SDV.")
            return None
        
        try:
            print(f"🚀 Training SDV {model_type} model...")
            
            if model_type == 'gaussian_copula':
                self.sdv_model = GaussianCopula(**model_kwargs)
            elif model_type == 'ctgan':
                default_kwargs = {'epochs': 100, 'batch_size': 500}
                default_kwargs.update(model_kwargs)
                self.sdv_model = CTGAN(**default_kwargs)
            else:
                raise ValueError("Unsupported model type. Use 'gaussian_copula' or 'ctgan'")
            
            self.sdv_model.fit(self.processed_data)
            print("✓ SDV model training completed!")
            
            return self.sdv_model
        
        except Exception as e:
            print(f"✗ Error training SDV model: {e}")
            return None
    
    def _generate_text_with_llama(self, column: str, context_data: pd.Series, num_samples: int) -> List[str]:
        """Generate text using Llama model based on context"""
        if not self.llama_model:
            print(f"⚠️ Llama model not available. Using pattern-based generation for {column}")
            return self._generate_text_patterns(column, num_samples)
        
        generated_texts = []
        patterns = self.text_patterns.get(column, {})
        sample_texts = patterns.get('sample_texts', [])
        
        print(f"🦙 Generating {num_samples} text samples for '{column}' using Llama...")
        
        for i in range(num_samples):
            # Create context-aware prompt
            if i < len(context_data):
                context_row = context_data.iloc[i]
                context_info = self._create_context_prompt(context_row, column)
            else:
                context_info = "Generate a feedback comment"
            
            # Use sample texts as examples
            examples = "\n".join([f"Example: {text}" for text in sample_texts[:3]])
            
            prompt = f"""Generate a realistic feedback comment similar to these examples:
{examples}

Context: {context_info}
Generate a feedback comment (keep it concise, 1-3 sentences):"""
            
            try:
                response = self.llama_model(
                    prompt,
                    max_tokens=100,
                    temperature=0.8,
                    top_p=0.9,
                    stop=["\n\n", "Example:", "Context:"]
                )
                
                generated_text = response['choices'][0]['text'].strip()
                if generated_text:
                    generated_texts.append(generated_text)
                else:
                    # Fallback to pattern-based
                    generated_texts.append(self._generate_single_text_pattern(column))
                    
            except Exception as e:
                print(f"⚠️ Error generating text sample {i+1}: {e}")
                generated_texts.append(self._generate_single_text_pattern(column))
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   Generated {i + 1}/{num_samples} samples...")
        
        return generated_texts
    
    def _create_context_prompt(self, context_row: pd.Series, text_column: str) -> str:
        """Create context-aware prompt based on other columns"""
        context_parts = []
        
        # Add rating context if available
        if 'rating' in context_row.index:
            rating = context_row['rating']
            if rating <= 2:
                context_parts.append("negative experience")
            elif rating >= 4:
                context_parts.append("positive experience")
            else:
                context_parts.append("neutral experience")
        
        # Add category context if available
        category_cols = [col for col in context_row.index if 'category' in col.lower() or 'product' in col.lower()]
        if category_cols:
            context_parts.append(f"about {context_row[category_cols[0]]}")
        
        # Add satisfaction context
        satisfaction_cols = [col for col in context_row.index if 'satisfaction' in col.lower() or 'score' in col.lower()]
        if satisfaction_cols:
            score = context_row[satisfaction_cols[0]]
            if score < 5:
                context_parts.append("low satisfaction")
            elif score > 7:
                context_parts.append("high satisfaction")
        
        return " ".join(context_parts) if context_parts else "general feedback"
    
    def _generate_text_patterns(self, column: str, num_samples: int) -> List[str]:
        """Fallback pattern-based text generation"""
        print(f"📝 Using pattern-based generation for '{column}'...")
        
        patterns = self.text_patterns.get(column, {})
        sample_texts = patterns.get('sample_texts', ['Good product', 'Average experience', 'Could be better'])
        
        generated_texts = []
        for i in range(num_samples):
            generated_texts.append(self._generate_single_text_pattern(column))
        
        return generated_texts
    
    def _generate_single_text_pattern(self, column: str) -> str:
        """Generate a single text sample using patterns"""
        patterns = self.text_patterns.get(column, {})
        
        # Use sample texts with some variation
        sample_texts = patterns.get('sample_texts', ['Good product', 'Average experience'])
        base_text = np.random.choice(sample_texts)
        
        # Add some variation
        variations = [
            " Really satisfied with this.",
            " Would recommend to others.",
            " Good value for money.",
            " Fast delivery.",
            " Easy to use.",
            " Could be improved.",
            " Met my expectations.",
            " Will buy again."
        ]
        
        if np.random.random() > 0.5:
            base_text += np.random.choice(variations)
        
        return base_text
    
    def generate_hybrid_data(self, num_samples: int = 1000, validate: bool = True) -> pd.DataFrame:
        """Generate synthetic data using hybrid approach"""
        if self.sdv_model is None:
            print("No SDV model trained. Please train SDV model first.")
            return None
        
        print(f"🎲 Generating {num_samples} hybrid synthetic samples...")
        
        # Step 1: Generate categorical and numerical data with SDV
        print("1️⃣ Generating categorical/numerical data with SDV...")
        sdv_synthetic = self.sdv_model.sample(num_samples)
        
        # Step 2: Generate text data with Llama
        text_cols = self.column_types.get('text', [])
        if text_cols:
            print("2️⃣ Generating text data with Llama...")
            for text_col in text_cols:
                generated_texts = self._generate_text_with_llama(
                    text_col, 
                    sdv_synthetic,  # Use SDV data as context
                    num_samples
                )
                sdv_synthetic[text_col] = generated_texts
        
        # Step 3: Validate if requested
        if validate and self.processed_data is not None:
            print("3️⃣ Validating synthetic data quality...")
            # Validate only non-text columns with SDV
            non_text_synthetic = sdv_synthetic.drop(columns=text_cols, errors='ignore')
            if not non_text_synthetic.empty:
                evaluation_results = evaluate(non_text_synthetic, self.processed_data)
                print(f"✓ SDV validation score: {evaluation_results:.3f}")
        
        print("✅ Hybrid synthetic data generation completed!")
        return sdv_synthetic
    
    def save_results(self, synthetic_data: Optional[pd.DataFrame] = None, prefix: str = 'hybrid_feedback'):
        """Save all results and reports"""
        # Save analysis report
        if self.report is not None:
            report_file = f'{prefix}_analysis_report.html'
            self.report.show_html(report_file)
            print(f"📊 Analysis report saved: {report_file}")
        
        # Save processed data (SDV input)
        if self.processed_data is not None:
            processed_file = f'{prefix}_sdv_data.csv'
            self.processed_data.to_csv(processed_file, index=False)
            print(f"🔧 SDV input data saved: {processed_file}")
        
        # Save synthetic data
        if synthetic_data is not None:
            synthetic_file = f'{prefix}_synthetic_data.csv'
            synthetic_data.to_csv(synthetic_file, index=False)
            print(f"🎲 Hybrid synthetic data saved: {synthetic_file}")
        
        # Save text patterns
        if self.text_patterns:
            patterns_file = f'{prefix}_text_patterns.json'
            with open(patterns_file, 'w') as f:
                json.dump(self.text_patterns, f, indent=2, default=str)
            print(f"📝 Text patterns saved: {patterns_file}")
        
        # Save analysis summary
        summary_file = f'{prefix}_analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("HYBRID FEEDBACK DATA ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Original data shape: {self.data.shape if self.data is not None else 'N/A'}\n")
            f.write(f"SDV data shape: {self.processed_data.shape if self.processed_data is not None else 'N/A'}\n")
            f.write(f"Synthetic data shape: {synthetic_data.shape if synthetic_data is not None else 'N/A'}\n\n")
            
            f.write("Generation Strategy:\n")
            f.write("- Categorical & Numerical: SDV (Synthetic Data Vault)\n")
            f.write("- Free-text: Llama-cpp-python\n\n")
            
            if self.column_types:
                f.write("Column Types:\n")
                for col_type, cols in self.column_types.items():
                    f.write(f"  {col_type}: {len(cols)} columns\n")
                    for col in cols:
                        f.write(f"    - {col}\n")
                f.write("\n")
            
            if self.text_patterns:
                f.write("Text Analysis:\n")
                for col, patterns in self.text_patterns.items():
                    f.write(f"  {col}:\n")
                    f.write(f"    - Average length: {patterns['avg_length']:.1f} characters\n")
                    f.write(f"    - Average words: {patterns['word_count_avg']:.1f}\n")
                    f.write(f"    - Common starts: {patterns['common_starts'][:3]}\n")
        
        print(f"📋 Analysis summary saved: {summary_file}")

def main():
    """Example usage of the hybrid processor"""
    # You'll need to download a Llama model file (e.g., from Hugging Face)
    # Example: llama-2-7b-chat.Q4_K_M.gguf
    llama_model_path = "path/to/your/llama-model.gguf"  # Update this path
    
    processor = HybridFeedbackProcessor(llama_model_path)
    
    # Create sample data for demonstration
    print("Creating sample data for demonstration...")
    sample_data = create_comprehensive_sample_data()
    
    # Load and process data
    processor.load_data('comprehensive_feedback_data.csv')
    processor.analyze_columns()
    
    # Train SDV model
    processor.train_sdv_model('gaussian_copula')
    
    # Generate hybrid synthetic data
    synthetic_data = processor.generate_hybrid_data(300, validate=True)
    
    # Save all results
    processor.save_results(synthetic_data, prefix='hybrid_feedback')
    
    print("\n🎉 Hybrid processing completed! Check the generated files.")

def create_comprehensive_sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)
    n_samples = 500
    
    ratings = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.15, 0.35, 0.35])
    
    feedback_templates = {
        1: ["Terrible experience with this product", "Very disappointed, would not recommend", "Poor quality, waste of money"],
        2: ["Not satisfied with the purchase", "Below my expectations", "Could be much better"],
        3: ["Average product, nothing special", "It's okay, does the job", "Neither good nor bad"],
        4: ["Good experience overall", "Satisfied with this purchase", "Would recommend to others"],
        5: ["Excellent product, exceeded expectations!", "Amazing quality, love it!", "Perfect, exactly what I needed"]
    }
    
    feedback_texts = []
    for rating in ratings:
        base_text = np.random.choice(feedback_templates[rating])
        # Add some variation
        if np.random.random() > 0.6:
            additional = [" Fast delivery.", " Great customer service.", " Good value for money.", 
                         " Will buy again.", " Easy to use.", " High quality materials."]
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
        'ease_of_use': np.random.choice(['Very Easy', 'Easy', 'Moderate', 'Difficult'], n_samples),
    }
    
    df = pd.DataFrame(data)
    df.to_csv('comprehensive_feedback_data.csv', index=False)
    print("✓ Sample data created: comprehensive_feedback_data.csv")
    return df

if __name__ == "__main__":
    main()