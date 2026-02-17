# Hybrid Feedback Data Processor

A comprehensive tool for analyzing feedback data and generating synthetic datasets using a hybrid approach:
- **SDV (Synthetic Data Vault)** for categorical and numerical data
- **Llama-cpp-python** for realistic free-text generation

## Features

- **Automatic Column Detection**: Identifies categorical, numerical, and free-text columns using pandas and sweetviz
- **Hybrid Generation Strategy**: 
  - Categorical/Numerical: SDV (Gaussian Copula or CTGAN)
  - Free-text: Llama language model with context awareness
- **Text Analysis**: Analyzes patterns, sentiment, and structure in feedback text
- **Visual Analysis**: Generates detailed HTML reports using sweetviz
- **Context-Aware Text Generation**: Uses other column values to generate relevant text
- **Data Validation**: Validates synthetic data quality

## Quick Start

```python
from hybrid_feedback_processor import HybridFeedbackProcessor

# Initialize with Llama model for text generation
processor = HybridFeedbackProcessor("path/to/llama-model.gguf")

# Load your data
processor.load_data('your_feedback_data.csv')

# Analyze columns automatically
processor.analyze_columns()

# Train SDV model for categorical/numerical data
processor.train_sdv_model('gaussian_copula')

# Generate hybrid synthetic data
synthetic_data = processor.generate_hybrid_data(1000)

# Save all results
processor.save_results(synthetic_data)
```

## Model Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download a Llama model** (see [Model Setup Guide](model_setup_guide.md)):
```bash
# Example: Download Llama-2 7B model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

3. **Run the quick start:**
```python
python hybrid_quick_start.py
```

## Files

- `hybrid_feedback_processor.py` - **Main hybrid processor** (SDV + Llama)
- `hybrid_quick_start.py` - Quick start example for hybrid approach
- `model_setup_guide.md` - Guide for setting up Llama models
- `feedback_data_processor.py` - Basic processor (SDV only)
- `advanced_feedback_processor.py` - Enhanced processor (SDV only)
- `quick_start_example.py` - Basic usage example
- `requirements.txt` - Required packages

## Hybrid Generation Strategy

### Categorical & Numerical Data (SDV)
- Uses Gaussian Copula or CTGAN
- Preserves statistical relationships
- Handles missing values intelligently
- Maintains data distributions

### Free-Text Data (Llama)
- Context-aware generation based on other columns
- Uses original text patterns and examples
- Generates realistic feedback comments
- Fallback to pattern-based if Llama unavailable

## Usage Examples

### 1. Hybrid Approach (Recommended)
```python
python hybrid_quick_start.py
```

### 2. Custom Model Path
```python
from hybrid_feedback_processor import HybridFeedbackProcessor

processor = HybridFeedbackProcessor("models/llama-2-7b-chat.Q4_K_M.gguf")
processor.load_data('your_data.csv')
processor.analyze_columns()
processor.train_sdv_model('gaussian_copula')
synthetic_data = processor.generate_hybrid_data(1000)
```

### 3. Different SDV Models
```python
# Gaussian Copula (faster, good for most cases)
processor.train_sdv_model('gaussian_copula')

# CTGAN (slower, better for complex relationships)
processor.train_sdv_model('ctgan', epochs=200)
```

### 4. Without Llama Model (Pattern-based text)
```python
# Will use pattern-based text generation
processor = HybridFeedbackProcessor()  # No model path
synthetic_data = processor.generate_hybrid_data(1000)
```

## Column Type Detection

The processor automatically identifies:

- **Categorical**: Rating scales, categories, yes/no responses
- **Numerical**: Scores, amounts, counts
- **Text**: Free-form feedback, comments, reviews

For text columns, it creates:
- **Context-aware text** using Llama model based on other column values
- **Pattern-based fallback** if Llama model unavailable
- **Sentiment-appropriate content** matching the rating/satisfaction scores

## Output Files

- `*_analysis_report.html` - Interactive sweetviz report
- `*_sdv_data.csv` - Processed data used for SDV training
- `*_synthetic_data.csv` - Final hybrid synthetic dataset
- `*_text_patterns.json` - Text analysis patterns for generation
- `*_analysis_summary.txt` - Text summary of analysis

## Data Format Support

- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)
- JSON files (`.json`)

## Tips for Best Results

1. **Download appropriate Llama model**: See [Model Setup Guide](model_setup_guide.md)
2. **Clean your data first**: Remove or handle extreme outliers
3. **Use meaningful column names**: Helps with automatic detection
4. **For large datasets**: Use CTGAN for better quality
5. **For quick prototyping**: Use Gaussian Copula
6. **Validate results**: Always check the generated data makes sense
7. **Context matters**: Llama generates better text when other columns provide context

## Example Data Structure

Your feedback data might look like:
```csv
customer_id,rating,satisfaction_score,category,feedback_text,recommend
1,5,8.5,Electronics,"Great product, fast delivery",Yes
2,3,6.2,Clothing,"Average quality, could be better",Maybe
3,1,2.1,Books,"Terrible experience, very disappointed",No
```

The processor will automatically:
- Detect `rating` and `recommend` as categorical → **SDV generation**
- Detect `satisfaction_score` as numerical → **SDV generation**
- Detect `feedback_text` as text → **Llama generation** (context-aware based on rating/category)
- Generate synthetic data maintaining these relationships