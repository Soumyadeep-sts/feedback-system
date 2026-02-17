# Llama Model Setup Guide

This guide helps you set up a Llama model for text generation in the hybrid feedback processor.

## Quick Setup

### 1. Install llama-cpp-python
```bash
pip install llama-cpp-python
```

### 2. Download a Llama Model

You need a `.gguf` format model file. Here are recommended options:

#### Option A: Hugging Face (Recommended)
Visit [Hugging Face GGUF models](https://huggingface.co/models?search=gguf) and download:

**For beginners (smaller, faster):**
- `llama-2-7b-chat.Q4_K_M.gguf` (~4GB)
- `mistral-7b-instruct-v0.1.Q4_K_M.gguf` (~4GB)

**For better quality (larger, slower):**
- `llama-2-13b-chat.Q4_K_M.gguf` (~7GB)
- `codellama-13b-instruct.Q4_K_M.gguf` (~7GB)

#### Option B: Direct Download Examples
```bash
# Create models directory
mkdir models
cd models

# Download Llama-2 7B (example)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# Or download Mistral 7B
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### 3. Update Your Code
```python
from hybrid_feedback_processor import HybridFeedbackProcessor

# Point to your downloaded model
llama_model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
processor = HybridFeedbackProcessor(llama_model_path)
```

## Model Recommendations by Use Case

### For Development/Testing
- **Model**: `llama-2-7b-chat.Q4_K_M.gguf`
- **Size**: ~4GB
- **Speed**: Fast
- **Quality**: Good for testing

### For Production
- **Model**: `llama-2-13b-chat.Q4_K_M.gguf`
- **Size**: ~7GB  
- **Speed**: Moderate
- **Quality**: High quality text generation

### For Code/Technical Feedback
- **Model**: `codellama-13b-instruct.Q4_K_M.gguf`
- **Size**: ~7GB
- **Speed**: Moderate
- **Quality**: Specialized for technical content

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB+ (for 7B models)
- **Storage**: 5GB+ free space
- **CPU**: Modern multi-core processor

### Recommended
- **RAM**: 16GB+ (for 13B models)
- **Storage**: 10GB+ free space
- **CPU**: 8+ cores for faster generation

## Configuration Options

You can customize the Llama model behavior:

```python
processor = HybridFeedbackProcessor(
    llama_model_path="models/llama-2-7b-chat.Q4_K_M.gguf"
)

# The model is initialized with these defaults:
# - n_ctx=2048 (context window)
# - n_threads=4 (CPU threads)
# - verbose=False
```

## Troubleshooting

### Common Issues

**1. "Model file not found"**
- Check the file path is correct
- Ensure the file downloaded completely
- Use absolute path if relative path doesn't work

**2. "Out of memory"**
- Try a smaller model (7B instead of 13B)
- Close other applications
- Reduce n_ctx parameter

**3. "Slow generation"**
- Increase n_threads parameter
- Use a smaller model
- Consider GPU acceleration (requires special build)

### GPU Acceleration (Optional)

For faster generation, you can use GPU acceleration:

```bash
# Install with CUDA support (NVIDIA GPUs)
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Install with Metal support (Apple Silicon)
pip uninstall llama-cpp-python  
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

## Testing Your Setup

Run this test to verify everything works:

```python
from hybrid_feedback_processor import HybridFeedbackProcessor

# Test with your model path
processor = HybridFeedbackProcessor("path/to/your/model.gguf")

if processor.llama_model:
    print("✅ Llama model loaded successfully!")
    
    # Test generation
    test_prompt = "Generate a product review:"
    response = processor.llama_model(test_prompt, max_tokens=50)
    print("Test generation:", response['choices'][0]['text'])
else:
    print("❌ Model failed to load")
```

## Alternative: Pattern-Based Generation

If you can't set up Llama, the system will fall back to pattern-based text generation:

```python
# This will work without Llama model
processor = HybridFeedbackProcessor()  # No model path
synthetic_data = processor.generate_hybrid_data(100)
```

The pattern-based approach uses the original text patterns but won't be as sophisticated as Llama generation.