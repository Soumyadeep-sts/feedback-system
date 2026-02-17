"""
Quick Start for Hybrid Feedback Processor
SDV for categorical/numerical + Llama for text generation
"""

from hybrid_feedback_processor import HybridFeedbackProcessor, create_comprehensive_sample_data
import os

def hybrid_quick_start(llama_model_path=None):
    """Quick example of hybrid approach"""
    
    print("🚀 HYBRID FEEDBACK PROCESSOR - QUICK START")
    print("="*60)
    print("📊 SDV for categorical/numerical data")
    print("🦙 Llama-cpp-python for text generation")
    print("="*60)
    
    # Step 1: Create sample data
    print("\n1️⃣ Creating sample feedback data...")
    create_comprehensive_sample_data()
    
    # Step 2: Initialize hybrid processor
    print("\n2️⃣ Initializing hybrid processor...")
    if llama_model_path and os.path.exists(llama_model_path):
        processor = HybridFeedbackProcessor(llama_model_path)
        print(f"✓ Llama model loaded from: {llama_model_path}")
    else:
        processor = HybridFeedbackProcessor()
        print("⚠️ No Llama model provided - will use pattern-based text generation")
        print("   To use Llama: download a .gguf model and provide the path")
    
    # Step 3: Load data
    print("\n3️⃣ Loading data...")
    processor.load_data('comprehensive_feedback_data.csv')
    
    # Step 4: Analyze columns
    print("\n4️⃣ Analyzing columns...")
    column_types = processor.analyze_columns()
    
    # Step 5: Train SDV model (for categorical/numerical only)
    print("\n5️⃣ Training SDV model for categorical/numerical data...")
    sdv_model = processor.train_sdv_model('gaussian_copula')
    
    # Step 6: Generate hybrid synthetic data
    print("\n6️⃣ Generating hybrid synthetic data...")
    synthetic_data = processor.generate_hybrid_data(num_samples=200, validate=True)
    
    # Step 7: Save results
    print("\n7️⃣ Saving results...")
    processor.save_results(synthetic_data, prefix='hybrid_quickstart')
    
    print("\n✅ HYBRID PROCESSING COMPLETED!")
    print("\nGenerated files:")
    print("   📊 hybrid_quickstart_analysis_report.html - Visual analysis")
    print("   🔧 hybrid_quickstart_sdv_data.csv - Data used for SDV")
    print("   🎲 hybrid_quickstart_synthetic_data.csv - Final synthetic dataset")
    print("   📝 hybrid_quickstart_text_patterns.json - Text analysis patterns")
    print("   📋 hybrid_quickstart_analysis_summary.txt - Summary report")
    
    return processor, synthetic_data

def show_comparison(synthetic_data):
    """Show comparison between original and synthetic data"""
    if synthetic_data is not None:
        print(f"\n📈 SYNTHETIC DATA PREVIEW:")
        print(f"Shape: {synthetic_data.shape}")
        print(f"Columns: {list(synthetic_data.columns)}")
        
        # Show sample of text generation
        text_cols = [col for col in synthetic_data.columns if 'text' in col.lower() or 'feedback' in col.lower() or 'comment' in col.lower()]
        if text_cols:
            print(f"\n📝 Sample generated text from '{text_cols[0]}':")
            for i, text in enumerate(synthetic_data[text_cols[0]].head(3)):
                print(f"   {i+1}. {text}")
        
        # Show categorical/numerical samples
        print(f"\n📊 Sample categorical/numerical data:")
        non_text_cols = [col for col in synthetic_data.columns if col not in text_cols][:5]
        print(synthetic_data[non_text_cols].head(3))

if __name__ == "__main__":
    # Example usage
    # Download a Llama model (e.g., from https://huggingface.co/models?search=gguf)
    # Popular options:
    # - llama-2-7b-chat.Q4_K_M.gguf (smaller, faster)
    # - llama-2-13b-chat.Q4_K_M.gguf (larger, better quality)
    
    # Update this path to your downloaded model
    llama_model_path = None  # e.g., "models/llama-2-7b-chat.Q4_K_M.gguf"
    
    processor, synthetic_data = hybrid_quick_start(llama_model_path)
    show_comparison(synthetic_data)