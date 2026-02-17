"""
Quick Start Example - Simple usage of the feedback processor
"""

from advanced_feedback_processor import AdvancedFeedbackProcessor, create_comprehensive_sample_data

def quick_example():
    """Quick example showing basic usage"""
    
    print("🚀 FEEDBACK DATA PROCESSOR - QUICK START")
    print("="*50)
    
    # Step 1: Create sample data (replace with your data loading)
    print("\n1️⃣ Creating sample feedback data...")
    create_comprehensive_sample_data()
    
    # Step 2: Initialize processor
    print("\n2️⃣ Initializing processor...")
    processor = AdvancedFeedbackProcessor()
    
    # Step 3: Load your data
    print("\n3️⃣ Loading data...")
    processor.load_data('comprehensive_feedback_data.csv')
    
    # Step 4: Analyze columns (identifies categorical, numerical, text)
    print("\n4️⃣ Analyzing columns...")
    column_types = processor.analyze_columns()
    
    # Step 5: Train synthetic data model
    print("\n5️⃣ Training synthetic data model...")
    model = processor.train_synthetic_model('gaussian_copula')
    
    # Step 6: Generate synthetic data
    print("\n6️⃣ Generating synthetic data...")
    synthetic_data = processor.generate_synthetic_data(num_samples=300, validate=True)
    
    # Step 7: Save results
    print("\n7️⃣ Saving results...")
    processor.save_results(synthetic_data, prefix='quickstart')
    
    print("\n✅ COMPLETED! Check these files:")
    print("   📊 quickstart_analysis_report.html - Visual analysis")
    print("   🔧 quickstart_processed_data.csv - Cleaned data")
    print("   🎲 quickstart_synthetic_data.csv - Generated data")
    print("   📋 quickstart_analysis_summary.txt - Summary report")
    
    return processor, synthetic_data

if __name__ == "__main__":
    processor, synthetic_data = quick_example()
    
    # Show some basic stats
    if synthetic_data is not None:
        print(f"\n📈 SYNTHETIC DATA PREVIEW:")
        print(f"Shape: {synthetic_data.shape}")
        print(f"Columns: {list(synthetic_data.columns)}")
        print("\nFirst 3 rows:")
        print(synthetic_data.head(3))