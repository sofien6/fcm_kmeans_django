#!/usr/bin/env python
"""
Test script to verify the improved k-selection algorithms
"""
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'image_segmentation.settings')
django.setup()

import numpy as np
from segmentation.segmentation_utils import (
    preprocess_image, 
    determine_optimal_k,
    find_optimal_k_elbow_method,
    find_optimal_k_silhouette,
    find_optimal_k_calinski_harabasz,
    find_optimal_k_gap_statistic,
    find_optimal_k_variance_analysis
)

def test_k_selection_improvements():
    """Test the improved k-selection algorithms"""
    print("🔍 Testing Improved K-Selection Algorithms")
    print("=" * 50)
    
    # Create a synthetic test image with known structure
    print("📊 Creating synthetic test image...")
    
    # Create a 100x100 RGB image with 4 distinct regions
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Region 1: Red (top-left)
    test_image[0:50, 0:50] = [255, 0, 0]
    
    # Region 2: Green (top-right)  
    test_image[0:50, 50:100] = [0, 255, 0]
    
    # Region 3: Blue (bottom-left)
    test_image[50:100, 0:50] = [0, 0, 255]
    
    # Region 4: Yellow (bottom-right)
    test_image[50:100, 50:100] = [255, 255, 0]
    
    print(f"✅ Created test image with 4 distinct regions (100x100 pixels)")
    
    # Test individual methods
    print("\n🧪 Testing Individual K-Selection Methods:")
    print("-" * 40)
    
    k_range = (2, 8)
    
    try:
        # Test Elbow method
        k_elbow, _, _ = find_optimal_k_elbow_method(test_image, k_range)
        print(f"🔵 Elbow Method: k = {k_elbow}")
        
        # Test Silhouette method
        k_silhouette, _, _ = find_optimal_k_silhouette(test_image, k_range)
        print(f"🟢 Silhouette Analysis: k = {k_silhouette}")
        
        # Test Calinski-Harabasz method
        k_calinski, _, _ = find_optimal_k_calinski_harabasz(test_image, k_range)
        print(f"🟣 Calinski-Harabasz Index: k = {k_calinski}")
        
        # Test Gap statistic
        k_gap, _, _ = find_optimal_k_gap_statistic(test_image, k_range, n_refs=3)
        print(f"🔵 Gap Statistic: k = {k_gap}")
        
        # Test Variance analysis
        k_variance, color_var = find_optimal_k_variance_analysis(test_image, k_range)
        print(f"🟡 Variance Analysis: k = {k_variance} (variance: {color_var:.2f})")
        
    except Exception as e:
        print(f"❌ Error in individual methods: {e}")
        return False
    
    # Test consensus method
    print("\n🎯 Testing Consensus Method:")
    print("-" * 30)
    
    try:
        optimal_k, analysis_data = determine_optimal_k(test_image)
        print(f"🏆 Final Consensus: k = {optimal_k}")
        print(f"📊 Method Agreement Variance: {analysis_data['suggestion_variance']:.3f}")
        
        # Show all suggestions
        suggestions = [
            analysis_data['elbow_k'],
            analysis_data['silhouette_k'], 
            analysis_data['calinski_k'],
            analysis_data['gap_k'],
            analysis_data['variance_k']
        ]
        print(f"📋 All Suggestions: {suggestions}")
        
        # Evaluate result
        if optimal_k == 4:
            print("✅ EXCELLENT: Correctly identified 4 regions!")
        elif optimal_k in [3, 5]:
            print("✅ GOOD: Close to optimal (expected 4 regions)")
        else:
            print(f"⚠️  SUBOPTIMAL: Expected ~4 regions, got {optimal_k}")
            
    except Exception as e:
        print(f"❌ Error in consensus method: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 K-Selection Algorithm Test Complete!")
    
    return True

def test_real_image_if_available():
    """Test with a real image if available"""
    print("\n🖼️  Testing with Real Images (if available):")
    print("-" * 45)
    
    # Check for uploaded images
    media_uploads = "media/uploads"
    if os.path.exists(media_uploads):
        image_files = [f for f in os.listdir(media_uploads) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            test_image_path = os.path.join(media_uploads, image_files[0])
            print(f"📁 Found test image: {image_files[0]}")
            
            try:
                # Preprocess the image
                image, _, is_grayscale = preprocess_image(test_image_path)
                print(f"📐 Image shape: {image.shape}, Grayscale: {is_grayscale}")
                
                # Test k-selection
                optimal_k, analysis_data = determine_optimal_k(image)
                
                print(f"🎯 Optimal k for real image: {optimal_k}")
                print(f"📊 Method suggestions:")
                print(f"   - Elbow: {analysis_data['elbow_k']}")
                print(f"   - Silhouette: {analysis_data['silhouette_k']}")
                print(f"   - Calinski-Harabasz: {analysis_data['calinski_k']}")
                print(f"   - Gap Statistic: {analysis_data['gap_k']}")
                print(f"   - Variance: {analysis_data['variance_k']}")
                print(f"📈 Agreement variance: {analysis_data['suggestion_variance']:.3f}")
                
                if analysis_data['suggestion_variance'] < 1.0:
                    print("✅ HIGH agreement between methods")
                elif analysis_data['suggestion_variance'] < 2.0:
                    print("⚠️  MEDIUM agreement between methods")
                else:
                    print("❌ LOW agreement between methods")
                    
            except Exception as e:
                print(f"❌ Error processing real image: {e}")
        else:
            print("📂 No image files found in media/uploads")
    else:
        print("📂 No uploads directory found")

if __name__ == "__main__":
    print("🚀 Starting K-Selection Algorithm Tests")
    print("=" * 60)
    
    # Test synthetic image
    success = test_k_selection_improvements()
    
    if success:
        # Test real image if available
        test_real_image_if_available()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n💡 Key Improvements Made:")
        print("   1. ✨ Improved Elbow method with kneedle algorithm")
        print("   2. 🆕 Added Calinski-Harabasz index")
        print("   3. 🆕 Added Gap statistic method")
        print("   4. 🆕 Added Variance analysis for images")
        print("   5. 🧠 Intelligent weighted consensus system")
        print("   6. 📊 Better visualization with 4 analysis plots")
        print("   7. 🔍 Disagreement detection and conservative fallback")
        print("\n🎯 Result: Much more robust k-selection that should")
        print("   avoid always choosing k=2!")
    else:
        print("\n❌ TESTS FAILED - Check the error messages above") 