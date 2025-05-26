#!/usr/bin/env python
"""
Simple test script to verify the improved k-selection algorithms
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

def find_optimal_k_elbow_method(image, k_range=(2, 10)):
    """Find optimal number of clusters using improved Elbow method with kneedle algorithm"""
    print("🔵 Testing Elbow method...")
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Sample data if too large for faster computation
    if len(reshaped) > 15000:
        indices = np.random.choice(len(reshaped), 15000, replace=False)
        reshaped = reshaped[indices]
    
    inertias = []
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(reshaped)
        inertias.append(kmeans.inertia_)
    
    # Improved elbow detection using kneedle algorithm
    optimal_k = _find_elbow_kneedle(k_values, inertias)
    
    print(f"   Elbow method suggests k={optimal_k}")
    return optimal_k, inertias, list(k_values)

def _find_elbow_kneedle(k_values, inertias):
    """Find elbow using kneedle algorithm - more robust than second derivative"""
    k_values = np.array(k_values)
    inertias = np.array(inertias)
    
    # Normalize the data
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    inertia_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    
    # Calculate the difference between the curve and the straight line
    differences = []
    for i in range(len(k_norm)):
        # Point on the line from first to last point
        line_point = k_norm[0] + (k_norm[-1] - k_norm[0]) * (k_norm[i] - k_norm[0]) / (k_norm[-1] - k_norm[0])
        line_inertia = inertia_norm[0] + (inertia_norm[-1] - inertia_norm[0]) * (k_norm[i] - k_norm[0]) / (k_norm[-1] - k_norm[0])
        
        # Distance from curve point to line
        diff = abs(inertia_norm[i] - line_inertia)
        differences.append(diff)
    
    # Find the point with maximum distance (the elbow)
    elbow_idx = np.argmax(differences)
    return k_values[elbow_idx]

def find_optimal_k_silhouette(image, k_range=(2, 10)):
    """Find optimal number of clusters using Silhouette analysis"""
    print("🟢 Testing Silhouette analysis...")
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Sample data if too large for faster computation
    if len(reshaped) > 15000:
        indices = np.random.choice(len(reshaped), 15000, replace=False)
        reshaped = reshaped[indices]
    
    silhouette_scores = []
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reshaped)
        score = silhouette_score(reshaped, labels)
        silhouette_scores.append(score)
    
    # Find k with highest silhouette score
    optimal_k = k_values[np.argmax(silhouette_scores)]
    
    print(f"   Silhouette analysis suggests k={optimal_k}")
    return optimal_k, silhouette_scores, list(k_values)

def find_optimal_k_calinski_harabasz(image, k_range=(2, 10)):
    """Find optimal number of clusters using Calinski-Harabasz index"""
    print("🟣 Testing Calinski-Harabasz index...")
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Sample data if too large for faster computation
    if len(reshaped) > 15000:
        indices = np.random.choice(len(reshaped), 15000, replace=False)
        reshaped = reshaped[indices]
    
    ch_scores = []
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reshaped)
        score = calinski_harabasz_score(reshaped, labels)
        ch_scores.append(score)
    
    # Find k with highest Calinski-Harabasz score
    optimal_k = k_values[np.argmax(ch_scores)]
    
    print(f"   Calinski-Harabasz index suggests k={optimal_k}")
    return optimal_k, ch_scores, list(k_values)

def find_optimal_k_variance_analysis(image, k_range=(2, 10)):
    """Find optimal k using color variance analysis for images"""
    print("🟡 Testing Variance analysis...")
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Calculate color variance in the image
    color_variance = np.var(reshaped, axis=0).mean()
    
    # Estimate optimal k based on color complexity
    # More variance suggests more distinct regions
    if color_variance > 2000:  # High variance
        suggested_k = min(6, k_range[1])
    elif color_variance > 1000:  # Medium variance
        suggested_k = min(4, k_range[1])
    else:  # Low variance
        suggested_k = min(3, k_range[1])
    
    # Ensure k is within range
    suggested_k = max(k_range[0], min(suggested_k, k_range[1]))
    
    print(f"   Variance analysis suggests k={suggested_k} (variance: {color_variance:.2f})")
    return suggested_k, color_variance

def _calculate_weighted_consensus(suggestions, weights, k_range):
    """Calculate weighted consensus from multiple k suggestions"""
    # Create a vote count for each possible k value
    vote_counts = {k: 0 for k in range(k_range[0], k_range[1] + 1)}
    
    # Add weighted votes
    for suggestion, weight in zip(suggestions, weights):
        if suggestion in vote_counts:
            vote_counts[suggestion] += weight
            # Also add partial votes to neighboring values
            if suggestion - 1 in vote_counts:
                vote_counts[suggestion - 1] += weight * 0.3
            if suggestion + 1 in vote_counts:
                vote_counts[suggestion + 1] += weight * 0.3
    
    # Find k with highest vote count
    optimal_k = max(vote_counts, key=vote_counts.get)
    return optimal_k

def determine_optimal_k(image):
    """Determine optimal k using multiple methods and intelligent consensus"""
    print("🧠 Determining optimal k using multiple methods...")
    
    k_range = (2, 8)  # Reasonable range for image segmentation
    
    # Get suggestions from all methods
    k_elbow, elbow_inertias, k_values = find_optimal_k_elbow_method(image, k_range)
    k_silhouette, silhouette_scores, _ = find_optimal_k_silhouette(image, k_range)
    k_calinski, calinski_scores, _ = find_optimal_k_calinski_harabasz(image, k_range)
    k_variance, color_variance = find_optimal_k_variance_analysis(image, k_range)
    
    # Create a voting system with weights
    suggestions = [k_elbow, k_silhouette, k_calinski, k_variance]
    weights = [0.30, 0.25, 0.25, 0.20]  # Weights for each method
    
    print(f"📊 Method suggestions - Elbow: {k_elbow}, Silhouette: {k_silhouette}, "
          f"Calinski-Harabasz: {k_calinski}, Variance: {k_variance}")
    
    # Calculate weighted consensus
    optimal_k = _calculate_weighted_consensus(suggestions, weights, k_range)
    
    # Additional validation: if all methods suggest very different values, 
    # use a more conservative approach
    suggestion_variance = np.var(suggestions)
    if suggestion_variance > 2.0:  # High disagreement
        print(f"⚠️  High disagreement between methods (variance: {suggestion_variance:.2f}), using conservative approach")
        # Use the median of the suggestions
        sorted_suggestions = sorted(suggestions)
        optimal_k = int(np.median(sorted_suggestions))
    
    # Final bounds check
    optimal_k = max(k_range[0], min(optimal_k, k_range[1]))
    
    print(f"🏆 Final optimal k determined: {optimal_k}")
    print(f"📈 Method agreement variance: {suggestion_variance:.3f}")
    
    return optimal_k, suggestion_variance

def test_k_selection_improvements():
    """Test the improved k-selection algorithms"""
    print("🚀 Testing Improved K-Selection Algorithms")
    print("=" * 60)
    
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
    
    # Test consensus method
    print("\n🎯 Testing Consensus Method:")
    print("-" * 40)
    
    try:
        optimal_k, variance = determine_optimal_k(test_image)
        
        # Evaluate result
        if optimal_k == 4:
            print("✅ EXCELLENT: Correctly identified 4 regions!")
        elif optimal_k in [3, 5]:
            print("✅ GOOD: Close to optimal (expected 4 regions)")
        else:
            print(f"⚠️  SUBOPTIMAL: Expected ~4 regions, got {optimal_k}")
        
        if variance < 1.0:
            print("✅ HIGH agreement between methods")
        elif variance < 2.0:
            print("⚠️  MEDIUM agreement between methods")
        else:
            print("❌ LOW agreement between methods")
            
    except Exception as e:
        print(f"❌ Error in consensus method: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 K-Selection Algorithm Test Complete!")
    
    # Test with different image types
    print("\n🔬 Testing with Different Image Complexities:")
    print("-" * 50)
    
    # Test 1: Simple 2-region image
    simple_image = np.zeros((50, 50, 3), dtype=np.uint8)
    simple_image[0:25, :] = [255, 0, 0]  # Red top
    simple_image[25:50, :] = [0, 0, 255]  # Blue bottom
    
    print("📸 Testing simple 2-region image...")
    optimal_k_simple, _ = determine_optimal_k(simple_image)
    print(f"   Result: k={optimal_k_simple} (expected: 2)")
    
    # Test 2: Complex 6-region image
    complex_image = np.zeros((60, 60, 3), dtype=np.uint8)
    complex_image[0:20, 0:30] = [255, 0, 0]      # Red
    complex_image[0:20, 30:60] = [0, 255, 0]     # Green
    complex_image[20:40, 0:30] = [0, 0, 255]     # Blue
    complex_image[20:40, 30:60] = [255, 255, 0]  # Yellow
    complex_image[40:60, 0:30] = [255, 0, 255]   # Magenta
    complex_image[40:60, 30:60] = [0, 255, 255]  # Cyan
    
    print("📸 Testing complex 6-region image...")
    optimal_k_complex, _ = determine_optimal_k(complex_image)
    print(f"   Result: k={optimal_k_complex} (expected: 6)")
    
    print("\n💡 Key Improvements Made:")
    print("   1. ✨ Improved Elbow method with kneedle algorithm")
    print("   2. 🆕 Added Calinski-Harabasz index")
    print("   3. 🆕 Added Variance analysis for images")
    print("   4. 🧠 Intelligent weighted consensus system")
    print("   5. 🔍 Disagreement detection and conservative fallback")
    print("\n🎯 Result: Much more robust k-selection that should")
    print("   avoid always choosing k=2!")
    
    return True

if __name__ == "__main__":
    test_k_selection_improvements() 