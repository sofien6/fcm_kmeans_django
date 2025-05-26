#!/usr/bin/env python3
"""
Comprehensive example of using the new robust FCM implementation
with predefined centroids for image segmentation.

This example demonstrates:
1. How to use FCM with custom centroids
2. How to compare K-means vs FCM results
3. How to visualize membership matrices
4. How to adjust fuzziness parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from segmentation.segmentation_utils import (
    preprocess_image,
    kmeans_segmentation_with_centroids,
    fcm_segmentation_with_centroids,
    fcm_segmentation_with_custom_centroids,
    FuzzyCMeans
)

def example_1_basic_fcm_with_custom_centroids():
    """
    Example 1: Basic FCM segmentation with user-defined centroids
    """
    print("=== Example 1: Basic FCM with Custom Centroids ===")
    
    # Create a simple synthetic image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add colored regions
    image[50:100, 50:100] = [255, 0, 0]    # Red square
    image[100:150, 50:100] = [0, 255, 0]   # Green square
    image[50:100, 100:150] = [0, 0, 255]   # Blue square
    image[100:150, 100:150] = [255, 255, 0] # Yellow square
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Define custom centroids (what we expect the colors to be)
    custom_centroids = np.array([
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green  
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [0, 0, 0]         # Black (background)
    ])
    
    k = len(custom_centroids)
    
    print(f"Image shape: {image.shape}")
    print(f"Using {k} custom centroids:")
    for i, centroid in enumerate(custom_centroids):
        print(f"  Centroid {i+1}: RGB{tuple(centroid)}")
    
    # Apply FCM with custom centroids
    fcm_segmented, refined_centroids, fcm_positions, membership = fcm_segmentation_with_custom_centroids(
        image, custom_centroids, k, m=2.0, max_iter=100
    )
    
    # Show results
    print("\nResults:")
    print("Original centroids -> Refined centroids:")
    for i, (orig, refined) in enumerate(zip(custom_centroids, refined_centroids)):
        movement = np.linalg.norm(refined - orig)
        print(f"  C{i+1}: {orig} -> [{refined[0]:.1f}, {refined[1]:.1f}, {refined[2]:.1f}] (movement: {movement:.2f})")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Synthetic Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(fcm_segmented, cmap='tab10')
    axes[0, 1].set_title('FCM Segmentation')
    axes[0, 1].axis('off')
    
    # Show membership for first few clusters
    for i in range(min(3, k)):
        row = i // 3
        col = (i % 3) + (0 if i < 3 else 0)
        if i < 3:
            axes[0, 2].imshow(membership[i].reshape(image.shape[:2]), cmap='viridis')
            axes[0, 2].set_title(f'Membership Map - Cluster {i+1}')
            axes[0, 2].axis('off')
        else:
            axes[1, col].imshow(membership[i].reshape(image.shape[:2]), cmap='viridis')
            axes[1, col].set_title(f'Membership Map - Cluster {i+1}')
            axes[1, col].axis('off')
    
    # Show uncertainty map (entropy)
    entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
    entropy_map = entropy.reshape(image.shape[:2])
    axes[1, 1].imshow(entropy_map, cmap='hot')
    axes[1, 1].set_title('Membership Uncertainty (Entropy)')
    axes[1, 1].axis('off')
    
    # Show true color reconstruction
    true_color_image = np.zeros_like(image)
    for i in range(k):
        mask = fcm_segmented == i
        true_color_image[mask] = refined_centroids[i].astype(np.uint8)
    
    axes[1, 2].imshow(true_color_image)
    axes[1, 2].set_title('True Color Reconstruction')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('fcm_example_1_custom_centroids.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return image, fcm_segmented, refined_centroids, membership

def example_2_compare_kmeans_vs_fcm():
    """
    Example 2: Compare K-means initialization vs custom centroid initialization
    """
    print("\n=== Example 2: K-means vs Custom FCM Comparison ===")
    
    # Load a real image (you can replace this with your own image path)
    try:
        # Try to find an uploaded image
        import os
        upload_dir = "media/uploads"
        if os.path.exists(upload_dir):
            image_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if image_files:
                image_path = os.path.join(upload_dir, image_files[0])
                print(f"Using image: {image_path}")
                image, original_image, is_grayscale = preprocess_image(image_path)
            else:
                raise FileNotFoundError("No images found")
        else:
            raise FileNotFoundError("Upload directory not found")
    except:
        print("No real image found, creating synthetic image...")
        # Create a more complex synthetic image
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Create gradient background
        for i in range(300):
            for j in range(300):
                image[i, j] = [i//3, j//3, (i+j)//6]
        
        # Add some distinct regions
        cv2.circle(image, (100, 100), 50, (255, 0, 0), -1)      # Red circle
        cv2.circle(image, (200, 100), 50, (0, 255, 0), -1)      # Green circle
        cv2.circle(image, (150, 200), 50, (0, 0, 255), -1)      # Blue circle
        
        original_image = image.copy()
        is_grayscale = False
    
    k = 4
    
    # Method 1: K-means initialization
    print("Running K-means segmentation...")
    kmeans_segmented, kmeans_centroids, kmeans_positions = kmeans_segmentation_with_centroids(image, k)
    
    print("Running FCM with K-means initialization...")
    fcm_kmeans_segmented, fcm_kmeans_centroids, fcm_kmeans_positions = fcm_segmentation_with_centroids(
        image, kmeans_segmented, kmeans_centroids, k
    )
    
    # Method 2: Custom centroids
    if is_grayscale:
        custom_centroids = np.array([
            [64, 64, 64],      # Dark
            [128, 128, 128],   # Medium
            [192, 192, 192],   # Light
            [255, 255, 255]    # White
        ])
    else:
        custom_centroids = np.array([
            [255, 100, 100],   # Reddish
            [100, 255, 100],   # Greenish
            [100, 100, 255],   # Bluish
            [200, 200, 100]    # Yellowish
        ])
    
    print("Running FCM with custom centroids...")
    fcm_custom_segmented, fcm_custom_centroids, fcm_custom_positions, membership = fcm_segmentation_with_custom_centroids(
        image, custom_centroids, k, m=2.0, max_iter=100
    )
    
    # Compare results
    print("\nComparison Results:")
    print("K-means -> FCM centroid evolution:")
    for i, (km, fcm_km) in enumerate(zip(kmeans_centroids, fcm_kmeans_centroids)):
        movement = np.linalg.norm(fcm_km - km)
        print(f"  C{i+1}: [{km[0]:.1f}, {km[1]:.1f}, {km[2]:.1f}] -> [{fcm_km[0]:.1f}, {fcm_km[1]:.1f}, {fcm_km[2]:.1f}] (movement: {movement:.2f})")
    
    print("Custom -> FCM centroid evolution:")
    for i, (custom, fcm_custom) in enumerate(zip(custom_centroids, fcm_custom_centroids)):
        movement = np.linalg.norm(fcm_custom - custom)
        print(f"  C{i+1}: [{custom[0]:.1f}, {custom[1]:.1f}, {custom[2]:.1f}] -> [{fcm_custom[0]:.1f}, {fcm_custom[1]:.1f}, {fcm_custom[2]:.1f}] (movement: {movement:.2f})")
    
    # Calculate differences between methods
    diff_kmeans_custom = np.sum(fcm_kmeans_segmented != fcm_custom_segmented)
    total_pixels = fcm_kmeans_segmented.size
    diff_percentage = (diff_kmeans_custom / total_pixels) * 100
    
    print(f"\nPixel differences between methods: {diff_kmeans_custom}/{total_pixels} ({diff_percentage:.2f}%)")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Top row: Original and segmentation results
    if is_grayscale:
        axes[0, 0].imshow(original_image, cmap='gray')
    else:
        axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(kmeans_segmented, cmap='tab10')
    axes[0, 1].set_title('K-means Segmentation')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(fcm_kmeans_segmented, cmap='tab10')
    axes[0, 2].set_title('FCM (K-means Init)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(fcm_custom_segmented, cmap='tab10')
    axes[0, 3].set_title('FCM (Custom Init)')
    axes[0, 3].axis('off')
    
    # Bottom row: Analysis
    # Difference map
    diff_map = (fcm_kmeans_segmented != fcm_custom_segmented).astype(int)
    axes[1, 0].imshow(diff_map, cmap='Reds')
    axes[1, 0].set_title(f'Differences ({diff_percentage:.1f}%)')
    axes[1, 0].axis('off')
    
    # Membership uncertainty for custom FCM
    entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
    entropy_map = entropy.reshape(image.shape[:2])
    im1 = axes[1, 1].imshow(entropy_map, cmap='hot')
    axes[1, 1].set_title('Membership Uncertainty')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # True color reconstructions
    true_color_kmeans = np.zeros_like(image)
    for i in range(k):
        mask = fcm_kmeans_segmented == i
        true_color_kmeans[mask] = fcm_kmeans_centroids[i].astype(np.uint8)
    
    true_color_custom = np.zeros_like(image)
    for i in range(k):
        mask = fcm_custom_segmented == i
        true_color_custom[mask] = fcm_custom_centroids[i].astype(np.uint8)
    
    axes[1, 2].imshow(true_color_kmeans)
    axes[1, 2].set_title('True Colors (K-means Init)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(true_color_custom)
    axes[1, 3].set_title('True Colors (Custom Init)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('fcm_example_2_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'kmeans_result': (kmeans_segmented, kmeans_centroids),
        'fcm_kmeans_result': (fcm_kmeans_segmented, fcm_kmeans_centroids),
        'fcm_custom_result': (fcm_custom_segmented, fcm_custom_centroids),
        'difference_percentage': diff_percentage
    }

def example_3_fuzziness_parameter_effects():
    """
    Example 3: Demonstrate the effect of different fuzziness parameters
    """
    print("\n=== Example 3: Fuzziness Parameter Effects ===")
    
    # Create test image with overlapping regions
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Create overlapping circles with gradual transitions
    center1, center2 = (70, 100), (130, 100)
    for i in range(200):
        for j in range(200):
            dist1 = np.sqrt((i - center1[0])**2 + (j - center1[1])**2)
            dist2 = np.sqrt((i - center2[0])**2 + (j - center2[1])**2)
            
            if dist1 < 50:
                intensity = max(0, 255 - dist1 * 3)
                image[i, j] = [intensity, 0, 0]  # Red gradient
            elif dist2 < 50:
                intensity = max(0, 255 - dist2 * 3)
                image[i, j] = [0, intensity, 0]  # Green gradient
            else:
                image[i, j] = [50, 50, 50]  # Gray background
    
    # Test different fuzziness parameters
    m_values = [1.1, 1.5, 2.0, 2.5, 3.0]
    custom_centroids = np.array([
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [50, 50, 50]    # Gray
    ])
    
    k = len(custom_centroids)
    
    print(f"Testing fuzziness parameters: {m_values}")
    
    fig, axes = plt.subplots(3, len(m_values), figsize=(4*len(m_values), 12))
    
    results = {}
    
    for i, m in enumerate(m_values):
        print(f"Testing m = {m}")
        
        # Run FCM with current m value
        fcm_segmented, fcm_centroids, fcm_positions, membership = fcm_segmentation_with_custom_centroids(
            image, custom_centroids, k, m=m, max_iter=100
        )
        
        results[m] = {
            'segmented': fcm_segmented,
            'centroids': fcm_centroids,
            'membership': membership
        }
        
        # Plot segmentation
        axes[0, i].imshow(fcm_segmented, cmap='tab10')
        axes[0, i].set_title(f'Segmentation (m={m})')
        axes[0, i].axis('off')
        
        # Plot membership uncertainty (entropy)
        entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
        entropy_map = entropy.reshape(image.shape[:2])
        
        im1 = axes[1, i].imshow(entropy_map, cmap='hot')
        axes[1, i].set_title(f'Uncertainty (m={m})')
        axes[1, i].axis('off')
        
        # Plot membership overlap (how fuzzy the boundaries are)
        # Calculate the maximum membership value for each pixel
        max_membership = np.max(membership, axis=0)
        fuzziness_map = (1 - max_membership).reshape(image.shape[:2])
        
        im2 = axes[2, i].imshow(fuzziness_map, cmap='viridis')
        axes[2, i].set_title(f'Fuzziness (m={m})')
        axes[2, i].axis('off')
        
        # Calculate average fuzziness
        avg_fuzziness = np.mean(fuzziness_map)
        print(f"  Average fuzziness: {avg_fuzziness:.3f}")
    
    # Add original image for reference
    fig.suptitle('Effect of Fuzziness Parameter (m) on FCM Segmentation', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('fcm_example_3_fuzziness_effects.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    print("🎯 Comprehensive FCM Examples with Custom Centroids")
    print("=" * 60)
    
    try:
        # Run all examples
        print("Running Example 1: Basic FCM with Custom Centroids")
        result1 = example_1_basic_fcm_with_custom_centroids()
        
        print("\nRunning Example 2: K-means vs Custom FCM Comparison")
        result2 = example_2_compare_kmeans_vs_fcm()
        
        print("\nRunning Example 3: Fuzziness Parameter Effects")
        result3 = example_3_fuzziness_parameter_effects()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n📋 Summary of New FCM Features:")
        print("1. ✅ Custom centroid initialization")
        print("2. ✅ Robust convergence algorithm")
        print("3. ✅ Membership matrix access")
        print("4. ✅ Adjustable fuzziness parameter")
        print("5. ✅ No dependency on problematic scikit-fuzzy")
        print("6. ✅ Full integration with existing K-means workflow")
        
        print("\n🚀 How to use in your Django app:")
        print("from segmentation.segmentation_utils import fcm_segmentation_with_custom_centroids")
        print("result = fcm_segmentation_with_custom_centroids(image, your_centroids, k)")
        
    except Exception as e:
        print(f"❌ Error during examples: {e}")
        import traceback
        traceback.print_exc() 