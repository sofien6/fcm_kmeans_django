#!/usr/bin/env python3
"""
Test script for the new robust FCM implementation with custom centroids
"""

import numpy as np
import matplotlib.pyplot as plt
from segmentation.segmentation_utils import (
    preprocess_image, 
    kmeans_segmentation_with_centroids,
    fcm_segmentation_with_centroids,
    fcm_segmentation_with_custom_centroids,
    visualize_fcm_result
)

def test_fcm_with_custom_centroids(image_path, custom_centroids=None):
    """
    Test FCM segmentation with custom centroids
    """
    print("Testing FCM with custom centroids...")
    
    # Load and preprocess image
    image, original_image, is_grayscale = preprocess_image(image_path)
    print(f"Image shape: {image.shape}, Grayscale: {is_grayscale}")
    
    # If no custom centroids provided, use some example ones
    if custom_centroids is None:
        if is_grayscale:
            # For grayscale images, use intensity-based centroids
            custom_centroids = np.array([
                [50, 50, 50],      # Dark region
                [128, 128, 128],   # Medium region  
                [200, 200, 200]    # Light region
            ])
            k = 3
        else:
            # For color images, use color-based centroids
            custom_centroids = np.array([
                [255, 0, 0],       # Red
                [0, 255, 0],       # Green
                [0, 0, 255],       # Blue
                [255, 255, 0],     # Yellow
                [128, 128, 128]    # Gray
            ])
            k = 5
    else:
        k = len(custom_centroids)
    
    print(f"Using {k} custom centroids:")
    for i, centroid in enumerate(custom_centroids):
        print(f"  Centroid {i+1}: {centroid}")
    
    # First, run K-means for comparison
    print("\nRunning K-means for comparison...")
    kmeans_segmented, kmeans_centroids, kmeans_positions = kmeans_segmentation_with_centroids(image, k)
    
    # Run FCM with K-means initialization
    print("\nRunning FCM with K-means initialization...")
    fcm_kmeans_segmented, fcm_kmeans_centroids, fcm_kmeans_positions = fcm_segmentation_with_centroids(
        image, kmeans_segmented, kmeans_centroids, k
    )
    
    # Run FCM with custom centroids
    print("\nRunning FCM with custom centroids...")
    fcm_custom_segmented, fcm_custom_centroids, fcm_custom_positions, membership_matrix = fcm_segmentation_with_custom_centroids(
        image, custom_centroids, k, m=2.0, max_iter=100
    )
    
    # Display results
    print("\nResults:")
    print(f"K-means centroids shape: {kmeans_centroids.shape}")
    print(f"FCM (K-means init) centroids shape: {fcm_kmeans_centroids.shape}")
    print(f"FCM (custom init) centroids shape: {fcm_custom_centroids.shape}")
    print(f"Membership matrix shape: {membership_matrix.shape}")
    
    # Show centroid evolution
    print("\nCentroid Evolution:")
    print("Original custom centroids:")
    for i, centroid in enumerate(custom_centroids):
        print(f"  C{i+1}: [{centroid[0]:6.1f}, {centroid[1]:6.1f}, {centroid[2]:6.1f}]")
    
    print("Final FCM centroids:")
    for i, centroid in enumerate(fcm_custom_centroids):
        print(f"  C{i+1}: [{centroid[0]:6.1f}, {centroid[1]:6.1f}, {centroid[2]:6.1f}]")
    
    # Calculate centroid movement
    movements = []
    for i, (orig, final) in enumerate(zip(custom_centroids, fcm_custom_centroids)):
        movement = np.linalg.norm(final - orig)
        movements.append(movement)
        print(f"  Movement C{i+1}: {movement:.2f}")
    
    print(f"Average centroid movement: {np.mean(movements):.2f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    if is_grayscale:
        axes[0, 0].imshow(original_image, cmap='gray')
    else:
        axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # K-means result
    axes[0, 1].imshow(kmeans_segmented, cmap='tab10')
    axes[0, 1].set_title('K-means Segmentation')
    axes[0, 1].axis('off')
    
    # FCM with K-means init
    axes[0, 2].imshow(fcm_kmeans_segmented, cmap='tab10')
    axes[0, 2].set_title('FCM (K-means Init)')
    axes[0, 2].axis('off')
    
    # FCM with custom centroids
    axes[1, 0].imshow(fcm_custom_segmented, cmap='tab10')
    axes[1, 0].set_title('FCM (Custom Centroids)')
    axes[1, 0].axis('off')
    
    # Membership visualization (show membership for first cluster)
    axes[1, 1].imshow(membership_matrix[0].reshape(image.shape[:2]), cmap='viridis')
    axes[1, 1].set_title('Membership Map (Cluster 1)')
    axes[1, 1].axis('off')
    
    # Difference between methods
    diff = (fcm_custom_segmented != fcm_kmeans_segmented).astype(int)
    axes[1, 2].imshow(diff, cmap='Reds')
    axes[1, 2].set_title('Difference: Custom vs K-means Init')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('fcm_custom_test_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'kmeans_result': (kmeans_segmented, kmeans_centroids, kmeans_positions),
        'fcm_kmeans_result': (fcm_kmeans_segmented, fcm_kmeans_centroids, fcm_kmeans_positions),
        'fcm_custom_result': (fcm_custom_segmented, fcm_custom_centroids, fcm_custom_positions),
        'membership_matrix': membership_matrix,
        'centroid_movements': movements
    }

def demonstrate_different_fuzziness_parameters(image_path, custom_centroids, m_values=[1.5, 2.0, 2.5, 3.0]):
    """
    Demonstrate how different fuzziness parameters affect FCM results
    """
    print(f"\nDemonstrating different fuzziness parameters: {m_values}")
    
    # Load image
    image, original_image, is_grayscale = preprocess_image(image_path)
    k = len(custom_centroids)
    
    fig, axes = plt.subplots(2, len(m_values), figsize=(4*len(m_values), 8))
    
    for i, m in enumerate(m_values):
        print(f"\nTesting with m={m}")
        
        # Run FCM with different m values
        fcm_segmented, fcm_centroids, fcm_positions, membership = fcm_segmentation_with_custom_centroids(
            image, custom_centroids, k, m=m, max_iter=100
        )
        
        # Plot segmentation result
        axes[0, i].imshow(fcm_segmented, cmap='tab10')
        axes[0, i].set_title(f'FCM Segmentation (m={m})')
        axes[0, i].axis('off')
        
        # Plot membership uncertainty (entropy)
        # Calculate entropy of membership for each pixel
        entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
        entropy_map = entropy.reshape(image.shape[:2])
        
        im = axes[1, i].imshow(entropy_map, cmap='hot')
        axes[1, i].set_title(f'Membership Uncertainty (m={m})')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('fcm_fuzziness_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default image path - you can change this
        image_path = "media/uploads/your_test_image.jpg"
        print(f"No image path provided, using default: {image_path}")
    
    try:
        # Test with default centroids
        print("=== Testing FCM with default centroids ===")
        results = test_fcm_with_custom_centroids(image_path)
        
        # Test with custom RGB centroids
        print("\n=== Testing FCM with custom RGB centroids ===")
        custom_rgb_centroids = np.array([
            [200, 50, 50],     # Reddish
            [50, 200, 50],     # Greenish
            [50, 50, 200],     # Bluish
            [200, 200, 50]     # Yellowish
        ])
        
        results_custom = test_fcm_with_custom_centroids(image_path, custom_rgb_centroids)
        
        # Demonstrate different fuzziness parameters
        print("\n=== Testing different fuzziness parameters ===")
        demonstrate_different_fuzziness_parameters(image_path, custom_rgb_centroids)
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 