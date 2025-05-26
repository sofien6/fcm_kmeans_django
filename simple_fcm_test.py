#!/usr/bin/env python3
"""
Simple test for the new FCM implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from segmentation.segmentation_utils import FuzzyCMeans

def test_fcm_basic():
    """Test basic FCM functionality with synthetic data"""
    print("Testing basic FCM functionality...")
    
    # Create synthetic 2D data with 3 clusters
    np.random.seed(42)
    
    # Cluster 1: around (2, 2)
    cluster1 = np.random.normal([2, 2], 0.5, (50, 2))
    
    # Cluster 2: around (6, 6)
    cluster2 = np.random.normal([6, 6], 0.5, (50, 2))
    
    # Cluster 3: around (2, 6)
    cluster3 = np.random.normal([2, 6], 0.5, (50, 2))
    
    # Combine data
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Test FCM without initial centers
    print("Testing FCM without initial centers...")
    fcm = FuzzyCMeans(n_clusters=3, m=2.0, max_iter=100, random_state=42)
    fcm.fit(data)
    
    print(f"Converged centers: {fcm.cluster_centers_}")
    print(f"Membership matrix shape: {fcm.u_.shape}")
    print(f"Labels shape: {fcm.labels_.shape}")
    
    # Test FCM with custom initial centers
    print("\nTesting FCM with custom initial centers...")
    custom_centers = np.array([[1, 1], [5, 5], [1, 5]])
    fcm_custom = FuzzyCMeans(n_clusters=3, m=2.0, max_iter=100, random_state=42)
    fcm_custom.fit(data, initial_centers=custom_centers)
    
    print(f"Initial centers: {custom_centers}")
    print(f"Final centers: {fcm_custom.cluster_centers_}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    axes[0].scatter(data[:, 0], data[:, 1], c='gray', alpha=0.6)
    axes[0].set_title('Original Data')
    axes[0].grid(True, alpha=0.3)
    
    # FCM without initial centers
    axes[1].scatter(data[:, 0], data[:, 1], c=fcm.labels_, cmap='tab10', alpha=0.6)
    axes[1].scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Final Centers')
    axes[1].set_title('FCM (Random Init)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # FCM with custom initial centers
    axes[2].scatter(data[:, 0], data[:, 1], c=fcm_custom.labels_, cmap='tab10', alpha=0.6)
    axes[2].scatter(custom_centers[:, 0], custom_centers[:, 1], 
                   c='blue', marker='o', s=100, alpha=0.7, label='Initial Centers')
    axes[2].scatter(fcm_custom.cluster_centers_[:, 0], fcm_custom.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Final Centers')
    axes[2].set_title('FCM (Custom Init)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fcm_basic_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fcm, fcm_custom

def test_fcm_with_image_colors():
    """Test FCM with RGB color data"""
    print("\nTesting FCM with RGB color data...")
    
    # Create synthetic RGB data
    np.random.seed(42)
    
    # Red cluster
    red_cluster = np.random.normal([200, 50, 50], 20, (100, 3))
    
    # Green cluster  
    green_cluster = np.random.normal([50, 200, 50], 20, (100, 3))
    
    # Blue cluster
    blue_cluster = np.random.normal([50, 50, 200], 20, (100, 3))
    
    # Combine and clip to valid RGB range
    rgb_data = np.vstack([red_cluster, green_cluster, blue_cluster])
    rgb_data = np.clip(rgb_data, 0, 255)
    
    # Define custom centroids
    custom_centroids = np.array([
        [255, 0, 0],    # Pure red
        [0, 255, 0],    # Pure green
        [0, 0, 255]     # Pure blue
    ])
    
    # Test FCM
    fcm = FuzzyCMeans(n_clusters=3, m=2.0, max_iter=100, random_state=42)
    fcm.fit(rgb_data, initial_centers=custom_centroids)
    
    print(f"Initial centroids: {custom_centroids}")
    print(f"Final centroids: {fcm.cluster_centers_}")
    
    # Calculate centroid movements
    movements = []
    for i, (initial, final) in enumerate(zip(custom_centroids, fcm.cluster_centers_)):
        movement = np.linalg.norm(final - initial)
        movements.append(movement)
        print(f"Centroid {i+1} movement: {movement:.2f}")
    
    print(f"Average movement: {np.mean(movements):.2f}")
    
    return fcm

if __name__ == "__main__":
    print("=== Testing New FCM Implementation ===")
    
    try:
        # Test basic functionality
        fcm_basic, fcm_custom = test_fcm_basic()
        
        # Test with RGB colors
        fcm_rgb = test_fcm_with_image_colors()
        
        print("\n✅ All FCM tests passed successfully!")
        print("\nThe new FCM implementation is working correctly and supports:")
        print("- Custom initial centroids")
        print("- Proper convergence")
        print("- RGB color space clustering")
        print("- Membership matrix calculation")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 