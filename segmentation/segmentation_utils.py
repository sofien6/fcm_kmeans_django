import numpy as np
import cv2
import skfuzzy as fuzz
from sklearn.cluster import KMeans
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import uuid
import logging

# Get logger for this module
logger = logging.getLogger('segmentation')

def preprocess_image(image):
    """Preprocess the image for segmentation"""
    # Resize for faster processing if needed
    if max(image.shape[0], image.shape[1]) > 1000:
        scale = 1000 / max(image.shape[0], image.shape[1])
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height))
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to RGB if it's BGR (OpenCV default)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def kmeans_segmentation(image, n_clusters=5):
    """Segment image using K-means clustering"""
    logger.info(f"Starting K-means segmentation with {n_clusters} clusters")
    logger.info(f"Input image shape for K-means: {image.shape}")
    
    start_time = datetime.now()
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    logger.info(f"Reshaped data for K-means: {reshaped.shape}")
    
    # Apply KMeans
    logger.info(f"Initializing K-means with {n_clusters} clusters and random_state=42")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    logger.info("Fitting K-means model to image data...")
    labels = kmeans.fit_predict(reshaped)
    
    # Reshape back to the original image shape
    segmented_image = labels.reshape(h, w)
    
    # Get cluster centers for coloring
    centers = kmeans.cluster_centers_
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"K-means segmentation completed in {duration:.2f} seconds")
    logger.info(f"K-means centers: min={centers.min():.2f}, max={centers.max():.2f}, shape={centers.shape}")
    logger.info(f"K-means labels distribution: {[np.sum(labels == i) for i in range(n_clusters)]}")
    
    return segmented_image, centers

def fcm_segmentation(image, n_clusters=5, m=2, error=0.005, max_iter=1000):
    """Segment image using Fuzzy C-means clustering"""
    logger.info(f"Starting FCM segmentation with {n_clusters} clusters")
    logger.info(f"FCM parameters: m={m}, error={error}, max_iter={max_iter}")
    logger.info(f"Input image shape for FCM: {image.shape}")
    
    start_time = datetime.now()
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels).T  # Transpose for skfuzzy format
    logger.info(f"Reshaped data for FCM: {reshaped.shape}")
    
    # Apply FCM
    logger.info(f"Starting FCM clustering process with {n_clusters} clusters...")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        reshaped, n_clusters, m, error=error, maxiter=max_iter, init=None
    )
    
    logger.info(f"FCM convergence information: iterations={p}, final fuzzy partition coefficient={fpc}")
    
    # Get cluster membership for each pixel
    cluster_membership = np.argmax(u, axis=0)
    
    # Reshape back to the original image shape
    segmented_image = cluster_membership.reshape(h, w)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"FCM segmentation completed in {duration:.2f} seconds")
    logger.info(f"FCM centers: min={cntr.min():.2f}, max={cntr.max():.2f}, shape={cntr.shape}")
    logger.info(f"FCM labels distribution: {[np.sum(cluster_membership == i) for i in range(n_clusters)]}")
    
    return segmented_image, cntr.T  # Transpose back for consistency with KMeans

def hybrid_segmentation(image, n_clusters=5, media_root=None):
    """
    Hybrid segmentation using both K-means and FCM.
    Uses K-means for initial clustering and then FCM for refinement.
    """
    logger.info(f"Starting hybrid segmentation with {n_clusters} clusters")
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Get initial segmentation with KMeans (faster than FCM)
    logger.info("Performing initial K-means segmentation in hybrid approach")
    kmeans_segments, kmeans_centers = kmeans_segmentation(processed_image, n_clusters)
    
    # Save K-means result if media_root is provided
    kmeans_result_path = None
    if media_root:
        results_dir = os.path.join(media_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        kmeans_filename = f"KMeans_initial_{timestamp}_{unique_id}"
        
        # Create a colormap for the segments
        cmap = generate_color_map(n_clusters)
        
        # Create a figure
        plt.figure(figsize=(10, 8))
        plt.imshow(kmeans_segments, cmap=cmap)
        plt.title('K-means Segmentation (Initial Step)')
        plt.axis('off')
        
        # Save figure
        kmeans_result_path = os.path.join(results_dir, f"{kmeans_filename}.png")
        plt.savefig(kmeans_result_path)
        plt.close()
        
        # Create a colored segmentation output
        colored_segments = np.zeros_like(processed_image)
        for i in range(n_clusters):
            colored_segments[kmeans_segments == i] = np.array(cmap(i)[:3]) * 255
        
        # Save colored segmentation as a separate image
        kmeans_colored_path = os.path.join(results_dir, f"{kmeans_filename}_colored.png")
        cv2.imwrite(kmeans_colored_path, cv2.cvtColor(colored_segments.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        logger.info(f"K-means initial segmentation saved to {kmeans_result_path}")
    
    # Use FCM for refinement, using KMeans centers as initial centroids
    h, w, channels = processed_image.shape
    reshaped = processed_image.reshape(h * w, channels).T  # Transpose for skfuzzy format
    
    # Initialize FCM with KMeans centers
    logger.info("Using K-means results as initialization for FCM refinement")
    u0 = np.zeros((n_clusters, h * w))
    for i in range(h * w):
        u0[kmeans_segments.flatten()[i], i] = 1.0
    
    # Apply FCM
    logger.info("Starting FCM refinement in hybrid approach")
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        reshaped, n_clusters, 2, error=0.005, maxiter=100, init=u0
    )
    
    # Get cluster membership for each pixel
    cluster_membership = np.argmax(u, axis=0)
    
    # Reshape back to the original image shape
    segmented_image = cluster_membership.reshape(h, w)
    
    logger.info("FCM refinement completed successfully")
    logger.info("Hybrid segmentation completed successfully")
    
    if kmeans_result_path:
        return segmented_image, cntr.T, f"results/{os.path.basename(kmeans_result_path)}"
    else:
        return segmented_image, cntr.T, None

def generate_color_map(n_clusters):
    """Generate a colormap for visualizing segments"""
    # Generate random colors for each cluster
    np.random.seed(42)  # For reproducibility
    colors = np.random.rand(n_clusters, 3)
    return ListedColormap(colors)

def visualize_segmentation(original_image, segmented_image, centers, media_root, method_name):
    """Visualize segmentation results and save to file"""
    logger.info(f"Visualizing segmentation results for {method_name}")
    
    # Create a directory for results if it doesn't exist
    results_dir = os.path.join(media_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate a unique identifier for this segmentation
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{method_name}_{timestamp}_{unique_id}"
    
    # Get number of clusters
    n_clusters = len(np.unique(segmented_image))
    
    # Create a colormap for the segments
    cmap = generate_color_map(n_clusters)
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Display segmented image
    ax2.imshow(segmented_image, cmap=cmap)
    ax2.set_title(f'{method_name} Segmentation')
    ax2.axis('off')
    
    # Save figure
    plt.tight_layout()
    result_path = os.path.join(results_dir, f"{filename}.png")
    plt.savefig(result_path)
    plt.close(fig)
    
    # Create a colored segmentation output
    colored_segments = np.zeros_like(original_image)
    for i in range(n_clusters):
        colored_segments[segmented_image == i] = np.array(cmap(i)[:3]) * 255
    
    # Save colored segmentation as a separate image
    colored_path = os.path.join(results_dir, f"{filename}_colored.png")
    cv2.imwrite(colored_path, cv2.cvtColor(colored_segments.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    logger.info(f"Segmentation results saved to {result_path} and {colored_path}")
    return f"results/{filename}.png", f"results/{filename}_colored.png"

def segment_image_with_method(image_path, media_root, method='hybrid', n_clusters=5):
    """Segment an image using the specified method"""
    logger.info(f"Starting image segmentation process with method: {method}, clusters: {n_clusters}")
    logger.info(f"Input image: {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        error_msg = f"Failed to read image from {image_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    logger.info(f"Image preprocessed. Shape: {processed_image.shape}")
    
    kmeans_result_path = None
    
    if method == 'kmeans':
        segmented_image, centers = kmeans_segmentation(processed_image, n_clusters)
        method_name = 'KMeans'
    elif method == 'fcm':
        segmented_image, centers = fcm_segmentation(processed_image, n_clusters)
        method_name = 'FCM'
    else:  # Default to hybrid
        segmented_image, centers, kmeans_result_path = hybrid_segmentation(processed_image, n_clusters, media_root)
        method_name = 'Hybrid'
    
    # Visualize and save results
    result_path, colored_path = visualize_segmentation(
        processed_image, segmented_image, centers, media_root, method_name
    )
    
    logger.info(f"Segmentation process completed. Method: {method_name}")
    
    results = {
        'result_path': result_path,
        'colored_path': colored_path,
        'kmeans_result_path': kmeans_result_path
    }
    
    return results 