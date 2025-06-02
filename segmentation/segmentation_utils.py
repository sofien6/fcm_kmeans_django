import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import base64
import io
import logging
from scipy.spatial.distance import cdist
from scipy.stats import zscore

# Get logger for this module
logger = logging.getLogger('segmentation')

class FuzzyCMeans:
    """
    Robust Fuzzy C-Means implementation with predefined centroid support
    """
    def __init__(self, n_clusters, m=2.0, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.m = m  # Fuzziness parameter
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.u_ = None  # Membership matrix
        self.labels_ = None
        
    def _initialize_membership(self, X, initial_centers=None):
        """Initialize membership matrix"""
        n_samples = X.shape[0]
        
        if initial_centers is not None:
            # Initialize based on provided centers
            distances = cdist(X, initial_centers, metric='euclidean')
            # Avoid division by zero
            distances = np.fmax(distances, np.finfo(np.float64).eps)
            
            # Calculate membership based on distances
            u = np.zeros((self.n_clusters, n_samples))
            for i in range(self.n_clusters):
                for j in range(self.n_clusters):
                    u[i] += (distances[:, i] / distances[:, j]) ** (2 / (self.m - 1))
                u[i] = 1.0 / u[i]
        else:
            # Random initialization
            np.random.seed(self.random_state)
            u = np.random.rand(self.n_clusters, n_samples)
            u = u / np.sum(u, axis=0)
            
        return u
    
    def _update_centers(self, X, u):
        """Update cluster centers"""
        um = u ** self.m
        centers = np.dot(um, X) / np.sum(um, axis=1, keepdims=True)
        return centers
    
    def _update_membership(self, X, centers):
        """Update membership matrix"""
        distances = cdist(X, centers, metric='euclidean')
        distances = np.fmax(distances, np.finfo(np.float64).eps)
        
        u = np.zeros((self.n_clusters, X.shape[0]))
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                u[i] += (distances[:, i] / distances[:, j]) ** (2 / (self.m - 1))
            u[i] = 1.0 / u[i]
            
        return u
    
    def fit(self, X, initial_centers=None):
        """Fit FCM to data"""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # Initialize membership matrix
        u = self._initialize_membership(X, initial_centers)
        
        for iteration in range(self.max_iter):
            # Update centers
            centers = self._update_centers(X, u)
            
            # Update membership
            u_new = self._update_membership(X, centers)
            
            # Check convergence
            if np.linalg.norm(u_new - u) < self.tol:
                logger.info(f"FCM converged after {iteration + 1} iterations")
                break
                
            u = u_new
        
        self.cluster_centers_ = centers
        self.u_ = u
        self.labels_ = np.argmax(u, axis=0)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        u = self._update_membership(X, self.cluster_centers_)
        return np.argmax(u, axis=0)

def preprocess_image(image_input):
    """Load and preprocess the image for segmentation
    
    Args:
        image_input: Can be either a file path (string) or a file-like object (Django UploadedFile)
    """
    if isinstance(image_input, str):
        # Traditional file path
        image = cv2.imread(image_input)
        if image is None:
            error_msg = f"Failed to read image from {image_input}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        # File-like object (Django UploadedFile)
        import numpy as np
        from PIL import Image
        
        # Read image from memory
        image_input.seek(0)  # Reset file pointer
        pil_image = Image.open(image_input)
        
        # Convert PIL image to OpenCV format
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'L':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array and BGR format (OpenCV standard)
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Check if image is grayscale or RGB
    is_grayscale = len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)
    
    if is_grayscale:
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_image = image.copy()
        # Convert to RGB for processing but keep track it was grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
    
    # Resize for faster processing if needed
    if max(image.shape[0], image.shape[1]) > 1000:
        scale = 1000 / max(image.shape[0], image.shape[1])
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height))
        original_image = cv2.resize(original_image, (width, height))
    
    return image, original_image, is_grayscale

def find_optimal_k_elbow_method(image, k_range=(2, 10)):
    """Find optimal number of clusters using improved Elbow method with kneedle algorithm"""
    logger.info("Finding optimal k using improved Elbow method")
    
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
    
    logger.info(f"Elbow method suggests k={optimal_k}")
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
    logger.info("Finding optimal k using Silhouette analysis")
    
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
    
    logger.info(f"Silhouette analysis suggests k={optimal_k}")
    return optimal_k, silhouette_scores, list(k_values)

def find_optimal_k_calinski_harabasz(image, k_range=(2, 10)):
    """Find optimal number of clusters using Calinski-Harabasz index"""
    logger.info("Finding optimal k using Calinski-Harabasz index")
    
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
    
    logger.info(f"Calinski-Harabasz index suggests k={optimal_k}")
    return optimal_k, ch_scores, list(k_values)

def find_optimal_k_gap_statistic(image, k_range=(2, 10), n_refs=10):
    """Find optimal number of clusters using Gap statistic"""
    logger.info("Finding optimal k using Gap statistic")
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Sample data if too large for faster computation
    if len(reshaped) > 10000:
        indices = np.random.choice(len(reshaped), 10000, replace=False)
        reshaped = reshaped[indices]
    
    gaps = []
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in k_values:
        # Calculate within-cluster sum of squares for actual data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reshaped)
        wk = _calculate_wk(reshaped, labels, kmeans.cluster_centers_)
        
        # Calculate expected within-cluster sum of squares for reference data
        ref_wks = []
        for _ in range(n_refs):
            # Generate reference data with same bounds as original
            ref_data = np.random.uniform(
                low=reshaped.min(axis=0),
                high=reshaped.max(axis=0),
                size=reshaped.shape
            )
            ref_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            ref_labels = ref_kmeans.fit_predict(ref_data)
            ref_wk = _calculate_wk(ref_data, ref_labels, ref_kmeans.cluster_centers_)
            ref_wks.append(ref_wk)
        
        # Calculate gap
        gap = np.log(np.mean(ref_wks)) - np.log(wk)
        gaps.append(gap)
    
    # Find optimal k using gap statistic
    # Look for the smallest k such that Gap(k) >= Gap(k+1) - s(k+1)
    optimal_k = k_values[0]  # Default
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1]:
            optimal_k = k_values[i]
            break
    
    logger.info(f"Gap statistic suggests k={optimal_k}")
    return optimal_k, gaps, list(k_values)

def _calculate_wk(data, labels, centers):
    """Calculate within-cluster sum of squares"""
    wk = 0
    for i in range(len(centers)):
        cluster_data = data[labels == i]
        if len(cluster_data) > 0:
            wk += np.sum((cluster_data - centers[i]) ** 2)
    return wk

def find_optimal_k_variance_analysis(image, k_range=(2, 10)):
    """Find optimal k using color variance analysis for images"""
    logger.info("Finding optimal k using color variance analysis")
    
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
    
    logger.info(f"Variance analysis suggests k={suggested_k} (variance: {color_variance:.2f})")
    return suggested_k, color_variance

def determine_optimal_k(image):
    """Determine optimal k using multiple methods and intelligent consensus"""
    logger.info("Determining optimal number of clusters using multiple methods")
    
    k_range = (2, 8)  # Reasonable range for image segmentation
    
    # Get suggestions from all methods
    k_elbow, elbow_inertias, k_values = find_optimal_k_elbow_method(image, k_range)
    k_silhouette, silhouette_scores, _ = find_optimal_k_silhouette(image, k_range)
    k_calinski, calinski_scores, _ = find_optimal_k_calinski_harabasz(image, k_range)
    k_gap, gap_scores, _ = find_optimal_k_gap_statistic(image, k_range, n_refs=5)  # Reduced refs for speed
    k_variance, color_variance = find_optimal_k_variance_analysis(image, k_range)
    
    # Create a voting system with weights
    suggestions = [k_elbow, k_silhouette, k_calinski, k_gap, k_variance]
    weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # Weights for each method
    
    logger.info(f"Method suggestions - Elbow: {k_elbow}, Silhouette: {k_silhouette}, "
                f"Calinski-Harabasz: {k_calinski}, Gap: {k_gap}, Variance: {k_variance}")
    
    # Calculate weighted consensus
    optimal_k = _calculate_weighted_consensus(suggestions, weights, k_range)
    
    # Additional validation: if all methods suggest very different values, 
    # use a more conservative approach
    suggestion_variance = np.var(suggestions)
    if suggestion_variance > 2.0:  # High disagreement
        logger.info(f"High disagreement between methods (variance: {suggestion_variance:.2f}), using conservative approach")
        # Use the median of the middle 3 suggestions
        sorted_suggestions = sorted(suggestions)
        optimal_k = int(np.median(sorted_suggestions[1:4]))
    
    # Final bounds check
    optimal_k = max(k_range[0], min(optimal_k, k_range[1]))
    
    logger.info(f"Final optimal k determined: {optimal_k}")
    logger.info(f"Method consensus: Elbow={k_elbow}, Silhouette={k_silhouette}, "
                f"Calinski={k_calinski}, Gap={k_gap}, Variance={k_variance}")
    
    return optimal_k, {
        'elbow_k': int(k_elbow),
        'silhouette_k': int(k_silhouette),
        'calinski_k': int(k_calinski),
        'gap_k': int(k_gap),
        'variance_k': int(k_variance),
        'elbow_inertias': [float(x) for x in elbow_inertias],
        'silhouette_scores': [float(x) for x in silhouette_scores],
        'calinski_scores': [float(x) for x in calinski_scores],
        'gap_scores': [float(x) for x in gap_scores],
        'color_variance': float(color_variance),
        'k_values': [int(x) for x in k_values],
        'suggestion_variance': float(np.var(suggestions))
    }

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

def kmeans_segmentation_with_centroids(image, k):
    """Segment image using K-means clustering and return centroids"""
    logger.info(f"Starting K-means segmentation with k={k} clusters")
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reshaped)
    
    # Reshape back to the original image shape
    segmented_image = labels.reshape(h, w)
    
    # Get cluster centers (centroids)
    centroids = kmeans.cluster_centers_
    
    # Calculate centroid positions in image coordinates
    centroid_positions = []
    for i in range(k):
        # Find pixels belonging to this cluster
        cluster_pixels = np.where(segmented_image == i)
        if len(cluster_pixels[0]) > 0:
            # Calculate the center of mass for this cluster
            center_y = float(np.mean(cluster_pixels[0]))
            center_x = float(np.mean(cluster_pixels[1]))
            centroid_positions.append((center_x, center_y))
        else:
            centroid_positions.append((float(w//2), float(h//2)))  # Default position
    
    logger.info(f"K-means segmentation completed with {k} clusters")
    return segmented_image, centroids, centroid_positions

def fcm_segmentation_with_centroids(image, segmented_image, initial_centroids, k):
    """Segment image using robust FCM implementation with predefined centroids"""
    logger.info(f"Starting robust FCM segmentation with k={k} clusters")
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Initialize FCM with provided centroids
    fcm = FuzzyCMeans(n_clusters=k, m=2.0, max_iter=100, tol=1e-4, random_state=42)
    
    # Fit FCM with initial centroids from K-means
    fcm.fit(reshaped, initial_centers=initial_centroids)
    
    # Get cluster membership for each pixel
    cluster_membership = fcm.labels_
    
    # Reshape back to the original image shape
    fcm_segmented_image = cluster_membership.reshape(h, w)
    
    # Get the refined centroids
    refined_centroids = fcm.cluster_centers_
    
    # Calculate FCM centroid positions in image coordinates
    fcm_centroid_positions = []
    for i in range(k):
        # Find pixels belonging to this cluster
        cluster_pixels = np.where(fcm_segmented_image == i)
        if len(cluster_pixels[0]) > 0:
            # Calculate the center of mass for this cluster
            center_y = float(np.mean(cluster_pixels[0]))
            center_x = float(np.mean(cluster_pixels[1]))
            fcm_centroid_positions.append((center_x, center_y))
        else:
            fcm_centroid_positions.append((float(w//2), float(h//2)))  # Default position
    
    logger.info(f"Robust FCM segmentation completed with {k} clusters")
    logger.info(f"Membership matrix shape: {fcm.u_.shape}")
    logger.info(f"Final centroids shape: {refined_centroids.shape}")
    
    return fcm_segmented_image, refined_centroids, fcm_centroid_positions

def _simple_fcm_alternative(image, segmented_image, initial_centroids, k):
    """Simple FCM alternative when skfuzzy is not available"""
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Use iterative refinement similar to FCM
    centroids = initial_centroids.copy()
    
    for iteration in range(10):  # Limited iterations
        # Calculate distances and soft assignments
        distances = np.zeros((h * w, k))
        for i in range(k):
            distances[:, i] = np.sum((reshaped - centroids[i]) ** 2, axis=1)
        
        # Soft assignment with fuzzy parameter m=2
        m = 2.0
        u = np.zeros((k, h * w))
        for i in range(k):
            for j in range(k):
                u[i] += (distances[:, i] / distances[:, j]) ** (2 / (m - 1))
            u[i] = 1.0 / u[i]
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            weights = u[i] ** m
            new_centroids[i] = np.sum(reshaped * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        # Check convergence
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            break
        centroids = new_centroids
    
    # Final assignment
    final_labels = np.argmin(distances, axis=1)
    fcm_segmented_image = final_labels.reshape(h, w)
    
    # Calculate FCM centroid positions in image coordinates
    fcm_centroid_positions = []
    for i in range(k):
        cluster_pixels = np.where(fcm_segmented_image == i)
        if len(cluster_pixels[0]) > 0:
            center_y = float(np.mean(cluster_pixels[0]))
            center_x = float(np.mean(cluster_pixels[1]))
            fcm_centroid_positions.append((center_x, center_y))
        else:
            fcm_centroid_positions.append((float(w//2), float(h//2)))
    
    logger.info(f"Simple FCM alternative completed with {k} clusters")
    return fcm_segmented_image, centroids, fcm_centroid_positions

def fcm_segmentation_with_custom_centroids(image, custom_centroids, k, m=2.0, max_iter=100):
    """
    Segment image using FCM with user-provided centroids
    
    Args:
        image: Input image (H, W, C)
        custom_centroids: User-provided centroids in color space (k, C)
        k: Number of clusters
        m: Fuzziness parameter (default 2.0)
        max_iter: Maximum iterations (default 100)
    
    Returns:
        fcm_segmented_image: Segmented image
        refined_centroids: Final centroids after FCM
        fcm_centroid_positions: Centroid positions in image coordinates
        membership_matrix: Fuzzy membership matrix
    """
    logger.info(f"Starting FCM segmentation with custom centroids, k={k}")
    
    # Reshape image to a 2D array of pixels
    h, w, channels = image.shape
    reshaped = image.reshape(h * w, channels)
    
    # Ensure custom centroids are in the right format
    custom_centroids = np.array(custom_centroids)
    if custom_centroids.shape[0] != k:
        raise ValueError(f"Number of custom centroids ({custom_centroids.shape[0]}) must match k ({k})")
    if custom_centroids.shape[1] != channels:
        raise ValueError(f"Centroid dimensions ({custom_centroids.shape[1]}) must match image channels ({channels})")
    
    # Initialize FCM with custom centroids
    fcm = FuzzyCMeans(n_clusters=k, m=m, max_iter=max_iter, tol=1e-4, random_state=42)
    
    # Fit FCM with custom centroids
    fcm.fit(reshaped, initial_centers=custom_centroids)
    
    # Get cluster membership for each pixel
    cluster_membership = fcm.labels_
    
    # Reshape back to the original image shape
    fcm_segmented_image = cluster_membership.reshape(h, w)
    
    # Get the refined centroids
    refined_centroids = fcm.cluster_centers_
    
    # Calculate FCM centroid positions in image coordinates
    fcm_centroid_positions = []
    for i in range(k):
        # Find pixels belonging to this cluster
        cluster_pixels = np.where(fcm_segmented_image == i)
        if len(cluster_pixels[0]) > 0:
            # Calculate the center of mass for this cluster
            center_y = float(np.mean(cluster_pixels[0]))
            center_x = float(np.mean(cluster_pixels[1]))
            fcm_centroid_positions.append((center_x, center_y))
        else:
            fcm_centroid_positions.append((float(w//2), float(h//2)))  # Default position
    
    logger.info(f"Custom FCM segmentation completed with {k} clusters")
    logger.info(f"Initial centroids: {custom_centroids}")
    logger.info(f"Refined centroids: {refined_centroids}")
    
    return fcm_segmented_image, refined_centroids, fcm_centroid_positions, fcm.u_

def generate_color_map(k):
    """Generate a colormap for visualizing segments"""
    np.random.seed(42)  # For reproducibility
    colors = np.random.rand(k, 3)
    return ListedColormap(colors)

def generate_true_color_map(centroids):
    """Generate a colormap using the actual cluster centroids (true colors)"""
    # Normalize centroids to [0, 1] range for matplotlib
    colors = centroids / 255.0 if centroids.max() > 1.0 else centroids
    # Ensure colors are in valid range
    colors = np.clip(colors, 0, 1)
    return ListedColormap(colors)

def create_true_color_segmented_image(original_image, segmented_image, centroids):
    """Create a segmented image using true colors from centroids"""
    try:
        h, w = segmented_image.shape
        true_color_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert centroids to numpy array if it isn't already
        centroids = np.array(centroids)
        
        # Ensure centroids are in the right format (0-255 range)
        if centroids.max() <= 1.0:
            centroids = (centroids * 255).astype(np.uint8)
        else:
            centroids = centroids.astype(np.uint8)
        
        # Ensure centroids have the right shape
        if len(centroids.shape) == 1:
            # If centroids is 1D, reshape it
            centroids = centroids.reshape(1, -1)
        
        # Get unique cluster labels in the segmented image
        unique_labels = np.unique(segmented_image)
        
        # Assign true colors to each pixel based on its cluster
        for label in unique_labels:
            if label < len(centroids):  # Make sure we have a centroid for this label
                mask = segmented_image == label
                if np.any(mask):  # Only assign if there are pixels in this cluster
                    centroid = centroids[label]
                    
                    # Ensure the centroid has 3 color channels
                    if len(centroid) >= 3:
                        true_color_image[mask] = centroid[:3]  # Take first 3 channels
                    elif len(centroid) == 1:
                        # Grayscale case - replicate to RGB
                        color_val = centroid[0]
                        true_color_image[mask] = [color_val, color_val, color_val]
                    else:
                        # Unexpected case - use mean
                        color_val = np.mean(centroid)
                        true_color_image[mask] = [color_val, color_val, color_val]
        
        return true_color_image
        
    except Exception as e:
        logger.error(f"Error in create_true_color_segmented_image: {str(e)}")
        logger.error(f"Centroids shape: {centroids.shape if hasattr(centroids, 'shape') else 'No shape'}")
        logger.error(f"Segmented image shape: {segmented_image.shape}")
        logger.error(f"Unique labels in segmented image: {np.unique(segmented_image)}")
        
        # Fallback: return a copy of the original image
        if len(original_image.shape) == 3:
            return original_image.copy()
        else:
            # Convert grayscale to RGB
            return np.stack([original_image] * 3, axis=-1)

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return image_base64

def visualize_k_analysis(k_analysis_data):
    """Visualize the comprehensive k-selection analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    k_values = k_analysis_data['k_values']
    
    # Plot Elbow method
    ax1.plot(k_values, k_analysis_data['elbow_inertias'], 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=k_analysis_data['elbow_k'], color='red', linestyle='--', alpha=0.7, 
                label=f'Elbow k={k_analysis_data["elbow_k"]}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia', fontsize=11)
    ax1.set_title('Elbow Method', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot Silhouette analysis
    ax2.plot(k_values, k_analysis_data['silhouette_scores'], 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=k_analysis_data['silhouette_k'], color='red', linestyle='--', alpha=0.7,
                label=f'Best k={k_analysis_data["silhouette_k"]}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add overall title with consensus information
    fig.suptitle(f'K-Selection Analysis\n'
                 f'Elbow Method suggests k={k_analysis_data["elbow_k"]}, '
                 f'Silhouette Analysis suggests k={k_analysis_data["silhouette_k"]}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return plot_to_base64(fig)

def visualize_kmeans_result(original_image, segmented_image, centroid_positions, k, is_grayscale, centroids=None):
    """Visualize K-means segmentation results with centroids"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Display original image
    if is_grayscale:
        ax1.imshow(original_image, cmap='gray')
    else:
        ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # Display segmented image with false colors (traditional view)
    cmap = generate_color_map(k)
    ax2.imshow(segmented_image, cmap=cmap)
    ax2.set_title(f'K-means Segmentation (k={k}) - False Colors', fontsize=14)
    ax2.axis('off')
    
    # Plot centroids on false color image
    for i, (x, y) in enumerate(centroid_positions):
        circle = patches.Circle((x, y), radius=8, color='red', fill=True, alpha=0.8)
        ax2.add_patch(circle)
        ax2.text(x, y-15, f'C{i+1}', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
    
    # Display segmented image with true colors if centroids are provided
    if centroids is not None:
        true_color_image = create_true_color_segmented_image(original_image, segmented_image, centroids)
        ax3.imshow(true_color_image)
        ax3.set_title(f'K-means Segmentation (k={k}) - True Colors', fontsize=14)
        ax3.axis('off')
        
        # Plot centroids on true color image
        for i, (x, y) in enumerate(centroid_positions):
            circle = patches.Circle((x, y), radius=8, color='red', fill=True, alpha=0.8)
            ax3.add_patch(circle)
            ax3.text(x, y-15, f'C{i+1}', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
    else:
        # If no centroids provided, show the same false color image
        ax3.imshow(segmented_image, cmap=cmap)
        ax3.set_title(f'K-means Segmentation (k={k}) - Centroids Not Available', fontsize=14)
        ax3.axis('off')
    
    plt.tight_layout()
    return plot_to_base64(fig)

def visualize_fcm_result(original_image, kmeans_segmented, fcm_segmented, 
                        kmeans_centroids, fcm_centroids, k, is_grayscale, kmeans_color_centroids=None, fcm_color_centroids=None):
    """Visualize FCM segmentation results with centroids comparison"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
    
    # Display original image
    if is_grayscale:
        ax1.imshow(original_image, cmap='gray')
    else:
        ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # Display K-means result with false colors
    cmap = generate_color_map(k)
    ax2.imshow(kmeans_segmented, cmap=cmap)
    ax2.set_title('K-means Segmentation - False Colors', fontsize=14)
    ax2.axis('off')
    
    # Plot K-means centroids
    for i, (x, y) in enumerate(kmeans_centroids):
        circle = patches.Circle((x, y), radius=6, color='red', fill=True, alpha=0.8)
        ax2.add_patch(circle)
        ax2.text(x, y-12, f'K{i+1}', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.8))
    
    # Display FCM result with false colors
    ax3.imshow(fcm_segmented, cmap=cmap)
    ax3.set_title('FCM Segmentation - False Colors', fontsize=14)
    ax3.axis('off')
    
    # Plot FCM centroids
    for i, (x, y) in enumerate(fcm_centroids):
        circle = patches.Circle((x, y), radius=6, color='blue', fill=True, alpha=0.8)
        ax3.add_patch(circle)
        ax3.text(x, y-12, f'F{i+1}', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.8))
    
    # Display K-means result with true colors
    if kmeans_color_centroids is not None:
        kmeans_true_color_image = create_true_color_segmented_image(original_image, kmeans_segmented, kmeans_color_centroids)
        ax5.imshow(kmeans_true_color_image)
        ax5.set_title('K-means Segmentation - True Colors', fontsize=14)
        ax5.axis('off')
        
        # Plot K-means centroids on true color image
        for i, (x, y) in enumerate(kmeans_centroids):
            circle = patches.Circle((x, y), radius=6, color='red', fill=True, alpha=0.8)
            ax5.add_patch(circle)
            ax5.text(x, y-12, f'K{i+1}', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.8))
    else:
        ax5.imshow(kmeans_segmented, cmap=cmap)
        ax5.set_title('K-means - True Colors Not Available', fontsize=14)
        ax5.axis('off')
    
    # Display FCM result with true colors
    if fcm_color_centroids is not None:
        fcm_true_color_image = create_true_color_segmented_image(original_image, fcm_segmented, fcm_color_centroids)
        ax6.imshow(fcm_true_color_image)
        ax6.set_title('FCM Segmentation - True Colors', fontsize=14)
        ax6.axis('off')
        
        # Plot FCM centroids on true color image
        for i, (x, y) in enumerate(fcm_centroids):
            circle = patches.Circle((x, y), radius=6, color='blue', fill=True, alpha=0.8)
            ax6.add_patch(circle)
            ax6.text(x, y-12, f'F{i+1}', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.8))
    else:
        ax6.imshow(fcm_segmented, cmap=cmap)
        ax6.set_title('FCM - True Colors Not Available', fontsize=14)
        ax6.axis('off')
    
    # Display difference analysis
    # Calculate pixel-wise differences
    diff_mask = (kmeans_segmented != fcm_segmented).astype(int)
    total_pixels = kmeans_segmented.size
    changed_pixels = np.sum(diff_mask)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    # Create difference visualization
    if changed_pixels > 0:
        # Show differences as highlighted regions
        diff_display = np.zeros_like(fcm_segmented)
        diff_display[diff_mask == 1] = 1  # Mark changed pixels
        ax4.imshow(fcm_segmented, cmap=cmap, alpha=0.7)
        ax4.imshow(diff_display, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        ax4.set_title(f'FCM Refinement: {change_percentage:.1f}% pixels changed', fontsize=14)
    else:
        ax4.imshow(fcm_segmented, cmap=cmap)
        ax4.set_title('FCM Result: No pixel changes detected', fontsize=14)
    ax4.axis('off')
    
    # Plot both sets of centroids for comparison
    for i, (x, y) in enumerate(kmeans_centroids):
        circle = patches.Circle((x, y), radius=6, color='red', fill=True, alpha=0.6)
        ax4.add_patch(circle)
        ax4.text(x, y-12, f'K{i+1}', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.8))
    
    for i, (x, y) in enumerate(fcm_centroids):
        circle = patches.Circle((x, y), radius=6, color='blue', fill=True, alpha=0.6)
        ax4.add_patch(circle)
        ax4.text(x+15, y, f'F{i+1}', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.8))
    
    # Add legend with statistics
    red_patch = patches.Patch(color='red', alpha=0.6, label='K-means Centroids')
    blue_patch = patches.Patch(color='blue', alpha=0.6, label='FCM Centroids')
    if changed_pixels > 0:
        change_patch = patches.Patch(color='red', alpha=0.5, label=f'Changed Pixels ({change_percentage:.1f}%)')
        ax4.legend(handles=[red_patch, blue_patch, change_patch], loc='upper right')
    else:
        ax4.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    plt.tight_layout()
    return plot_to_base64(fig)

def calculate_centroid_movement_stats(kmeans_centroids, fcm_centroids):
    """Calculate statistics about centroid movement from K-means to FCM"""
    movements = []
    total_movement = 0
    
    for i, (kmeans_pos, fcm_pos) in enumerate(zip(kmeans_centroids, fcm_centroids)):
        # Calculate Euclidean distance
        distance = float(np.sqrt((kmeans_pos[0] - fcm_pos[0])**2 + (kmeans_pos[1] - fcm_pos[1])**2))
        movements.append({
            'centroid': i + 1,
            'kmeans_pos': (float(kmeans_pos[0]), float(kmeans_pos[1])),
            'fcm_pos': (float(fcm_pos[0]), float(fcm_pos[1])),
            'distance': distance
        })
        total_movement += distance
    
    avg_movement = float(total_movement / len(movements)) if movements else 0.0
    max_movement = max(movements, key=lambda x: x['distance']) if movements else None
    
    return {
        'movements': movements,
        'total_movement': float(total_movement),
        'average_movement': avg_movement,
        'max_movement': max_movement,
        'num_centroids': len(movements)
    }

def create_detailed_difference_visualization(kmeans_segmented, fcm_segmented, k, original_image=None, kmeans_centroids=None, fcm_centroids=None):
    """Create a detailed visualization showing differences between K-means and FCM"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generate consistent colormap
    cmap = generate_color_map(k)
    
    # Display K-means result - use true colors if available
    if original_image is not None and kmeans_centroids is not None:
        kmeans_true_color = create_true_color_segmented_image(original_image, kmeans_segmented, kmeans_centroids)
        ax1.imshow(kmeans_true_color)
        ax1.set_title('K-means Segmentation - True Colors', fontsize=14)
    else:
        ax1.imshow(kmeans_segmented, cmap=cmap)
        ax1.set_title('K-means Segmentation - False Colors', fontsize=14)
    ax1.axis('off')
    
    # Display FCM result - use true colors if available
    if original_image is not None and fcm_centroids is not None:
        fcm_true_color = create_true_color_segmented_image(original_image, fcm_segmented, fcm_centroids)
        ax2.imshow(fcm_true_color)
        ax2.set_title('FCM Segmentation - True Colors', fontsize=14)
    else:
        ax2.imshow(fcm_segmented, cmap=cmap)
        ax2.set_title('FCM Segmentation - False Colors', fontsize=14)
    ax2.axis('off')
    
    # Calculate and display differences
    diff_mask = (kmeans_segmented != fcm_segmented).astype(int)
    total_pixels = kmeans_segmented.size
    changed_pixels = np.sum(diff_mask)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    # Difference map
    ax3.imshow(diff_mask, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax3.set_title(f'Difference Map: {change_percentage:.2f}% pixels changed', fontsize=14)
    ax3.axis('off')
    
    # Overlay visualization
    ax4.imshow(fcm_segmented, cmap=cmap, alpha=0.8)
    if changed_pixels > 0:
        ax4.imshow(diff_mask, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        ax4.set_title('FCM Result with Changes Highlighted', fontsize=14)
    else:
        ax4.set_title('FCM Result (No Changes Detected)', fontsize=14)
    ax4.axis('off')
    
    plt.tight_layout()
    return plot_to_base64(fig), change_percentage

def segment_image_with_auto_k(image_input):
    """
    Automatic segmentation with optimal k detection:
    1. Automatically determine optimal k using Elbow and Silhouette methods
    2. Implement K-means algorithm with optimal k
    3. Display image with centroid positions
    4. Implement FCM based on segmented image and centroids
    5. Display centroid positions after FCM
    6. Return base64 encoded images for web display
    
    Args:
        image_input: Can be either a file path (string) or a file-like object (Django UploadedFile)
    """
    logger.info("Starting automatic segmentation with optimal k detection")
    
    # Step 1: Preprocess image
    image, original_image, is_grayscale = preprocess_image(image_input)
    logger.info(f"Image loaded. Shape: {image.shape}, Grayscale: {is_grayscale}")
    
    # Step 2: Determine optimal k automatically
    optimal_k, k_analysis_data = determine_optimal_k(image)
    logger.info(f"Optimal k determined: {optimal_k}")
    
    # Step 3: Visualize k-selection analysis
    k_analysis_image = visualize_k_analysis(k_analysis_data)
    
    # Step 4: K-means segmentation with optimal k
    kmeans_segmented, kmeans_centroids, kmeans_centroid_positions = kmeans_segmentation_with_centroids(image, optimal_k)
    
    # Step 5: Visualize K-means result with centroids
    logger.info(f"K-means centroids shape: {kmeans_centroids.shape}")
    logger.info(f"K-means segmented image unique values: {np.unique(kmeans_segmented)}")
    kmeans_result_image = visualize_kmeans_result(
        original_image, kmeans_segmented, kmeans_centroid_positions, optimal_k, is_grayscale, kmeans_centroids
    )
    
    # Step 6: FCM segmentation based on K-means results
    fcm_segmented, fcm_centroids, fcm_centroid_positions = fcm_segmentation_with_centroids(
        image, kmeans_segmented, kmeans_centroids, optimal_k
    )
    
    # Step 7: Visualize FCM result with centroids comparison
    logger.info(f"FCM centroids shape: {fcm_centroids.shape}")
    logger.info(f"FCM segmented image unique values: {np.unique(fcm_segmented)}")
    fcm_result_image = visualize_fcm_result(
        original_image, kmeans_segmented, fcm_segmented,
        kmeans_centroid_positions, fcm_centroid_positions, optimal_k, is_grayscale, kmeans_centroids, fcm_centroids
    )
    
    # Step 8: Calculate centroid movement statistics
    centroid_movement_stats = calculate_centroid_movement_stats(kmeans_centroid_positions, fcm_centroid_positions)
    
    # Step 9: Create detailed difference visualization
    difference_visualization, change_percentage = create_detailed_difference_visualization(
        kmeans_segmented, fcm_segmented, optimal_k, original_image, kmeans_centroids, fcm_centroids
    )
    
    logger.info("Automatic segmentation process completed successfully")
    
    return {
        'k_analysis': k_analysis_image,
        'kmeans_result': kmeans_result_image,
        'fcm_result': fcm_result_image,
        'difference_visualization': difference_visualization,
        'change_percentage': float(change_percentage),
        'kmeans_centroids': kmeans_centroid_positions,
        'fcm_centroids': fcm_centroid_positions,
        'is_grayscale': bool(is_grayscale),
        'k': int(optimal_k),
        'k_analysis_data': k_analysis_data,
        'centroid_movement_stats': centroid_movement_stats
    }

# Keep the old function for backward compatibility
def segment_image_with_new_principle(image_path, k):
    """Legacy function - now redirects to auto k detection"""
    return segment_image_with_auto_k(image_path) 