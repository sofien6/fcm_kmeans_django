# Hybrid Image Segmentation using K-means and FCM

This Django-based web application provides image segmentation functionality using a hybrid approach that combines K-means and Fuzzy C-means (FCM) clustering algorithms.

## Features

- **Multiple Segmentation Methods:**
  - K-means clustering (fast but deterministic)
  - Fuzzy C-means (FCM) clustering (probabilistic approach)
  - Hybrid approach (combines K-means speed with FCM refinement)
  
- **Adjustable Parameters:**
  - Customize the number of clusters (segments)
  
- **Results Visualization:**
  - Side-by-side comparison of original and segmented images
  - Colored segment visualization

## Installation

1. Clone the repository:
   ```
   git clone <repository_url>
   cd fcm_kmeans_django
   ```

2. Install the required packages:
   ```
   pip install django numpy scikit-learn scikit-fuzzy opencv-python pillow matplotlib
   ```

3. Run the Django development server:
   ```
   python manage.py runserver
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

## How It Works

### Hybrid Segmentation Approach

1. **Preprocessing**:
   - Resize the image if needed
   - Convert to RGB color space

2. **Initial Clustering with K-means**:
   - Fast clustering to identify initial segments
   - Provides initial centroids for FCM

3. **Refinement with FCM**:
   - Uses K-means results as initialization
   - Applies fuzzy clustering for more nuanced segmentation
   - Each pixel gets a membership degree to each cluster

4. **Visualization**:
   - Original image side-by-side with segmentation result
   - Color-coded segments for better visualization

## Technical Details

### K-means Clustering

K-means is a centroid-based clustering algorithm that partitions data into K clusters, where each data point belongs to the cluster with the nearest mean.

### Fuzzy C-means (FCM)

FCM is a soft clustering algorithm where each data point can belong to multiple clusters with varying degrees of membership, rather than belonging to just one cluster.

### Hybrid Approach Benefits

- Faster convergence than pure FCM
- More refined results than K-means alone
- Better handling of ambiguous regions in images

## Usage

1. Upload an image (JPG, PNG, BMP formats supported)
2. Select segmentation method (Hybrid, K-means, or FCM)
3. Choose the number of clusters (2-10)
4. View and compare segmentation results

## Future Improvements

- Support for more segmentation algorithms
- Interactive parameter tuning
- Batch processing of multiple images
- Segment-level statistics and analysis 