# FCM-KMeans Image Segmentation App

A Django web application that performs image segmentation using a hybrid approach combining K-means and Fuzzy C-means (FCM) algorithms.

## Deployment on Render

### Setup Instructions

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn [your_project_name].wsgi:application`
   - **Python Version**: 3.10 or later

### Environment Variables

Add the following environment variables in the Render dashboard:
- `DEBUG`: Set to `False` for production
- `SECRET_KEY`: Your Django secret key
- `ALLOWED_HOSTS`: Add your Render domain (e.g., `your-app-name.onrender.com`)
- `DATABASE_URL`: Added automatically by Render if using their PostgreSQL service

## Local Development

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run migrations: `python manage.py migrate`
6. Start the server: `python manage.py runserver`

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