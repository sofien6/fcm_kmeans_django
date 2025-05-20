from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from .segmentation_utils import segment_image_with_method
from django.http import HttpResponse
import json
import logging

# Get logger for this module
logger = logging.getLogger('segmentation')

def home(request):
    """Home page view"""
    logger.info("Home page accessed")
    return render(request, 'segmentation/home.html')

def segment_image(request):
    """Handle image upload and segmentation"""
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image
        image_file = request.FILES['image']
        logger.info(f"Image upload received: {image_file.name}, size: {image_file.size} bytes")
        
        fs = FileSystemStorage()
        
        # Save the uploaded image
        filename = fs.save(f"uploads/{image_file.name}", image_file)
        uploaded_file_path = os.path.join(settings.MEDIA_ROOT, filename)
        logger.info(f"Image saved to: {uploaded_file_path}")
        
        # Get segmentation parameters
        method = request.POST.get('method', 'hybrid')
        n_clusters = int(request.POST.get('n_clusters', 5))
        logger.info(f"Segmentation requested with method: {method}, clusters: {n_clusters}")
        
        try:
            # Perform segmentation
            results = segment_image_with_method(
                uploaded_file_path, 
                settings.MEDIA_ROOT, 
                method=method, 
                n_clusters=n_clusters
            )
            
            # Get just the filename for the redirect
            result_filename = os.path.basename(results['result_path'])
            logger.info(f"Segmentation successful, redirecting to result page with filename: {result_filename}")
            
            # Store kmeans result in session if available
            if results['kmeans_result_path']:
                request.session['kmeans_result'] = results['kmeans_result_path']
                logger.info(f"K-means intermediate result saved: {results['kmeans_result_path']}")
            else:
                request.session['kmeans_result'] = None
            
            # Redirect to result page
            return redirect('result', filename=result_filename)
        except Exception as e:
            logger.error(f"Error during segmentation: {str(e)}")
            # In a production app, you would handle this error better and show a user-friendly message
            return redirect('home')
    
    # If not POST request, redirect to home
    logger.info("Redirect to home page - not a valid POST request with image")
    return redirect('home')

def result(request, filename):
    """Display segmentation results"""
    logger.info(f"Result page accessed for file: {filename}")
    
    # Extract the base filename without extension
    base_filename = os.path.splitext(filename)[0]
    
    # Construct paths for the result images
    result_path = f"{settings.MEDIA_URL}results/{filename}"
    colored_path = f"{settings.MEDIA_URL}results/{base_filename}_colored.png"
    
    # Get kmeans result if available
    kmeans_result = request.session.get('kmeans_result')
    if kmeans_result:
        kmeans_path = f"{settings.MEDIA_URL}{kmeans_result}"
        kmeans_colored_path = f"{settings.MEDIA_URL}{os.path.splitext(kmeans_result)[0]}_colored.png"
        logger.info(f"Including K-means result in template: {kmeans_path}")
    else:
        kmeans_path = None
        kmeans_colored_path = None
    
    # Prepare context data
    context = {
        'result_image': result_path,
        'colored_image': colored_path,
        'filename': filename,
        'kmeans_image': kmeans_path,
        'kmeans_colored_image': kmeans_colored_path
    }
    
    return render(request, 'segmentation/result.html', context)
