from django.shortcuts import render, redirect
from .segmentation_utils import segment_image_with_auto_k
import logging

# Get logger for this module
logger = logging.getLogger('segmentation')

def home(request):
    """Home page view"""
    logger.info("Home page accessed")
    return render(request, 'segmentation/home.html')

def segment_image(request):
    """Handle image upload and automatic segmentation with optimal k detection (in-memory processing)"""
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image
        image_file = request.FILES['image']
        logger.info(f"Image upload received: {image_file.name}, size: {image_file.size} bytes")
        
        logger.info("Starting automatic segmentation with optimal k detection (in-memory processing)")
        
        try:
            # Perform automatic segmentation with optimal k detection directly from memory
            # No need to save the image to disk
            results = segment_image_with_auto_k(image_file)
            
            # Store results in session
            request.session['segmentation_results'] = results
            logger.info(f"Automatic segmentation successful with optimal k={results['k']} (processed in memory)")
            
            # Redirect to result page
            return redirect('result')
        except Exception as e:
            logger.error(f"Error during automatic segmentation: {str(e)}")
            # In a production app, you would handle this error better and show a user-friendly message
            return redirect('home')
    
    # If not POST request, redirect to home
    logger.info("Redirect to home page - not a valid POST request with image")
    return redirect('home')

def result(request):
    """Display automatic segmentation results"""
    logger.info("Result page accessed")
    
    # Get results from session
    results = request.session.get('segmentation_results')
    if not results:
        logger.warning("No segmentation results found in session")
        return redirect('home')
    
    # Prepare context data
    context = {
        'k_analysis': results['k_analysis'],
        'kmeans_result': results['kmeans_result'],
        'fcm_result': results['fcm_result'],
        'difference_visualization': results.get('difference_visualization', ''),
        'change_percentage': results.get('change_percentage', 0),
        'kmeans_centroids': results['kmeans_centroids'],
        'fcm_centroids': results['fcm_centroids'],
        'is_grayscale': results['is_grayscale'],
        'k': results['k'],
        'k_analysis_data': results['k_analysis_data'],
        'centroid_movement_stats': results.get('centroid_movement_stats', {})
    }
    
    return render(request, 'segmentation/result.html', context)

def custom_fcm_segmentation(request):
    """Handle custom FCM segmentation with user-provided centroids"""
    if request.method == 'POST' and request.FILES.get('image'):
        import json
        import numpy as np
        from .segmentation_utils import fcm_segmentation_with_custom_centroids, preprocess_image
        
        # Get the uploaded image
        image_file = request.FILES['image']
        logger.info(f"Custom FCM: Image upload received: {image_file.name}, size: {image_file.size} bytes")
        
        # Get custom centroids from form
        try:
            centroids_data = request.POST.get('centroids', '[]')
            custom_centroids = np.array(json.loads(centroids_data))
            
            if len(custom_centroids) == 0:
                raise ValueError("No centroids provided")
                
            k = len(custom_centroids)
            logger.info(f"Custom FCM: Using {k} custom centroids")
            
            # Process image in memory
            image, original_image, is_grayscale = preprocess_image(image_file)
            
            # Apply FCM with custom centroids
            fcm_segmented, refined_centroids, fcm_positions, membership = fcm_segmentation_with_custom_centroids(
                image, custom_centroids, k, m=2.0, max_iter=100
            )
            
            # Create visualization (you can add a specific visualization function for custom FCM)
            from .segmentation_utils import visualize_fcm_result, plot_to_base64
            import matplotlib.pyplot as plt
            
            # Simple visualization for custom FCM
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            if is_grayscale:
                axes[0].imshow(original_image, cmap='gray')
            else:
                axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(fcm_segmented, cmap='tab10')
            axes[1].set_title('FCM Segmentation')
            axes[1].axis('off')
            
            # Show membership uncertainty
            entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
            uncertainty_map = entropy.reshape(image.shape[:2])
            im = axes[2].imshow(uncertainty_map, cmap='hot')
            axes[2].set_title('Membership Uncertainty')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            result_image = plot_to_base64(fig)
            
            # Calculate centroid movements
            movements = []
            for i, (orig, refined) in enumerate(zip(custom_centroids, refined_centroids)):
                movement = float(np.linalg.norm(refined - orig))
                movements.append({
                    'centroid': i + 1,
                    'original': orig.tolist(),
                    'refined': refined.tolist(),
                    'movement': movement
                })
            
            response_data = {
                'success': True,
                'result_image': result_image,
                'original_centroids': custom_centroids.tolist(),
                'refined_centroids': refined_centroids.tolist(),
                'centroid_positions': fcm_positions,
                'movements': movements,
                'k': k,
                'is_grayscale': is_grayscale
            }
            
            from django.http import JsonResponse
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Error in custom FCM segmentation: {str(e)}")
            from django.http import JsonResponse
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    # GET request - show the custom FCM form
    return render(request, 'segmentation/custom_fcm.html')
