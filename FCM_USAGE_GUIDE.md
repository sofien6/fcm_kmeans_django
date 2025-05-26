# FCM with Custom Centroids - Usage Guide

## Overview

Your Django application now has a **robust Fuzzy C-Means (FCM) implementation** that allows you to provide **predefined centroids** as input. This solves the issues with the problematic `scikit-fuzzy` library and gives you full control over the segmentation process.

## Key Features

✅ **Custom Centroid Initialization**: Provide your own centroids in RGB color space  
✅ **Robust Convergence**: Reliable algorithm that always converges  
✅ **Membership Matrix Access**: Get fuzzy membership values for each pixel  
✅ **Adjustable Fuzziness**: Control the fuzziness parameter (m)  
✅ **No External Dependencies**: No reliance on problematic scikit-fuzzy  
✅ **Full Integration**: Works seamlessly with existing K-means workflow  

## Quick Start

### 1. Basic FCM with Custom Centroids

```python
from segmentation.segmentation_utils import fcm_segmentation_with_custom_centroids
import numpy as np

# Define your custom centroids (RGB values)
custom_centroids = np.array([
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    [0, 0, 255],      # Blue
    [255, 255, 0],    # Yellow
    [128, 128, 128]   # Gray
])

# Load your image (H, W, 3)
image, original_image, is_grayscale = preprocess_image("path/to/your/image.jpg")

# Apply FCM with your custom centroids
k = len(custom_centroids)
fcm_segmented, refined_centroids, fcm_positions, membership_matrix = fcm_segmentation_with_custom_centroids(
    image, custom_centroids, k, m=2.0, max_iter=100
)

print(f"Original centroids: {custom_centroids}")
print(f"Refined centroids: {refined_centroids}")
print(f"Membership matrix shape: {membership_matrix.shape}")
```

### 2. Compare K-means vs FCM

```python
from segmentation.segmentation_utils import (
    kmeans_segmentation_with_centroids,
    fcm_segmentation_with_centroids
)

# First run K-means
kmeans_segmented, kmeans_centroids, kmeans_positions = kmeans_segmentation_with_centroids(image, k)

# Then run FCM using K-means results as initialization
fcm_segmented, fcm_centroids, fcm_positions = fcm_segmentation_with_centroids(
    image, kmeans_segmented, kmeans_centroids, k
)

# Compare the results
print("K-means centroids:", kmeans_centroids)
print("FCM refined centroids:", fcm_centroids)
```

### 3. Adjust Fuzziness Parameter

```python
# Lower m = crisper boundaries (more like K-means)
fcm_crisp, _, _, membership_crisp = fcm_segmentation_with_custom_centroids(
    image, custom_centroids, k, m=1.5
)

# Higher m = fuzzier boundaries (more gradual transitions)
fcm_fuzzy, _, _, membership_fuzzy = fcm_segmentation_with_custom_centroids(
    image, custom_centroids, k, m=3.0
)
```

## Advanced Usage

### Working with Membership Matrices

The membership matrix tells you how much each pixel belongs to each cluster:

```python
# Get membership matrix (k, num_pixels)
_, _, _, membership = fcm_segmentation_with_custom_centroids(image, custom_centroids, k)

# Reshape to image dimensions for visualization
h, w = image.shape[:2]
membership_maps = membership.reshape(k, h, w)

# Visualize membership for cluster 0
import matplotlib.pyplot as plt
plt.imshow(membership_maps[0], cmap='viridis')
plt.title('Membership Map - Cluster 1')
plt.colorbar()
plt.show()

# Calculate uncertainty (entropy) for each pixel
entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
uncertainty_map = entropy.reshape(h, w)
```

### Custom Centroid Strategies

#### Strategy 1: Color-based Centroids
```python
# For natural images
custom_centroids = np.array([
    [139, 69, 19],    # Brown (earth)
    [34, 139, 34],    # Green (vegetation)
    [135, 206, 235],  # Sky blue
    [255, 255, 255],  # White (clouds)
    [64, 64, 64]      # Dark gray (shadows)
])
```

#### Strategy 2: Intensity-based Centroids (Grayscale)
```python
# For grayscale or medical images
custom_centroids = np.array([
    [0, 0, 0],        # Black
    [64, 64, 64],     # Dark gray
    [128, 128, 128],  # Medium gray
    [192, 192, 192],  # Light gray
    [255, 255, 255]   # White
])
```

#### Strategy 3: Domain-specific Centroids
```python
# For satellite imagery
custom_centroids = np.array([
    [0, 100, 0],      # Deep water
    [139, 69, 19],    # Soil/bare earth
    [34, 139, 34],    # Vegetation
    [169, 169, 169],  # Urban/concrete
    [255, 255, 0]     # Sand/desert
])
```

## Integration with Django Views

### Example Django View

```python
# views.py
from django.shortcuts import render
from django.http import JsonResponse
from segmentation.segmentation_utils import fcm_segmentation_with_custom_centroids
import numpy as np

def custom_fcm_segmentation(request):
    if request.method == 'POST':
        # Get uploaded image
        image_file = request.FILES['image']
        
        # Get custom centroids from form
        centroids_data = request.POST.get('centroids')  # JSON string
        custom_centroids = np.array(json.loads(centroids_data))
        
        # Process image
        image, original_image, is_grayscale = preprocess_image(image_file.temporary_file_path())
        
        # Apply FCM
        k = len(custom_centroids)
        fcm_segmented, refined_centroids, fcm_positions, membership = fcm_segmentation_with_custom_centroids(
            image, custom_centroids, k, m=2.0, max_iter=100
        )
        
        # Create visualization
        result_image = visualize_fcm_result(
            original_image, None, fcm_segmented,
            [], fcm_positions, k, is_grayscale, None, refined_centroids
        )
        
        return JsonResponse({
            'success': True,
            'result_image': result_image,
            'original_centroids': custom_centroids.tolist(),
            'refined_centroids': refined_centroids.tolist(),
            'centroid_positions': fcm_positions
        })
    
    return render(request, 'segmentation/custom_fcm.html')
```

## Performance Tips

1. **Image Size**: For large images, the preprocessing function automatically resizes to max 1000px
2. **Convergence**: The algorithm typically converges in 10-50 iterations
3. **Memory**: Membership matrices can be large for high-resolution images
4. **Fuzziness Parameter**: 
   - `m = 1.1-1.5`: Crisp boundaries (faster)
   - `m = 2.0`: Standard fuzziness (recommended)
   - `m = 2.5-3.0`: Very fuzzy boundaries (slower)

## Troubleshooting

### Common Issues

**Issue**: "Number of custom centroids must match k"
```python
# Solution: Ensure centroids array has correct shape
custom_centroids = np.array([...])  # Shape should be (k, 3) for RGB
k = len(custom_centroids)
```

**Issue**: "Centroid dimensions must match image channels"
```python
# Solution: For RGB images, centroids need 3 values
custom_centroids = np.array([
    [R, G, B],  # Not [R, G] or [R, G, B, A]
    # ...
])
```

**Issue**: Slow convergence
```python
# Solution: Reduce fuzziness parameter or max iterations
fcm_segmentation_with_custom_centroids(image, centroids, k, m=1.5, max_iter=50)
```

## Testing Your Implementation

Run the test scripts to verify everything works:

```bash
# Basic functionality test
py simple_fcm_test.py

# Comprehensive examples
py fcm_example_usage.py

# Custom test with your image
py test_fcm_custom.py path/to/your/image.jpg
```

## What's Different from Before

| Before (scikit-fuzzy) | Now (Custom Implementation) |
|----------------------|----------------------------|
| ❌ Import errors with newer Python | ✅ No external dependencies |
| ❌ Limited control over initialization | ✅ Full control with custom centroids |
| ❌ Complex parameter tuning | ✅ Simple, intuitive parameters |
| ❌ Inconsistent results | ✅ Reliable, reproducible results |
| ❌ No membership matrix access | ✅ Full membership matrix available |

## Next Steps

1. **Test with your images**: Try different centroid strategies
2. **Experiment with fuzziness**: Find the best `m` value for your use case
3. **Integrate with UI**: Add centroid selection interface
4. **Optimize performance**: Adjust image sizes and iteration limits
5. **Analyze results**: Use membership matrices for uncertainty analysis

Your FCM implementation is now **robust, reliable, and fully customizable**! 🎯 