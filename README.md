#color-Analysis
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load image
img = cv2.imread('your_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_flat = img.reshape((-1, 3))

# Extract dominant color(s) with k-means
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(img_flat)
dominant = kmeans.cluster_centers_.astype(int)

def classify_hue(h, s):
    # Neutral: low saturation
    if s < 40:
        return 'Neutral'
    # Warm if hue between 0-50 or 330-360 (red/yellow)
    if (h <= 50) or (h >= 330):
        return 'Warm'
    # Cool if hue between 180-270 (blue)
    if 180 <= h <= 270:
        return 'Cool'
    # Otherwise, may be intermediate
    return 'Intermediate'

for rgb in dominant:
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    classification = classify_hue(h, s)
    print(f"Dominant Color RGB: {rgb} | HSV: ({h},{s},{v}) | Category: {classification}")

    
