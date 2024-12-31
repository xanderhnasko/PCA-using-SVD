# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:58:45 2024

@author: Morteza
"""
import numpy as np
import spectral
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from PIL import Image

# Load the hyperspectral image 
image = Image.open("starry_night.png")  
data = np.array(image)  

if len(data.shape) == 2:  # Grayscale
    data = data[:, :, np.newaxis]


# Reshape the data to 2D (samples, bands)
n_samples, n_lines, n_bands = data.shape
reshaped_data = data.reshape((n_samples * n_lines, n_bands))
reshaped_data = reshaped_data/255.0

# Center the data by subtracting the mean of each band (feature)
mean_centered_data = reshaped_data - np.mean(reshaped_data, axis=0)

# Apply PCA using SVD
n_components = 3  
pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(mean_centered_data)

# Reshape the principal components back to image dimensions for visualization
pca_images = pca_data.reshape((n_samples, n_lines, n_components))

# Plot the first 6 principal components
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

pca_images_normalized = (pca_images - np.min(pca_images, axis=(0, 1))) / \
                        (np.max(pca_images, axis=(0, 1)) - np.min(pca_images, axis=(0, 1)))

for i in range(n_components):
    pc_image = pca_images_normalized[:, :, i]
    im = axes[i].imshow(pc_image, cmap='viridis')  # Use a color map like 'viridis'
    axes[i].set_title(f'Principal Component {i+1}')
    axes[i].axis('off')
    fig.colorbar(im, ax=axes[i], orientation='vertical')  # Add a color bar for reference


plt.tight_layout()
plt.show()

# ----- Compression and Reconstruction -----

# Function to reconstruct image using a given number of components
def reconstruct_image(pca, pca_data, n_components, mean):
    """
    Reconstruct the image from the PCA-transformed data using 'n_components' principal components.
    
    Parameters:
    pca: PCA object already fitted to the data
    pca_data: Transformed PCA data (low-dimensional representation)
    n_components: Number of components to use for reconstruction
    mean: Mean of the original data (for reversing mean-centering)
    
    Returns:
    Reconstructed data in the original space
    """
    # Only use the first 'n_components' components for reconstruction
    reduced_data = pca_data[:, :n_components]
    
    # Reconstruct the data by multiplying with the first 'n_components' eigenvectors
    reconstructed_data = np.dot(reduced_data, pca.components_[:n_components, :])
    
    # Add the mean back to the data (reversing mean-centering)
    return reconstructed_data + mean

# Calculate reconstruction errors and store them
reconstruction_errors = []

# Iterate over the number of components (1 to n_components)
for n in range(1, n_components + 1):
    reconstructed_data = reconstruct_image(pca, pca_data, n, np.mean(reshaped_data, axis=0))
    
    # Calculate the Mean Squared Error between the original and reconstructed data
    mse = mean_squared_error(reshaped_data, reconstructed_data)
    reconstruction_errors.append(mse)
    
    # Print the reconstruction error after adding each eigenvector
    print(f'Reconstruction error with {n} components: {mse}')

# Plot the reconstruction error after adding each component
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_components + 1), reconstruction_errors, marker='o', linestyle='--')
plt.title('Reconstruction Error vs Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.show()

