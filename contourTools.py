import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from noise import snoise2
from scipy.interpolate import splprep, splev
from scipy.integrate import cumulative_trapezoid
from skimage.measure import find_contours



def genNoiseMap(width, height, scale=0.04, octaves=2, persistence=0.6, lacunarity=3):
    """
    Generate a 2D OpenSimplex noise grid.

    :param width: Width of the noise grid.
    :param height: Height of the noise grid.
    :param scale: Scale of the noise (controls frequency).
    :param octaves: Number of octaves.
    :param persistence: Controls amplitude of each octave.
    :param lacunarity: Controls frequency of each octave.
    :return: A 2D numpy array with noise values.
    """
    noise_grid = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            noise_grid[y][x] = snoise2(
                x * scale,
                y * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
    return noise_grid


def findClosed(contours):
    c = []
    for cont in contours:
        if (
            np.allclose(cont[0], cont[-1], atol=0.001) and len(cont) > 30
        ):  #  SI el contorno es cerrado y tiene más de 10 puntos
            c.append(cont)
    return c


def resize_contour(cont, scale=100):
    """
    Resizes and Interpolates contour with cm resolution
    """
    # Normalizamos
    maxPoint = np.max(cont, axis=0)  # Right Up corner
    minPoint = np.min(cont, axis=0)  # Low Left corner
    diag = maxPoint - minPoint  # Diagonal
    normDim = max(diag)  # Normalizamos por la dimensión más grande
    print(f"Max:{maxPoint}, Min:{minPoint}")
    cont -= (minPoint) + diag / 2  # Centramos sobre 0
    cont /= normDim
    # Escalamos el contorno
    cont *= scale
    return cont


def resample_contour(points, spacing_m= 0.01):
    """
    Resample a contour with specified spacing in meters.
    
    Parameters:
    points : ndarray, shape (N, 2)
        Input contour points as (x, y) coordinates in meters
    spacing_m : float
        Desired spacing between points in meters
        
    Returns:
    ndarray, shape (M, 2)
        Resampled contour points with constant spacing
    """
    # Close the contour if it's not already closed
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # Fit a B-spline to the points
    try:
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=1)
    except Exception:
        # If periodic spline fails, try non-periodic
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=0)
    
    # Generate dense points for accurate arc length calculation
    u_fine = np.linspace(0, 1, 10000)
    x_fine, y_fine = splev(u_fine, tck)
    
    # Calculate cumulative arc length
    dx = np.diff(x_fine)
    dy = np.diff(y_fine)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = cumulative_length[-1]
    
    # Calculate number of points needed
    num_points = int(np.ceil(total_length / spacing_m))
    
    # Generate evenly spaced points by arc length
    target_lengths = np.linspace(0, total_length, num_points)
    u_resampled = np.interp(target_lengths, cumulative_length, u_fine)
    
    # Get final points
    x_resampled, y_resampled = splev(u_resampled, tck)
    
    return np.column_stack([x_resampled, y_resampled])

def analyze_contour(points):
    """
    Calculate slope and curvature for an ordered set of points.
    
    Parameters:
    points : ndarray, shape (N, 2)
        Input contour points as (x, y) coordinates, assumed to be ordered
        
    Returns:
    ndarray, shape (N,)
        Slope at each point (in radians)
    ndarray, shape (N,)
        Curvature at each point (1/meters)
    """
    # Use central differences for first derivatives
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    
    # Second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    # Calculate slope (in radians)
    slopes = np.arctan2(dy, dx)
    
    # Calculate curvature
    # κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    numerator = dx * d2y - dy * d2x
    denominator = (dx**2 + dy**2)**(3/2)
    curvature = numerator / denominator
    
    return slopes, curvature


def plot_curvature(points, curvature, figsize=(10, 10), cmap='viridis', 
                  show_colorbar=True, abs_curvature=True):
    """
    Plot a contour colored by its curvature.
    
    Parameters:
    points : ndarray, shape (N, 2)
        Contour points as (x, y) coordinates
    curvature : ndarray, shape (N,)
        Curvature value at each point
    figsize : tuple
        Figure size in inches
    cmap : str
        Matplotlib colormap name
    show_colorbar : bool
        Whether to show the colorbar
    abs_curvature : bool
        Whether to plot absolute curvature values
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create line segments
    segments = np.column_stack([points[:-1], points[1:]])
    segments = segments.reshape(-1, 2, 2)
    
    # Process curvature values
    if abs_curvature:
        curv_values = np.abs(curvature)
        label = 'Absolute Curvature (1/m)'
    else:
        curv_values = curvature
        label = 'Curvature (1/m)'
    
    # Average curvature for each segment
    segment_curvatures = (curv_values[:-1] + curv_values[1:]) / 2
    
    # Create line collection
    norm = Normalize(vmin=np.min(curv_values), vmax=np.max(curv_values))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(segment_curvatures)
    
    # Add to plot
    line = ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(line)
        cbar.set_label(label)
    
    # Labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Contour Colored by Curvature')
    plt.show()




