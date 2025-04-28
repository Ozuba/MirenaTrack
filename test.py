import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import LineCollection

# Step 1: Create a set of (x, y) points for the contour (for example, on a sine wave)
x = np.linspace(0, 10, 100)
y = np.sin(x)  # y values as a sine function

# Step 2: Calculate the intensity (for example, based on the sine values)
intensity = np.abs(np.sin(x))  # Intensity can be the absolute value of sine

# Step 3: Create a path for the contour from the (x, y) points
vertices = np.column_stack((x, y))  # Combine x and y into (x, y) pairs
codes = [Path.MOVETO] + [Path.LINETO] * (len(x) - 1)  # Define path segments
path = Path(vertices, codes)

# Step 4: Create a colormap and normalize the intensity values
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=intensity.min(), vmax=intensity.max())

# Step 5: Create a collection of line segments with different colors
segments = []
for i in range(len(x) - 1):
    segment = [[x[i], y[i]], [x[i + 1], y[i + 1]]]
    segments.append(segment)

# Create a LineCollection object to handle the coloring of the line segments
lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, array=intensity[:-1])

# Step 6: Plot the path with colored segments
fig, ax = plt.subplots()
ax.add_collection(lc)

# Add colorbar to indicate intensity
plt.colorbar(lc, ax=ax, label='Intensity')

# Step 7: Set axis limits and labels
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()
