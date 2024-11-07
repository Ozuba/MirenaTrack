import numpy as np
import matplotlib.pyplot as plt
from noise import snoise2
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours
import json
import random


def genNoiseMap(width, height, scale=0.04, octaves=2, persistence=0.8, lacunarity=2):
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


def resizeContour(cont, scale=100):
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

    # Generamos un eje distancia e interpolamos sobre el
    d = np.linspace(0, 1, scale * 100)  # Interpolamos en un espacio de cm
    rCont = np.zeros((len(d), 2))  # Array to hold the points
    tck, u = splprep([cont[:, 0], cont[:, 1]], s=0)  # Create parametric spline
    rCont[:, 0], rCont[:, 1] = splev(d, tck)  # Interpolate it over the metric space
    # Calculamos su distancia
    length = np.sum(
        np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    )  # Calculate the distance of the curve (m)
    print(f"Length: {length}")
    return rCont


def genGates(contour, gateDist: float = 4, trackWidth: float = 3):
    coneTrack = []
    dist = 0  # Distancia a 0
    for i in range(0, len(contour) - 1):  # Recorremos todos los puntos
        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]  # Index with wrap
        tan = p1 - p2  # Tangent Vector
        norm = np.linalg.norm(tan)  # sacamos la disntancia
        dist += norm  # Acumulamos distancia
        if dist > gateDist:  # Si toca poner un Gate lo ponemos
            tan /= norm  # Normalizamos
            perp = np.array([-tan[1], tan[0]])  # Get the perpendicular Vector
            gate = {
                "blue": (p1 + perp * trackWidth / 2).tolist(),
                "yellow": (p1 - perp * trackWidth / 2).tolist(),
            }
            coneTrack.append(gate)
            dist = 0  # Ponemos a 0 la distancia
    return coneTrack


def genTrackFile(cones):
    bCones = [
        {"x": gate["blue"][0], "y": gate["blue"][1], "type": "blue"} for gate in cones
    ]
    yCones = [
        {"x": gate["yellow"][0], "y": gate["yellow"][1], "type": "yellow"}
        for gate in cones
    ]
    data = {"cones": bCones + yCones}
    with open("track.json", "w") as file:
        json.dump(data, file, indent=4)


nmap = genNoiseMap(100, 100)
cut = random.uniform(np.min(nmap),np.max(nmap))
contours = find_contours(nmap, level=0.2)
contours = findClosed(contours)


plt.imshow(nmap, cmap="viridis", origin="lower")
plt.show()
contour = contours[random.randint(0,len(contours))-1]
contour = resizeContour(contour, 90)
plt.plot(contour[:, 0], contour[:, 1], color="r")
gates = genGates(contour)
for gate in gates:
    plt.scatter(*gate["blue"], color="blue", s=20)
    plt.scatter(*gate["yellow"], color="yellow", s=20)
plt.show()


genTrackFile(gates)


plt.title("Contours from 2D Function")

