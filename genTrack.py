import json
import random
import numpy as np
from contourTools import *


def genGates(contour, gateDist: float = 4, trackWidth: float = 3):
    coneTrack = []
    # Resampleamos el contorno con distancia entre las puertas
    gatePoints = resample_contour(contour, spacing_m=gateDist)
    for i in range(0, len(gatePoints)-1):  # Recorremos todos los puntos
        p1 = gatePoints[i]
        p2 = gatePoints[(i + 1) % len(gatePoints)]  # Index with wrap
        tan = p2 - p1  # Tangent Vector
        tan /= np.linalg.norm(tan)   # Normalizamos
        perp = np.array([tan[1], -tan[0]])  # Get the perpendicular Vector
        perp /= np.linalg.norm(perp)
        gate = {
            "blue": (p1 + perp * trackWidth / 2).tolist(),
            "yellow": (p1 - perp * trackWidth / 2).tolist(),
        }
        coneTrack.append(gate)
    slope, curvature = analyze_contour(contour)
    # get spawn from the minimum place
    spawn_pos = contour[np.argmin(abs(curvature))]
    spawn_dir = contour[np.argmin(abs(curvature))+1] - contour[np.argmin(abs(curvature))]
    spawn_rot = np.arctan2(spawn_dir[0],spawn_dir[1])
    spawn = {"x": spawn_pos[0], "y": spawn_pos[1],"theta" : spawn_rot}
    return coneTrack, spawn


def genTrackFile(cones, spawn):
    bCones = [
        {"x": gate["blue"][0], "y": gate["blue"][1], "type": "blue"} for gate in cones
    ]
    yCones = [
        {"x": gate["yellow"][0], "y": gate["yellow"][1], "type": "yellow"}
        for gate in cones
    ]

    data = {"spawn": spawn,
            "cones": bCones + yCones}
    with open("track.json", "w") as file:
        json.dump(data, file, indent=4)


nmap = genNoiseMap(100, 100)
cut = random.uniform(np.min(nmap), np.max(nmap))
contours = find_contours(nmap, level=0.2)
contours = findClosed(contours)


plt.imshow(nmap, cmap="viridis", origin="lower")
plt.show()
contour = contours[random.randint(0, len(contours))-1]
contour = resize_contour(contour, 90)
contour = resample_contour(contour)
slope, curvature = analyze_contour(contour)
plot_curvature(contour, curvature, figsize=(8, 8), cmap='RdBu_r')
plt.plot(contour[:, 0], contour[:, 1], color="r")


gates, spawn = genGates(contour)


for gate in gates:
    plt.scatter(*gate["blue"], color="blue", s=20)
    plt.scatter(*gate["yellow"], color="yellow", s=20)
plt.scatter(spawn["x"],spawn["y"], color="orange", s=100)
plt.show()


genTrackFile(gates, spawn)
