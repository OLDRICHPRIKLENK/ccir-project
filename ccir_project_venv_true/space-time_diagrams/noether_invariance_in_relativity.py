import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants
m = 1.0     # mass units
c = 1.0     # natural units
dt = 0.05
steps = 200

# Particle momenta
momenta = [0.0, 2.0, -2.0]
colors = ['red', 'blue', 'green']

# Create cube vertices for 1D space + time
space_min, space_max = -3, 3
time_min, time_max = 0, steps*dt

cube_vertices = np.array([
    [space_min, 0, time_min],
    [space_max, 0, time_min],
    [space_max, 0, time_max],
    [space_min, 0, time_max],
    [space_min, 1, time_min],
    [space_max, 1, time_min],
    [space_max, 1, time_max],
    [space_min, 1, time_max]
])

edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

# Draw cube edges
for e in edges:
    xline = [cube_vertices[e[0],0], cube_vertices[e[1],0]]
    yline = [cube_vertices[e[0],1], cube_vertices[e[1],1]]
    zline = [cube_vertices[e[0],2], cube_vertices[e[1],2]]
    ax.plot3D(xline, yline, zline, color='black')

# Plot each particle's worldline
y_offsets = [-0.2, 0, 0.2]  # just to separate lines in y-axis visually
for p, col, y_off in zip(momenta, colors, y_offsets):
    gamma = np.sqrt(1 + (p/(m*c))**2)
    E = np.sqrt((m*c**2)**2 + (p*c)**2)
    v = p / (gamma * m)
    
    xs = [0 + v*dt*k for k in range(steps+1)]
    ys = [y_off]*len(xs)
    zs = [dt*k for k in range(steps+1)]
    
    ax.plot(xs, ys, zs, color=col, label=f"p={p}, E={E:.2f}")
    
# Labels
ax.set_xlabel("Space (1D)")
ax.set_ylabel("Offset for clarity")
ax.set_zlabel("Time - Energy invariance")
ax.set_title("Worldlines in 1D Space + Time Cub - momentum invariance")

ax.legend()
ax.view_init(elev=22, azim=30)
plt.show()

