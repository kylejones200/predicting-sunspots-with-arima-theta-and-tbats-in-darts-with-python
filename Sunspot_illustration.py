"""Generated from Jupyter notebook: Sunspot_illustration

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Create figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 1]}
)

# Set up the transit parameters
transit_depth = 0.98
total_frames = 100
time_points = np.linspace(-3, 3, total_frames)
light_curve = []

# Create static star circle
star = plt.Circle((0.5, 0.5), 0.4, color="lightgray", zorder=1)
ax1.add_artist(star)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis("off")

# Create planet circle (initially outside view)
planet = plt.Circle((-0.2, 0.5), 0.1, color="black", zorder=2)
ax1.add_artist(planet)

# Set up light curve plot
(line,) = ax2.plot([], [], "k-", linewidth=1)
(points,) = ax2.plot([], [], "k.", markersize=10)
ax2.set_xlabel("Time - T$_c$ (hours)")
ax2.set_xlim(-3, 3)
ax2.set_ylim(0.97, 1.01)
ax2.grid(True, alpha=0.3)

# Initialize empty data
x_data, y_data = [], []


def init():
    line.set_data([], [])
    points.set_data([], [])
    return line, points, planet


def animate(frame):
    # Update planet position
    x = frame / total_frames
    planet.center = (x * 1.0, 0.5)

    # Calculate light curve point
    t = time_points[frame]
    if -1 <= t <= 1:
        y = transit_depth
    elif -2 <= t < -1:
        y = 1 - (1 - transit_depth) * (t + 2)
    elif 1 < t <= 2:
        y = transit_depth + (1 - transit_depth) * (t - 1)
    else:
        y = 1.0

    x_data.append(t)
    y_data.append(y)

    # Update line and point
    line.set_data(x_data, y_data)
    points.set_data([t], [y])

    return line, points, planet



def main():
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=total_frames, interval=50, blit=True
    )

    plt.tight_layout()
    plt.show()

    # Uncomment the following line to save the animation (requires additional setup)
    anim.save("transit_animation.gif", writer="pillow")


if __name__ == "__main__":
    main()
