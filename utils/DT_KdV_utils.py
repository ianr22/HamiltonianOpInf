# IMPORTS
# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# function to return POD base
def get_POD_basis(UU, n=150):
    return UU[:,:n]

# helper function to save paths
def set_path_name_list(dir, subdirectory: str, names: list):
    return [f"{dir}/{subdirectory}/{name}" for name in names]

def plot_trajectory(x_plot, x_value, dt, title, name_of_file):
    """
    Parameters
    -----------
    x_plot : ndarray
    x_value : ndarray
    """
    fig, ax = plt.subplots()
    line, = ax.plot(x_plot, x_value[:, 0], lw=2)

    ax.set_xlim([-20, 20])
    ax.set_ylim([-1, 9])
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)

    def update(frame):
        line.set_ydata(x_value[:, frame])
        ax.set_title(f't = {frame * dt:.2f}')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(x_value.shape[1]), blit=True, interval=20)
    plt.show()
    ani.save(name_of_file, writer='pillow')