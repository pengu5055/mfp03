import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as anim
import matplotlib.animation

# To see writers anim.writers.list()

fig, ax = plt.subplots()
xdata = []
ydata = []
line, = plt.plot([], [])


def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    return line,


def animate(frame):
    print(frame)
    xdata.append(frame)
    ydata.append(np.sin(frame))
    line.set_data(xdata, ydata)
    return line,


ani = FuncAnimation(fig, animate, frames=np.linspace(0, 10, 200), interval=10, init_func=init, blit=True, repeat=False)

ani.save("test2.mp4", writer="ffmpeg", fps=30)
plt.show()
