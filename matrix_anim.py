import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ArtistAnimation

x = int(input("Matrix dimensions: "))

fig, ax = plt.subplots()
img = []
for i in range(0, x):
    img.append([plt.imshow(np.random.rand(x, x), cmap="hot")])

print(img)
# Artist animation over range?:
# https://stackoverflow.com/questions/18019226/matplotlib-artistanimation-gives-typeerror-axesimage-object-is-not-iterable

ani = ArtistAnimation(fig, img, interval=30, repeat=False, blit=False)
# ani = FuncAnimation(fig, animate, frames=10, interval=10, init_func=init, blit=True, repeat=False)

plt.title("Random matrix heat map")
plt.axis("off")  # Turn off x and y axis
plt.colorbar()  # Legend for heat map

# ani.save("test_artist.mp4", "ffmpeg", fps=30)
plt.show()
