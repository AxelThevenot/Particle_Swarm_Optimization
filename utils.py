import os
import glob
import shutil

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter




def make_gif_from_folder(folder, out_file_path, remove_folder=True):
    files = os.path.join(folder, '*.png')
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(files))]
    img.save(fp=out_file_path, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
    shutil.rmtree(folder, ignore_errors=True)



plt.rcParams['figure.figsize'] = [12, 6] # default = [6.0, 4.0]
plt.rcParams['figure.dpi']     = 100     # default = 72.0
plt.rcParams['font.size']      = 7.5     # default = 10.0

cmap = cm.colors.LinearSegmentedColormap.from_list('Custom',
                                                   [(0, '#2f9599'),
                                                    (0.45, '#eee'),
                                                    (1, '#8800ff')], N=256)


def plot_2d_pso(meshgrid, function, particles=None, velocity=None, normalize=True, color='#000', ax=None):
    X_grid, Y_grid = meshgrid
    Z_grid = function(X_grid, Y_grid)
    # get coordinates and velocity arrays
    if particles is not None:
        X, Y = particles.swapaxes(0, 1)
        Z = function(X, Y)
        if velocity is not None:
            U, V = velocity.swapaxes(0, 1)
            if normalize:
                N = np.sqrt(U**2+V**2)
                U, V = U/N, V/N

    # create new ax if None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    # add contours and contours lines
    ax.contour(X_grid, Y_grid, Z_grid, levels=30, linewidths=0.5, colors='#999')
    cntr = ax.contourf(X_grid, Y_grid, Z_grid, levels=30, cmap=cmap, alpha=0.7)
    if particles is not None:
        ax.scatter(X, Y, color=color)
        if velocity is not None:
            ax.quiver(X, Y, U, V, color=color, headwidth=2, headlength=2, width=5e-3)

    # add labels and set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(np.min(X_grid), np.max(X_grid))
    ax.set_ylim(np.min(Y_grid), np.max(Y_grid))
    ax.set_aspect(aspect='equal')


def plot_3d_pso(meshgrid, function, particles=None, velocity=None, normalize=True, color='#000', ax=None):
    X_grid, Y_grid = meshgrid
    Z_grid = function(X_grid, Y_grid)
    # get coordinates and velocity arrays
    if particles is not None:
        X, Y = particles.swapaxes(0, 1)
        Z = function(X, Y)
        if velocity is not None:
            U, V = velocity.swapaxes(0, 1)
            W = function(X + U, Y + V) - Z

    # create new ax if None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap,
                           linewidth=0, antialiased=True, alpha=0.7)
    ax.contour(X_grid, Y_grid, Z_grid, zdir='z', offset=0, levels=30, cmap=cmap)
    if particles is not None:
        ax.scatter(X, Y, Z, color=color, depthshade=True)
        if velocity is not None:
            ax.quiver(X, Y, Z, U, V, W, color=color, arrow_length_ratio=0., normalize=normalize)

    len_space = 10
    # Customize the axis
    max_z = (np.max(Z_grid) // len_space + 1).astype(np.int) * len_space
    ax.set_xlim3d(np.min(X_grid), np.max(X_grid))
    ax.set_ylim3d(np.min(Y_grid), np.max(Y_grid))
    ax.set_zlim3d(0, max_z)
    ax.zaxis.set_major_locator(LinearLocator(max_z // len_space + 1))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # Rmove fills and set labels
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf)


if __name__ == '__main__':
    N = 1000
    t = np.arange(0, N + 1)
    w = (0.4 / N**2) * (t - N) ** 2 + 0.4
    c_1 = -3 * t / N + 3.5
    c_2 =  3 * t / N + 0.5

    plt.plot(t, w, color='#999', label=r'$w = 0.4\frac{(t - n)}{n^2} + 0.4$')
    plt.plot(t, c_1, color='#80f', label=r'$c_1 = -3\frac{t}{n} + 3.5$')
    plt.plot(t, c_2, color='#80f', label=r'$c_2 = 3\frac{t}{n} + 0.5$')
    plt.legend()
    plt.show()
