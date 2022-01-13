import matplotlib.pyplot as plt
import numpy as np

from dylightful.utilities import make_name, parse_file_path, get_dir


def make_barplot(time_ser, prefix=None, save_path=None):
    """Craeates and saves bar plot


    Args:
        time_ser ([type]): by paser.get_timeseries processed pml time series of features
        path ([type], optional): Where to save. Defaults to None.
        prefix ([type], optional): Name of the file
    """
    save_path = get_dir(save_path)
    name = "_barplot.png"
    file_name = make_name(prefix=prefix, name=name, dir=save_path)
    code = time_ser

    pixel_per_bar = 4
    dpi = 300

    fig = plt.figure(figsize=(20, 13), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
    ax.imshow(code.T, cmap="binary", aspect="auto", interpolation="nearest")
    plt.xlabel("Timestep")
    plt.ylabel("Superfeature")
    plt.savefig(file_name, dpi=dpi)
