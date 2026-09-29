import numpy as np 
import matplotlib.pyplot as plt 
import h5py


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{txfonts}",
})


def flush_color_bar_single_column(xlim,ylim,width_mm=88):
    
    # Margins include room for 11 pt labels and colorbar ticks/label.
    left_mm = 16
    right_mm = 15
    top_mm = 8
    bottom_mm = 14
    colorbar_width_mm = 2.5
    gap_mm = 0

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    axes_width_mm = (
        width_mm - left_mm - right_mm - colorbar_width_mm - gap_mm
    )
    axes_height_mm = axes_width_mm * y_span / x_span
    height_mm = top_mm + axes_height_mm + bottom_mm

    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    
    axis = fig.add_axes([
        left_mm / width_mm,
        bottom_mm / height_mm,
        axes_width_mm / width_mm,
        axes_height_mm / height_mm,
    ])
    
    cax = fig.add_axes([
        (left_mm + axes_width_mm + gap_mm) / width_mm,
        bottom_mm / height_mm,
        colorbar_width_mm / width_mm,
        axes_height_mm / height_mm,
    ])
    return fig, axis, cax