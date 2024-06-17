import numpy as np

###
### Functions for ternary plots (not really needed with mpltern)
###


def polycorners(ncorners=3):
    """
    Return 2D cartesian coordinates of a regular convex polygon of a specified
    number of corners.
    Args:
        ncorners (int, optional) number of corners for the polygon (default 3).
    Returns:
        (ncorners, 2) np.ndarray of cartesian coordinates of the polygon.
    """

    center = np.array([0.5, 0.5])
    points = []

    for i in range(ncorners):
        angle = (float(i) / ncorners) * (np.pi * 2) + (np.pi / 2)
        x = center[0] + np.cos(angle) * 0.5
        y = center[1] + np.sin(angle) * 0.5
        points.append(np.array([x, y]))

    return np.array(points)


def bary2cart(bary, corners):
    """
    Convert barycentric coordinates to cartesian coordinates given the
    cartesian coordinates of the corners.
    Args:
        bary (np.ndarray): barycentric coordinates to convert. If this matrix
            has multiple rows, each row is interpreted as an individual
            coordinate to convert.
        corners (np.ndarray): cartesian coordinates of the corners.
    Returns:
        2-column np.ndarray of cartesian coordinates for each barycentric
        coordinate provided.
    """

    cart = None

    if len(bary.shape) > 1 and bary.shape[1] > 1:
        cart = np.array([np.sum(b / np.sum(b) * corners.T, axis=1) for b in bary])
    else:
        cart = np.sum(bary / np.sum(bary) * corners.T, axis=1)

    return cart


def CLR(input_array):
    """Transform chemical composition in colors

    Inputs
    ------
    input_array: n*4 array
        4 chemical inputs with sio2, al2o3, k2o and na2o in 4 columns, n samples in rows

    Returns
    -------
    out: n*3 array
        RGB colors
    """
    XXX = input_array.copy()
    XXX[:, 2] = XXX[:, 2] + XXX[:, 3]  # adding alkalis
    out = np.delete(XXX, 3, 1)  # remove 4th row
    # min max scaling to have colors in the full RGB scale
    out[:, 0] = (out[:, 0] - out[:, 0].min()) / (out[:, 0].max() - out[:, 0].min())
    out[:, 1] = (out[:, 1] - out[:, 1].min()) / (out[:, 1].max() - out[:, 1].min())
    out[:, 2] = (out[:, 2] - out[:, 2].min()) / (out[:, 2].max() - out[:, 2].min())
    return out


def make_ternary(
    ax,
    t,
    l,
    r,
    z,
    labelt,
    labell,
    labelr,
    levels,
    levels_l,
    c_m,
    norm,
    boundaries_SiO2,
    annotation="(a)",
):

    ax.plot([1.0, 0.5], [0.0, 0.5], [0.0, 0.5], "--", color="black")

    ax.tricontourf(t, l, r, z, levels=levels, cmap=c_m, norm=norm)

    tc = ax.tricontour(t, l, r, z, levels=levels_l, colors="k", norm=norm)

    ax.clabel(tc, inline=1, fontsize=7, fmt="%1.1f")

    ax.set_tlabel(labelt)
    # ax.set_llabel(labell)
    # ax.set_rlabel(labelr)

    ax.taxis.set_label_rotation_mode("horizontal")
    # ax.laxis.set_tick_rotation_mode('horizontal')
    # ax.raxis.set_label_rotation_mode('horizontal')

    make_arrow(ax, labell, labelr)

    ax.raxis.set_ticks([])

    # Using ``ternary_lim``, you can limit the range of ternary axes.
    # Be sure about the consistency; the limit values must satisfy:
    # tmax + lmin + rmin = tmin + lmax + rmin = tmin + lmin + rmax = ternary_scale
    ax.set_ternary_lim(
        boundaries_SiO2[0],
        boundaries_SiO2[1],  # tmin, tmax
        0.0,
        boundaries_SiO2[0],  # lmin, lmax
        0.0,
        boundaries_SiO2[0],  # rmin, rmax
    )

    ax.annotate(annotation, xy=(-0.1, 1.0), xycoords="axes fraction", fontsize=12)

    ax.spines["tside"].set_visible(False)

    # ax.annotate(labell, xy=(-0.1,-0.07), xycoords="axes fraction", ha="center")
    # ax.annotate(labelr, xy=(1.1,-0.07), xycoords="axes fraction", ha="center")

    ax.tick_params(labelrotation="horizontal")


def make_arrow(ax, labell, labelr, sx1=-0.1, sx2=1.02, fontsize=9, linewidth=2):
    ax.annotate(
        "",
        xy=(sx1, 0.03),
        xycoords="axes fraction",
        xytext=(sx1 + 0.08, 0.18),
        arrowprops=dict(arrowstyle="->", color="k", linewidth=linewidth),
    )

    ax.annotate(
        labell,
        xy=(sx1 + 0.03, 0.08),
        xycoords="axes fraction",
        ha="center",
        rotation=60,
        fontsize=fontsize,
    )

    ax.annotate(
        "",
        xy=(sx2, 0.18),
        xycoords="axes fraction",
        xytext=(sx2 + 0.08, 0.03),
        arrowprops=dict(arrowstyle="<-", color="k", linewidth=linewidth),
    )

    ax.annotate(
        labelr,
        xy=(sx2 + 0.05, 0.08),
        xycoords="axes fraction",
        ha="center",
        rotation=-60,
        fontsize=fontsize,
    )


def plot_loss(ax, loss, legends, scale="linear"):
    for count, i in enumerate(loss):
        ax.plot(i, label=legends[count])

    ax.legend()
    ax.set_yscale(scale)
    ax.set_xlabel("Epoch")
