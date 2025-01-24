import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import yaml
from yaml import Loader
from sklearn.preprocessing import MinMaxScaler
from test_disc_functions.load_acetone import acetone_Xmm as acetone_pts
from test_disc_functions.load_acetone import tri as acetone_tri
import warnings

discfuncs_dict = {}


def dfunc_basic_001(X):
    """
    Circular discontinuity: center=(0, 0.5), radius 0.5;
    Constant jump: 2
    Functions: x1 + x2 (outside circ.), x1 + x2 + 2 (inside circ.)
    """

    yIN = (X[:, 0] + X[:, 1])

    isIN = ((X - np.array([[0., 0.5]])) ** 2).sum(axis=1) <= 0.25

    y = yIN * isIN + (yIN + 2) * np.logical_not(isIN)

    return y


discfuncs_dict['dfunc_basic_001'] = dfunc_basic_001


def dfunc_basic_002(X):
    """
    Circular discontinuity: center=(0.5, 0.5), radius 0.316 c.a.;
    Constant jump: 2
    Functions: x1 + x2 (outside circ.), x1 + x2 + 2 (inside circ.)
    """

    yIN = (X[:, 0] + X[:, 1])

    isIN = ((X - np.array([[0.5, 0.5]])) ** 2).sum(axis=1) <= 0.1

    y = yIN * isIN + (yIN + 2) * np.logical_not(isIN)

    return y


discfuncs_dict['dfunc_basic_002'] = dfunc_basic_002


def dfunc_basic_003(X):
    """
    Circular discontinuity: center=(0.5, 0.5), radius 0.316 c.a.;
    Constant jump: 2
    Functions: exp(-(x1 - 0.5)^2 + (x2-0.5)^2) (outside circ.), sin(x1) + 4 sin(x2) (inside circ.)
    """

    yIN = (np.array([[1., 4.]]) * np.sin(X)).sum(axis=1)
    yOUT = np.exp(
        -((X[:, 0] - 0.5) ** 2) + (X[:, 1] - 0.5) ** 2
    )
    isIN = ((X - np.array([[0.5, 0.5]])) ** 2).sum(axis=1) <= 0.08

    y = yIN * isIN + yOUT * np.logical_not(isIN)

    return y


discfuncs_dict['dfunc_basic_003'] = dfunc_basic_003


def dfunc_basic_004(X):
    """
    Circular discontinuity: center=(0.5, 0.5), radius 0.316 c.a.;
    Constant jump: 2
    Functions: exp(-(x1 - 0.5)^2 + (x2-0.5)^2) (outside circ.), sin(x1) + 4 sin(x2) (inside circ.)
    """

    yIN = (np.array([[1., 4.]]) * np.sin(X)).sum(axis=1)
    yOUT = - np.exp(
        -((X[:, 0] - 0.5) ** 2) + (X[:, 1] - 0.5) ** 2
    )
    isIN = ((X - np.array([[0.5, 0.5]])) ** 2).sum(axis=1) <= 0.08

    y = yIN * isIN + yOUT * np.logical_not(isIN)

    return y


discfuncs_dict['dfunc_basic_004'] = dfunc_basic_004


def franke_function(X):
    """
    Fake discontinuos function (i.e., it is continuos).
    """
    y = (
            0.75 * np.exp(-((9 * X[:, 0] - 2) ** 2) / 4.0 - (9 * X[:, 1] - 2) ** 2 / 4.0) +
            0.75 * np.exp(-((9 * X[:, 0] + 1) ** 2) / 49.0 - (9 * X[:, 1] + 1) / 10.0) +
            0.5 * np.exp(-((9 * X[:, 0] - 7) ** 2) / 4.0 - (9 * X[:, 1] - 3) ** 2 / 4.0) -
            0.2 * np.exp(-((9 * X[:, 0] - 4) ** 2) - (9 * X[:, 1] - 7) ** 2)
    )

    return y


discfuncs_dict['franke_function'] = franke_function


def disc_twoexps(X):
    """
    Functions:
    sin(0.4 pi(x1 + x2)) if x2 >= exp(x1)
    sin(0.7 pi(x1 + x2)) - 4 if x2 < exp(x1) - 1
    sin(pi(x1 + x2)) + 4 otherwise
    """

    X_ = (X * 2) - 1

    yF1 = X_[:, 1] >= np.exp(X_[:, 0])
    yF2 = X_[:, 1] < np.exp(X_[:, 0]) - 1
    yF3 = np.logical_and(np.logical_not(yF1), np.logical_not(yF2))

    y1 = np.sin(0.4 * np.pi * X_.sum(axis=1))
    y2 = np.sin(0.7 * np.pi * X_.sum(axis=1)) - 4
    y3 = np.sin(np.pi * X_.sum(axis=1)) + 4

    y = y1 * yF1 + y2 * yF2 + y3 * yF3

    return y


discfuncs_dict['disc_twoexps'] = disc_twoexps


def acetone_surface(X):
    """
    Interpolates the value on the surface defined by the given normalized acetone's points for a set of query points.

    Parameters:
    - X: (M, 2) array of points (x, y) for which to compute the interpolated z-values.

    Returns:
    - interpolated_values: (M,) array of interpolated z-values for each point.
    """
    # Ensure points and query_points are numpy arrays
    points = acetone_pts
    query_points = np.asarray(X)

    # Perform Delaunay triangulation on the 3D points in the XY plane
    tri = acetone_tri

    # Interpolate each query point
    interpolated_values = []
    for qp in query_points:
        simplex = tri.find_simplex(qp, bruteforce=True)  # Find the containing triangle for (x, y)
        # simplex, bary_coords = my_find_simplex(qp, acetone_pts, acetone_tri.simplices)
        if simplex == -1:
            warnings.warn(f"Query point {qp} is outside the triangulation")
        else:
            # Get vertices of the containing triangle
            vertices = tri.simplices[simplex]
            tri_points = points[vertices]

            # Compute barycentric coordinates
            T = np.c_[tri_points[:, :2], np.ones(3)]  # Build the 2D triangle matrix
            bary_coords = np.linalg.solve(T.T, np.r_[qp, 1])  # Solve for barycentric coordinates

            # Interpolate z-value using barycentric coordinates
            z_value = np.dot(bary_coords, tri_points[:, 2])
            interpolated_values.append(z_value)

    return np.array(interpolated_values)


discfuncs_dict['acetone_surface'] = acetone_surface




