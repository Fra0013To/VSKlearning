import numpy as np

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

