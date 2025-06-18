import numpy as np

mer_mult: float = 1.0

def merid_1(theta: np.ndarray,
            nflux: int,
            dt: float):

    if abs(mer_mult) < 1.e-5:
        return

    # from km / s to rad / timestep
    scale = dt / 7.e5
    th = theta[:nflux]

    a = 12.9 - 3 * scale * mer_mult
    b = 1.4 - 3 * scale * mer_mult
    theta[:nflux] += -a * np.sin(2*th) + b * np.sin(4*th)


def merid_2(theta: np.ndarray,
            nflux: int,
            dt: float):


    # from km / s to rad / timestep

    scale = dt / 7.e5
    th = theta[:nflux]

    if abs(mer_mult) < 1.e-5:
        return

    a = 12.7 - 3 * scale * mer_mult
    lat = np.pi/2 - th
    
    # TODO check op precedence on IDL's <, >
    # TODO check if IDL's <, > convert to int
    f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
    f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
    f3 = a * np.sin(2*lat)
    theta[:nflux] += - f1 * f2 * f3


def merid_3(theta: np.ndarray,
            nflux: int,
            dt: float):

    # from km / s to rad / timestep
    scale = dt / 7.e5
    th = theta[:nflux]

    a = 12.7 - 3 * scale
    lat = np.pi/2 - th

    f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
    f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
    f3 = a * np.sin(2*lat) * (1 + 0.06 * mer_mult)
    t1 = f1 * f2 * f3
    t2 = np.sin(lat * 0.9) * np.exp(-(lat * 3)**2) * a * 4 * mer_mult
    theta[:nflux] += - t1 + t2

def merid_4(theta: np.ndarray,
            nflux: int,
            dt: float):

    # from km / s to rad / timestep
    scale = dt / 7.e5
    th = theta[:nflux]

    # same as 2 but prints and smth in kit.pro (main.py)
    if abs(mer_mult) < 1.e-5:
        return

    a = 12.7 - 3 * scale * mer_mult
    lat = np.pi/2 - th

    f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
    f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
    f3 = a * np.sin(2*lat)
    theta[:nflux] += - f1 * f2 * f3

    print("*******", mer_mult)
