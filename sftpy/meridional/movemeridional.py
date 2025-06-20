import numpy as np

def move_meridional(theta: np.ndarray,
                    nflux: int,
                    dt: float,
                    amplifier: float,
                    mode: int=2):

    # from km / s to rad / timestep
    scale = dt / 7.e5
    th = theta[:nflux]

    if mode == 1:
        if abs(amplifier) < 1.e-5:
            return

        a = 12.9 - 3 * scale * amplifier
        b = 1.4 - 3 * scale * amplifier
        theta[:nflux] += -a * np.sin(2*th) + b * np.sin(4*th)

    elif mode == 2:
        if abs(amplifier) < 1.e-5:
            return

        a = 12.7 - 3 * scale * amplifier
        lat = np.pi/2 - th
        
        # TODO check op precedence on IDL's <, >
        # TODO check if IDL's <, > convert to int
        f1 = 1 - np.exp(-np.clip(3 * th**3), a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = a * np.sin(2*lat)
        theta[:nflux] += - f1 * f2 * f3

    elif mode == 3:
        a = 12.7 - 3 * scale
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3) a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = a * np.sin(2*lat) * (1 + 0.06 * amplifier)
        t1 = f1 * f2 * f3
        t2 = np.sin(lat * 0.9) * np.exp(-(lat * 3)**2) * a * 4 * amplifier
        theta[:nflux] += - t1 + t2

    elif mode == 4:
        # same as 2 but prints and smth in kit.pro (main.py)
        if abs(amplifier) < 1.e-5:
            return

        a = 12.7 - 3 * scale * amplifier
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = a * np.sin(2*lat)
        theta[:nflux] += - f1 * f2 * f3

        print("*******", amplifier)
