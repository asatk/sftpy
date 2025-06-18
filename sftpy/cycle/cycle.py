import numpy as np

min_lat = 0             # minimum latitude for source emergence
max_lat = np.pi / 2     # maximum latitude for source emergence
cyl_t = 21.9            # cycle time
cyl_overlap = 3.0       # overlap between consecutive cycles
cyl_peak = 4.0          # time of peak in activity and total flux
cyl_mult = 1.0          # relative strength of activity cycle

def cycle_0(time: float):
    ...

def cycle_1(time: float):

    a = np.full(2, 2 * np.pi / (cyl_t + 2 * cyl_overlap), dtype=np.float64)
    a[0] *= np.mod(time + cyl_t, cyl_t)
    a[1] *= np.mod(time + cyl_t / 2, cyl_t)

    if cyl_overlap < 0.01:
        b = 5
        c = 5.81
    elif np.abs(cyl_overlap - 2) < 0.01:
        b = 8
        c = 8.39
    else:
        a[0] = 2 * np.pi * cyl_peak / (cyl_t + 2 * cyl_overlap)
        b = (a[0] * np.cos(a[0]) + np.sin(a[0])) / \
                (2 * np.sin(a[0]) * a[0]**2) * np.pi**2
        c = a[0] * np.exp(-(a[0] / np.pi)**2 * b) / np.pi / \
                np.clip(np.sin(a[0]), a_min=0, a_max=None)

    # clipped sin
    sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

    source = np.array([1, -1]) * sin_clip * cyl_mult * a / np.pi * \
            np.exp(-(a / np.pi)**2 * b) * c
    latsource = (max_lat - (max_lat - min_lat) * a / np.pi) * sin_clip

    return source, latsource

def cycle_2():

    a = np.full(2, 2 * np.pi / (cyl_t + 2 * cyl_overlap), dtype=np.float64)
    a[0] *= np.mod(time + cyl_t, cyl_t)
    a[1] *= np.mod(time + cyl_t / 2, cyl_t)

    if cyl_overlap < 0.01:
        b = 5
        c = 5.81
    elif np.abs(cyl_overlap - 2) < 0.01:
        b = 8
        c = 8.39
    else:
        a[0] = 2 * np.pi * cyl_peak / (cyl_t + 2 * cyl_overlap)
        b = (a[0] * np.cos(a[0]) + np.sin(a[0])) / \
                (2 * np.sin(a[0]) * a[0]**2) * np.pi**2
        c = a[0] * np.exp(-(a[0] / np.pi)**2 * b) / np.pi / \
                np.clip(np.sin(a[0]), a_min=0, a_max=None)

    # clipped sin
    sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

    source = np.array([1, -1]) * sin_clip * cyl_mult * a / np.pi * \
            np.exp(-(a / np.pi)**2 * b) * c
    latsource = (max_lat - (max_lat - min_lat) * a / np.pi) * sin_clip

    # TODO yearssn
    source *= yearssn(time, yssn, ssn)

    return source, latsource

def cycle_3():
    return None, None

# matchsolarrecords
def cycle_4(time: float):
    
    minima = np.array([
        1635.1, 1646.0, 1657, 1668, 1679, 1690, 1700, 1713.5, 1724.0, 1733.5,
        1745.0, 1756.0, 1767.0, 1775.5, 1784.0, 1798.5, 1811.0, 1825.0, 1833.5,
        1844.0, 1856.5, 1867.0, 1879.0, 1890.0, 1901.4, 1913.4, 1923.8, 1934.4,
        1944.3, 1954.3, 1964.5, 1976.6, 1986.5, 1996.7, 2006.9, 2018.2, 2029.2,
        2040.1]) - 1646.001

    polarity = np.mod(np.arange(len(minima), dtype=np.float64), 2) * 2 - 1

    # force solar conditions
    cyl_t = 21.9
    cyl_overlap = 3.0
    min_lat = 0.0

    # determine index of first cycle
    yi = np.nonzero(time < minima)[0][0] - 2
    yd = 2 * np.array([minima[yi+1] - minima[yi], \
                       minima[yi+2] - minima[yi+1]])

    a = np.pi * np.array([(time - minima[yi]) / (yd[0] / 2 + cyl_overlap), \
                          (time - minima[yi+1]) / (yd[1] / 2 + cyl_overlap)])
    sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

    source = cyl_mult * sin_clip * a / np.pi * polarity[[yi,yi+1]] * \
            np.exp(-(a / np.pi) ** 2 * 8) * 8.39
    latsource = (max_lat - (max_lat - min_lat) * a / np.pi) * sin_clip

    if polarity[yi] < 0:
        source = np.roll(source, 1)
        latsource = np.roll(latsource, 1)

    # TODO yearssn
    source *= yearssn(time, yssn, ssn)

    return source, latsource
