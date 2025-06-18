# TODO needs review

def cycle_strength(time: float,
                   mode: int,
                   overlap: int,
                   pol_t: float,
                   strength: float,
                   max_lat: float,
                   peak: float=4.15):

    if mode == 4:
        # goto matchsolarrecords
        pass

    min_lat = 0

    a = np.full(2, 2 * np.pi / (pol_t + 2 * overlap), dtype=np.float64)
    a[0] *= np.mod(time + pol_t, pol_t)
    a[1] *= np.mod(time + pol_t / 2, pol_t)

    if overlap < 0.01:
        b = 5
        c = 5.81
    elif np.abs(overlap - 2) < 0.01:
        b = 8
        c = 8.39
    else:
        a[0] = 2 * np.pi * peak / (ysource + 2 * overlap)
        b = (a_max * np.cos(a_max) + np.sin(a_max)) / \
                (2 * np.sin(a_max) * a_max**2) * np.pi**2
        c = a_max * np.exp(-(a_max / np.pi)**2 * b) / np.pi / \
                np.clip(np.sin(a_max), a_min=0, a_max=None)

    # clipped sin
    sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

    source = np.array([1, -1]) * sin_clip * csource * a / np.pi * \
            np.exp(-(a / np.pi)**2 * b) * c
    latsource = (max_lat - (max_lat - min_lat) * a / np.pi) * sin_clip

    if mode == 2:
        source *= yearssn(time, yssn, ssn)

    #return source, latsource



    # matchsolarrecords
    minima = np.array([
        1635.1, 1646.0, 1657, 1668, 1679, 1690, 1700, 1713.5, 1724.0, 1733.5,
        1745.0, 1756.0, 1767.0, 1775.5, 1784.0, 1798.5, 1811.0, 1825.0, 1833.5,
        1844.0, 1856.5, 1867.0, 1879.0, 1890.0, 1901.4, 1913.4, 1923.8, 1934.4,
        1944.3, 1954.3, 1964.5, 1976.6, 1986.5, 1996.7, 2006.9, 2018.2, 2029.2,
        2040.1]) - 1646.001

    polarity = np.mod(np.arange(len(minima), dtype=np.float64), 2) * 2 - 1

    pol_t = 21.9
    overlap = 3.0

    # determine index of first cycle
    yi = np.nonzero(time < minima)[0][0] - 2
    yd = 2 * np.array([minima[yi+1] - minima[yi], \
                       minima[yi+2] - minima[yi+1])
    min_lat = 0.0
    a = np.pi * np.array([(time - minima[yi]) / (yd[0] / 2 + overlap), \
                          (time - minima[yi+1]) / (yd[1] / 2 + overlap)])
    sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

    source = csource * sin_clip * a / np.pi * polarity[[yi,yi+1]] \
            np.exp(-(a / np.pi) ** 2 * 8) * 8.39
    latsource = (max_lat - (max_lat - min_lat) * a / np.pi) * sin_clip

    if polarity[yi] < 0:
        source = np.roll(source, 1)
        latsource = np.roll(latsource, 1)
    source *= yearssn(time, yssn, ssn)

    return source, latsource
