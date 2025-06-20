import numpy as np

from .newbipolefluxes import new_bipole_fluxes

from ..constants import binflux

from ..cycle import cyl_mult

joy = np.pi * 4.2 / 180.0
joy_width = np.pi * 90.0 / 18.0
joy_fold = 80.0
sjzero = 18.0

max_lat = np.pi * 25.0 / 180.0
lat_width = np.pi * 25.0 / 180.0
lat_fold = 500.0

turbulent = 0.7
psource = 1.9

avefluxd = 180.0
miniflux = 6
maxflux = 15000

def add_sources(phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int,
                dt: float,
                rng,
                source,
                latsource,
                specified=None,
                as_specified: bool=False,
                initialize: bool=False,
                assimilation: bool=False,
                gradual: bool=False):
    
    # TODO place outside of add_sources

    if initialize:
        newflux = float(maxflux)
        ntotal = 1
        newphi = 0.0
        newtheta = np.pi / 2 - latsource[0] * np.pi / 180
        orient = joy * np.pi / 180
        # goto specified sources

    # TODO what can we vectorize / pull out of loop?
    for i in range(len(source)):
        if specified is not None:
            newflux = specified[:,0]
            newphi = specified[:,1]
            newtheta = specified[:,2]
            ntotal = specified.shape[0]
            orient = np.full(ntotal, joy * np.pi / 180, dtype=np.float64)
            hemi_south = newtheta > np.pi / 2
            orient[hemi_south] = -orient[hemi_south] + np.pi
            orient += (np.max(source) < 0) * np.pi
            #print
            # goto specified sources
        
        if np.fabs(source[i] < 1e-5):
            continue

        a = 8.0
        a *= np.fabs(source[i])
        p = psource

        minpp = miniflux

        minflux = minpp / binflux
        scale = 1. / ((1.0 - p) * 1.5 ** (1.0 - p) * avefluxd ** (1.0 - p))
        rangefactor = maxflux ** (1.0 - p) - (minpp * 2) ** (1.0 - p)
        ntotal = 2 * a * dt / 86400 * scale * rangefactor

        frac = ntotal - int(ntotal)
        ntotal = int(ntotal) + (rng.uniform() < frac)

        newflux = new_bipole_fluxes(ntotal, p, binflux, minflux, maxflux, rng)

        a = 8.0
        a *= np.fabs(source[i])**(1.0/3) * turbulent + (1 - turbulent)
        p = psource + 1
        scale = 1. / ((1.0 - p) * 1.5 ** (1.0 - p) * avefluxd ** (1.0 - p))
        rangefactor = maxflux ** (1.0 - p) - (minpp * 2) ** (1.0 - p)
        ntotal = 2 * a * dt / 86400 * scale * rangefactor

        frac = ntotal - int(ntotal)
        ntotal = int(ntotal) + (rng.uniform() < frac)

        newflux2 = new_bipole_fluxes(ntotal, p, binflux, minflux, maxflux, rng)

        if len(newflux) == 0 and len(newflux2) == 0:
            continue

        newflux = np.r_[newflux, newflux2]
    
        ntotal = len(newflux)
        #print

        if not as_specified and cyl_mult > 1e-5:
            index = newflux > (3 * avefluxd / binflux)
            if not np.any(index):
                return
            newflux = newflux[index]
            ntotal = len(newflux)

        if cyl_mult < -1e-5:
            index = newflux < (3 * avefluxd / binflux)
            if not np.any(index):
                return
            newflux = newflux[index]
            ntotal = len(newflux)

        # step 2 -- determine positions
        newphi = rng.uniform(high=2*np.pi, size=ntotal)
        newtheta = latsource[i] * np.pi / 180 * rng.choice([-1, 1], size=ntotal)
        width = latwidth * (np.exp(-newflux * binflux / latfold) + 0.15)
        newtheta += rng.normal(scale=width*np.pi/180, size=ntotal)
        newtheta = np.pi - newtheta

        # nesting
        nesting = 1.5 * avefluxd * 1.47562 / 2 / binflux
        ind_nest = newflux >= nesting
        if np.any(ind_nest):
            ind_pick = rng.uniform(size=len(ind_nest)) < 0.4
            if not np.any(ind_pick):
                # goto nonests
                pass
            ind_pick = ind_nest[ind_pick]

            xx = np.zeros(180, dtype=np.byte)
            xx[21:180-21] = 1
            yy = np.ones(360, dtype=np.byte)
            ind = (synoptic * np.outer(xx, yy)) != 0
            if not np.any(ind):
                # goto nonests
                pass

            nreplace = min(len(ind_pick), len(ind)) - 1
            point = np.choice(ind, replace=False, size=nreplace)
            lat = point / 360
            newphi[ind_pick[:nreplace+1]] = point - 360 * lat
            newtheta[ind_pick[:nreplace+1]] = np.pi / 2 - np.arcsin(lat / 90 - 1)
            
        # assimilating
        no_data_assimilated = False
        if assimilation and not no_data_assimilated:

            if l0 is None:
                l0 = 0.0

            if b0 is None:
                b0 = 0.0

            # TODO phithetaxyz
            phithetaxyz(newphi+l0, newtheta, xe, ye, ze)
            # TODO tiltmatrix
            # TODO this two-hash mtx mult
            pos = tiltmatrix(b0) ## [[xe], [ye], [ze]]
            edge = np.sin(radassim * np.pi / 180)
            ind = ((pos[:,1]**2 + pos[:,2]**2) < edge) and pos[:,0] > 0
            if np.any(ind):
                newflux[ind] = 0

            ind = newflux != 0
            newflux = newflux[ind]
            newphi = newphi[ind]
            newtheta = newtheta[ind]
            ntotal = len(newflux)
        
        # step 3 -- orientation of bipole axes
        width = joywidth * np.exp(-binflux * newflux / joyfold) + sjzero
        orient = rng.normal(loc=joy, scale=width, size=ntotal) * np.pi / 180
        itheta = newtheta > np.pi / 2
        if np.any(itheta):
            orient[itheta] = np.pi - orient[itheta]

        orient += (source[i] < 0) * np.pi


        # SPECIFIED SOURCES

        # step 4 -- position concentrations
        r = (np.sqrt(binflux * newflux * 1e18 / avefluxd / np.pi) + 7e8) / 7e10
        sep = np.clip(r, a_min=9000/7e5/2, a_max=None)
        percon = np.asarray(np.clip(newflux / 3., a_min=1, a_max=None), 
                            dtype=np.int64)
        percon[newflux > 3 * 15 / binflux] = 15 / binflux

        bulk = np.clip(newflux // percon, a_min=1, a_max=None)
        rest = np.clip(neflux - percon * bulk, a_min=0, a_max=None)
        
        nadd = np.ones(ntotal)
        nadd[newflux >= bulk * percon] = bulk + (rest > 0)

        r_nadd = np.repeat(r, nadd)
        nadd_tot = np.sum(nadd)

        offset1 = rng.uniform(high=r_nadd)
        angle1 = rng.uniform(high=2*np.pi, size=nadd_tot)

        offset2 = rng.uniform(high=r_nadd)
        angle2 = rng.uniform(high=2*np.pi, size=nadd_tot)

        x_tmp = np.r_[sep + offset1 * np.cos(angle1),
                      -sep + offset2 * np.cos(angle2)]
        y_tmp = np.r_[offset1 * np.sin(angle1),
                      offset2 * np.sin(angle2)]

        orient_nadd = np.repeat(orient, nadd)

        orient_tmp = orient_nadd + np.pi / 2
        coso = np.cos(orient_tmp)
        sino = np.sin(orient_tmp)
        xo = coso * x_tmp + sino * y_tmp
        yo = -sino * x_tmp + coso * y_tmp

        newphi_nadd = np.repeat(newphi, nadd)
        newtheta_nadd = np.repeat(newtheta, nadd)

        cosphi = np.cos(newphi_nadd)
        sinphi = np.sin(newphi_nadd)
        costheta = np.cos(newtheta_nadd)
        sintheta = np.sin(newtheta_nadd)

        x = cosphi * sintheta + xo * cosphi * costheta - yo * sinphi
        y = sinphi * sintheta + xo * sinphi * costheta + yo * cosphi
        z = costheta - xo * sintheta

        aphi = np.fmod(np.arctan2(y, x) + 2 * np.pi, 2 * np.pi)
        atheta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))

        scale_nadd = np.repeat(np.sqrt(percon), nadd)
        noise = rng.normal(scale=scale_nadd)

        # TODO uhh confirm this funky index magic i made up
        aflux = np.r_[percon + noise, -percon - noise]
        if rest != 0:
            aflux[nadd] = rest
            aflux[nadd + 1 + nadd_tot] = -rest
        
        # TODO vectorize this loop for gradual
        # gradual introduction of active regions
        if gradual:
            for j in range(ntotal):

                dur = np.asarray(np.sum(np.fabs(aflux)) * binflux / 0.05 / dt)+1
                
                if dur > 1 or np.any((latime == 0) & (laflux != 0)):

                    an = len(aflux) / 2
                    ai = np.asarray(rng.uniform(high=dur, size=an))
                    laphi = np.r_[laphi, aphi]
                    latheta = np.r_[latheta, atheta]
                    laflux = np.r_[laflux, aflux]
                    latime = np.r_[latime, ai, ai]

                    ia = (latime == 0) & (laflux != 0)
                    if np.any(ia):
                        aphi = laphi[ia]
                        atheta = latheta[ia]
                        aflux = laflux[ia]
                        nadd = len(ia) / 2
                    else:
                        aphi = 0.0
                        atheta = 0.0
                        aflux = 0
                        nadd = 0

                    ia = (latime != 0) & (laflux != 0)
                    if np.any(ia):
                        laphi = laphi[ia]
                        latheta = latheta[ia]
                        laflux = laflux[ia]
                        latime = latime[ia]
                    else:
                        laphi = 0.0
                        latheta = 0.0
                        laflux = 0
                        latime = 0

        phi[nflux:nflux+2*nadd_tot] = aphi
        theta[nflux:nflux+2*nadd_tot] = atheta
        flux[nflux:nflux+2*nadd_tot] = aflux

        sourceinput += 2.0 * np.sum(np.fabs(newflux))
        nflux += nadd_tot * 2

    # return sourceinput?
    return nflux
