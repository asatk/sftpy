import numpy as np

from .newbipolefluxes import new_bipole_fluxes

from ..constants import binflux

from ..cycle import cyl_mult

# defaults from kit_iocontrol.pro

joy = 7.0           # deg
joy_width = 90.0    # deg
joy_fold = 80.0     # Mx
sjzero = 18.0       # deg

max_lat = 25.0      # deg
lat_width = 45.0    # deg
lat_fold = 500.0    # Mx

turbulent = 0.7     # fraction of regions not w/ turbulent dynamo
psource = 1.9       # power law index

avefluxd = 180.0    # G
miniflux = 6        # Mx
maxflux = 15000     # Mx

def add_sources(phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int,
                dt: float,
                rng,
                source,
                latsource,
                synoptic: np.ndarray,
                specified=None,
                as_specified: bool=True,
                initialize: bool=False,
                assimilation: bool=False,
                gradual: bool=False):
    
    # TODO figure out use of sourceinput: "total absolute flux added"
    sourceinput = 0.0

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
        ntotal2 = 2 * a * dt / 86400 * scale * rangefactor

        frac = ntotal2 - int(ntotal2)
        ntotal2 = int(ntotal2) + (rng.uniform() < frac)

        newflux2 = new_bipole_fluxes(ntotal2, p, binflux, minflux, maxflux, rng)

        if len(newflux) == 0 and len(newflux2) == 0:
            continue

        newflux = np.r_[newflux, newflux2]
    
        print(f"[addsrc] ---- ntotal = {ntotal}   ntotal2 = {ntotal2}")
        ntotal = len(newflux)
        print(f"[addsrc] ---- ntotal (all) = {ntotal}")

        # TODO -- this mode emerges nothing...
        # accelerated-time mode; leave out ephemeral regions
        if not as_specified and cyl_mult > 1e-5:
            print(f"[addsrc] ---- NEWFLUX {newflux}")
            index = newflux > (3 * avefluxd / binflux)
            if not np.any(index):
                print("[addsrc] ---- NO SOURCES PASS THRESHOLD")
                return nflux
            newflux = newflux[index]
            ntotal = len(newflux)

        if cyl_mult < -1e-5:
            index = newflux < (3 * avefluxd / binflux)
            if not np.any(index):
                return nflux
            newflux = newflux[index]
            ntotal = len(newflux)

        # step 2 -- determine positions
        newphi = rng.uniform(high=2*np.pi, size=ntotal)
        newtheta = latsource[i] * np.pi / 180 * rng.choice([-1, 1], size=ntotal)
        width = lat_width * (np.exp(-newflux * binflux / lat_fold) + 0.15)
        newtheta += rng.normal(scale=width*np.pi/180, size=ntotal)
        newtheta = np.pi/2 - newtheta

        # nesting
        nesting = 1.5 * avefluxd * 1.47562 / 2 / binflux
        ind_nest = np.nonzero(newflux >= nesting)[0]
        nnest = len(ind_nest)
        if nnest > 0:
            ind_pick = rng.uniform(size=nnest) < 0.4
            npick = np.sum(ind_pick)

            if npick > 0:
                print(f"[addsrc] ---- nnest={nnest} npick={npick}")
                print(f"[addsrc] ---- ind_nest {ind_nest}")
                print(f"[addsrc] ---- ind_pick {ind_pick}")
                ind_nest_picks = ind_nest[ind_pick]

                xx = np.zeros(180, dtype=np.byte)
                xx[21:180-21] = 1
                yy = np.ones(360, dtype=np.byte)
                # TODO check this matmult
                ind = (synoptic * np.outer(yy, xx)) != 0
                nind = np.sum(ind)

                if nind > 0:
                    nreplace = min(npick, nind) - 1
                    print(f"[addrc] ---- NEST nreplace = {nreplace}")
                    point = rng.choice(ind, replace=False, size=nreplace)
                    lat = point / 360
                    # TODO check the  +1 on these
                    newphi[ind_nest_picks[:nreplace+1]] = point - 360 * lat
                    newtheta[ind_nest_picks[:nreplace+1]] = np.pi / 2 - np.arcsin(lat / 90 - 1)
                else:
                    # goto nonests
                    pass
            else:
                # goto nonests
                pass
            
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
        width = joy_width * np.exp(-binflux * newflux / joy_fold) + sjzero
        orient = rng.normal(loc=joy, scale=width, size=ntotal) * np.pi / 180
        itheta = newtheta > np.pi / 2
        if np.any(itheta):
            orient[itheta] = np.pi - orient[itheta]

        orient += (source[i] < 0) * np.pi


        # SPECIFIED SOURCES

        # step 4 -- position concentrations
        r = (np.sqrt(binflux * newflux * 1.e18 / avefluxd / np.pi) + 7.e8) / 7.e10
        sep = np.clip(r, a_min=9000./7.e5/2, a_max=None)
        percon = np.astype(np.clip(newflux / 3., a_min=1, a_max=None), 
                           np.int64)
        percon[newflux > 3 * 15. / binflux] = 15. / binflux

        bulk = np.clip(newflux // percon, a_min=1, a_max=None)
        rest = np.clip(newflux - percon * bulk, a_min=0, a_max=None)
        
        nadd = np.ones(ntotal, dtype=np.int64)
        nadd[newflux >= bulk * percon] = bulk + (rest > 0)

        r_nadd = np.repeat(r, nadd)
        sep_nadd = np.repeat(sep, nadd)
        percon_nadd = np.repeat(percon, nadd)
        nadd_tot = np.sum(nadd)

        # one polarity
        offset1 = rng.uniform(high=r_nadd)
        angle1 = rng.uniform(high=2*np.pi, size=nadd_tot)

        # opposite polarity
        offset2 = rng.uniform(high=r_nadd)
        angle2 = rng.uniform(high=2*np.pi, size=nadd_tot)

        x_tmp = np.r_[sep_nadd + offset1 * np.cos(angle1),
                      -sep_nadd + offset2 * np.cos(angle2)]
        y_tmp = np.r_[offset1 * np.sin(angle1),
                      offset2 * np.sin(angle2)]

        # orientation of bipolar spot
        #TODO better way to double these?
        orient_nadd_half = np.repeat(orient, nadd)
        orient_nadd = np.r_[orient_nadd_half, orient_nadd_half]

        orient_tmp = orient_nadd + np.pi / 2
        coso = np.cos(orient_tmp)
        sino = np.sin(orient_tmp)
        xo = coso * x_tmp + sino * y_tmp
        yo = -sino * x_tmp + coso * y_tmp

        # location of bipolar active region / concentration
        newphi_nadd_half = np.repeat(newphi, nadd)
        newphi_nadd = np.r_[newphi_nadd_half, newphi_nadd_half]
        newtheta_nadd_half = np.repeat(newtheta, nadd)
        newtheta_nadd = np.r_[newtheta_nadd_half, newtheta_nadd_half]

        cosphi = np.cos(newphi_nadd)
        sinphi = np.sin(newphi_nadd)
        costheta = np.cos(newtheta_nadd)
        sintheta = np.sin(newtheta_nadd)

        x = cosphi * sintheta + xo * cosphi * costheta - yo * sinphi
        y = sinphi * sintheta + xo * sinphi * costheta + yo * cosphi
        z = costheta - xo * sintheta

        aphi = np.fmod(np.arctan2(y, x) + 2 * np.pi, 2 * np.pi)
        atheta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))

        print(f"[addsrc] ---- aphi {aphi}")
        print(f"[addsrc] ---- atheta {atheta}")
        print(f"[addsrc] ---- mean atheta {np.mean(atheta)} std atheta "
              f"{np.std(atheta)}")

        scale_nadd = np.sqrt(percon_nadd)
        noise = rng.normal(scale=scale_nadd)

        # TODO uhh confirm this funky index magic i made up
        aflux = np.r_[percon_nadd + noise, -percon_nadd - noise]
        rest_nz = np.nonzero(rest)
        aflux[nadd[rest_nz]] = rest[rest_nz]
        aflux[(nadd+1+nadd_tot)[rest_nz]] = -rest[rest_nz]

        """
        if rest != 0:
            aflux[nadd] = rest
            aflux[nadd + 1 + nadd_tot] = -rest
        """
        
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

        print(f"[addsrc] ---- add {nadd_tot}")


    # return sourceinput?
    return nflux
