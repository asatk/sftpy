# sad :(

import numpy as np
from ..constants import rng

from numpy import int64 as long

def addsources(years,
               phi: np.ndarray,
               theta: np.ndarray,
               flux: np.ndarray,
               nflux: int,
               nfluxmax,
               cyclemode: int,
               cycleoverlap,
               csource,
               ysource,
               psource,
               turbulent,
               miniflux,
               maxflux,
               dt: float,
               iseed: int,
               radius: float,
               binflux,
               synoptic,
               yssn,
               ssn,
               laphi,
               latheta,
               laflux,
               latime=None,
               radassim=None,
               l0=None,
               b0=None,
               invertpolarity: bool=False,
               updated: bool=False,
               specified: bool=False,
               gradual: bool=False,
               peak=None,
               controlfile: str=None,
               runnumber: int=0):
    dr = np.pi / 180
    avefluxd = 180.

    # TODO is latime a scalar or array?
    if latime is None:
        latime = 0
    if gradual:
        latime = max(latime-1, 0)
    else:
        lattime = latime * 0

    if cyclemode == 0:
        source = np.zeros(1)
        source[0] = csource
        if invertpolarity:
            source *= -1
    else:
        cyclestrength(years, cyclemode, cycleoverlap, ysource, csource, maxlat,
            yssn, ssn, source, latsource, silent=silent, peak=peak,
            controlfile=controlfile, runnumber=runnumber)
        
        cs = ['+', '-']
        if invertpolarity:
            source *= -1
        if not silent:
            for ip in [0, 1]:
                print(strcompress(f"Addsources{cs[ip]}: {source[ip]} / " + \
                    f"{csource} at {years} yrs (cycle {ysource} yrs) " + \
                    f"around {latsource[ip]} degrees"))
    
    # TODO apparently there is a keyword initialize?
    if initialize:
        newflux = maxflux
        ntotal = 1
        newphi = 0.0
        newtheta = np.pi/2 - latsource[0]*dr
        orient = joy * dr
        # TODO goto SPECIFIED SOURCES
    
    for ic in range(len(source)):
        # Specified is a collection of specified sources... this code sucks
        if specified:
            # TODO nint
            newflux = specified[:,0]
            newphi = specified[:,1]
            newtheta = specified[:,2]
            ntotal = specified.shape[1]
            orient = joy * dr * np.zeros(ntotal, dtype=np.float64)
            itheta = np.nonzero(newtheta > np.pi/2)
            if itheta[0] >= 0:
                orient[itheta] = np.pi - orient[itheta]
            # TODO what?
            # ; assume the orientation of the largest cycle for the new regions:
            # orient=orient+(max(source) lt 0)*!dpi
            orient += (max(source < 0)) * np.pi

            print("addsources: ", orient, source, cyclemode, invertpolarity)

            # TODO goto SPECIFIED SOURCES
        
        if abs(source[ic]) < 1.0e-5:
            # TODO goto nextcycle
            # ; if 'nothing' to add, skip remainder of loop
            pass

        a = 8.0 * np.abs(source[ic])
        p = psource

        if miniflux is None or len(miniflux) == 0:
            minpp = 6
        else:
            minpp = miniflux
        
        minflux = minpp / binflux
        mp = 1 - p
        scale = 1 / (mp * 1.5 ** mp * avefluxd ** mp)
        rangefactor = maxflux ** mp - (2 * minpp) ** mp

        ntotal = 2 * (a * dt / 86400 * scale * rangefactor)

        # what
        ntotal = long(ntotal) + (rng.uniform() < (ntotal - long(ntotal)))

        newbipolefluxes(newfluxflag, newflux, ntotal, p, binflux, minflux,
            maxflux, iseed)

        a = 8 * (np.power(np.abs(source[ic]), 1/3)) * turbulent + 1 - turbulent
        p = psource + 1
        mp = 1 - p
        scale = 1 / mp / 1.5 ** mp / avefluxd ** mp
        rangefactor = maxflux ** mp - (minpp * 2) ** mp

        ntotal_fp = 2 * a * dt / 86400 * scale * rangefactor
        # TODO huh?
        ntotal_trunc = np.astype(ntotal_fp, dtype=np.int64)
        ntotal = ntotal_trunc + (rng.uniform(size=1) < (ntotal_fp - ntotal_trunc))

        newbipolefluxes(newflux2flag, newflux2, ntotal, p, binflux, minflux, maxflux, iseed)

        if newfluxflag == 0 and newflux2flag == 0:
            #TODO goto nextcycle
            pass
        elif newfluxflag > 0 and newflux2flag > 0:
            #list/array append?
            newflux = [newflux, newflux2]
        elif newfluxflag == 0 and newfluxflag > 0:
            newflux = newflux2
        ntotal = len(newflux)
            
        if not silent:
            print("Adding (" + strcompress(ic, rem=True) + \
                ") " + strcompress(ntotal, rem=True) + " bipoles")
        
        if as_specified == 0 and csource > 1e-5:
            index = newflux > 3 * avefluxd / binflux
            if index[0] < 0:
                return
            newflux = newflux[index]
            ntotal = len(newflux)
        
        if csource < -1e-5:
            index = newflux < 3 * avefluxd / binflux
            if index[0] < 0:
                return
            newflux = newflux[index]
            ntotal = len(newflux)
        
        newphi = rng.uniform(size=ntotal) * 2 * np.pi
        newtheta = latsource[ic] * dr * ((2 * np.asarray(rng.normal(size=ntotal), np.int64) > 0) - 1)
        width = latwidth * np.exp(-newflux * binflux / latfold + 0.15)
        newtheta += rng.normal(size=ntotal) * width * dr
        newtheta = np.pi / 2 - newtheta

        nesting = 2.5 * avefluxd * 1.47562 / 2 / binflux
        ii = newflux >= nesting
        if ii[0] >= 0:
            pick = rng.uniform(size=len(ii)) < 0.4
            if pick[0] < 0:
                # TODO goto nonests
                pass
            pick = ii[pick]

            xx = np.zeroes(180, dtype=np.byte)
            xx[21:-21] = 1
            index = synoptic * (np.ones(360, dtype=np.byte) @ xx) != 0
            if index[0] < 0:
                #TODO goto nonests
                pass
            nreplace = min(len(pick), len(index))
            point = (index[np.argsort(rng.uniform(size=len(index)))])[:nreplace]
            lat = point / 360
            newphi[pick[:nreplace]] = point - 360 * lat
            newtheta[pick[:nreplace]] = np.pi / 2 - np.arcsin(lat/90 - 1)
        
        # NONESTS
        

        if assimilation is not None:
            if strpos(update, "No data assimilated") >= 0:
                #TODO goto skipremoval
                pass
            if l0 is None:
                l0 = 0
            if b0 is None:
                b0 = 0
            
            phithetaxyx(newphi + l0, newtheta, xe, ye, ze)
            pos = tiltmatrix(b0) @ np.array([[xe],[ye],[ze]])
            edge = np.sin(radassim * np.pi / 180)
            index = ((pos[:,1]**2 + pos[:,2]**2) < edge) & (pos[:,0] > 0)

            if index[0] >= 0:
                newflux[index] = 0
            
            index = newflux != 0
            newflux = newflux[index]
            newphi = newphi[index]
            newtheta = newtheta[index]
            ntotal = len(newflux)
        
        # SKIPREMOVAL
        sjzero = 18
        width = joywidth * np.exp(-binflux * newflux / joyfold) + sjzero
        orient = (rng.normal(size=ntotal) * width + joy) * dr

        itheta = newtheta > np.pi/2
        if itheta[0] >= 0:
            orient[itheta] = np.pi - orient[itheta]
        
        # TODO huh
        orient += np.pi * (source[ic] < 0)


        # SPECIFIED SOURCES
        sourceinput = 0.0
        for i in range(ntotal):
            newfluxi = newflux[i]
            r = (np.sqrt(binflux * newfluxi * 1e18 / avefluxd / np.pi) + 7e8) / 7e10
            separation = max(r, 9000 / 7e5 / 2)
            if newfluxi > 3 * 15 / binflux:
                percon = 15 / binflux
            else:
                percon = max(newfluxi//3,1)
            bulk = max(np.int64(newfluxi / percon), 1)
            rest = max(newfluxi - percon * bulk, 0)
            if newfluxi >= bulk * percon:
                # TODO huh
                nadd = bulk + (rest > 0)
            else:
                nadd = 1
            
            offset1 = rng.uniform(size=nadd) * r
            angle1 = np.uniform(size=nadd) * 2 * np.pi

            offset2 = rng.uniform(size=nadd) * r
            angle2 = np.uniform(size=nadd) * 2 * np.pi

            x = [separation + offset1 * np.cos(angle1), -separation + offset2 * np.cos(angle2)]
            y = [offset1 * np.sin(angle1), offset2 * np.sin(angle2)]

            orienti = orient[i] + np.pi/2
            coso = np.cos(orienti)
            sino = np.sin(orienti)

            newphii = newphi[i]
            newthetai = newtheta[i]
            cosphi = np.cos(newphii)
            sinphi = np.sin(newphii)
            costheta = np.cos(newthetai)
            sintheta = np.sin(newthetai)

            x = cosphi * sintheta + xo * cosphi * costheta - yo * sinphi
            y = sintheta * sinphi + xo * sinphi * costheta + yo * cosphi
            z = costheta - xo * sintheta

            aphi = np.mod(np.arctan(y, x) + 2 * np.pi, 2 * np.pi)
            atheta = np.arccos(z / np.sqrt(x**2 + y**2 + z ** 2))

            noise = np.sqrt(percon) * rng.normal(size=nadd)
            if rest != 0:
                aflux = [percon + noise[0:nadd-1], rest, -percon-noise[:nadd-1],-rest]
            else:
                aflux = [percon + noise, -percon - noise]
            
            aflux = np.asarray(aflux, dtype=np.int64)

            if gradual is not None:
                dur = np.int64(np.sum(np.abs*aflux) * binflux / 0.05 / dt) + 1

                if not (dur > 1 or ((latime == 0) | (laflux != 0))[0] >= 0):
                    # TODO goto skipgradual
                    pass
                an = len(aflux) / 2
                # TODO huh... hmmm ahh logical expressions do not get sep type (T/F), just 0 and 1?
                ai = np.asarray(rng.uniform(size=an) * dur, dtype=np.int64) * (gradual is not None)

                # append
                laphi = [laphi, aphi]
                latheta = [latheta, atheta]
                laflux = [laflux, aflux]
                latime = [latime, ai, ai]

                ia = (latime == 0) & (laflux != 0)

                if ia[0] >= 0:
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
                
                if ia[0] >= 0:
                    laphi = laphi[ia]
                    latheta = latheta[ia]
                    laflux = laflux[ia]
                    latime = latime[ia]
                else:
                    aphi = 0.0
                    atheta = 0.0
                    aflux = 0
                    nadd = 0
            
            # SKIPGRADUAL
            phi[nflux] = aphi
            theta[nflux] = atheta
            flux[nflux] = aflux
            sourceinput += 2 * abs(newfluxi)
            nflux += 2 * nadd
        
        # NEXTCYCLE
    return
