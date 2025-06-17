avefluxd = 180.0
def add_sources():
    
    if mode == 0:
        source = np.array([csource])
        source = -source if inv_pol else source
    else:
        source = cycle_strength()
        source = -source if inv_pol else source
        # print

    if initialize:
        newflux = float(maxflux)
        ntotal = 1
        newphi = 0.0
        newtheta = np.pi / 2 - latsource[0] * np.pi / 180
        orient = joy * np.pi / 180
        # goto specified sources

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
            # goto nextcycle
            continue

        a = 8.0
        a *= np.fabs(source[i])

        if len(miniflux) == 0:
            minpp = 6
        else:
            minpp = miniflux

        minflux = minpp / binflux
        scale = 1. / ((1.0 - p) * 1.5 ** (1.0 - p) * avefluxd ** (1.0 - p))
        rangefactor = maxflux ** (1.0 - p) - (minpp * 2) ** (1.0 - p)
        ntotal = 2 * a * dt / 86400 * scale * rangefactor

        frac = ntotal - int(ntotal)
        ntotal = int(ntotal) + (rng.uniform() < frac)

        newflux = newbipolefluxes(newfluxflag, ntotal, p, binflux, minflux, maxflux)

        a = 8.0
        a *= np.fabs(source[i])**(1.0/3) * turbulent + (1 - turbulent)
        p = psource + 1
        scale = 1. / ((1.0 - p) * 1.5 ** (1.0 - p) * avefluxd ** (1.0 - p))
        rangefactor = maxflux ** (1.0 - p) - (minpp * 2) ** (1.0 - p)
        ntotal = 2 * a * dt / 86400 * scale * rangefactor

        frac = ntotal - int(ntotal)
        ntotal = int(ntotal) + (rng.uniform() < frac)

        newflux2 = newbipolefluxes(newflux2flag, ntotal, p, binflux, minflux, maxflux)

        if not newfluxflag and not newflux2flag:
            # goto nextcycle
            continue

        if newfluxflag and newflux2flag:
            newflux = [newflux, newflux2]
    
        if not newfluxflag and newflux2flag:
            newflux = newflux2

        ntotal = len(newflux)
        #print

        if not as_specified and csource > 1e-5:
            index = newflux > (3 * avefluxd / binflux)
            if not np.any(index):
                return
            newflux = newflux[index]
            ntotal = len(newflux)

        if csource < -1e-5:
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
        if assimilation:
            if 'no data assimilated':
                # goto skipremoval
                pass

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
        
        # SKIP REMOVAL

        # step 3 -- orientation of bipole axes
        sjzero = 18.0
        width = joywidth * np.exp(-binflux * newflux / joyfold) + sjzero
        orient = rng.normal(loc=joy, scale=width, size=ntotal) * np.pi / 180
        itheta = newtheta > np.pi / 2
        if np.any(itheta):
            orient[itheta] = np.pi - orient[itheta]

        orient += (source[i] < 0) * np.pi


        # SPECIFIED SOURCES

        # step 4 -- position concentrations
        sourceinput = 0.0
        for j in range(ntotal):
            newfluxi = newflux[j]
            r = (np.sqrt(binflux * newfluxi * 1e18 / axefluxd / np.pi) + 7e8) / 7e10
            separation = max(r, 9000 / 7e5 / 2)

            if newfluxi > 3 * 15 / binflux:
                percon = 15 / binflux
            else:
                percon = int(max(newfluxi/3., 1))

            bulk = max(int(newfluxi/percon))
            rest = max(newfluxi - percon*bulk, 0)

            if newfluxi >= bulk*percon:
                nadd = bulk + (rest > 0)
            else:
                nadd = 1

            offset1 = rng.uniform(high=r, size=nadd)
            angle1 = rng.uniform(high=2*np.pi, size=nadd)

            offset2 = rng.uniform(high=r, size=nadd)
            angle2 = rng.uniform(high=2*np.pi, size=nadd)

            x = np.array([separation + offset1 * np.cos(angle1),
                          -separation + offset2 * np.cos(angle2)])
            y = np.array([offset1 * np.sin(angle1),
                          offset2 * np.sin(angle2)])

            orienti = orient[j] + np.pi / 2
            coso = np.cos(orienti)
            sino = np.sin(orienti)
            xo = coso * x + sino * y
            yo = -sino * x + coso * y

            newphii = newphi[j]
            newthetai = newtheta[j]
            cosphi = np.cos(newphii)
            sinphi = np.sin(newphii)
            costheta = np.cos(newthetai)
            sintheta = np.sin(newthetai)

            x = cosphi * sintheta + xo * cosphi * costheta - yo * sinphi
            y = sinphi * sintheta + xo * sinphi * costheta + yo * cosphi
            z = costheta - xo * sintheta

            aphi = np.fmod(np.arctan2(y, x) + 2 * np.pi, 2 * np.pi)
            atheta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))

            noise = np.sqrt(percon) * rng.normal(size=nadd)

            if rest != 0:
                aflux = np.asarray([percon + noise[:nadd-1],
                                    rest,
                                    -percon - noise[:nadd-1],
                                    -rest],
                                   dtype=np.int64)
            else:
                aflux = np.asarray([percon + noise, -percon - noise],
                                   dtype=np.int64)

            # gradual introduction of active regions
            if gradual:
                dur = np.asarray(np.sum(np.fabs(aflux)) * binflux / 0.05 / dt)+1
                
                if dur > 1 or np.any((latime == 0) & (laflux != 0)):
                    # else: goto skipgradual

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

            # skip gradual

            phi[nflux:nflux+2*nadd] = aphi
            theta[nflux:nflux+2*nadd] = atheta
            flux[nflux:nflux+2*nadd] = aflux

            sourceinput += 2.0 * np.fabs(newfluxi)
            nflux += nadd * 2

        # nextcycle
    return





