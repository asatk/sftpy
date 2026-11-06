'''
class ARInflow(Component):

    def __init__(self,
                 prop: float=2.0,
                 power: float=0.5,
                 offx: float=0.0,
                 offy: float=0.0,
                 loglvl: int=0):
        super().__init__(loglvl=loglvl)
        self._prop = prop
        self._power = power
        self._offx = offx
        self._offy = offy

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):
        # synoptic map
        sss = [synoptic] * 3
        sss = [[reverse(sss, 2)], [sss], [reverse(sss, 2)]]
        ns = 4
        nt = 4 * ns
        nu = 2 * nt
        sss = smear(np.abs(sss), ns, edge=True)
        sss = np.clip(sss, a_min=0.0, a_max=None) ** self._power
        synopt = sss[120-nt:239+nt,60-nt:119+nt]

        # use avg of left and right gradient of a givel pixel
        gradx = ((congrid( (shift(synopt, -1, 0) - shift(synopt, 1, 0)), 360+3*nu, 180+3*nu,
                        cubic=True, center=True))(3*nt:359+3*nt,3*nt:179+3*nt)) / 2.
        grady = ((congrid(-(shift(synopt, 0, -1) - shift(synopt, 0, 1)), 360+3*nu, 180+3*nu,
                        cubic=True, center=True))(3*nt:359+3*nt,3*nt:179+3*nt)) / 2.

        gradx[:,0] = 0.0
        gradx[:,179] = 0.0
        grady[:,0] = 0.0
        grady[:,179] = 0.0

        # Normalization: 1 m / s = 1 * 86400. * 360 / (2 *!pi * 7.e8) = 0.0071
        # degrees / day, or expressed in radians / day:
        rdnorm = 86400 * 360 / 2 / np.pi / 7e8 * np.pi / 180

        # field gradient comuted in units of 10^18 Mx / sq deg / deg spacing
        # convert to G / 10^4 km
        gkm = 1.e18 / (2 * np.pi * 7e10 / 360.) ** 3 * 1.e9

        gradx *= gkm * rdnorm * self._prop
        grady *= gkm * rdnorm * self._prop

        xin = ((phi[:nflux] + 2 * np.pi) % (2 * np.pi))
        xin *= 360 / 2 / np.pi
        xin = (xin + self._offx - 0.5) % 360

        yin = 90 + np.cos(theta[:nflux]) * 90
        yin = np.clip(yin + self._offy - 0.5, a_min=0.0, a_max=180.0)

        phi[:nflux] += bilinear(gradx,
                                reform(xin, 1, nflux),
                                reform(yin, 1, nflux) * np.sin(theta[:nflux]))
        phi[:nflux] = (phi[:nflux] + 2 * np.pi) % (2 * np.pi)

        theta[:nflux] += bilinear(grady,
                                  reform(xin, 1, nflux),
                                  reform(yin, 1, nflux) / np.clip(np.sin(theta[:nflux]), a_min=0.01, a_max=None))
        theta[:nflux] = np.clip(theta[:nflux], a_min=0.0, a_max=np.pi)

        # what is testphi

'''