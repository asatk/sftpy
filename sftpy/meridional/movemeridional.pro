;+
;  NAME: movemeridional.pro
;    
;  PURPOSE: 
;    apply meridional flow displacement to a set of flux
;    concentrations computed for a time step dt
;    
;  CALLING SEQUENCE:
;    movemeridional,theta,nflux,dt,amplifier,profile=mode
;
;  INPUTS:
;    theta  array of co-latitudes (in radians)
;    nflux  number of elements to be processed (0:nflux-1)
;    dt     time interval (seconds)
;    amplifier  a multiplicative factor to be applied to for the flow profile
;
;  OUTPUTS:
;    
;  OPTIONAL INPUT KEYWORDS:
;    profile  1  profile from Komm et al. (1993)
;             2  default: modified after v Ballegooijen et al. (1998)
;                         with tapered flow near poles
;             3  with modified flow near equator; amplifier=1 has
;                reversed flow near equator
;             4... custom
;
;  OPTIONAL OUTPUT KEYWORDS:
;
;  METHOD:
;
;  MODIFICATION HISTORY:
;  2007/12/07: mode=3 with an equatorward cell south of the Carrington
;              latitude, modifiable with an amplifier.
;
;-
;
pro movemeridional,theta,nflux,dt,amplifier,profile=profile
; 
if keyword_set(profile) then profile=profile else profile=2
;
scale=dt/7.e5 ; from km/s to rad/timestep
;
th=theta(0:nflux-1l)
pi=3.1415926536d0
;
; profile from Komm et al. 93
if profile eq 1 then begin
; if (essentially) zero, apply no meridional displacements
  if abs(amplifier) lt 1.e-5 then return
;
  a=12.9d-3*scale*amplifier  ; rescale from km/s!
  b=1.4d-3*scale*amplifier
  theta(0)=th-a*sin(2*th)+b*sin(4*th)
  return
endif
;
; modified after v Ballegooijen et al. (1998) ApJ 501, 866 below 40deg, but
; with tapered flow near poles, as in Schrijver & Title (2001)
;
if profile eq 2 then begin
; if (essentially) zero, apply no meridional displacements
  if abs(amplifier) lt 1.e-5 then return
;
  a=12.7d-3*scale*amplifier
; N.B.
;   Schrijver (2000) used: Komm's curve
;   Schrijver and Title (2000) used: curve by van Ballegooijen, with
;   tapering, which required slight increase on constant of proportionality
;   from 11 in v.B.etal. to 12.7 to match profile up to ~40 degrees.
  lat=(pi/2-th)
  theta(0)=th-(1.-exp( -( (3.*(     th)^3)>(-40)<40) ) )*$
              (1.-exp( -( (3.*(!dpi-th)^3)>(-40)<40) ) )*$
              a*sin(2*lat)
  return
endif
; Add your profile of choice (does not need to be symmetric)
if profile eq 3 then begin
  a=12.7d-3*scale
; N.B.
;   Schrijver (2000) used: Komm's curve
;   Schrijver and Title (2000) used: curve by van Ballegooijen, with
;   tapering, which required slight increase on constant of proportionality
;   from 11 in v.B.etal. to 12.7 to match profile up to ~40 degrees.
  lat=(pi/2-th)
  theta(0)=th-(1.-exp( -( (3.*(     th)^3)>(-40)<40) ) )*$
              (1.-exp( -( (3.*(!dpi-th)^3)>(-40)<40) ) )*$
              a*sin(2*lat)*(1.+.06*amplifier)+$
              sin(lat*.9)*exp(-(lat*3.)^2)*a*4*amplifier
  return
endif

;  profile=4: same as profile 2 but with time-varying amplitude (implemented
;  in kit.pro immediately before calling this routine)
if profile eq 4 then begin
  if abs(amplifier) lt 1.e-5 then return
  a=12.7d-3*scale*amplifier
  lat=(pi/2-th)
  theta(0)=th-(1.-exp( -( (3.*(     th)^3)>(-40)<40) ) )*$
              (1.-exp( -( (3.*(!dpi-th)^3)>(-40)<40) ) )*$
              a*sin(2*lat)
  print,'*******',amplifier 
 return
endif
;
stop,'Stop: selected meridional-flow option not available'
;
end
