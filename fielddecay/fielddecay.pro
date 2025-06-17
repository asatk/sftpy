;+
;  NAME: fielddecay.pro
;    
;  PURPOSE: 
;   let flux on the stellar surface 'decay' with time by randomly
;   removing concentrations
;    
;  CALLING SEQUENCE: fielddecay,phi,theta,flux,nflux,timescale,dt,iseed
;
;  INPUTS:
;     phi,theta,flux: nflux positions (radians) and fluxes (binflux * 10^18Mx)
;     timescale: decay time scale (yrs) [no effect if >999]
;     dt: time step of the simulation (s)
;     iseed: random number generator seed value
;
;  OUTPUTS:
;     flux: array with reduced fluxes, conserved total flux.
;    
;  OPTIONAL INPUT KEYWORDS:
;    
;  OPTIONAL OUTPUT KEYWORDS:
;
;  METHOD:
;    The procedure lets the field decay by removing units of flux from
;    randomly selected concentrations. Beware that the decay time should
;    be long enough that the flux to be removed can be taken from an adequate
;    number of concentrations!
; 
;  MODIFICATION HISTORY:
;
;-
pro fielddecay,phi,theta,flux,nflux,timescale,dt,iseed
; if timescale 999 yearas or above, then ignore:
if timescale gt 999. then return
; how much `flux' is to be removed for each polarity
; (N.B. dt in seconds, timescale in years)
remove=((1-exp(-alog(2.)*dt/(365.25*86400.*timescale)))*total(abs(flux))/2)
; remove the appropriate fraction statistically speaking
rest=remove-fix(remove)
if randomu(iseed) lt rest then remove=long(remove)+1 else remove=long(remove)
if remove eq 0 then return 
positive=where(flux(0:nflux-1) gt 0)
negative=where(flux(0:nflux-1) lt 0)
;
if positive(0) lt 0 or negative(0) lt 0 then return
;
j=(sort(randomu(iseed,n_elements(positive))))(0:remove-1)
k=(sort(randomu(iseed,n_elements(negative))))(0:remove-1)
flux(positive(j))=flux(positive(j))-1
flux(negative(k))=flux(negative(k))+1
; remove ``empty'' concentrations, if necessary
ii=where(flux(0:nflux-1) ne 0)
if n_elements(ii) eq nflux then return
phi(0)=phi(ii)
theta(0)=theta(ii)
flux(0)=flux(ii)
nflux=n_elements(ii)
return
end
