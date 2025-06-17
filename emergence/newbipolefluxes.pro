;+
;  NAME:
;    
;  PURPOSE:
;    select random numbers within a specified range from a power-law
;    parent distribution function (compare Numerical Recipes), called
;    by 
;    
;  CALLING SEQUENCE:
;
;  INPUTS:
;    ntotal        number of regions to be selected (return if 0)
;    p             slope of power-law distribution
;    binflux       units for flux
;    minflux       minimum acceptable flux
;    maxflux       maximum acceptable flux
;    iseed         seed for random number generator
;
;  OUTPUTS:
;    newfluxflag   logical: are there any regions to be selected?
;    newflux       flux values (in units of binflux)
;    
;  OPTIONAL INPUT KEYWORDS:
;    
;  OPTIONAL OUTPUT KEYWORDS:
;
;  METHOD:
;
;  MODIFICATION HISTORY:
;
;-
; 
pro newbipolefluxes,newfluxflag,newflux,ntotal,p,binflux,minflux,maxflux,iseed
;
newfluxflag=ntotal ge 0.5d0
if newfluxflag eq 0 then return
;
newflux=lonarr(ntotal)
inew=0l
; the devision by 2 is because Karen Harvey counts the sum of both polarities
ep=1./(1.-p) & bf2=binflux*2l
mbinflux=maxflux/binflux
while (inew lt ntotal) do begin
  new=long(p*randomu(iseed)^ep+0.5)/bf2
  if (new ge minflux and new lt mbinflux) then begin
    newflux(inew)=new
    inew=inew+1l
  endif
endwhile
return
end
