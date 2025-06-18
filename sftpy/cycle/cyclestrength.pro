;+
;  NAME:
;    
;  PURPOSE:
; procedure to compute the cycle's strength and emergence latitude as a
; function of time. Called by addsources.pro
;    
;  CALLING SEQUENCE:
;
;  INPUTS:
;   years	time in years (must be a single number!)
;   cyclemode:	0	fixed strength, independent of time (do not use 0
;			for cycle strength in this case, because that is
;			for a single-region test case
;		1	fixed amplitude cycle
;		2	cycle amplitude modulated according to file spec.
;		3	no regions added at all
;		4	match sunspot observations as closely as possible
;   overlap     If overlap is set to either 0 or 2, and peak is not
;               set, then the sunspot number profiles  are like that
;               of the Sun, peaking around 4.15 years into the
;               cycle. If not, the peak is used as the maximum.
;   ysource     duration of the polarity cycle
;   csource     cycle strength multiplier
;   maxlat      maximum latitude of the butterfly pattern
;   yssn,ssn    sunspot record time and number (see yearssn.pro)
;
;  OUTPUTS:
;   source      strengths of up to two overlapping cycles
;   latsource   mean latitudes of up to two overlapping cycles
;    
;  OPTIONAL INPUT KEYWORDS:
;   silent      if set, suppress diagnostic statements
;   peak        if specified, year at which sunspot cycle peaks
;   controlfile path of control file
;   runnumber   number within control file
;    
;  OPTIONAL OUTPUT KEYWORDS:
;
;  METHOD:
;
;  MODIFICATION HISTORY:
;   N.B. Use plotcyclestrength.pro to visualize the cycle profile(s)
;
;-
;
pro cyclestrength,years,cyclemode,overlap,ysource,csource,maxlat,$
  yssn,ssn,source,latsource,silent=silent,peak=peak,$
  runnumber=runnumber,controlfile=controlfile
;
  if n_elements(years) ne 1 then stop,'Stop: error in call to cyclestrength.'
;
; model allows only a periodic change in magnitude, with drift in latitude
  source=fltarr(2)
  latsource=fltarr(2)
;
 if cyclemode eq 4 then goto, matchsolarrecords
;
; skewed to steep rise, slower decay, treating polarity halfs separately
; compute the positive and negative sources separately, offset by ysource/2.
;
; also introduce a time-dependent latitude, from maxlat to minlat over cycle
  minlat=0.
  a=2*!dpi*((years+ysource) mod ysource)/(ysource+2*overlap)
; note that the source profiles need modification, depending on cycle overlap.
; this has only been implemented for overlaps of 0 and 2 years:
  if overlap lt 0.01 then $
    source(0)=(csource*(sin(a)>0)*(a/!dpi)*exp(-(a/!dpi)^2*5)*5.81) 
  if abs(overlap-2) lt 0.01 then $
    source(0)=(csource*(sin(a)>0)*(a/!dpi)*exp(-(a/!dpi)^2*8)*8.39)
  if not(overlap lt 0.01 or abs(overlap-2) lt 0.01) then begin
    if not(keyword_set(peak)) then peak=4.15
    amax=2*!dpi*peak/(ysource+2*overlap)
    b=(amax*cos(amax)+sin(amax))/(2*sin(amax)*amax^2)*!dpi^2
    c=1./((sin(amax)>0)*(amax/!dpi)*exp(-(amax/!dpi)^2*b))
    source(0)=(csource*(sin(a)>0)*(a/!dpi)*exp(-(a/!dpi)^2*b)*c)
  endif
  latsource(0)=(maxlat-(maxlat-minlat)*a/!dpi)*(sin(a) gt 0)
  a=2*!dpi*((years+ysource/2) mod ysource)/(ysource+2*overlap)
  if overlap lt 0.01 then $
    source(1)=-((csource*(sin(a)>0)*(a/!dpi)*exp(-(a/!dpi)^2*5)*5.81))
  if abs(overlap-2) lt 0.01 then $
    source(1)=-((csource*(sin(a)>0)*(a/!dpi)*exp(-(a/!dpi)^2*8)*8.39))
  if not(overlap lt 0.01 or abs(overlap-2) lt 0.01) then $
    source(1)=-(csource*(sin(a)>0)*(a/!dpi)*exp(-(a/!dpi)^2*b)*c)
  latsource(1)=(maxlat-(maxlat-minlat)*a/!dpi)*(sin(a) gt 0)
; N.B. above curves are normalized to nearly the same integrated strength only
; for overlaps of 0 and 2 years.
;
; modulate cycle strength using data in data file
  if cyclemode eq 2  then source=source*yearssn(years,yssn,ssn,silent=silent,$
    runnumber=runnumber,controlfile=controlfile)
return
;
matchsolarrecords:
; If cyclemode=4, match solar observations as closely as possible.
; List of cycle minima, taken to be the starting points of the new cycles, 
; projecting into the future, and back from 1712.0, at a 21.9 year pol. cycle.
; The minima were slightly moved to obtain a good match, with a
; detailed comparison study yielding 1997.2 and 2006.5. 
  minyear=[1635.1,1646.,1657,1668,1679,1690,1700,1713.5,1724.,1733.5,1745.,$
           1756.0,1767.,1775.5,1784.,1798.5,1811.,1825.,1833.5,1844.,$
           1856.5,1867.0,1879.0,1890.0,1901.4,1913.4,1923.8,1934.4,1944.3,$
           1954.3,1964.5,1976.6,1986.5,1996.7,2006.9,2018.2,2029.2,2040.1]
  minyear=minyear-1646.001
  polarity=(findgen(n_elements(minyear-1)) mod 2)*2-1
  ysource=21.9  ; set to the average between 1744.5 and 1996.3; not used here
  overlap=3.    ; forced to match average overlap of cycles
; Method: determine the two cycles that are possibly involved, estimate their
; durations, and compute, as above, their strengths given the year/time
;
; determine the indices of the first cycle involved:
  yi=(where(years lt minyear))(0)-2
; determine duration of the two cycles involved (full polarity cycle: *2)
  yd=[minyear(yi+1)-minyear(yi),minyear(yi+2)-minyear(yi+1)]*2
; cycle 1:
  minlat=0. ; hardwired
  a=!dpi*(years - minyear(yi))/(yd(0)/2+overlap)
  source(0)=(csource*(sin(a)>0)*(a/!dpi)*exp(-(a/!dpi)^2*8)*8.39)*polarity(yi)
  latsource(0)=(maxlat-(maxlat-minlat)*a/!dpi)*(sin(a) gt 0)
; cycle 2:
  b=!dpi*(years - minyear(yi+1))/(yd(1)/2+overlap)
  source(1)=((csource*(sin(b)>0)*(b/!dpi)*exp(-(b/!dpi)^2*8)*8.39))*$
    polarity(yi+1)
  latsource(1)=(maxlat-(maxlat-minlat)*b/!dpi)*(sin(b) gt 0)
;
  if polarity(yi) lt 0 then begin
    source=shift(source,1)
    latsource=shift(latsource,1)
  endif
;
  source=source*yearssn(years,yssn,ssn,silent=silent,$
    runnumber=runnumber,controlfile=controlfile) 
return
end



