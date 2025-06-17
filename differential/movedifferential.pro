;+
;  NAME: movedifferential.pro
;    
;  PURPOSE: 
;           apply differential rotation (excluding mean solid-body rotation), 
; 	    optionally modified by a factor differential, relative to 
;	    sidereal Carrington rotation rate
;    
;  CALLING SEQUENCE:
;	movedifferential,phi,theta,nflux,dt,differential,profile=2
;  INPUTS:
;	phi, theta	arrays of longitude and colatitude (radians)
;	nflux		number of elements to be processed
;	dt		time step (seconds)
;	differential	muliplier for differential rotation
;
;  OUTPUTS:
;       phi		longitudes modified by differential rotation in dt
;  
;  OPTIONAL INPUT KEYWORDS:
;	profile		1) Komm et al. (1993) profile (default)
; 			2) Compatibility with MDI data assimilation requires, 
;			   in zeroth order, an increased rotation rate over
;			   Komm et al. of ~465nHz at the equator
;			3) As 2, but tapered with an exponential+pedestal
;			   towards higher latitudes: leaves `2' mostly in tact
;			   up to about 40 degrees, and has less than half the
;			   diff. rot at the poles, with a more or less rigidly
;			   rotating polar cap poleward of about 70 degrees.
;                       4) as 2 but with expanding flux (see note below)
;    
;  OPTIONAL OUTPUT KEYWORDS:
;	
;  METHOD:
;	
;  MODIFICATION HISTORY:
;	02 April 2002:  modified to enforce corotation in the
;			Carrington frame, i.e. at 16 degrees,
;                       also using the equatorial rate corresponding to
; 			the latitude-dependent coefficients.
;	11 April 2002:  modified to increase net rotation rate to
;			be compatible with MDI assimilation data,
;	07 June 2002:   that sets the equatorial rotation rate to
;			for profile=2 to 464.9nHz (with an estimated accuracy
;			of +/- 1.5 nHz from eye estimates), which according to 
;			Beck and Schou (1999) is intermediate to the
;			rates for supergranulation and small magnetic
;			features, corresonding to a mean of their scale
;			spectrum, or a value of m~150. This revised value
;			is compatible with the rotation rate derived by
;			Brouwer and Zwaan (1990) for nests (463.7 +/- 0.3 nHz
;			at the equator).
;       31 Jan. 2008:   added the option to let flux expand, i.e. to
;                       have the leading polarity move ahead while the
;                       trailing polarity lags behind. Normalized to a
;                       speed of 10 m/s at the equator for differential=1.
;
;-
pro movedifferential,phi,theta,nflux,dt,differential,profile=profile,$
  flux=flux,showmap=showmap,runnumber=runnumber
;
if keyword_set(profile) then profile=profile else profile=1

; if (essentially) zero, then return immediataly
if abs(differential) lt 1.e-5 then return

if profile eq 4 then begin
  mdiff=1.d0              ; cannot change magnitude of diffrot in this mode,
  diff=abs(differential)  ; only the magnitude of the relative flow
  sdiff=differential/abs(differential)
endif else begin
  mdiff=differential
endelse
;
scale=(1/86400.)*(!dpi/180.)*dt ; from deg/day to rad/timestep
;
if profile ge 1 and profile le 4 then begin
;
; a is computed relative to the sidereal Carrington rotation rate
; (period of 25.38 days, or 14.18 deg/day), based on Komm et al. (1993) data 
; at 16 deg. latitude.
;
 a=(14.255984-14.18)*scale*mdiff
 if profile eq 2 or profile eq 3 then a=a+0.2*scale*mdiff
;
 b=-2.00*scale*mdiff
 c=-2.09*scale*mdiff
;
 sinlatitude=sin(!dpi/2-theta(0l:nflux-1l))^2
;
 if profile eq 1 or profile eq 2 or profile eq 4 then $
   phi(0)=phi(0l:nflux-1l)+b*sinlatitude+c*sinlatitude^2+a

; taper high-latitude differential rotation for simulation of AB Dor-like star:
 if profile eq 3 then phi(0)=phi(0l:nflux-1l)+(b*sinlatitude+c*sinlatitude^2)*$
   (0.8*exp(-abs(theta(0l:nflux-1l)*!radeg-90.)/80.)+0.2)+a 

; let leading and trailing polarity lead or trail to mimic continued emergence:
 if profile eq 4 then begin
   thr=findsetting('differential threshold',run=abs(runnumber))

; N.B. The sign of 'differential' signifies the leading polarity on
; the southern hemisphere.
   if sdiff gt 0 then $
   fflux=where((flux gt  thr and theta lt !pi/2 and theta gt !pi/6) or $
               (flux lt -thr and theta ge !pi/2 and theta lt 5./6*!pi)) else $
   fflux=where((flux lt -thr and theta lt !pi/2 and theta gt !pi/6) or $
               (flux gt  thr and theta ge !pi/2 and theta lt 5./6*!pi)) 
   if sdiff gt 0 then $
   sflux=where((flux lt -thr and theta lt !pi/2 and theta gt !pi/6) or $
               (flux gt  thr and theta ge !pi/2 and theta lt 5./6*!pi)) else $
   sflux=where((flux gt  thr and theta lt !pi/2 and theta gt !pi/6) or $
               (flux lt -thr and theta ge !pi/2 and theta lt 5./6*!pi))

   scale2=10.d0/7.e8*dt ; rad/sec for 10m/s at equator of Sun.
   if fflux(0) ge 0 then phi(fflux)=phi(fflux)+diff*scale2
   if sflux(0) ge 0 then phi(sflux)=phi(sflux)-diff*scale2

;stop,'Test stop in movedifferential.pro'
   if keyword_set(showmap) then begin
; show a synoptic map
     ssflux=flux                                    
     ssflux(where(abs(ssflux) lt 50))=0              
     synopticmap,phi,theta,ssflux,synoptic,xsize=360
     synoptic=congrid(bytscl(synoptic>(-50)<50),3*360,3*180)
     tv_match,synoptic
     tv,synoptic/4+96
; and overlay the fast and slow points
     for iii=0,n_elements(fflux)-1 do xyouts,3*phi(fflux(iii))*180/!pi,$
         3*(89+90*sin(!pi/2-theta(fflux(iii)))),'+',/device,align=0.5,color=255
     for iii=0,n_elements(sflux)-1 do xyouts,3*phi(sflux(iii))*180/!pi,$
         3*(89+90*sin(!pi/2-theta(sflux(iii)))),'o',/device,align=0.5,color=1
   endif
 endif
;
 return
endif
;
stop,'Stop: differential-rotation option not (yet) avalable'
;
return
end
