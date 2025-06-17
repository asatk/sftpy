;+
;  NAME: movecharges.pro
;    
;  PURPOSE:
; procedure to move charges subject to random walk only
;
; called by kit.pro to simulate a random walk 
;    and by testfragment.pro to offset fragments in a random direction
;
;    
;  CALLING SEQUENCE:
;
;  INPUTS:
;  dependence	specifies whether stepping is to be 
;            0  independent of flux and local flux density
;            1  dependent on local flux density 
;            2  dependent on flux in concentration
;
; If source lt 0 (testruns for diffusion models), then difftest returns
; the flux in the synoptic map above a threshold of three gauss.
;;
;  OUTPUTS:
;    
;  OPTIONAL INPUT KEYWORDS:
;    
; fln           if specified, a latitude distribution of the synoptic chart
;               is appended to the file named fln.
; savelat       if set, fln is used as filename
;
;  OPTIONAL OUTPUT KEYWORDS:
;
;  METHOD:
;
;  MODIFICATION HISTORY:
;
;-
pro movecharges,phi,theta,flux,binflux,nflux,dt,diffusion,iseed,dependence,$
  synoptic,source,difftest,fln=fln,savelat=savelat
;
step=fltarr(nflux)+1.  ; step-size array 
;
; make a map of the absolute flux density, thresholded at thr
; N.B. This map is also used to nest active regions
thr=40.						; threshold in gauss
synoptic=fltarr(360,180)			; synoptic bitmap (output)
dr=180/!dpi					; radians to degrees
; (x,y) coordinates on synoptic map; binning does not require rounding off
x=fix(((phi(0l:nflux-1l)+2*!dpi) mod (2*!dpi))*dr)>0<359
y=(fix(( sin(theta(0l:nflux-1l)+!dpi/2) +1.)*90))>0<179
c=x+y*360l & aflux=abs(flux)
;for i=0l,nflux-1l do begin
;  ci=c(i)
;  synoptic(ci)=synoptic(ci)+aflux(i)
;endfor
synoptic=reform(synoptic,360*180l,/overwrite)

; first make arrays with longitudinally summed (absolute) fluxes
if keyword_set(savelat) and keyword_set(fln) then begin
  s=fltarr(180)
  sumarr,nflux,y,aflux,s
  s2=fltarr(180)
  sumarr,nflux,y,flux,s2
  get_lun,lun
  openu,lun,fln,/append
  str=''
  for jj=0,179 do str=str+string(s(jj),format='(f8.0," ")')
  for jj=0,179 do str=str+string(s2(jj),format='(f8.0," ")')
  printf,lun,strcompress(str)
  free_lun,lun
endif

; then make map of absolute fluxes:
sumarr,nflux,c,aflux,synoptic
synoptic=reform(synoptic,360,180,/overwrite)
;
; former, much slower statement
;for i=0l,nflux-1l do synoptic(x(i),y(i))=synoptic(x(i),y(i))+abs(flux(i))
if source lt 0 then difftest=$
  total(abs(synoptic(where(abs(synoptic) gt 10/(binflux/1.4752)))))*dt
synoptic=synoptic gt thr/(binflux/1.4752) 	; transform to gauss
; smooth slightly, and require at least 6 neighbors to be part of a plage too;
; then delate to add an extra ring of pixels to the plage
synoptic=dilate((smooth(float(synoptic),3) gt 5.9/9.),replicate(1,3,3))
;
; if required, adjust the step size using the ratio of diffusion coefficients
; taken from Schrijver&Martin90; but NOT in conjunction with flux-dens. dep.
if dependence eq 1 then is=where(synoptic(x,y) gt 0) else is=[-1]
if is(0) ge 0 then step(is)=110./250
;
; evaluate the actual stepping distance
step=sqrt(4*diffusion*step*dt)/7.e5 ; fraction of circumference
;
; if flux-dependent steps are required, apply that correction here
; also to concentrations that are contained in a plage
; (idea from Schrijver et al. 1996; the e-folding value comes from
; the PhD thesis by Hagenaar, p 119, Fig 7.4, for t=1.5ksec, which
; shows a decrease in the ms velocity by a factor of 2.4^2 between
; a flux of 0 and of 3 10^19 Mx.) 
if dependence eq 2 then $
  step=step*(240./140.)*exp(-(abs(flux(0l:nflux-1l)))*binflux/35.)
;
; Move the sources, given the above step size. The proper recipe for stepping 
; on a sphere is the following:
; 1) rotate sphere so that point is on the pole
; 2) tilt over step in theta
; 3) rotate over random phi angle
; 4) apply inverse of 1
; This is approximated in step 1 by letting concentrations move on a plane 
; tangent to the sphere at the pole, neglecting curvature.
;
; random direction to step in:
rphi=randomu(iseed,nflux)*2*!dpi
cosrphi=cos(rphi) & sinrphi=sin(rphi)
sinphi=sin(phi(0:nflux-1))
cosphi=cos(phi(0:nflux-1))
sintheta=sin(theta(0:nflux-1))
costheta=cos(theta(0:nflux-1))
crct=cosrphi*costheta
x=cosphi*sintheta+step*(crct*cosphi-sinrphi*sinphi)
y=sinphi*sintheta+step*(crct*sinphi+sinrphi*cosphi)
z=costheta-step*cosrphi*sintheta
; new coordinates, including wrap around
;phi(0)= (( atan(y/x)+(x lt 0)*!dpi    ) +2*!dpi) mod (2*!dpi)
phi(0)=(atan(y,x)+2*!dpi) mod (2*!dpi)
theta(0)=acos(z/sqrt(x^2+y^2+z^2)) 
end
