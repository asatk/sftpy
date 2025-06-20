;+
;  NAME: addsources.pro
;    
;  PURPOSE: Determine the number and properties of new bipolar regions
;           to be added to the stellar surface, determine positions
;           for a range of fragments of a specified flux, and append
;           these to the exising lists (optionally flagged for gradual
;           emergence)
;    
;  CALLING SEQUENCE: 
;  addsources,years,phi,theta,flux,nflux,nfluxmax,cyclemode,cycleoverlap,$
;      source,ysource,psource,turbulent,miniflux,maxflux,dt,iseed,radius,$
;      binflux,joy,joywidth,joyfold,latwidth,latfold,maxlat,sourceinput,$
;      as_specified,synoptic,yssn,ssn,laphi,latheta,laflux,latime,$
;      silent=silent,assimilation=assimilation,l0=l0,b0=b0,gradual=gradual,$
;      invertpolarity=invertpolarity,radassim=radassim,updated=updated,$
;      peak=peak,controlfile=controlfile,runnumber=runnumber
;
;  INPUTS:
;    years                   time in years for cycle variation
;    phi,theta,flux,nflux    positions, fluxes, and number of concentrations
;    nfluxmax                [not used]
;    cyclemode,cycleoverlap  cycle properties (see cyclestrength.pro)
;    source,ysource,psource  cycle properties (see cyclestrength.pro)
;    turbulent               fraction of ERs that varies with cycle
;    miniflux,maxflux        range of fluxes for bipoles (one polarity)
;    dt                      time step (seconds)
;    iseed                   seed for random number generator
;    radius                  [not used]
;    binflux                 flux unit
;    joy,joywidth,joyfold    dipole orientations and spreads
;    latwidth,latfold,maxlat dipolar latitudes and spreads
;    sourceinput             total absolute flux added
;    as_specified            if '0' leave out ERs [not recommended]
;    synoptic                lat-long bitmap where the flux density
;                            exceeds a threshold
;    yssn,ssn                sunspot record (see cyclestrength.pro)
;    laphi,latheta,laflux,latime  properties of concentrations yet to
;                            be added (when \gradual is set)
;
;  OUTPUTS:
;    phi,theta,flux,nflux    positions, fluxes, and number of concentrations
;    laphi,latheta,laflux,latime  properties of concentrations yet to
;                            be added (when \gradual is set)
;    
;  OPTIONAL INPUT KEYWORDS:
;      silent                if set, suppress diagnostic statements
;      assimilation          if set, import magnetogram information
;      l0, b0                carrington longitude, tilt angle
;      gradual               if set, grow each AR at 0.05e18 Mx/s
;      invertpolarity        if set, invert polarities of sources
;      radassim              radius (degrees) of assimilation window
;      updated               time stamp of magnetogram
;      peak                  specified peak year in cycle (cyclestrength.pro)
;      controlfile           control file path name
;      runnumber             run number within control file
;    
;  OPTIONAL OUTPUT KEYWORDS:
;
;  METHOD:
;
;  MODIFICATION HISTORY:
;
; Modification: 26 Dec. 2001/CJS: allow overlapping cycles
;   Assumption: each polarity cycle overlaps with the next for a period of
;   approximately 2 years. The cycle amplitude is modified to keep the same
;   integrated ``strength'' over the duration of each polarity cycle.
; Modification: 27 Dec. 2001/CJS: allow cycle strength modulation
; Modification: 28 Dec. 2001/CJS: split cyclestrength into separate routine
; Modification: 03 Jan. 2002/CJS: corrected scatter about Joy's law
; Modification: 20 Mar. 2002/CJS: modified for data assimilation, in which
;			case no new sources are added within 60deg of DC.
; Modification: 18 Jun. 2002/CJS: allow specification of AR size and position
;			on call (e.g., for farside assimilation)
; Modification: 16 Oct. 2002/CJS: allow gradual injection of AR flux
;
;-
pro addsources,years,phi,theta,flux,nflux,nfluxmax,cyclemode,cycleoverlap,$
  csource,ysource,psource,turbulent,miniflux,maxflux,dt,iseed,radius,binflux,$
  joy,joywidth,joyfold,latwidth,latfold,maxlat,sourceinput,as_specified,$
  synoptic,yssn,ssn,laphi,latheta,laflux,latime,$
  initialize=initialize,silent=silent,assimilation=assimilation,$
  radassim=radassim,l0=l0,b0=b0,invertpolarity=invertpolarity,updated=updated,$
  specified=specified,gradual=gradual,peak=peak,$
  controlfile=controlfile,runnumber=runnumber
;
dr=!dpi/180.     ; degrees to radians
avefluxd=180.    ; average flux density in active regions
; move the clock on gradual assimilation
if n_elements(latime) eq 0 then latime=0
if keyword_set(gradual) then latime=(latime-1)>0 else latime=latime*0
;
; how strong is the cycle at this phase, if indeed there is a cycle?
if cyclemode eq 0 then begin
  source=fltarr(1)
  source(0)=csource
  if keyword_set(invertpolarity) then source=-source
endif else begin
;
  cyclestrength,years,cyclemode,cycleoverlap,$
    ysource,csource,maxlat,yssn,ssn,source,latsource,silent=silent,peak=peak,$
    controlfile=controlfile,runnumber=runnumber

;stop,'Test stop in addsources.pro'

  cs=['+','-']
  if keyword_set(invertpolarity) then source=-source
  if not(keyword_set(silent)) then for ip=0,1 do $
    print,strcompress('Addsources'+cs(ip)+': '+string(source(ip))+' / '+$
      string(csource)+' at '+string(years)+' yrs (cycle '+string(ysource)+$
      ' yrs) around '+string(latsource(ip))+' degrees')
endelse
;
if keyword_set(initialize) then begin
  newflux=float(maxflux)
  ntotal=1
  newphi=0.
  newtheta=(!dpi/2-latsource(0)*dr)
  orient=joy*dr
  goto,specifiedsources ; N.B. not tested since rewrite 26/Dec/2001!
end
;
; add the sources of the one or two cycles present on the stellar surface:
for ic=0,n_elements(source)-1 do begin
; if sources specified, set variables and jump to step 4.
if keyword_set(specified) then begin
  newflux=nint(specified(*,0))
  newphi=specified(*,1)
  newtheta=specified(*,2)
  ntotal=n_elements(specified(*,0))
  orient=joy*dr+fltarr(ntotal)
  itheta=where(newtheta gt !dpi/2)
  if itheta(0) ge 0 then orient(itheta)=-orient(itheta)+!dpi
; assume the orientation of the largest cycle for the new regions:
  orient=orient+(max(source) lt 0)*!dpi

  print,'addsources: ', orient, source, cyclemode, keyword_set(invertpolarity)

  goto,specifiedsources
endif
; if 'nothing' to add, skip remainder of loop
if abs(source(ic)) lt 1.e-5 then goto,nextcycle
;
; step 1: determine size distribution
;
; fit made to Fig 17 in Harvey's PhD thesis, page 173: a A^{-1.9}+a A^{-2.9};
; see paper I (ApJ 547, 475) for a description of this fit; implement by 
; selecting random numbers from two simple power-law distributions
; 1) high-flux tail dominant for large regions
a=8.        ; A in square deg, t in d; matched to phase ``c''
a=a*abs(source(ic)) ; modify source strength
p=psource   ; 1.9 is the default value from K. Harvey's PhD thesis
; smallest is set to 2*6.e18
; modified 2004/09/28 to allow cutoff at small bipoles as for large:
if n_elements(miniflux) eq 0 then minpp=6 else minpp=miniflux
;was: minpp=6
minflux=minpp/binflux
scale=1/((1.-p)*1.5^(1.-p)*avefluxd^(1-p))
rangefactor=maxflux^(1.-p)-(minpp*2)^(1.-p)
; double number to convert from hemisphere to full surface 
; 2004/10/07: modified the following line
;ntotal=2*(a*(dt/86400.)*scale*rangefactor+0.5) ;number of regions to be added
; to one that does not add the 0.5 regions (taken care of 2 lines below)
ntotal=2*(a*(dt/86400.)*scale*rangefactor) ;number of regions to be added
; include restfraction by appropriate probability
ntotal=long(ntotal)+(randomu(iseed) lt (ntotal-long(ntotal)))
; determine the actual distribution
newbipolefluxes,newfluxflag,newflux,ntotal,p,binflux,minflux,maxflux,iseed
; 2) low-flux tail dominated by ephemeral regions
a=8.    ; for A in square degrees, t in days = cycle-averaged value
; modify source strength, also using a `turbulent-dynamo fraction'
a=a*((abs(source(ic)))^(1./3)*turbulent+(1-turbulent))
p=psource+1     ; coupled to value of 1.9 for high-flux tail
scale=1/((1.-p)*1.5^(1.-p)*avefluxd^(1-p))
rangefactor=maxflux^(1.-p)-(minpp*2)^(1.-p)
ntotal=2*a*(dt/86400.)*scale*rangefactor ; regions to be added
ntotal=long(ntotal)+(randomu(iseed) lt (ntotal-long(ntotal)))
newbipolefluxes,newflux2flag,newflux2,ntotal,p,binflux,minflux,maxflux,iseed
; append the two arrays, where applicable, or skip remainder
if newfluxflag eq 0 and newflux2flag eq 0 then goto,nextcycle
if newfluxflag gt 0 and newflux2flag gt 0 then newflux=[newflux,newflux2]
if newfluxflag eq 0 and newflux2flag gt 0 then newflux=newflux2
ntotal=n_elements(newflux)
;
if not(keyword_set(silent)) then print,'Adding ('+strcompress(ic,/rem)+$
 ') '+strcompress(ntotal,/rem)+' bipoles.'
;
; If in ``accelerated-time'' mode, leave out the ephemeral regions,
; including only regions that are 2 square degrees and larger, i.e.
; 2*1.5e18*avefluxd/1.e18=3 avefluxd units of 1.e18.
; But: if in testrun mode (csource<0), include all - and only - ephemeral reg.
if as_specified eq 0 and csource gt 1.e-5 then begin
  index=where(newflux gt (3*avefluxd/binflux))
  if index(0) lt 0 then return
  newflux=newflux(index)
  ntotal=n_elements(newflux)
endif
if csource lt -1.e-5 then begin
  index=where(newflux lt (3*avefluxd/binflux))
  if index(0) lt 0 then return
  newflux=newflux(index)
  ntotal=n_elements(newflux)
endif
;
; step 2: determine positions
;
; in longitude: random until otherwise specified (e.g., clustering into nests?)
newphi=randomu(iseed,ntotal)*2*!dpi
; in latitude: centered on latsource degrees north & south
newtheta=latsource(ic)*dr*(2*fix(randomn(iseed,ntotal) gt 0.d0)-1)
; add latitude spread (in degrees), depending on flux content
width=latwidth*(exp(-newflux*binflux/latfold)+0.15)
newtheta=newtheta+randomn(iseed,ntotal)*(width*dr)
; transform latitude to colatitude
newtheta=(!dpi/2-newtheta)
;
; Harvey's PhD thesis (p 48): ~40% of active regions emerge inside existing
; regions. This is applied here to all regions larger than 2.5 square degrees,
; or 2.5*avefluxd*1.47562/(2*binflux) units (factor two for two polarities)
nesting=2.5*avefluxd*1.47562/(2*binflux)
ii=where(newflux ge nesting)
if ii(0) ge 0 then begin
; pick nest regions from the set of sufficiently large regions
  pick=where(randomu(iseed,n_elements(ii)) lt 0.4)
  if pick(0) lt 0 then goto,nonests
  pick=ii(pick)
; pick a new location inside any of the plage regions for the selected regions
; but not at the polar caps; limit the emergence to +-50 degrees
; sin(50.*!dpi/180)*90.=68.9, i.e. the 21 pixels nearest the poles are excluded
  xx=bytarr(180) & xx(21:179-21)=1
  index=where(synoptic*((bytarr(360)+1)#xx) ne 0)
  if index(0) lt 0 then goto,nonests
  nreplace=min([n_elements(pick),n_elements(index)])-1
  point=(index((sort(randomu(iseed,n_elements(index))))))(0:nreplace)
  lat=point/360
  newphi(pick(0:nreplace))=point-360*lat
  newtheta(pick(0:nreplace))=!dpi/2-asin(lat/90.-1)
endif
nonests:
; if assimilating data, remove sources from within radassim deg of the 
; magnetograph subobservation point, but only if updating info is available!
if keyword_set(assimilation) then begin
  if strpos(updated,'No data assimilated') ge 0 then goto,skipremoval
  if not(keyword_set(l0)) then l0=0.
  if not(keyword_set(b0)) then b0=0.
  phithetaxyz,newphi+l0,newtheta,xe,ye,ze
  pos=tiltmatrix(b0)##[[xe],[ye],[ze]]
  edge=sin(radassim*!pi/180.)
  index=where((pos(*,1)^2+pos(*,2)^2) lt edge and pos(*,0) gt 0)
; set source fluxes to zero if within assimilated region, then remove
  if index(0) ge 0 then newflux(index)=0
  index=where(newflux ne 0)
  newflux=newflux(index)
  newphi=newphi(index)
  newtheta=newtheta(index)
  ntotal=n_elements(newflux)
endif
skipremoval:
;
; step 3: determine orientations of bipole axes
;
; until 27 Dec. 2007: 
;sjzero=0.2 ; was 0.02 until 3 Jan. 2002
;width=joywidth*(exp(-binflux*newflux/joyfold)+sjzero) ; narrowing with size
; from 27 Dec. 2007: 
; modified to leave the largest regions with the observed spread, but 
; increase it for smaller regions N.B.: joywidth has always been set
; to 90 degrees until now according to the controlfile):
sjzero=18.
width=joywidth*(exp(-binflux*newflux/joyfold))+sjzero ; narrowing with size
orient=(randomn(iseed,ntotal)*width+joy)*dr
; flip sign for opposite-polarity regions and incorporate Hale's law
itheta=where(newtheta gt !dpi/2)
if itheta(0) ge 0 then orient(itheta)=-orient(itheta)+!dpi
; polarity inversion depending on phase of corresponding cycle
orient=orient+(source(ic) lt 0)*!dpi
;
specifiedsources:    ; skip to here to add specific region(s):
;
; step 4: position concentrations
;
sourceinput=0.
for i=0l,ntotal-1l do begin
; area in which flux is distrib. assuming a specified average flux density,
; which covers at least a minimum separation
  newfluxi=newflux(i)
  r=(sqrt(binflux*newfluxi*1.e18/(avefluxd*!dpi))+7.e8)/7.e10
; impose min. separation of ~0.5 supergr. of 18Mm.
  separation=r>((9000./7.e5)/2)   
; determine the number of new concentrations, and the flux contained in them
; roughly 15 10^18 Mx each, but always at least three (equal) conc. per pol.
  if newfluxi gt 3*(15./binflux) then $
              percon=15./binflux else percon=fix((newfluxi/3.)>1)
  bulk=(fix((newfluxi)/(percon)))>1
  rest=(newfluxi-(percon)*bulk)>0       ; one or no residual conc.
  if (newfluxi ge bulk*percon) then nadd=bulk+(rest gt 0) else nadd=1
; one polarity (uniform in radius, 1/r in distance)
  offset1=(randomu(iseed,nadd))*r
  angle1=randomu(iseed,nadd)*2*!dpi
; opposite polarity (uniform in radius, 1/r in distance)
  offset2=(randomu(iseed,nadd))*r
  angle2=randomu(iseed,nadd)*2*!dpi
;
  x=[separation+offset1*cos(angle1),-separation+offset2*cos(angle2)]
  y=[offset1*sin(angle1),offset2*sin(angle2)]
; apply orientation
  orienti=orient(i)+!dpi/2
  coso=cos(orienti)   & sino=sin(orienti)
  xo=coso*x+sino*y    & yo=-sino*x+coso*y
; position on disk by rotating vecors off the pole
  newphii=newphi(i)   & newthetai=newtheta(i)
  cosphi=cos(newphii) & sintheta=sin(newthetai)
  sinphi=sin(newphii) & costheta=cos(newthetai)
  x=cosphi*sintheta+xo*cosphi*costheta-yo*sinphi
  y=sintheta*sinphi+xo*sinphi*costheta+yo*cosphi
  z=costheta-xo*sintheta
; coordinates and fluxes to be added
  aphi=(atan(y,x)+2*!dpi) mod (2*!dpi)
  atheta=acos(z/sqrt(x^2+y^2+z^2))
; add a spread to the flux proportional to sqrt(flux) in each concentration,
; use noise of opposite sign on matching opposite concentrations to preserve
; flux balance
  noise=sqrt(percon)*randomn(iseed,nadd)
  if rest ne 0 then $
    aflux=[percon+noise(0:nadd-2),rest,-percon-noise(0:nadd-2),-rest] else $
    aflux=[percon+noise,-percon-noise]
  aflux=fix(aflux)
;
; add gradually to the list of existing concentrations?
  if keyword_set(gradual) then begin
; How many steps to grow this region (growth: 1.5e22Mx/3.5days=0.05e18 Mx/s)?
    dur=fix(total(abs(aflux))*binflux/0.05/dt)+1
; If everything can be added in a single step, and nothing else is waiting to
; be added, skip the next bit
    if not(dur gt 1 or (where(latime eq 0 and laflux ne 0))(0) ge 0) then $
      goto,skipgradual
; How many pairs are there to be added
    an=n_elements(aflux)/2
; Use uniform random number to identify delay before a conc. is added
    ai=fix(randomu(iseed,an)*dur)*keyword_set(gradual)
; Add this to the list of concentrations yet to be added
    laphi=[laphi,aphi]
    latheta=[latheta,atheta]
    laflux=[laflux,aflux]
    latime=[latime,ai,ai]
; Determine which are to be added now
    ia=where(latime eq 0 and laflux ne 0)
; Select those to be added
    if ia(0) ge 0 then begin
      aphi=laphi(ia)
      atheta=latheta(ia)
      aflux=laflux(ia)
      nadd=n_elements(ia)/2 ; divide by 2 to match earlier convention
    endif else begin
      aphi=0.
      atheta=0.
      aflux=0
      nadd=0
    endelse
; Determine the remaining list of concentrations
    ia=where(latime ne 0 and laflux ne 0)
    if ia(0) ge 0 then begin
      laphi=laphi(ia)
      latheta=latheta(ia)
      laflux=laflux(ia)
      latime=latime(ia)
    endif else begin
      laphi=0.
      latheta=0.
      laflux=0
      latime=0
    endelse
  endif
skipgradual:
; add to current coordinate list
  phi(nflux)=aphi
  theta(nflux)=atheta
  flux(nflux)=aflux
;
  sourceinput=sourceinput+2.*abs(newfluxi)
  nflux=nflux+nadd*2l
endfor
;
; end of for loop that allows for overlapping cycles
nextcycle:
endfor
;
return
end







