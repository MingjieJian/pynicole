; Thus function needs to be run in the NICOLE run directory
; The model for which the response functions are required must be
; in tmp.model and NICOLE.input must be set to synthesis mode, with
; tmp.model as input model and tmp.prof as output profile
;
; Note: tmp.model is changed after running this function
;
function rf
  lambda=read_wavelengths()

  nl=n_elements(lambda)
  refmodel=read_model('tmp.model')
  nx=(size(refmodel.tau))(1)
  ny=(size(refmodel.tau))(2)
  nz=(size(refmodel.tau))(3)

  spawn,'./run_nicole.py'
  refi=read_profile('tmp.prof',refq,refu,refv)
  
  rf=fltarr(nx,ny,nz,nl,4)
  for iz=0,nz-1 do begin
     pertmodel=refmodel
     pertmodel.t[*,*,iz]=pertmodel.t[*,*,iz]+100
     idl_to_nicole,file='tmp.model',model=pertmodel
     perti=read_profile('tmp.prof',pertq,pertu,pertv)
     rf[*,*,iz,*,0]=perti-refi
     rf[*,*,iz,*,1]=pertq-refq
     rf[*,*,iz,*,2]=pertu-refu
     rf[*,*,iz,*,3]=pertv-refv
  endfor
  rf=rf/100.
  
  return,rf
  
end


