  GRID_PROC subroutine sbp63_steady_mms_source(position,gamma,mach,reynolds,source)
    real(real_kind),intent(in) :: position(3),gamma,mach,reynolds
    real(real_kind),intent(out) :: source(5)
    real(real_kind),parameter :: amplitude=0.05e0_real_kind,modulation=0.1e0_real_kind
    real(real_kind),parameter :: normal_amplitude=0.002e0_real_kind,span_amplitude=0.001e0_real_kind
    real(real_kind) :: sx,cx,sy,cy,sz,cz,v(3),g(3,3),laplacian(3),grad_div(3)
    real(real_kind) :: stress(3,3),unused_heat(3),viscous(3),advection(3),divergence,mu,pressure,energy
    sx=sin(position(1));cx=cos(position(1));sy=sin(position(2));cy=cos(position(2))
    sz=sin(position(3));cz=cos(position(3))
    v=[amplitude*sy*(1.e0_real_kind+modulation*cx*cz), &
       normal_amplitude*sx*sy**2,span_amplitude*cz*sy**2]
    g(1,:)=[-amplitude*modulation*sy*sx*cz,amplitude*cy*(1.e0_real_kind+modulation*cx*cz), &
       -amplitude*modulation*sy*cx*sz]
    g(2,:)=[normal_amplitude*cx*sy**2,2.e0_real_kind*normal_amplitude*sx*sy*cy,0.e0_real_kind]
    g(3,:)=[0.e0_real_kind,2.e0_real_kind*span_amplitude*cz*sy*cy,-span_amplitude*sz*sy**2]
    divergence=g(1,1)+g(2,2)+g(3,3)
    laplacian=[-amplitude*sy*(1.e0_real_kind+3.e0_real_kind*modulation*cx*cz), &
       normal_amplitude*sx*(2.e0_real_kind*cy**2-3.e0_real_kind*sy**2), &
       span_amplitude*cz*(2.e0_real_kind*cy**2-3.e0_real_kind*sy**2)]
    grad_div=[-amplitude*modulation*sy*cx*cz+2.e0_real_kind*normal_amplitude*cx*sy*cy, &
       -amplitude*modulation*cy*sx*cz+2.e0_real_kind*normal_amplitude*sx*(cy**2-sy**2) &
       -2.e0_real_kind*span_amplitude*sz*sy*cy, &
       amplitude*modulation*sy*sx*sz-span_amplitude*cz*sy**2]
    mu=1.e0_real_kind/reynolds
    viscous=mu*(laplacian+grad_div/3.e0_real_kind)
    advection=matmul(g,v)
    call sbp63_material_flux(mu,0.e0_real_kind,v,g,[0.e0_real_kind,0.e0_real_kind,0.e0_real_kind],stress,unused_heat)
    pressure=1.e0_real_kind/(gamma*mach**2)
    energy=pressure/(gamma-1.e0_real_kind)+0.5e0_real_kind*dot_product(v,v)
    source(1)=divergence
    source(2:4)=advection+v*divergence-viscous
    source(5)=(energy+pressure)*divergence+dot_product(v,advection)-sum(stress*g)-dot_product(v,viscous)
  end subroutine
