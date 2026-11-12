program numerical_forms_probe
  use numerical_forms, only: filter_scheme,effective_filter_scheme,compact_filter_selected,viscous_form, &
    me6_configuration_error,convection_form,keep_configuration_error
  use numerical_forms, only: simple_physical_boundary_axis
  use numerical_forms, only: ldc_boundary_layout,profile_state_boundary_layout
  implicit none
  integer :: boundaries(6),axis,kind,n
  integer,parameter :: simple_kinds(7)=[31,41,42,411,421,50,60]
  filter_scheme=''
  if(effective_filter_scheme('643e','643e',.true.)/='explicit10') error stop 'explicit default changed'
  if(effective_filter_scheme('643c','643c',.true.)/='compact10') error stop 'compact default changed'
  filter_scheme='explicit10'
  if(compact_filter_selected('643c','643c')) error stop 'explicit filter tied to compact derivative'
  filter_scheme='compact10'
  if(.not.compact_filter_selected('643e','643e')) error stop 'compact filter tied to explicit derivative'
  if(effective_filter_scheme('643e','643e',.false.)/='none') error stop 'disabled filter remained active'
  viscous_form='me6_opti'
  if(len_trim(me6_configuration_error(0,.false.,[1,1,1,1,1,1],'none',.true.,'643e'))/=0) error stop 'periodic ME6 selection rejected'
  convection_form='keep'
  if(len_trim(me6_configuration_error(0,.false.,[1,1,41,41,1,1],'none',.true.,'643e'))/=0) error stop 'ME6 nodal wall rejected'
  if(len_trim(me6_configuration_error(0,.false.,[1,1,41,41,1,1],'none',.true.,'643c'))/=0) error stop 'ME6 compact nodal wall rejected'
  if(len_trim(me6_configuration_error(0,.true.,[1,1,1,1,1,1],'none',.true.,'643e'))==0) error stop 'reacting ME6 admitted'
  if(len_trim(me6_configuration_error(0,.false.,[1,1,1,1,1,1],'none',.false.,'643e'))==0) error stop 'inactive ME6 admitted'
  viscous_form='standard'
  convection_form='keep'
  if(len_trim(keep_configuration_error('643e','643e',0,.false.,[1,1,41,41,1,1],'none'))/=0) &
    error stop 'explicit KEEP low-order y wall rejected'
  if(len_trim(keep_configuration_error('643c','643e',0,.false.,[1,1,41,41,1,1],'none'))/=0) &
    error stop 'compact KEEP low-order y wall rejected'
  if(len_trim(keep_configuration_error('643e','643e',0,.false.,[11,22,41,52,1,1],'none'))/=0) &
    error stop 'KEEP low-order profile open boundary rejected'
  if(len_trim(keep_configuration_error('643e','643e',0,.false.,[1,1,42,52,1,1],'none'))==0) &
    error stop 'unimplemented KEEP adiabatic-wall/open-top coupling admitted'
  do axis=1,3
    do n=1,size(simple_kinds)
      kind=simple_kinds(n);boundaries=1;boundaries(2*axis-1:2*axis)=kind
      if((kind==42.and.axis==3).or.((kind==31.or.kind==411.or.kind==421).and.axis/=2)) then
        if(simple_physical_boundary_axis(boundaries)/=0) error stop 'unsupported simple physical direction selected'
      else
        if(simple_physical_boundary_axis(boundaries)/=axis) error stop 'simple physical axis classification differs'
        if(len_trim(keep_configuration_error('643e','643c',0,.false.,boundaries,'none'))/=0.or. &
           len_trim(keep_configuration_error('643c','643e',0,.false.,boundaries,'none'))/=0) &
          error stop 'implemented paired KEEP family rejected'
        viscous_form='me6_opti'
        convection_form='standard'
        if(len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643e'))/=0) &
          error stop 'implemented paired ME6 family rejected with original convection'
        viscous_form='standard'
        convection_form='keep'
      endif
    enddo
  enddo
  boundaries=[41,41,41,0,1,1]
  if(.not.ldc_boundary_layout(boundaries)) error stop 'existing LDC layout not classified'
  if(len_trim(keep_configuration_error('643c','643e',0,.false.,boundaries,'none'))/=0) &
    error stop 'KEEP LDC layout rejected'
  viscous_form='me6_opti'
  if(len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643e'))/=0.or. &
     len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643c'))/=0) &
    error stop 'ME6 LDC layout rejected'
  print '(A)', 'FILTER_SELECTION_PASS default explicit_override compact_override disabled'
  print '(A)', 'ME6_SELECTION_PASS periodic paired_physical_explicit_compact_nodal reacting_rejected inactive_rejected'
  print '(A)', 'KEEP_WALL_SELECTION_PASS explicit_compact_y_wall profile_open rejected_partial_open_configuration'
  print '(A)', 'KEEP_SIMPLE_SELECTION_PASS paired_families_and_actual_directions'
  print '(A)', 'LDC_SELECTION_PASS explicit_compact_nodal_band'
  boundaries=[12,22,1,1,1,1]
  if(len_trim(keep_configuration_error('643c','643e',0,.false.,boundaries,'none'))/=0.or. &
    len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643c'))/=0) &
    error stop 'new-form single-x NSCBC layout rejected'
  boundaries=[11,21,1,1,1,1]
  if(len_trim(keep_configuration_error('643e','643c',0,.false.,boundaries,'none'))/=0.or. &
    len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643e'))/=0) &
    error stop 'new-form single-x static layout rejected'
  print '(A)', 'X_OPEN_SELECTION_PASS static_and_nscbc_explicit_compact'
  boundaries=[11,22,41,52,1,1]
  if(len_trim(keep_configuration_error('643e','643e',0,.false.,boundaries,'none'))/=0.or. &
     len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643c'))/=0) &
    error stop 'KEEP profile/ME6 fixed nodal boundary band rejected'
  print '(A)', 'PROFILE_ME6_SELECTION_PASS explicit_compact_nodal_band'
  do kind=50,51
    boundaries=[11,21,41,kind,1,1]
    if(.not.profile_state_boundary_layout(boundaries)) error stop 'state-profile layout not classified'
    if(len_trim(keep_configuration_error('643c','643e',0,.false.,boundaries,'none'))/=0) &
      error stop 'KEEP state-profile layout rejected'
    convection_form='standard'
    if(len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643c'))/=0) &
      error stop 'original convection/ME6 state-profile layout rejected'
    convection_form='keep'
  enddo
  if(profile_state_boundary_layout([11,21,41,51,1,60])) error stop 'nonperiodic profile span admitted'
  print '(A)', 'STATE_PROFILE_SELECTION_PASS top50_top51_keep_and_original_me6'
  convection_form='standard'
  boundaries=[11,22,41,52,1,1]
  if(len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643e'))/=0.or. &
     len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643c'))/=0) &
    error stop 'original central convection/ME6 profile NSCBC rejected'
  print '(A)', 'CENTRAL_ME6_PROFILE_NSCBC_SELECTION_PASS explicit_compact_nodal_band'
  convection_form='keep'
  boundaries=[1,1,41,52,1,1]
  if(len_trim(keep_configuration_error('643e','643e',0,.false.,boundaries,'none'))/=0.or. &
     len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643e'))/=0) &
    error stop 'explicit wall/nonreflecting-y coupling rejected'
  if(len_trim(keep_configuration_error('643c','643e',0,.false.,boundaries,'none'))==0.or. &
     len_trim(me6_configuration_error(0,.false.,boundaries,'none',.true.,'643c'))==0) &
    error stop 'unverified compact wall/nonreflecting-y coupling admitted'
  boundaries=[11,21,41,52,1,1]
  if(len_trim(keep_configuration_error('643e','643e',0,.false.,boundaries,'none'))/=0) &
    error stop 'explicit profile wall/nonreflecting-y coupling rejected'
  print '(A)', 'WALL_NONREFLECTING_Y_SELECTION_PASS explicit_geometry_existing_source'
  boundaries=[11,22,421,52,1,1]
  if(len_trim(keep_configuration_error('643e','643e',0,.false.,boundaries,'none'))/=0) &
    error stop 'existing inviscid incident layout rejected for explicit KEEP'
  if(len_trim(keep_configuration_error('643c','643e',0,.false.,boundaries,'none'))/=0.or. &
     len_trim(keep_configuration_error('643e','643c',0,.false.,boundaries,'none'))/=0.or. &
     len_trim(keep_configuration_error('643c','643c',0,.false.,boundaries,'none'))/=0) &
    error stop 'independent compact incident convection/auxiliary selection rejected'
  viscous_form='me6_opti'
  if(len_trim(me6_configuration_error(0,.false.,boundaries,'none',.false.,'643e'))==0) &
    error stop 'ME6 admitted in the inviscid incident layout'
  print '(A)', 'INVISCID_INCIDENT_SELECTION_PASS explicit_compact_keep no_me6'
end program
