program numerical_forms_probe
  use numerical_forms, only: filter_scheme,effective_filter_scheme,compact_filter_selected,viscous_form,me6_configuration_error
  implicit none
  filter_scheme=''
  if(effective_filter_scheme('643e','643e',.true.)/='explicit10') error stop 'explicit default changed'
  if(effective_filter_scheme('643c','643c',.true.)/='compact10') error stop 'compact default changed'
  filter_scheme='explicit10'
  if(compact_filter_selected('643c','643c')) error stop 'explicit filter tied to compact derivative'
  filter_scheme='compact10'
  if(.not.compact_filter_selected('643e','643e')) error stop 'compact filter tied to explicit derivative'
  if(effective_filter_scheme('643e','643e',.false.)/='none') error stop 'disabled filter remained active'
  viscous_form='me6_opti'
  if(len_trim(me6_configuration_error(0,.false.,[1,1,1,1,1,1],'none',.true.))/=0) error stop 'periodic ME6 selection rejected'
  if(len_trim(me6_configuration_error(0,.false.,[1,1,41,41,1,1],'none',.true.))==0) error stop 'unimplemented ME6 wall admitted'
  if(len_trim(me6_configuration_error(0,.true.,[1,1,1,1,1,1],'none',.true.))==0) error stop 'reacting ME6 admitted'
  if(len_trim(me6_configuration_error(0,.false.,[1,1,1,1,1,1],'none',.false.))==0) error stop 'inactive ME6 admitted'
  viscous_form='standard'
  print '(A)', 'FILTER_SELECTION_PASS default explicit_override compact_override disabled'
  print '(A)', 'ME6_SELECTION_PASS periodic rejected_unimplemented_wall reacting inactive'
end program
