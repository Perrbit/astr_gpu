program insitu_catalog_probe
  use insitu_catalog, only: read_product_catalog,validate_catalog_grid
  use insitu_run_config, only: catalog_product,catalog_field_names
  implicit none
  type(catalog_product),allocatable :: products(:)
  character(:),allocatable :: message,path
  character(32) :: argument
  logical :: ok
  integer :: i,j,n,numq
  call get_command_argument(1,length=n)
  allocate(character(n)::path)
  call get_command_argument(1,path)
  call read_product_catalog(path,products,ok,message)
  if(.not.ok) then
    print '(a)','FILE_ERROR '//message
    stop 1
  endif
  print '(a,i0)','FILE_OK ',size(products)
  numq=5
  call get_command_argument(2,argument)
  if(len_trim(argument)>0) read(argument,*) numq
  do i=1,size(products)
    call validate_catalog_grid(products(i),[2250,260,480],numq)
    print '(a,i0,a,l1,a,a)','PRODUCT ',i,' valid=',products(i)%valid,' id=',products(i)%id
    print '(a)','reason='//products(i)%error
    if(.not.products(i)%valid) cycle
    print '(a,a)','source=',products(i)%source
    print '(a,i0)','request=',products(i)%enable_request
    print '(a,i0)','axis=',products(i)%axis
    print '(a,i0)','index=',products(i)%index
    print '(a,6(i0,1x))','ranges=',products(i)%ranges
    print '(a,i0)','steps=',products(i)%interval_steps
    print '(a,es24.16)','time=',products(i)%interval_time
    do j=1,size(products(i)%fields)
      print '(a,a)','field='//trim(catalog_field_names(products(i)%fields(j)))
    enddo
  enddo
end program
