module insitu_temporal_analysis
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes,real_to_word,word_to_real
  use ieee_arithmetic, only: ieee_is_finite
  use singleton, only: fftn
  implicit none
  private
  public :: welch_state,configure_welch,push_welch,read_welch,welch_info,welch_state_file
  public :: pack_welch,unpack_welch,time_block,configure_time_block,push_time_block,clear_time_block
  public :: pack_time_block,unpack_time_block
  type :: welch_state
    private
    integer :: length=0,components=0,filled=0
    integer(int64) :: segments=0,generation=0
    real(real_kind) :: spacing=0
    logical :: uniform=.false.
    real(real_kind),allocatable :: buffer(:,:),window(:),sum_psd(:,:)
  end type
  type :: time_block
    integer :: nodes=0,fields=0,capacity=0,filled=0
    integer(int64) :: number=0,samples=0,last_step=-1
    real(real_kind) :: last_time=0
    real(real_kind) :: last_coverage(3)=0
    real(real_kind),allocatable :: coordinates(:,:),times(:),values(:,:,:)
    real(real_kind),allocatable :: mean_coverage(:,:)
    integer(int64),allocatable :: steps(:)
    logical,allocatable :: gaps(:)
  end type
contains
  subroutine configure_time_block(s,coordinates,fields,capacity,ok,with_coverage)
    type(time_block),intent(out) :: s
    real(real_kind),intent(in) :: coordinates(:,:)
    integer,intent(in) :: fields,capacity
    logical,intent(out) :: ok
    logical,optional,intent(in) :: with_coverage
    integer :: status,n
    ok=.false.;n=size(coordinates,2)
    if(size(coordinates,1)/=3.or.n<1.or.fields<1.or.capacity<1) return
    if(.not.all(ieee_is_finite(coordinates))) return
    allocate(s%coordinates(3,n),s%times(capacity),s%steps(capacity),s%gaps(capacity), &
      s%values(n,fields,capacity),stat=status)
    if(status/=0) return
    if(present(with_coverage)) then
      if(with_coverage) then
        allocate(s%mean_coverage(3,capacity),stat=status)
        if(status/=0) return
        s%mean_coverage=0
      endif
    endif
    s%nodes=n;s%fields=fields;s%capacity=capacity;s%coordinates=coordinates
    s%times=0;s%steps=-1;s%gaps=.false.;s%values=0;ok=.true.
  end subroutine

  subroutine push_time_block(s,step,time,values,continuous,ok,coverage)
    type(time_block),intent(inout) :: s
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time,values(:,:)
    logical,intent(in) :: continuous
    logical,intent(out) :: ok
    real(real_kind),optional,intent(in) :: coverage(3)
    integer :: p
    ok=.false.
    if(s%capacity==0.or.s%filled==s%capacity.or.any(shape(values)/=[s%nodes,s%fields])) return
    if(step<0.or..not.ieee_is_finite(time).or.time<0.or..not.all(ieee_is_finite(values))) return
    if(s%last_step>=0) then
      if(step<=s%last_step.or.time<=s%last_time) return
    endif
    if(allocated(s%mean_coverage)) then
      if(.not.present(coverage)) return
      if(.not.all(ieee_is_finite(coverage))) return
      if(coverage(2)<=coverage(1).or.coverage(3)<=0) return
      s%mean_coverage(:,s%filled+1)=coverage;s%last_coverage=coverage
    endif
    p=s%filled+1;s%steps(p)=step;s%times(p)=time;s%values(:,:,p)=values
    s%gaps(p)=.not.continuous.and.s%last_step>=0
    s%filled=p;s%samples=s%samples+1;s%last_step=step;s%last_time=time;ok=.true.
  end subroutine

  subroutine clear_time_block(s)
    type(time_block),intent(inout) :: s
    if(s%filled==0) return
    s%number=s%number+1;s%filled=0;s%times=0;s%steps=-1;s%gaps=.false.;s%values=0
    if(allocated(s%mean_coverage)) s%mean_coverage=0
  end subroutine

  subroutine pack_time_block(s,words,ok)
    type(time_block),intent(in) :: s
    integer(int64),allocatable,intent(out) :: words(:)
    logical,intent(out) :: ok
    integer :: status,p,n
    integer(int64) :: count
    ok=.false.
    if(s%capacity==0) return
    count=13_int64+size(s%coordinates,kind=int64)+3_int64*s%capacity+size(s%values,kind=int64)
    if(allocated(s%mean_coverage)) count=count+size(s%mean_coverage,kind=int64)
    allocate(words(count),stat=status)
    if(status/=0) return
    words(:13)=[int(z'4153545254423032',int64),int(s%nodes,int64),int(s%fields,int64),int(s%capacity,int64), &
      int(s%filled,int64),s%number,s%samples,s%last_step,real_to_word(s%last_time), &
      int(merge(1,0,allocated(s%mean_coverage)),int64),real_to_word(s%last_coverage)]
    p=14;n=size(s%coordinates);words(p:p+n-1)=reshape(real_to_word(s%coordinates),[n]);p=p+n
    n=s%capacity;words(p:p+n-1)=s%steps;p=p+n
    words(p:p+n-1)=real_to_word(s%times);p=p+n
    words(p:p+n-1)=int(merge(1,0,s%gaps),int64);p=p+n
    n=size(s%values);words(p:p+n-1)=reshape(real_to_word(s%values),[n]);p=p+n
    if(allocated(s%mean_coverage)) words(p:)=reshape(real_to_word(s%mean_coverage),[3*s%capacity])
    ok=.true.
  end subroutine

  subroutine unpack_time_block(words,s,ok)
    integer(int64),intent(in) :: words(:)
    type(time_block),intent(out) :: s
    logical,intent(out) :: ok
    real(real_kind),allocatable :: coordinates(:,:)
    integer :: n,f,capacity,p,m,status
    integer(int64) :: remaining
    ok=.false.
    if(size(words)<13) return
    if(words(1)/=int(z'4153545254423032',int64).or.any(words(2:4)<1).or.any(words(2:4)>huge(n))) return
    if(words(10)/=0.and.words(10)/=1) return
    n=int(words(2));f=int(words(3));capacity=int(words(4))
    remaining=size(words,kind=int64)-13_int64-3_int64*n-(3_int64+3_int64*words(10))*capacity
    if(remaining<0) return
    if(int(n,int64)>remaining/f/capacity) return
    if(int(n,int64)*f*capacity/=remaining) return
    if(words(5)<0.or.words(5)>capacity.or.any(words(6:7)<0).or.words(8)<-1) return
    allocate(coordinates(3,n),stat=status)
    if(status/=0) return
    p=14;m=3*n;coordinates=reshape(word_to_real(words(p:p+m-1)),[3,n]);p=p+m
    call configure_time_block(s,coordinates,f,capacity,ok,with_coverage=words(10)==1)
    if(.not.ok) return
    ok=.false.;s%filled=int(words(5));s%number=words(6);s%samples=words(7)
    s%last_step=words(8);s%last_time=word_to_real(words(9))
    s%last_coverage=word_to_real(words(11:13))
    m=capacity;s%steps=words(p:p+m-1);p=p+m
    s%times=word_to_real(words(p:p+m-1));p=p+m
    if(any(words(p:p+m-1)<0).or.any(words(p:p+m-1)>1)) return
    s%gaps=words(p:p+m-1)==1;p=p+m
    m=size(s%values);s%values=reshape(word_to_real(words(p:p+m-1)),shape(s%values));p=p+m
    if(allocated(s%mean_coverage)) then
      s%mean_coverage=reshape(word_to_real(words(p:)),shape(s%mean_coverage))
      if(.not.all(ieee_is_finite(s%mean_coverage)).or..not.all(ieee_is_finite(s%last_coverage))) return
      if(s%filled>0) then
        if(any(s%mean_coverage(:,s%filled)/=s%last_coverage)) return
      endif
    endif
    if(.not.ieee_is_finite(s%last_time).or..not.all(ieee_is_finite(s%times)).or. &
       .not.all(ieee_is_finite(s%values))) return
    if(s%filled>0) then
      if(s%steps(s%filled)/=s%last_step.or.s%times(s%filled)/=s%last_time) return
      if(any(s%steps(2:s%filled)<=s%steps(:s%filled-1)).or. &
         any(s%times(2:s%filled)<=s%times(:s%filled-1))) return
    endif
    ok=.true.
  end subroutine

  subroutine configure_welch(s,length,components,ok)
    type(welch_state),intent(out) :: s
    integer,intent(in) :: length,components
    logical,intent(out) :: ok
    integer :: status,j
    real(real_kind) :: pi
    ok=.false.
    if(length<2.or.mod(length,2)/=0.or.components<1) return
    allocate(s%buffer(length,components),s%window(length),s%sum_psd(length/2+1,components),stat=status)
    if(status/=0) return
    s%length=length;s%components=components;s%buffer=0;s%sum_psd=0
    pi=acos(-1.e0_real_kind)
    do j=0,length-1
      s%window(j+1)=.5e0_real_kind*(1.e0_real_kind-cos(2.e0_real_kind*pi*real(j,real_kind)/length))
    enddo
    ok=.true.
  end subroutine

  subroutine push_welch(s,values,spacing,continuous,uniform,ok)
    type(welch_state),intent(inout) :: s
    real(real_kind),intent(in) :: values(:),spacing
    logical,intent(in) :: continuous,uniform
    logical,intent(out) :: ok
    complex(real_kind),allocatable :: transformed(:)
    real(real_kind),allocatable :: periodogram(:,:)
    real(real_kind) :: factor,average
    integer :: status,c,bins,half
    ok=.false.
    if(s%length==0.or.size(values)/=s%components) return
    if(.not.all(ieee_is_finite(values))) return
    if(.not.uniform) then
      s%filled=0;s%buffer=0;s%uniform=.false.;ok=.true.;return
    endif
    if(.not.ieee_is_finite(spacing).or.spacing<=0) return
    if(.not.continuous.or..not.s%uniform) then
      s%filled=0;s%buffer=0
    endif
    if(s%spacing/=spacing) then
      s%filled=0;s%buffer=0;s%sum_psd=0;s%segments=0
      s%spacing=spacing;s%generation=s%generation+1
    endif
    s%uniform=.true.
    s%filled=s%filled+1;s%buffer(s%filled,:)=values
    if(s%filled<s%length) then
      ok=.true.;return
    endif
    bins=s%length/2+1;half=s%length/2
    allocate(transformed(s%length),periodogram(bins,s%components),stat=status)
    if(status/=0) return
    ! Singleton uses a unitary FFT; recover the conventional unnormalised power.
    factor=real(s%length,real_kind)*s%spacing/sum(s%window**2)
    do c=1,s%components
      average=sum(s%buffer(:,c))/real(s%length,real_kind)
      transformed=cmplx((s%buffer(:,c)-average)*s%window,0.e0_real_kind,kind=real_kind)
      call fftn(transformed,[s%length],stat=status)
      if(status/=0) return
      periodogram(:,c)=factor*(real(transformed(:bins),real_kind)**2+aimag(transformed(:bins))**2)
      periodogram(2:bins-1,c)=2.e0_real_kind*periodogram(2:bins-1,c)
    enddo
    if(.not.all(ieee_is_finite(periodogram))) return
    s%sum_psd=s%sum_psd+periodogram
    if(.not.all(ieee_is_finite(s%sum_psd))) return
    s%segments=s%segments+1
    s%buffer(:half,:)=s%buffer(half+1:,:);s%buffer(half+1:,:)=0;s%filled=half
    ok=.true.
  end subroutine

  subroutine read_welch(s,frequencies,psd,available)
    type(welch_state),intent(in) :: s
    real(real_kind),allocatable,intent(out) :: frequencies(:),psd(:,:)
    logical,intent(out) :: available
    integer :: k
    available=s%uniform.and.s%segments>0
    if(.not.available) return
    allocate(frequencies(s%length/2+1),psd(s%length/2+1,s%components))
    do k=0,s%length/2
      frequencies(k+1)=real(k,real_kind)/(s%length*s%spacing)
    enddo
    psd=s%sum_psd/real(s%segments,real_kind)
  end subroutine

  subroutine welch_info(s,segments,filled,generation,spacing,uniform,bytes)
    type(welch_state),intent(in) :: s
    integer(int64),intent(out) :: segments,generation,bytes
    integer,intent(out) :: filled
    real(real_kind),intent(out) :: spacing
    logical,intent(out) :: uniform
    segments=s%segments;filled=s%filled;generation=s%generation;spacing=s%spacing;uniform=s%uniform
    bytes=0
    if(allocated(s%buffer)) bytes=size(s%buffer,kind=int64)*real_bytes
    if(allocated(s%window)) bytes=bytes+size(s%window,kind=int64)*real_bytes
    if(allocated(s%sum_psd)) bytes=bytes+size(s%sum_psd,kind=int64)*real_bytes
  end subroutine

  subroutine welch_state_file(unit,writing,s,ok)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    type(welch_state),intent(inout) :: s
    logical,intent(out) :: ok
    integer(int64),allocatable :: words(:)
    character(8) :: magic
    integer :: status
    integer(int64) :: n,position,bytes
    ok=.false.;magic='ASTRWP02'
    if(writing) then
      call pack_welch(s,words,ok)
      if(.not.ok) return
      write(unit,iostat=status) magic,size(words,kind=int64),words
      ok=status==0;return
    endif
    read(unit,iostat=status) magic,n
    if(status/=0.or.magic/='ASTRWP02'.or.n<8) return
    inquire(unit=unit,pos=position,size=bytes,iostat=status)
    if(status/=0) return
    if(n>(bytes-position+1)/8) return
    allocate(words(n),stat=status)
    if(status/=0) return
    read(unit,iostat=status) words
    if(status/=0) return
    call unpack_welch(words,s,ok)
  end subroutine

  subroutine pack_welch(s,words,ok)
    type(welch_state),intent(in) :: s
    integer(int64),allocatable,intent(out) :: words(:)
    logical,intent(out) :: ok
    integer :: status,n
    ok=.false.
    if(s%length==0) return
    allocate(words(8_int64+size(s%buffer,kind=int64)+size(s%sum_psd,kind=int64)),stat=status)
    if(status/=0) return
    words(:8)=[int(z'4153545257503032',int64),int(s%length,int64),int(s%components,int64), &
      int(s%filled,int64),s%segments,s%generation,real_to_word(s%spacing),int(merge(1,0,s%uniform),int64)]
    n=size(s%buffer);words(9:8+n)=reshape(real_to_word(s%buffer),[n])
    words(9+n:)=reshape(real_to_word(s%sum_psd),[size(s%sum_psd)]);ok=.true.
  end subroutine

  subroutine unpack_welch(words,s,ok)
    integer(int64),intent(in) :: words(:)
    type(welch_state),intent(out) :: s
    logical,intent(out) :: ok
    integer :: n,components,m
    real(real_kind) :: spacing
    integer(int64) :: remaining
    ok=.false.
    if(size(words)<8) return
    if(words(1)/=int(z'4153545257503032',int64).or.words(2)<2.or.words(2)>huge(n).or. &
       words(3)<1.or.words(3)>huge(n)) return
    n=int(words(2));components=int(words(3));remaining=size(words,kind=int64)-8
    if(mod(n,2)/=0.or.(int(n,int64)+n/2+1)>remaining/components) return
    if((int(n,int64)+n/2+1)*components/=remaining) return
    spacing=word_to_real(words(7))
    if(words(4)<0.or.words(4)>=n.or.any(words(5:6)<0).or.words(8)<0.or.words(8)>1) return
    if(.not.ieee_is_finite(spacing).or.spacing<0.or.(words(8)==1.and.spacing==0)) return
    call configure_welch(s,n,components,ok)
    if(.not.ok) return
    ok=.false.;m=size(s%buffer)
    s%buffer=reshape(word_to_real(words(9:8+m)),shape(s%buffer))
    s%sum_psd=reshape(word_to_real(words(9+m:)),shape(s%sum_psd))
    if(.not.all(ieee_is_finite(s%buffer)).or..not.all(ieee_is_finite(s%sum_psd))) return
    s%filled=int(words(4));s%segments=words(5);s%generation=words(6);s%spacing=spacing;s%uniform=words(8)==1
    ok=.true.
  end subroutine
end module
