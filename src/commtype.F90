!+---------------------------------------------------------------------+
!| The module declares new types.                                      |
!+---------------------------------------------------------------------+
!| ==============                                                      |
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 02-Jul-2021  | Created by J. Fang STFC                              |
!+---------------------------------------------------------------------+
module commtype
  use astr_precision, only: real_kind
  !
  implicit none
  private :: real_kind

  type :: compact_scheme
    
    integer :: first_node,last_node,dimension,nbctype
    integer :: direction=0
    real(real_kind),allocatable :: a(:),c(:),ac(:,:)
    character(len=1) :: wind

    contains

    procedure :: init=>allocate_compact_scheme_lhs

  end type compact_scheme

  type :: triangle
    real(real_kind) :: a(3),b(3),c(3),normdir(3),area,cen(3)
    !+-------------------+---------------------------------------------+
    !|            a,b,c  | coordinates of the 3 vertex of the triangle.|
    !|           normdir | normal direction of the face.               |
    !|              area | area of of the triangle.                    |
    !+-------------------+---------------------------------------------+
  end type triangle
  !
  type :: lsegment
    real(real_kind) :: a(2),b(2),normdir(2),length,cen(2)
    !+-------------------+---------------------------------------------+
    !|            a,b,c  | coordinates of the 3 vertex of the triangle.|
    !|           normdir | normal direction of the face.               |
    !|              area | area of of the triangle.                    |
    !+-------------------+---------------------------------------------+
  end type lsegment
  !
  type :: solid
    !
    character(len=32) :: name
    real(real_kind) :: xmin(3),xmax(3),xref(3),xcen(3)
    integer :: num_face,num_edge
    type(triangle),allocatable :: face(:)
    type(lsegment),allocatable :: edge(:)
    !
    contains
    !
    procedure :: alloface
    procedure :: alloedge
    !
  end type solid
  !
  type :: sboun
    !
    character(len=1) :: nodetype
    !
    real(real_kind) :: x(3),normdir(3),ximag(3)
    real(real_kind) :: dis_imga_inmg,dis2image,dis2ghost
    real(real_kind),allocatable :: qimag(:),q(:)
    integer :: igh(3),inmg(3)
    integer :: icell(3),icell_bnode(8),icell_ijk(8,3)
    integer,allocatable :: isup(:,:)
    real(real_kind),allocatable :: weig(:)
    real(real_kind),allocatable :: coef_dirichlet(:),coef_neumann(:),invmatrx(:,:)
    !
    logical :: localin,cellin
    integer :: icell_rank,ighst_rank
    integer,allocatable :: ighst_rank_all(:)
    !
    character(len=1) :: locality
    !
    !+-------------------+---------------------------------------------+
    !|               igh | i,j,k of the ghost point                    |
    !|              inmg | i,j,k that closest to the image point       |
    !|              isup | i,j,k that supports the image point         |
    !|              weig | wright from supporting node to image node   |
    !|                   | based on the distance.                      |
    !|           normdir | normal direction of the face.               |
    !|             ximag | coordinate of the boundary nodes.           |
    !|           localin | is the boundary node local in.              |
    !+-------------------+---------------------------------------------+
    contains
    !
  end type sboun
  !
  type :: nodcel
    real(real_kind),allocatable :: x(:,:)
    real(real_kind) :: xmax(3),xmin(3)
    real(real_kind) :: vol
    character(len=1) :: celltype
  end type nodcel
  !!
  !+---------------------+---------------------------------------------+
  !|           solidbody | a type of describing immersed solid body    |
  !|            triangle | a type of describing a  triangle            |
  !|               sboun | a boundary at the solid.                    |
  !|                cell | a cell formed with nodes.                   |
  !+---------------------+---------------------------------------------+
  !
  type :: varray
    !
    integer,allocatable :: vint(:)
    !
  end type varray
  !
  contains 
  !
  !+-------------------------------------------------------------------+
  !| This subroutine initialise faces of a solid.                      |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 02-Jul-2021  | Created by J. Fang STFC                            |
  !+-------------------------------------------------------------------+
  subroutine allocate_compact_scheme_lhs(ascheme)
    !
    class(compact_scheme),target :: ascheme
    !
    allocate( ascheme%a(ascheme%first_node:ascheme%last_node), &
              ascheme%c(ascheme%first_node:ascheme%last_node), &
              ascheme%ac(1:3,ascheme%first_node:ascheme%last_node) )
    !
  end subroutine allocate_compact_scheme_lhs
  !+-------------------------------------------------------------------+
  !| The end of the subroutine allocate_compact_scheme_lhs             |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine initialise faces of a solid.                      |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 02-Jul-2021  | Created by J. Fang STFC                            |
  !+-------------------------------------------------------------------+
  subroutine alloface(asolid)
    !
    class(solid),target :: asolid
    !
    allocate(asolid%face(asolid%num_face))
    !
  end subroutine alloface
  !+-------------------------------------------------------------------+
  !| The end of the subroutine alloface                                |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine initialise edges of a solid.                      |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 07-Jul-2021  | Created by J. Fang STFC                            |
  !+-------------------------------------------------------------------+
  subroutine alloedge(asolid)
    !
    class(solid),target :: asolid
    !
    allocate(asolid%edge(asolid%num_edge))
    !
  end subroutine alloedge
  !+-------------------------------------------------------------------+
  !| The end of the subroutine alloedge                                |
  !+-------------------------------------------------------------------+
  !
end module commtype
!+---------------------------------------------------------------------+
!| The end of the module commtype.                                     |
!+---------------------------------------------------------------------+
