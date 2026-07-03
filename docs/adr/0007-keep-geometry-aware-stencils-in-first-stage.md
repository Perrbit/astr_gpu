# Keep geometry-aware stencils in first-stage kernels

First-stage GPU gradient, convection, and diffusion kernels use `dxi_d` and `jacob_d` rather than hard-coded TGV grid spacings. RK may temporarily assume `jacob=1` for TGV after checking that assumption, but the stencil path remains aligned with ASTR's geometry representation to avoid a throwaway uniform-grid implementation.
