"""Small host-side wall curves; no volume or plane-field download."""
import ctypes
import io
from pathlib import Path

import numpy as np

PLANE_TITLES={
    'velocity_slice':'Speed / U_inf',
    'streamwise_velocity_slice':'u / U_inf',
    'pressure_ratio_slice':'p / p_inf',
    'temperature_slice':'T / T_inf',
    'density_gradient_slice':'log10(1 + Lx |grad_xy rho| / rho_inf)',
}
WALL_COLUMNS={
    'wall_pressure_curve':(1,r'$p_w/p_\infty$','p_wall_over_p_inf'),
    'skin_friction_curve':(2,r'$C_f$','Cf'),
    'wall_heat_flux_curve':(3,r'$q_{w,\mathrm{into\ wall}}/(\rho_\infty U_\infty^3)$','q_wall_over_rho_inf_U_inf_cubed'),
}


def wall_values():
    function=ctypes.CDLL(None).astr_insitu_sbli_wall_data
    function.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
    function.restype=ctypes.c_void_p
    count,real_bytes=ctypes.c_int(),ctypes.c_int()
    address=function(ctypes.byref(count),ctypes.byref(real_bytes))
    if count.value<2:
        raise ValueError('Missing SBLI wall curve')
    scalar={4:ctypes.c_float,8:ctypes.c_double}[real_bytes.value]
    pointer=ctypes.cast(address,ctypes.POINTER(scalar))
    data=np.ctypeslib.as_array(pointer,shape=(4*count.value,)).reshape(-1,4)
    if not np.isfinite(data).all() or np.any(np.diff(data[:,0])<=0.):
        raise ValueError('Nonfinite or unordered SBLI wall curve')
    return data


def curve_image_bytes(x,value,label,*,image_size,dpi,jpeg_quality,x_range,y_range=None,
                      line_width=None,write_eps=False,zero_line=False,x_label=r'$x/L_\mathrm{ref}$',break_before=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science','ieee','std-colors'])
    plt.rcParams['axes.grid']=False
    plt.rcParams['grid.alpha']=0.0
    plt.rcParams.update({'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
                        'savefig.bbox':None})
    fig,axis=plt.subplots(figsize=(image_size[0]/dpi,image_size[1]/dpi))
    try:
        if break_before is not None:
            gaps=np.flatnonzero(break_before)
            x=np.insert(np.asarray(x,dtype=float),gaps,np.nan)
            value=np.insert(np.asarray(value,dtype=float),gaps,np.nan)
        axis.plot(x,value,color='tab:blue',linewidth=line_width)
        if zero_line:
            axis.axhline(0.,color='0.5',linewidth=.7)
        axis.set_xlabel(x_label)
        axis.set_ylabel(label)
        axis.set_xlim(*x_range)
        if y_range is not None:
            axis.set_ylim(*y_range)
        fig.tight_layout()
        jpeg=io.BytesIO()
        fig.savefig(jpeg,format='jpeg',dpi=dpi,pil_kwargs={'quality':jpeg_quality})
        eps=io.BytesIO() if write_eps else None
        if write_eps:
            fig.savefig(eps,format='eps')
        return jpeg.getvalue(),eps.getvalue() if eps else None
    finally:
        plt.close(fig)


def encode_wall_curve(name,data,step,time,output,write_eps):
    from image_publication import publish_pair
    column,label,header=WALL_COLUMNS[name]
    jpeg,eps=curve_image_bytes(data[:,0],data[:,column],label,image_size=(3840,1440),dpi=384,
        jpeg_quality=95,x_range=(data[0,0],data[-1,0]),write_eps=write_eps,
        zero_line=name=='skin_friction_curve')
    base=Path(output)/f'{name}.step{step:08d}'
    with base.with_suffix(base.suffix+'.csv').open('x') as stream:
        np.savetxt(stream,data[:,[0,column]],delimiter=',',fmt='%.17e',
            header=f'step={step},time={time:.17e}\nx_over_Lref,{header}')
    failure=publish_pair(base.with_suffix(base.suffix+'.jpeg'),jpeg,eps)
    if failure:
        raise OSError(failure['errno'],failure['reason'])
