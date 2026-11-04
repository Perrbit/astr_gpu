"""Images of reduced line histories; these functions never receive a 3D flow field."""
import io

import numpy as np


def time_space_image_bytes(coordinates,times,values,gaps,*,axis,unit,flags,coordinate_values,quantity,
                           image_size,dpi,jpeg_quality,color_map,color_range,write_eps=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import scienceplots
    plt.style.use(['science','ieee','std-colors'])
    plt.rcParams['axes.grid']=False
    plt.rcParams['grid.alpha']=0.
    plt.rcParams.update({'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
                         'savefig.bbox':None})
    physical=np.asarray(coordinates)[:,axis]
    origin=coordinate_values[axis];length=coordinate_values[3];scale=coordinate_values[4+axis]
    shown=origin+scale*(physical-origin)
    times=np.asarray(times);values=np.asarray(values);gaps=np.asarray(gaps,dtype=bool)
    starts=np.r_[0,np.flatnonzero(gaps[1:])+1];stops=np.r_[starts[1:],len(times)]
    fig,plot=plt.subplots(figsize=(image_size[0]/dpi,image_size[1]/dpi))
    try:
        cmap={'diverging_blue_red':'RdBu_r','grayscale':'gray'}[color_map]
        artist=None
        for first,last in zip(starts,stops):
            if last-first==1:
                artist=plot.scatter(shown,np.full(len(shown),times[first]),c=values[first],s=4,
                    cmap=cmap,vmin=color_range[0],vmax=color_range[1])
            else:
                artist=plot.pcolormesh(shown,times[first:last],values[first:last],shading='nearest',
                    cmap=cmap,vmin=color_range[0],vmax=color_range[1],rasterized=True)
        plot.set_xlim(float(shown.min()),float(shown.max()))
        if len(times)>1:
            plot.set_ylim(float(times[0]),float(times[-1]))
        else:
            plot.set_yticks([float(times[0])])
        plot.xaxis.set_major_formatter(FuncFormatter(lambda value,_:
            f'{((value-origin)/scale/length if flags[1] else (value-origin)/scale+origin):g}'))
        symbol='xyz'[axis]
        plot.set_xlabel(f'$({symbol}-{origin:g})/L_{{ref}}$' if flags[1] else f'${symbol}$ [{unit}]')
        plot.set_ylabel('$t$' if unit=='1' else '$t$ [s]')
        if not flags[0]:
            plot.set_xticks([]);plot.set_yticks([])
        if scale!=1:
            plot.text(.01,.99,f'{symbol} coordinate map x{scale:g}; fixed viewport',transform=plot.transAxes,
                      ha='left',va='top')
        fig.colorbar(artist,ax=plot,label=quantity.replace('_',r'\_'))
        fig.tight_layout()
        jpeg=io.BytesIO();fig.savefig(jpeg,format='jpeg',dpi=dpi,pil_kwargs={'quality':jpeg_quality})
        eps=io.BytesIO() if write_eps else None
        if eps is not None: fig.savefig(eps,format='eps')
        return jpeg.getvalue(),eps.getvalue() if eps is not None else None
    finally:
        plt.close(fig)
