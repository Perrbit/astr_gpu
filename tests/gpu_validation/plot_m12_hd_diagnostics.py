"""Plot compact instantaneous spanwise means, never full flow fields."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee', 'std-colors'])
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.0
plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 14,
                     'ytick.labelsize': 14, 'legend.fontsize': 14})


def plot_curves(directory):
    directory = Path(directory)
    output = directory/'curves'
    output.mkdir(exist_ok=True)
    steps = sorted({int(path.name.split('.')[1][4:]) for path in directory.glob('wall.step*.rank*.csv')})
    for step in steps:
        wall = np.concatenate([np.genfromtxt(path, delimiter=',', names=True)
            for path in sorted(directory.glob(f'wall.step{step:08d}.rank*.csv'))])
        wall.sort(order='global_i')
        if not np.isfinite(np.column_stack([wall[k] for k in wall.dtype.names])).all():
            raise ValueError('Nonfinite wall diagnostics; do not plot')
        for field, label in (('Cf', r'$C_f$'), ('q_into_wall_rhoinf_Uinf3', r'$q_w/(\rho_\infty U_\infty^3)$')):
            for window, right in (('global', 1100.), ('inlet', 300.)):
                path = output/f'{field}.{window}.step{step:08d}'
                if all(Path(str(path)+suffix).is_file() and Path(str(path)+suffix).stat().st_size > 0
                       for suffix in ('.jpeg', '.eps')):
                    continue
                mask = wall['x'] <= right
                fig, ax = plt.subplots(figsize=(8., 3.6))
                ax.plot(wall['x'][mask], wall[field][mask], color='tab:red', linewidth=1.2)
                ax.set(xlabel=r'$x$', ylabel=label, xlim=(0., right))
                fig.tight_layout()
                fig.savefig(str(path)+'.jpeg', dpi=300)
                fig.savefig(str(path)+'.eps')
                plt.close(fig)
