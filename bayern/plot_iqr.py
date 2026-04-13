import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.qc import compute_log_iqr_bounds
from code.runtime import resolve_output_root

def plot_iqr_distribution(nc_file, var='SSC', outdir='./figs'):
    ds = nc.Dataset(nc_file)

    data = ds.variables[var][:].astype(float)
    flag = ds.variables[f'{var}_flag'][:]

    ds.close()

    # 只保留正值
    valid = np.isfinite(data) & (data > 0)
    data = data[valid]
    flag = flag[valid]

    log_data = np.log10(data)

    # IQR bounds
    lower, upper = compute_log_iqr_bounds(data)
    if lower is None:
        print(f'Not enough data for IQR: {nc_file}')
        return

    log_lower = np.log10(lower)
    log_upper = np.log10(upper)

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6, 4))

    plt.hist(log_data, bins=40, color='lightgray', edgecolor='k')

    plt.axvline(log_lower, color='red', linestyle='--', label='IQR lower')
    plt.axvline(log_upper, color='red', linestyle='--', label='IQR upper')

    plt.xlabel(f'log10({var})')
    plt.ylabel('Count')
    plt.title(f'{var} log-IQR screening')

    plt.legend()
    plt.tight_layout()

    outpng = os.path.join(
        outdir,
        f'{os.path.basename(nc_file)}_{var}_IQR.png'
    )
    plt.savefig(outpng, dpi=200)
    plt.close()

    print(f'Saved: {outpng}')

if __name__ == '__main__':
    nc_file = resolve_output_root(start=__file__) / 'daily' / 'Bayern' / 'qc' / 'Bayern_10026293.nc'
    plot_iqr_distribution(nc_file, var='SSC', outdir='./figs')
