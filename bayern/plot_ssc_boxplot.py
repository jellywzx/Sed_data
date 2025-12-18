import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_ssc_boxplot(nc_file, outdir='./figs'):
    ds = nc.Dataset(nc_file)

    SSC = ds.variables['SSC'][:].astype(float)
    flag = ds.variables['SSC_flag'][:]

    ds.close()

    # 只保留正值
    valid = np.isfinite(SSC) & (SSC > 0)
    SSC = SSC[valid]
    flag = flag[valid]

    log_ssc = np.log10(SSC)

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(3, 5))

    plt.boxplot(
        log_ssc,
        vert=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker='o', markersize=4, markerfacecolor='orange')
    )

    plt.ylabel('log10(SSC) [mg L⁻¹]')
    plt.title('SSC log-IQR boxplot')

    outpng = os.path.join(
        outdir,
        f'{os.path.basename(nc_file)}_SSC_boxplot.png'
    )
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

    print(f'Saved: {outpng}')

if __name__ == '__main__':
    nc_file = '/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r/daily/Bayern/qc/Bayern_10026293.nc'  # replace with your netCDF file
    plot_ssc_boxplot(nc_file, outdir='./figs')



