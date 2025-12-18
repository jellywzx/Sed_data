import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress


def plot_ssc_q(nc_file, outdir='./figs'):
    ds = nc.Dataset(nc_file)

    Q = ds.variables['Q'][:]
    SSC = ds.variables['SSC'][:]
    flag = ds.variables['SSC_flag'][:]

    ds.close()

    # 去掉 fill value 和非正值（log 轴必须）
    valid = (Q > 0) & (SSC > 0)

    Q = Q[valid]
    SSC = SSC[valid]
    flag = flag[valid]
    
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(5, 5))

    # good
    idx_good = flag == 0
    plt.scatter(Q[idx_good], SSC[idx_good],
                s=10, c='black', alpha=0.5, label='good')

    # suspect
    idx_sus = flag == 2
    plt.scatter(Q[idx_sus], SSC[idx_sus],
                s=20, c='orange', alpha=0.8, label='suspect')

    logQ = np.log10(Q[idx_good])
    logSSC = np.log10(SSC[idx_good])

    slope, intercept, r, _, _ = linregress(logQ, logSSC)

    Q_fit = np.logspace(np.log10(Q.min()), np.log10(Q.max()), 100)
    SSC_fit = 10**intercept * Q_fit**slope

    plt.plot(Q_fit, SSC_fit, 'r--',
         label=f'SSC ∝ Q^{slope:.2f}, R²={r**2:.2f}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Discharge Q (m³ s⁻¹)')
    plt.ylabel('Suspended sediment concentration (mg L⁻¹)')
    plt.legend()
    plt.title('SSC–Q relationship with QC flags')

    outpng = os.path.join(
        outdir,
        f'{os.path.basename(nc_file)}_SSC_Q.png'
    )
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

    print(f'Saved: {outpng}')

if __name__ == '__main__':
    nc_file = '/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r/daily/Bayern/qc/Bayern_10026293.nc'  # replace with your netCDF file
    plot_ssc_q(nc_file, outdir='./figs')


