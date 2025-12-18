import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_station_qc_timeseries(nc_file, var='SSC', outdir='./figs'):
    ds = nc.Dataset(nc_file)

    time = nc.num2date(ds.variables['time'][:],
                       ds.variables['time'].units,
                       ds.variables['time'].calendar)
    
    time = pd.to_datetime(
    [t.strftime('%Y-%m-%d') for t in time])

    data = ds.variables[var][:]
    flag = ds.variables[f'{var}_flag'][:]

    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(12, 4))

    # good
    idx_good = flag == 0
    plt.scatter(np.array(time)[idx_good], data[idx_good],
                s=6, c='black', label='good')

    # suspect
    idx_sus = flag == 2
    plt.scatter(np.array(time)[idx_sus], data[idx_sus],
                s=10, c='orange', label='suspect')

    plt.yscale('log')
    plt.ylabel(var)
    plt.xlabel('Time')
    plt.legend()
    plt.title(f'{var} QC time series')

    outpng = os.path.join(outdir, f'{os.path.basename(nc_file)}_{var}_QC.png')
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

    print(f'Saved: {outpng}')

if __name__ == '__main__':
    nc_file = '/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r/daily/Bayern/qc/Bayern_10026293.nc'  # replace with your netCDF file
    plot_station_qc_timeseries(nc_file, var='SSC', outdir='./figs')

