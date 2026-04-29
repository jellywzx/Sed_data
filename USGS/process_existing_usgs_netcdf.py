
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from code.constants import FILL_VALUE_FLOAT, FILL_VALUE_INT

def get_flag(value, thresholds, meanings):
    if pd.isna(value):
        return FILL_VALUE_INT
    if value < thresholds.get('negative', -float('inf')):
        return np.int8(3)
    if value == thresholds.get('zero', -1):
        return np.int8(2)
    if value > thresholds.get('extreme', float('inf')):
        return np.int8(0)
    return np.int8(0)

def process_existing_usgs_netcdf():
    input_dir = Path("/Users/zhongwangwei/Downloads/Sediment/Output/daily/USGS")
    output_dir = Path("/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/USGS")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_station_summary_data = []
    nc_files = sorted(list(input_dir.glob('*.nc')))

    for nc_file in nc_files:
        station_id = nc_file.stem.split('_')[1]
        print(f"Processing station {station_id} from {nc_file.name}...")

        try:
            ds = xr.open_dataset(nc_file)
        except Exception as e:
            print(f"  Skipping station {station_id}: could not open NetCDF file. Error: {e}")
            continue

        df = ds.to_dataframe()

        # Time cropping
        valid_data = df.dropna(subset=['discharge', 'ssc'], how='all')
        if valid_data.empty:
            print(f"  Skipping station {station_id}: No valid data.")
            continue
        start_date = valid_data.index.min()
        end_date = valid_data.index.max()
        date_index = pd.date_range(start=f"{start_date.year}-01-01", end=f"{end_date.year}-12-31", freq='D')
        df = df.reindex(date_index)

        # Correct SSL calculation
        if 'discharge' in df and 'ssc' in df:
            df['SSL'] = df['discharge'] * df['ssc'] * 0.0864
        else:
            df['SSL'] = np.nan

        # QC Flags
        q_thresholds = {'negative': 0, 'zero': 0, 'extreme': 50000}
        ssc_thresholds = {'negative': 0, 'extreme': 3000}
        ssl_thresholds = {'negative': 0}
        flag_meanings = "good_data suspect_data bad_data missing_data"

        if 'discharge' in df:
            df['Q_flag'] = df['discharge'].apply(lambda x: get_flag(x, q_thresholds, flag_meanings))
        if 'ssc' in df:
            df['SSC_flag'] = df['ssc'].apply(lambda x: get_flag(x, ssc_thresholds, flag_meanings))
        if 'SSL' in df:
            df['SSL_flag'] = df['SSL'].apply(lambda x: get_flag(x, ssl_thresholds, flag_meanings))

        # Create new xarray Dataset
        new_ds = xr.Dataset()
        new_ds['time'] = ('time', df.index)

        variables = {
            'Q': {'name': 'discharge', 'units': 'm3 s-1', 'long_name': 'River Discharge', 'standard_name': 'river_discharge'},
            'SSC': {'name': 'ssc', 'units': 'mg L-1', 'long_name': 'Suspended Sediment Concentration', 'standard_name': 'mass_concentration_of_suspended_matter_in_water'},
            'SSL': {'name': 'SSL', 'units': 'ton day-1', 'long_name': 'Suspended Sediment Load', 'standard_name': 'load_of_suspended_matter'},
        }

        for var_key, attrs in variables.items():
            var_name = attrs['name']
            if var_name in df.columns:
                new_ds[var_key] = ('time', df[var_name].astype(np.float32).values)
                new_ds[var_key].attrs = {
                    'long_name': attrs['long_name'], 'standard_name': attrs['standard_name'], 'units': attrs['units'],
                    '_FillValue': FILL_VALUE_FLOAT, 'ancillary_variables': f'{var_key}_flag',
                    'comment': "Source: Original data from USGS. Calculated if applicable."
                }
                if f'{var_key}_flag' in df.columns:
                    new_ds[f'{var_key}_flag'] = ('time', df[f'{var_key}_flag'].astype(np.byte).values)
                    new_ds[f'{var_key}_flag'].attrs = {
                        'long_name': f'Quality flag for {attrs["long_name"]}', '_FillValue': FILL_VALUE_INT,
                        'flag_values': np.array([0, 2, 3, 9], dtype=np.byte), 'flag_meanings': flag_meanings,
                        'comment': "Flag definitions: 0=good_data, 2=suspect_data, 3=bad_data, 9=missing_data"
                    }

        # Coordinates and other metadata
        def _metadata_or_fill(source_ds, name):
            if name not in source_ds:
                return FILL_VALUE_FLOAT
            value = source_ds[name].item()
            return value if np.isfinite(value) else FILL_VALUE_FLOAT

        new_ds['lat'] = ((), ds.latitude.item())
        new_ds['lon'] = ((), ds.longitude.item())
        new_ds['altitude'] = ((), _metadata_or_fill(ds, 'altitude'), {'_FillValue': FILL_VALUE_FLOAT})
        new_ds['upstream_area'] = ((), _metadata_or_fill(ds, 'upstream_area'), {'_FillValue': FILL_VALUE_FLOAT})

        # Global attributes
        new_ds.attrs = {
            'title': 'Harmonized Global River Discharge and Sediment',
            'Data_Source_Name': 'USGS NWIS',
            'station_name': ds.attrs.get('station_name', 'N/A'),
            'river_name': ds.attrs.get('river_name', 'N/A'),
            'Source_ID': station_id,
            'Type': 'In-situ station data',
            'Temporal_Resolution': 'daily',
            'Temporal_Span': f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
            'Geographic_Coverage': f'{ds.attrs.get("state", "N/A")}, USA',
            'Variables_Provided': ', '.join([var for var in ['Q', 'SSC', 'SSL'] if var in new_ds.variables]),
            'Reference': 'https://waterdata.usgs.gov/nwis',
            'summary': 'This dataset contains daily river discharge and suspended sediment data from the USGS National Water Information System (NWIS).',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
            'Conventions': 'CF-1.8, ACDD-1.3',
            'history': f'{ds.attrs.get("history", "")}; Processed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        }

        # Save to NetCDF
        output_file = output_dir / f'USGS_{station_id}.nc'
        new_ds.to_netcdf(output_file, format='NETCDF4', encoding={'time': {'units': f'days since {start_date.year}-01-01'}})

        # Summary for CSV
        for var_key in ['Q', 'SSC', 'SSL']:
            if f'{var_key}_flag' in df.columns:
                good_data = df[df[f'{var_key}_flag'] == 0]
                if not good_data.empty:
                    all_station_summary_data.append({
                        'Source_ID': station_id,
                        'station_name': new_ds.attrs['station_name'],
                        'river_name': new_ds.attrs['river_name'],
                        'longitude': new_ds.lon.item(),
                        'latitude': new_ds.lat.item(),
                        'altitude': new_ds.altitude.item(),
                        'upstream_area': new_ds.upstream_area.item(),
                        'Variable': var_key,
                        'Start_Date': good_data.index.min().strftime('%Y-%m-%d'),
                        'End_Date': good_data.index.max().strftime('%Y-%m-%d'),
                        'Percent_Complete': 100 * len(good_data) / len(df.loc[good_data.index.min():good_data.index.max()]),
                        'Mean': good_data[variables[var_key]['name']].mean(),
                        'Median': good_data[variables[var_key]['name']].median(),
                        'Range': f"{good_data[variables[var_key]['name']].min()} - {good_data[variables[var_key]['name']].max()}"
                    })
        ds.close()

    # Create and save summary CSV
    summary_df = pd.DataFrame(all_station_summary_data)
    summary_df.to_csv(output_dir / 'USGS_station_summary.csv', index=False)

    print("Processing complete.")

if __name__ == '__main__':
    process_existing_usgs_netcdf()
