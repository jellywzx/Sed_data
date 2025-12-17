# core/cf_writer.py
from netCDF4 import Dataset
import numpy as np
from datetime import datetime


def write_station_netcdf(
    filepath,
    dimensions,
    variables,
    global_attrs,
):
    """
    Fully generic CF/ACDD NetCDF writer.

    Parameters
    ----------
    filepath : str
    dimensions : dict
        e.g. {"time": None}
    variables : dict
        key = variable name
        value = {
            "dtype": "f4" / "i1" / ...
            "dims": ("time",) or ()
            "fill_value": -9999
            "data": scalar or array
            "attrs": {attribute_name: attribute_value}
        }
    global_attrs : dict
        global NetCDF attributes
    """

    nc = Dataset(filepath, "w", format="NETCDF4")

    # -------------------------
    # Dimensions
    # -------------------------
    for dim, size in dimensions.items():
        nc.createDimension(dim, size)

    # -------------------------
    # Variables
    # -------------------------
    for varname, meta in variables.items():
        var = nc.createVariable(
            varname,
            meta["dtype"],
            meta.get("dims", ()),
            fill_value=meta.get("fill_value"),
        )

        # write data
        if "data" in meta and meta["data"] is not None:
            var[...] = meta["data"]

        # write attributes
        for k, v in meta.get("attrs", {}).items():
            setattr(var, k, v)

    # -------------------------
    # Global attributes
    # -------------------------
    for k, v in global_attrs.items():
        setattr(nc, k, v)

    # minimal provenance
    nc.history = (
        global_attrs.get("history", "")
        + f"\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC: "
        "written by generic cf_writer"
    )

    nc.close()
