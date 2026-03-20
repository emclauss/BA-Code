#function_donut
import intake
import numpy as np
import matplotlib.pylab as plt
import xarray as xr
import matplotlib.colors as colors
from shapely.geometry import Polygon, Point, box
from shapely import contains_xy
import time
import shapely
from shapely.affinity import scale
 
def dif_mean_calculation(eddy, sst, dt, var, n_min):
    """function for calculating difference in mean in/outside of eddy and creating new dataset with eddy_data, sst_data and var_data"""
    
    npoints_ed = f"{var}_npoints_ed"
    npoints_box = f"{var}_npoints_donut"
    
    #selecting eddies with npoints > n_min (drops any NaNs from dt)
    dt_n_min = dt.where((dt[npoints_ed] > n_min) & (dt[npoints_box] > n_min), drop = True)

    #calculate difference in mean and add it to dataset
    dt_n_min[f"dif_{var}"] = dt_n_min[f"{var}_mean_ed"] - dt_n_min[f"{var}_mean_donut"]
    
    #selecting eddies and sst with same ID as dt_n_min
    id_dt_n_min = dt_n_min["ID"].values
    eddy_n_min = eddy.where(eddy["ID"].isin(id_dt_n_min), drop = True)
    sst_n_min = sst.where(sst["ID"].isin(id_dt_n_min), drop = True)

    #create new dataset with eddy_data, var_data and sst_data
    eddy_n_min = eddy_n_min.set_index(obs="ID")
    dt_n_min = dt_n_min.set_index(obs="ID")
    sst_n_min = sst_n_min.set_index(obs="ID")

    # Align all three datasets on the same ID values
    eddy_var_n_min = xr.merge([eddy_n_min, dt_n_min, sst_n_min], join="inner")

    return(eddy_var_n_min)

















    
