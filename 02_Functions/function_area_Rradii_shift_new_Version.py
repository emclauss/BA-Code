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

def eddyR_donut_sst(ocean_data, eddy_data, ex_out, ex_in, R):
    """Function for calculating sst in/outside of eddy with 2 radii for eddy area and donut around eddy centre"""
    
    def weighted_mean(x, w):
        mask = np.isfinite(x) & np.isfinite(w)
        if mask.sum() == 0:
            return np.nan, 0
        return np.average(x[mask], weights=w[mask]), mask.sum()

    def dlon(lon, lon0):
        return ((lon - lon0 + 180) % 360) - 180

   
    #starting timer
    start_time = time.time()
        
    #select variable sst and conc for ocean
    ocean_data_sst = ocean_data["to"].squeeze() 

    #creating new data.frame with only the effective_contour_lat/lon and time of the eddies
    dt_contour = xr.Dataset(
        {
            "contour_lat": eddy_data["effective_contour_latitude"],
            "contour_lon": eddy_data["effective_contour_longitude"],
            "time" : eddy_data["time"],
            "radius" : eddy_data["effective_radius"],
            "lat" : eddy_data["latitude"],
            "lon" : eddy_data["longitude"],
            "ID" : eddy_data["ID"]
        }
    )
    
    n_obs = dt_contour.sizes["obs"]
        
    #changing radius in m to lat/lon, using simplification 
    # 1° lat = 111.32 km = 111320 m
    # 1° lon = 111.32 km * cos(lat of eddy cetnre) = 111320 m * cos(lat of eddy centre)
        
    radius_lat = dt_contour["radius"].values / 111320
    radius_lon = dt_contour["radius"].values /(111320 * np.cos(np.deg2rad(dt_contour["lat"].values)))
        
     #add new radius_lat/lon to dt_contour
    dt_contour = dt_contour.assign(
        radius_lat=("obs", radius_lat),
        radius_lon=("obs", radius_lon))
    
    # Ocean grid
    lat_vals = ocean_data_sst["lat"].values
    lon_vals = ocean_data_sst["lon"].values
    
    # Eddy properties
    eddy_lat = dt_contour["lat"].values
    eddy_lon = dt_contour["lon"].values
    eddy_time = dt_contour["time"].values
    eddy_radiusR_lat = dt_contour["radius_lat"].values * R
    eddy_radiusR_lon = dt_contour["radius_lon"].values * R
    donut_in_lat = dt_contour["radius_lat"].values * ex_in
    donut_in_lon = dt_contour["radius_lon"].values * ex_in
    donut_out_lat = dt_contour["radius_lat"].values * ex_out
    donut_out_lon = dt_contour["radius_lon"].values * ex_out

    
   # Results
    mean_arr_ed = np.full(n_obs, np.nan)
    npoints_arr_ed = np.zeros(n_obs, dtype=int)

    mean_arr_dn = np.full(n_obs, np.nan)
    npoints_arr_dn = np.zeros(n_obs, dtype=int)
    
    # Process eddies grouped by unique time
    unique_times = np.unique(eddy_time)
    
    for t in unique_times:
        # Select ocean data for this time only
        ocean_data_time = ocean_data_sst.sel(time=t, method="nearest").values
    
        # Indices of eddies at this time
        idx_time = np.where(eddy_time == t)[0]
    
        for i in idx_time:
            lat0, lon0 = eddy_lat[i], eddy_lon[i]
            ed_rad_lat_i, ed_rad_lon_i = eddy_radiusR_lat[i], eddy_radiusR_lon[i]
            dn_in_lat_i, dn_in_lon_i = donut_in_lat[i], donut_in_lon[i]
            dn_out_lat_i, dn_out_lon_i = donut_out_lat[i], donut_out_lon[i]
            
            # Local window around eddy
            lat_mask = (lat_vals >= lat0 - dn_out_lat_i) & (lat_vals <= lat0 + dn_out_lat_i)
            dlon_vals = ((lon_vals - lon0 + 180) % 360) - 180
            lon_mask = np.abs(dlon_vals) <= dn_out_lon_i
            lat_sub = lat_vals[lat_mask]
            lon_sub = lon_vals[lon_mask]
    
            lon_grid_sub, lat_grid_sub = np.meshgrid(lon_sub, lat_sub)
    
            # Ellipse mask eddy area (2 eddy radii)
            mask_eddy = (dlon(lon_grid_sub, lon0)/ed_rad_lon_i)**2 + ((lat_grid_sub - lat0)/ed_rad_lat_i)**2 <= 1

            # Ellipse mask donut in area (ex_in eddy radii)
            mask_donut_in = (dlon(lon_grid_sub, lon0)/dn_in_lon_i)**2 + ((lat_grid_sub - lat0)/dn_in_lat_i)**2 <= 1

            # Ellipse mask donut in area (ex_in eddy radii)
            mask_donut_out = (dlon(lon_grid_sub, lon0)/dn_out_lon_i)**2 + ((lat_grid_sub - lat0)/dn_out_lat_i)**2 <= 1

            #donut mask
            mask_donut = mask_donut_out & (~ mask_donut_in)

            eddy_masked = ocean_data_time[lat_mask, :][:, lon_mask][mask_eddy]
            donut_masked = ocean_data_time[lat_mask, :][:, lon_mask][mask_donut]

            #create weight for weighted mean (eddy)
            lat_masked_eddy = lat_grid_sub[mask_eddy]
            weights_eddy = np.cos(np.deg2rad(lat_masked_eddy))

            #create weight for weighted mean (donut)
            lat_masked_donut = lat_grid_sub[mask_donut]
            weights_donut = np.cos(np.deg2rad(lat_masked_donut))

            if eddy_masked.size > 0:
                mean_ed, n_points_ed = weighted_mean(eddy_masked, weights_eddy)
            
                mean_arr_ed[i] = mean_ed
                npoints_arr_ed[i] = n_points_ed


            if donut_masked.size > 0:
                mean_dn, n_points_dn = weighted_mean(donut_masked, weights_donut)
                
                mean_arr_dn[i] = mean_dn
                npoints_arr_dn[i] = n_points_dn


    #build dataset
    dt_masked = xr.Dataset({
        "sst_mean_ed": ("obs", mean_arr_ed),
        "sst_npoints_ed": ("obs", npoints_arr_ed),
    
        "sst_mean_donut": ("obs", mean_arr_dn),
        "sst_npoints_donut": ("obs", npoints_arr_dn),
    
        "dif_sst": ("obs", mean_arr_ed - mean_arr_dn),

        "ID": ("obs", dt_contour["ID"].values[:n_obs])
    }, coords={"obs": np.arange(n_obs)})
 

    dt_masked["dif_sst"].attrs.update({
    "comment": "difference between sst_mean in eddy center and sst_mean outside eddy",
    "units": "°C"})

    return dt_masked
        
def eddyR_donut_shift(atmos_data, eddy_data, var, ex_out, ex_in, shift, R):
    """Function for calculating sst in/outside of eddy with 2 radii for eddy area and donut around eddy centre"""

    def weighted_mean(x, w):
        mask = np.isfinite(x) & np.isfinite(w)
        if mask.sum() == 0:
            return np.nan, 0
        return np.average(x[mask], weights=w[mask]), mask.sum()

    def dlon(lon, lon0):
        return ((lon - lon0 + 180) % 360) - 180


    
    #starting timer
    start_time = time.time()
        
    #select variable var for atmos
    atmos_data_var = atmos_data[var]

    #creating new data.frame with only the effective_contour_lat/lon and time of the eddies
    dt_contour = xr.Dataset(
        {
            "contour_lat": eddy_data["effective_contour_latitude"],
            "contour_lon": eddy_data["effective_contour_longitude"],
            "time" : eddy_data["time"],
            "radius" : eddy_data["effective_radius"],
            "lat" : eddy_data["latitude"],
            "lon" : eddy_data["longitude"],
            "ID" : eddy_data["ID"]
        }
    )
    
    n_obs = dt_contour.sizes["obs"]
        
    #changing radius in m to lat/lon, using simplification 
    # 1° lat = 111.32 km = 111320 m
    # 1° lon = 111.32 km * cos(lat of eddy cetnre) = 111320 m * cos(lat of eddy centre)
        
    radius_lat = dt_contour["radius"].values / 111320
    radius_lon = dt_contour["radius"].values /(111320 * np.cos(np.deg2rad(dt_contour["lat"].values)))
        
     #add new radius_lat/lon to dt_contour
    dt_contour = dt_contour.assign(
        radius_lat=("obs", radius_lat),
        radius_lon=("obs", radius_lon))
    
    # atmos grid
    lat_vals = atmos_data_var["lat"].values
    lon_vals = atmos_data_var["lon"].values
    
    # Eddy properties
    eddy_lat = dt_contour["lat"].values
    eddy_lon = dt_contour["lon"].values
    eddy_time = dt_contour["time"].values
    eddy_radius_lon = dt_contour["radius_lon"].values
    eddy_radiusR_lat = dt_contour["radius_lat"].values * R
    eddy_radiusR_lon = dt_contour["radius_lon"].values * R
    donut_in_lat = dt_contour["radius_lat"].values * ex_in
    donut_in_lon = dt_contour["radius_lon"].values * ex_in
    donut_out_lat = dt_contour["radius_lat"].values * ex_out
    donut_out_lon = dt_contour["radius_lon"].values * ex_out

    
    # Results
    mean_arr_ed = np.full(n_obs, np.nan)
    npoints_arr_ed = np.zeros(n_obs, dtype=int)

    mean_arr_dn = np.full(n_obs, np.nan)
    npoints_arr_dn = np.zeros(n_obs, dtype=int)
    
    # Process eddies grouped by unique time
    unique_times = np.unique(eddy_time)
    
    for t in unique_times:

        # Select ocean data for this time only
        atmos_data_time = atmos_data_var.sel(time=t, method="nearest").values
        
        # Indices of eddies at this time
        idx_time = np.where(eddy_time == t)[0]
    
        for i in idx_time:
            lat0, lon0 = eddy_lat[i], eddy_lon[i] + shift*eddy_radius_lon[i]
            ed_rad_lat_i, ed_rad_lon_i = eddy_radiusR_lat[i], eddy_radiusR_lon[i]
            dn_in_lat_i, dn_in_lon_i = donut_in_lat[i], donut_in_lon[i]
            dn_out_lat_i, dn_out_lon_i = donut_out_lat[i], donut_out_lon[i]
            
            # Local window around eddy
            lat_mask = (lat_vals >= lat0 - dn_out_lat_i) & (lat_vals <= lat0 + dn_out_lat_i)
            dlon_vals = ((lon_vals - lon0 + 180) % 360) - 180
            lon_mask = np.abs(dlon_vals) <= dn_out_lon_i
            lat_sub = lat_vals[lat_mask]
            lon_sub = lon_vals[lon_mask]
    
            lon_grid_sub, lat_grid_sub = np.meshgrid(lon_sub, lat_sub)
    
            # Ellipse mask eddy area (2 eddy radii)
            mask_eddy = (dlon(lon_grid_sub, lon0)/ed_rad_lon_i)**2 + ((lat_grid_sub - lat0)/ed_rad_lat_i)**2 <= 1

            # Ellipse mask donut in area (ex_in eddy radii)
            mask_donut_in = (dlon(lon_grid_sub, lon0)/dn_in_lon_i)**2 + ((lat_grid_sub - lat0)/dn_in_lat_i)**2 <= 1

            # Ellipse mask donut in area (ex_in eddy radii)
            mask_donut_out = (dlon(lon_grid_sub, lon0)/dn_out_lon_i)**2 + ((lat_grid_sub - lat0)/dn_out_lat_i)**2 <= 1

            #donut mask
            mask_donut = mask_donut_out & (~ mask_donut_in)

            eddy_masked = atmos_data_time[lat_mask, :][:, lon_mask][mask_eddy]
            donut_masked = atmos_data_time[lat_mask, :][:, lon_mask][mask_donut]

            #create weight for weighted mean (eddy)
            lat_masked_eddy = lat_grid_sub[mask_eddy]
            weights_eddy = np.cos(np.deg2rad(lat_masked_eddy))

            #create weight for weighted mean (donut)
            lat_masked_donut = lat_grid_sub[mask_donut]
            weights_donut = np.cos(np.deg2rad(lat_masked_donut))


            if eddy_masked.size > 0:
                mean_ed, n_points_ed = weighted_mean(eddy_masked, weights_eddy)
            
                mean_arr_ed[i] = mean_ed
                npoints_arr_ed[i] = n_points_ed
            
            if donut_masked.size > 0:
                mean_dn, n_points_dn = weighted_mean(donut_masked, weights_donut)
            
                mean_arr_dn[i] = mean_dn
                npoints_arr_dn[i] = n_points_dn



    #build dataset
    dt_masked = xr.Dataset({
        f"{var}_mean_ed" : ("obs", mean_arr_ed),
        f"{var}_npoints_ed" : ("obs", npoints_arr_ed),
        
        f"{var}_mean_donut" : ("obs", mean_arr_dn),
        f"{var}_npoints_donut" : ("obs", npoints_arr_dn),
        "ID" : ("obs", dt_contour["ID"].values[:n_obs])},
        coords = {"obs" : np.arange(n_obs)}) 

    return dt_masked

def eddyR_donut_shift_var(atmos_data, eddy_data, var, ex_out, ex_in, shift, R):
    """Function for calculating sst in/outside of eddy with 2 radii for eddy area and donut around eddy centre"""
    
    def weighted_mean(x, w):
        mask = np.isfinite(x) & np.isfinite(w)
        if mask.sum() == 0:
            return np.nan, 0
        return np.average(x[mask], weights=w[mask]), mask.sum()

    def dlon(lon, lon0):
        return ((lon - lon0 + 180) % 360) - 180

    
    #starting timer
    start_time = time.time()
        
    #select variable var for atmos
    atmos_data_var = atmos_data

    #creating new data.frame with only the effective_contour_lat/lon and time of the eddies
    dt_contour = xr.Dataset(
        {
            "contour_lat": eddy_data["effective_contour_latitude"],
            "contour_lon": eddy_data["effective_contour_longitude"],
            "time" : eddy_data["time"],
            "radius" : eddy_data["effective_radius"],
            "lat" : eddy_data["latitude"],
            "lon" : eddy_data["longitude"],
            "ID" : eddy_data["ID"]
        }
    )
    
    n_obs = dt_contour.sizes["obs"]
        
    #changing radius in m to lat/lon, using simplification 
    # 1° lat = 111.32 km = 111320 m
    # 1° lon = 111.32 km * cos(lat of eddy cetnre) = 111320 m * cos(lat of eddy centre)
        
    radius_lat = dt_contour["radius"].values / 111320
    radius_lon = dt_contour["radius"].values /(111320 * np.cos(np.deg2rad(dt_contour["lat"].values)))
        
     #add new radius_lat/lon to dt_contour
    dt_contour = dt_contour.assign(
        radius_lat=("obs", radius_lat),
        radius_lon=("obs", radius_lon))
    
    # atmos grid
    lat_vals = atmos_data_var["lat"].values
    lon_vals = atmos_data_var["lon"].values
    
    # Eddy properties
    eddy_lat = dt_contour["lat"].values
    eddy_lon = dt_contour["lon"].values
    eddy_time = dt_contour["time"].values
    eddy_radius_lon = dt_contour["radius_lon"].values
    eddy_radiusR_lat = dt_contour["radius_lat"].values * R
    eddy_radiusR_lon = dt_contour["radius_lon"].values * R
    donut_in_lat = dt_contour["radius_lat"].values * ex_in
    donut_in_lon = dt_contour["radius_lon"].values * ex_in
    donut_out_lat = dt_contour["radius_lat"].values * ex_out
    donut_out_lon = dt_contour["radius_lon"].values * ex_out

    
    # Results
    mean_arr_ed = np.full(n_obs, np.nan)
    npoints_arr_ed = np.zeros(n_obs, dtype=int)

    mean_arr_dn = np.full(n_obs, np.nan)
    npoints_arr_dn = np.zeros(n_obs, dtype=int)
    
    # Process eddies grouped by unique time
    unique_times = np.unique(eddy_time)
    
    for t in unique_times:

        # Select ocean data for this time only
        atmos_data_time = atmos_data_var.sel(time=t, method="nearest").values
    
        # Indices of eddies at this time
        idx_time = np.where(eddy_time == t)[0]
    
        for i in idx_time:
            lat0, lon0 = eddy_lat[i], eddy_lon[i] + shift*eddy_radius_lon[i]
            ed_rad_lat_i, ed_rad_lon_i = eddy_radiusR_lat[i], eddy_radiusR_lon[i]
            dn_in_lat_i, dn_in_lon_i = donut_in_lat[i], donut_in_lon[i]
            dn_out_lat_i, dn_out_lon_i = donut_out_lat[i], donut_out_lon[i]
            
            # Local window around eddy
            lat_mask = (lat_vals >= lat0 - dn_out_lat_i) & (lat_vals <= lat0 + dn_out_lat_i)
            dlon_vals = ((lon_vals - lon0 + 180) % 360) - 180
            lon_mask = np.abs(dlon_vals) <= dn_out_lon_i
            lat_sub = lat_vals[lat_mask]
            lon_sub = lon_vals[lon_mask]
    
            lon_grid_sub, lat_grid_sub = np.meshgrid(lon_sub, lat_sub)
    
            # Ellipse mask eddy area (2 eddy radii)
            mask_eddy = (dlon(lon_grid_sub, lon0)/ed_rad_lon_i)**2 + ((lat_grid_sub - lat0)/ed_rad_lat_i)**2 <= 1

            # Ellipse mask donut in area (ex_in eddy radii)
            mask_donut_in = (dlon(lon_grid_sub, lon0)/dn_in_lon_i)**2 + ((lat_grid_sub - lat0)/dn_in_lat_i)**2 <= 1

            # Ellipse mask donut in area (ex_in eddy radii)
            mask_donut_out = (dlon(lon_grid_sub, lon0)/dn_out_lon_i)**2 + ((lat_grid_sub - lat0)/dn_out_lat_i)**2 <= 1

            #donut mask
            mask_donut = mask_donut_out & (~ mask_donut_in)

            eddy_masked = atmos_data_time[lat_mask, :][:, lon_mask][mask_eddy]
            donut_masked = atmos_data_time[lat_mask, :][:, lon_mask][mask_donut]

            #create weight for weighted mean (eddy)
            lat_masked_eddy = lat_grid_sub[mask_eddy]
            weights_eddy = np.cos(np.deg2rad(lat_masked_eddy))

            #create weight for weighted mean (donut)
            lat_masked_donut = lat_grid_sub[mask_donut]
            weights_donut = np.cos(np.deg2rad(lat_masked_donut))


            if eddy_masked.size > 0:
                mean_ed, n_points_ed = weighted_mean(eddy_masked, weights_eddy)
            
                mean_arr_ed[i] = mean_ed
                npoints_arr_ed[i] = n_points_ed
            
            if donut_masked.size > 0:
                mean_dn, n_points_dn = weighted_mean(donut_masked, weights_donut)
            
                mean_arr_dn[i] = mean_dn
                npoints_arr_dn[i] = n_points_dn



    #build dataset
    dt_masked = xr.Dataset({
        f"{var}_mean_ed" : ("obs", mean_arr_ed),
        f"{var}_npoints_ed" : ("obs", npoints_arr_ed),
        
        f"{var}_mean_donut" : ("obs", mean_arr_dn),
        f"{var}_npoints_donut" : ("obs", npoints_arr_dn),
        "ID" : ("obs", dt_contour["ID"].values[:n_obs])},
        coords = {"obs" : np.arange(n_obs)}) 

    return dt_masked
