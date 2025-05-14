import pandas as pd
import numpy as np
import earthaccess
import xarray as xr
import earthaccess
from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import os
import requests
import sys
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from cartopy.feature import NaturalEarthFeature
from shapely.geometry import Polygon, LineString
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import calendar
sys.path.append("../../NSO/NSO_data_processing")
from Functions_general import *



sys.path.append("../satellite_data")
sys.path.append("../buoy_data")

from Read_satellites import read_ssa


from Functions_Oracle import *






#%%  Read datasets



# calc_new_sat = 0


# # calculate a new combined satellite dataset
# if calc_new_sat == 1:
    
#     # Open NBS sea wind data
#     save_path = "../satellite_data/NBS_winds"
#     files = sorted(glob.glob(save_path +'/NBS*wind_stress*20*.nc'))[:]
#     nbs = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
#     nbs = nbs.where(nbs>-100)
#     nbs = nbs.sel(zlev=10)[['mask', 'x_tau', 'y_tau']].drop(['zlev'])
#     files = sorted(glob.glob(save_path +'/NBS*wind_6hourly_20*.nc'))[:5]
#     nbs_winds = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
#     nbs_winds = nbs_winds.sel(zlev=10)[['mask', 'u_wind', 'v_wind']].drop(['zlev'])
    
#     # Make daily averages (all other datasets are daily averages)    
#     nbs = nbs.resample(time='1D').mean()
#     nbs_winds = nbs_winds.resample(time='1D').mean()
    
#     # SSH/ SSA 
#     save_path = "../satellite_data/SSA"
#     files = sorted(glob.glob(save_path +'/*sla*20*.nc'))[:]
#     sla = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
    
#     # Oscar current data
#     save_path = 'C:/Users/uegerer/Desktop/Oracle/satellite_data/Oscar/'
#     files = sorted(glob.glob(save_path +'*Oscar_*.nc'))[:-1]
#     oscar = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
#     oscar = oscar.sel(time=slice(nbs.time[0], nbs.time[-1]))
#     oscar = oscar.rename({'latitude': 'lat', 'longitude': 'lon' })
    
#     # Interpolate sla grid onto nbs grid
#     if nbs['lon'].values.max()>180:
#         nbs['lon'] = nbs['lon']-360
#     nbs = nbs.where(nbs['mask'] == 1) # filter mask 1 value (only satellite data over ocean)
#     if nbs_winds['lon'].values.max()>180:
#         nbs_winds['lon'] = nbs_winds['lon']-360
#     nbs_winds = nbs_winds.where(nbs_winds['mask'] == 1) # filter mask 1 value (only satellite data over ocean)
#     sla_interp = sla.interp(latitude=nbs['lat'], longitude=nbs['lon'], time=nbs["time"], method="linear")
#     if oscar['lon'].values.max()>180:
#         oscar['lon'] = oscar['lon']-360
#     oscar_interp = oscar.interp(lat=nbs['lat'], lon=nbs['lon'], time=nbs["time"], method="linear")
        
    
#     # Concatenate datasets along the time dimension
#     sat = xr.merge([  nbs, 
#                       nbs_winds[['u_wind', 'v_wind']], 
#                       sla_interp[['sla', 'ugos', 'vgos']].drop(['latitude', 'longitude']),
#                       oscar_interp
#                       ])
#     # sat = sat.sel(time=slice("2020-01-01", "2022-07-01"))  # this is to where I have all datasets currently.
        
#     # Add the coastline angle values to the sat dataset
#     sat = add_costline_angle_to_sat(sat, plot = 0)

    
#     # Coriolis parameter
#     sat["f"] = calculate_coriolis_parameter(sat.lat)  
    
#     # sea water density
#     rho_ocean = 1025 # kg/m3
    
#     # Calculate along-and cross shore wind stress
#     sat['tau_along'], sat['tau_cross'] = calculate_along_cross_shore_components(
#                                             sat.x_tau, sat.y_tau, coastline_angle=sat.coastline_angle)
    
#     # Ekman transport
#     sat['U_Ek'] = sat['tau_along'] / rho_ocean / sat["f"]
#     # remove outliers
#     sat['U_Ek'] = sat['U_Ek'].where(sat['U_Ek']>-10)
#     sat['U_Ek'] = sat['U_Ek'].where(sat['U_Ek']<20)
       
#     # Calculate along-and cross shore geostrophic ocean velocities from SLA
#     sat['gos_along'], sat['gos_cross'] = calculate_along_cross_shore_components(
#                                             sat.ugos, sat.vgos, coastline_angle=sat.coastline_angle)
    
#     # Calculate along-and cross shore geostrophic velocities from Oscar
#     sat['go_along'], sat['go_cross'] = calculate_along_cross_shore_components(
#                                             sat.ug, sat.vg, coastline_angle=sat.coastline_angle)
#     sat['curr_along'], sat['curr_cross'] = calculate_along_cross_shore_components(
#                                                 sat.u, sat.v, coastline_angle=sat.coastline_angle)
    
#     # Mixed layer depth (fixed depth assumtopn - first guess)
#     MLD = 10
    
#     # GEostrophic transport (positive when offshore) with mixed layer depth
#     sat["U_Geo"] = sat['gos_cross'] * MLD
#     sat["U_Geo_Oscar"] = sat['go_cross'] * 30
    
#     # Calculate CUTI from satellite data
#     sat['CUTI_SLA'] = (sat['U_Ek'] + sat['U_Geo'])
#     sat['CUTI_Oscar'] = (sat['U_Ek'] + sat['U_Geo_Oscar'])       
    
    
#     # Add geostrophic currents from Copernicus
#     save_path = "../satellite_data/Copernicus_SLA"
#     files = sorted(glob.glob(save_path +'/*sliced*20*.nc'))[:]
#     cop = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
#     cop = cop.interp(lat=sat['lat'], lon=sat['lon'], time=sat["time"])[["cop_ugos", "cop_vgos", "err_sla"]]
#     sat = xr.merge([  sat,
#                       cop
#                       ])
#     sat['cop_gos_along'], sat['cop_gos_cross'] = calculate_along_cross_shore_components(
#                                             sat.cop_ugos, sat.cop_vgos, coastline_angle=sat.coastline_angle)
    
#     # Save combined sat dataset
#     sat.to_netcdf("Sat_data_combinded_new", engine = "netcdf4")   


# # or read combined satellite data file
# else:
#    sat = xr.open_dataset("Sat_data_combinded", engine = "netcdf4")


sat = xr.open_dataset("Sat_data_combinded", engine = "netcdf4")


# CUTI data
CUTI = pd.read_csv('C:/Users/uegerer/Desktop/Oracle/CUTI_data/CUTI_daily.csv', header = 0, index_col=None).apply(pd.to_numeric, errors='coerce')
CUTI.index = pd.to_datetime(CUTI[['year', 'month', 'day']])


# Coastline
coastline = np.load("coastline.npy")


# MLD from Argo floats (interpolated data)
mld = xr.open_dataset("MLD_data", engine = "netcdf4")
mld = mld.drop_vars(['lat', 'lon', 'month']).rename({'iLAT': 'lat', 'iLON': 'lon', 'iMONTH': 'month'})
mld_humb = mld.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest")
mld_morro = mld.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest")


# MLD from ARMOR3 (sate and in-situ)
mld_sat = xr.open_dataset('../satellite_data/Copernicus_MLD/mld_sat_2000_2022.nc')
mld_sat_humb = mld_sat.sel(latitude=Humboldt_coords[0], longitude=Humboldt_coords[1], method = "nearest").to_dataframe()
mld_sat_morro = mld_sat.sel(latitude=Morro_coords[0], longitude=Morro_coords[1], method = "nearest").to_dataframe()    
 


  



#%% Improve CUTI calc with new MLD



### Use MLD from ARGO floats climatology to calculate CUTI

# # Add MLD to sat dataset - Argo
# mld_interpolated = mld.interp(lat=sat['lat'], lon=sat['lon'])
# sat['mld_argo'] =  mld_interpolated['mld_da_median'].sel(month=sat['time'].dt.month, lat=sat['lat'], lon=sat['lon'])
# # sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").mld_argo.plot()


# Add MLD to sat dataset - ARMOR3 (sat and in-situ)
mld_sat = mld_sat.rename({"latitude": "lat", "longitude": "lon"})
mld_interpolated = mld_sat.interp(lat=sat['lat'], lon=sat['lon'], time=sat['time'])
sat['mld_sat'] =  mld_interpolated.sel(time=sat['time'], lat=sat['lat'], lon=sat['lon']).mlotst
sat['month'] =  sat['time'].dt.month
# sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").mld_sat.plot()



# Geostrophic transport (positive when offshore) with new mixed layer depth
sat["U_Geo"] = sat['gos_cross'] *  sat.mld_sat
sat["U_Geo_Oscar"] = sat['go_cross'] * sat.mld_sat
sat["U_Geo_cop"] = sat['cop_gos_cross'] *  sat.mld_sat

# Calculate CUTI from satellite data
sat['CUTI_SLA'] = (sat['U_Ek'] + sat['U_Geo'])
sat['CUTI_Oscar'] = (sat['U_Ek'] + sat['U_Geo_Oscar'])
sat['CUTI_Cop'] = (sat['U_Ek'] + sat['U_Geo_cop'])

# Save combined sat dataset
sat = sat.sel(time=slice("2000-01-01", "2023-01-01"))  # current data availability
# sat.to_netcdf("Sat_data_combinded_ARGO_mld", engine = "netcdf4")   
# sat.to_netcdf("Sat_data_combinded_ARMOR_mld", engine = "netcdf4")   


# sat_humb = sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest")


# plt.figure()
# plt.plot(sat_humb.time, sat_humb.U_Geo,".-")
# plt.plot(sat_humb.time, sat_humb.U_Geo_Oscar,".-")
# plt.plot(sat_humb.time, sat_humb.U_Geo_cop,".-")
# plt.plot(sat_humb.time, sat_humb.U_Ek,".-")

# # Add day of year
# sat = sat.assign_coords(DOY=pd.to_datetime(sat['time'].values).dayofyear)






#%% Compare CUTI from satellite and model for latitude bins


lat_range = np.arange(31,45)   #np.arange(33,45) #  [40,41]
offshore_diff = 75 # 75 is the distance used in Jacox 2018


compare_CUTI_sat = 1 

if compare_CUTI_sat == 1:
    
    
    # Create a DataFrame with latitude values as rows and multiple columns for each value
    MAE = pd.DataFrame(index=lat_range, columns=["CUTI_Oscar", "CUTI_SLA", "CUTI_Cop", "U_Ek"])
    
    
    num_lats = len(lat_range)-1
    num_rows = int(num_lats**0.5)
    num_cols = (num_lats + num_rows - 1) // num_rows
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i in np.arange(0, num_lats): 
    
        ax = axes[i]
        
        lat = lat_range[i]
        
        max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0]
        min_lon = max_lon + lon_difference(offshore_diff, max_lon)
        
        
        
        # Average satellite data over 1 deg in N-S direction and up to Xkm offshore
        sat_local = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 
    
        # Merge model CUTI and satellite area average
        all = CUTI.merge(
                sat_local[["CUTI_Oscar", "CUTI_SLA", "U_Ek", "U_Geo", "U_Geo_Oscar", "CUTI_Cop"]].to_dataframe(), 
                left_index=True, right_index=True, how="inner")
        all = all.replace([np.inf, -np.inf], np.nan)

        ax.set_title("{}$^\circ$N".format(lat)) 
        ax.plot(all[str(lat)+"N"], all[str(lat)+"N"], "--", color="grey", zorder=-10, label = "")
        for parameter in ["U_Ek", "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop" ]:
            mae = mean_absolute_error(all[[str(lat)+"N", parameter]].dropna()[parameter], 
                                    all[[str(lat)+"N", parameter]].dropna()[str(lat)+"N"]).round(2)
            ax.plot(all[str(lat)+"N"], all[parameter], ".", 
                      label = parameter + ", MAE=" + 
                      str( mae )  )
            MAE[parameter][lat] = mae
        ax.set_xlabel("CUTI from model data")
        ax.set_ylabel("CUTI from sat data")
        ax.grid()
        ax.legend()
        
    plt.tight_layout()
    
    
    
    
    plt.figure()
    for parameter in ["U_Ek", "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop"]:
        plt.plot(MAE.index, MAE[parameter], label = parameter)
    plt.ylabel("MAE of CUTI")
    plt.xlabel("latitude")
    plt.legend()
    plt.grid()
    
    
    
    
#%% USe model MLD - Compare CUTI from satellite and model for Humboldt location



compare_CUTI_sat_model_mld = 0  

if compare_CUTI_sat_model_mld == 1:
    
    mld_model = pd.read_csv('MLDs_from_CCSROMS_34_119.csv', index_col='Date', parse_dates=True)
    
    lat = 34 # round(Humboldt_coords[0],0)
        
    max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0] 
    min_lon = max_lon + lon_difference(offshore_diff, max_lon)    
    
    # Average satellite data over 1 deg in N-S direction and up to Xkm offshore
    sat_local = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon'])     
    
    # Merge model CUTI and satellite area average
    all = CUTI.merge(
         sat_local[["gos_cross", "go_cross", "U_Ek", "cop_gos_cross"]].to_dataframe(), 
         left_index=True, right_index=True, how="inner")
    all = all.replace([np.inf, -np.inf], np.nan)
  
    
    # RE-calculate with model MLD
    all = all.merge(
         mld_model.Height, 
         left_index=True, right_index=True, how="inner")
    all["U_Geo"] = all['gos_cross'] *  all.Height
    all["U_Geo_Oscar"] = all['go_cross'] * all.Height
    all["U_Geo_cop"] = all['cop_gos_cross'] *  all.Height

    # Calculate CUTI from allellite data
    all['CUTI_SLA'] = (all['U_Ek'] + all['U_Geo'])
    all['CUTI_Oscar'] = (all['U_Ek'] + all['U_Geo_Oscar'])
    all['CUTI_Cop'] = (all['U_Ek'] + all['U_Geo_cop'])
    
    
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_title("Using MLD from CCS-ROMS, {}$^\circ$N".format(lat)) 
    
    ax.plot(all[str(int(lat))+"N"], all[str(int(lat))+"N"], "--", color="grey", zorder=-10, label = "")
    for parameter in ["U_Ek", "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop" ]:
        mae = mean_absolute_error(all[[str(int(lat))+"N", parameter]].dropna()[parameter], 
                                all[[str(int(lat))+"N", parameter]].dropna()[str(int(lat))+"N"]).round(2)
        ax.plot(all[str(int(lat))+"N"], all[parameter], ".", 
                 label = parameter + ", MAE=" + 
                 str( mae )  )
    ax.set_xlabel("CUTI from model data")
    ax.set_ylabel("CUTI from sat data")
    ax.grid()
    ax.legend()
        
    plt.tight_layout()
    
    
    
    
    plt.figure()
    for parameter in ["U_Ek", "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop"]:
        plt.plot(MAE.index, MAE[parameter], label = parameter)
    plt.plot(lat, mae,".", color="C3", ms=10, label = "CUTI_Cop with model MLD" )
    plt.ylabel("MAE of CUTI")
    plt.xlabel("latitude")
    plt.legend()
    plt.grid()
    
    

    
    
    
    
#%% Make lat - month - MAE plot

lat_month_MAE = 0
if  lat_month_MAE == 1:

    
    
    # Create a DataFrame with latitude values as rows and months as columns
    months = np.arange(1,13)
    MAE_CUTI = pd.DataFrame(index=lat_range, columns=months )
    parameter = "CUTI_Cop"
    
    for lat in lat_range: 
    
        max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0] 
        min_lon = max_lon + lon_difference(offshore_diff, max_lon)
      
        # Average satellite data over 1 deg in N-S direction and up to Xkm offshore
        sat_local = sat.drop_vars(["month"]).sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 
    
        # Merge model CUTI and satellite area average
        all = CUTI.merge(
                sat_local[["CUTI_Oscar", "CUTI_SLA", "U_Ek", "U_Geo", "U_Geo_Oscar", "CUTI_Cop"]].to_dataframe(), 
                left_index=True, right_index=True, how="inner")
        all = all.replace([np.inf, -np.inf],np.nan)
        
        for month in months:
            
            all_filt = all[all.month == month]
        
            mae = mean_absolute_error(all_filt[[str(lat)+"N", parameter]].dropna()[parameter], 
                                        all_filt[[str(lat)+"N", parameter]].dropna()[str(lat)+"N"]).round(2)
     
            MAE_CUTI[month][lat] = mae
    
    
    
    
    # Scatter plot   
    
         
    plt.figure()
    plt.scatter(*np.meshgrid(MAE_CUTI.columns, MAE_CUTI.index), c=MAE_CUTI.values, cmap='viridis', s=500, marker = "s", vmin=0, vmax = 5.5)
    plt.colorbar(label='MAE of CUTI estimate')
    plt.xlabel('Month')
    plt.ylabel('Latitude')
    plt.title('MAE of CUTI estimate {} across latitude and seasons'.format(parameter))
    plt.grid(False)
    plt.xticks(MAE_CUTI.columns.astype(int).values, [calendar.month_abbr[m] for m in MAE_CUTI.columns.astype(int).values]) 
    plt.tight_layout()
    

     
    
    
    












#%% Map of CUTI during July 2021 upwelling period

plot =0 
if plot == 1:
    
    # Calculate the longitude of the 100km offshore line based on latitude
    
    lon_diff = get_offshore_parallel_coastline(coastline, offshore_diff=75)
    
    
    sat_masked = sat.copy()

    # Loop through each latitude value and remove latitudes outside of the 100 km coastline
    for i, lat_sat in enumerate(sat_masked.lat.values):
        lon_sat = lon_diff[i] 
        sat_masked[[ "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop" ]].loc[dict(lat=lat_sat, lon=slice(-150, lon_sat-0.25 ) )] =np.nan
        

    # days to plot
    days = ['2020-10-22', '2020-11-07', '2022-04-01', '2022-04-10']
    days = ['2021-07-20', '2021-07-21', '2021-07-22', '2021-07-23', '2021-07-24', '2021-07-25', '2021-07-26', '2021-07-27']
    days = ['2021-07-23', '2021-07-24']
    days = ['2021-08-30', '2021-08-31']
    days = ['2020-11-01', '2020-11-02', '2020-11-03', '2020-11-04', '2020-11-05', '2020-11-06',
            '2020-11-07', '2020-11-08', '2020-11-09', '2020-11-10', '2020-11-11', '2020-11-12']
    # from datetime import datetime, timedelta
    # days = [datetime(2021, 8, 1) + timedelta(days=i*2) 
    #         for i in range((datetime(2021, 8, 31) - datetime(2021, 8, 1)).days // 2 + 1)]
    
    # SST data
    ostia = xr.load_dataset("../satellite_data/ostia_all_2000_2023.nc", engine="netcdf4")  # all coordinates close to California coast
    
    sat_masked["wind_speed"] = (sat_masked.u_wind**2 + sat_masked.v_wind**2)**0.5
    
    
    # Number of columns and rows
    num_days = len(days)
    num_rows = int(num_days**0.5)
    num_cols = (num_days + num_rows - 1) // num_rows
    
    parameter = "analysed_sst"  # "CUTI_Cop"  "U_Geo_cop"  "U_Ek"  'analysed_sst' "wind_speed"
    min_cuti = -4
    max_cuti = 4   
    
    

    # Plot
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 6 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, day in enumerate(axes):
        ax = axes[i]
        ax.axis('off')         
    
    # Iterate over days and create subplots
    for i, day in enumerate(days):
        

        ax = axes[i]
        ax = plt.subplot(num_rows, num_cols, i + 1, projection=ccrs.PlateCarree())  # Stereographic(dict(central_latitude=40, central_longitude=-125)))
        
        #fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 8))  
        
        # stations
        ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        #ax.plot(Buoy_coords[1], Buoy_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='grey',  alpha=0.9)

        # map
        ax.add_feature(cartopy.feature.BORDERS)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.LAND)
        cs = ax.coastlines(resolution='10m', linewidth=1)

                
        # Plotting the extracted coastline and a line 75km offshore
        ax.plot(coastline[:, 0], coastline[:, 1], linewidth=1, color="r") 
       # ax.plot(lon_diff, coastline[:, 1], linewidth=1, color="r")
                
        # Plot other geographic features
        states_provinces = cartopy.feature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
        ax.add_feature(states_provinces, edgecolor='black', linestyle=':')
        filename = r'../BOEM_Renewable_Energy_Shapefiles_8/Wind_Lease_Outlines_11_16_2023.shp'
        shape_feature = ShapelyFeature(Reader(filename).geometries(), ccrs.PlateCarree(), lw = 2,  facecolor='None',edgecolor = 'black')
        ax.add_feature(shape_feature)
        
        # Plot CUTI map
        df = sat_masked.sel(time= day, method='nearest')
        
        if parameter == 'analysed_sst':
             df  = ostia.sel(time= day, method='nearest')
        
        #Plot color map               
        if parameter == "analysed_sst":
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='coolwarm', transform=ccrs.PlateCarree(), shading='auto', 
                              vmin = np.nanpercentile(df[parameter].values.flatten()[df[parameter].values.flatten() != 0], 5), 
                              vmax = np.nanpercentile(df[parameter].values.flatten()[df[parameter].values.flatten() != 0], 95)) 
        elif parameter == "wind_speed":
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', 
                              vmin = 0, 
                              vmax = 16) 
        else:
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', vmin=min_cuti, vmax=max_cuti)
        
        
        # Add arrows for wind vectors 
        if parameter == "wind_speed":
            u =   df['u_wind'].values
            v =   df['v_wind'].values
            scale_factor = 200  # Adjust the scale factor for arrow length
            ax.quiver(df['lon'], df['lat'],
                      u[ :, :], v[ :, :], scale=scale_factor, color='white', transform=ccrs.PlateCarree())
        
        
        # Add colorbar
        #if i%num_cols ==1:
        cbar = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.03, shrink=0.7, label= parameter ) #'CUTI (m$^2$/s)')
                    
        gl = ax.gridlines(draw_labels=True)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False

        ax.set_extent([-128,-120, 35, 42], crs=ccrs.PlateCarree())  
        
        # ax.set_title('{:%Y-%m-%d}, \nC$_H$={}, C$_M$={}'.format(day,
        #     round(CUTI.loc[pd.to_datetime(day), '41N'], 2), 
        #     round(CUTI.loc[pd.to_datetime(day), '36N'], 2))  , fontsize=10)   
        ax.set_title(day)   
        plt.tight_layout()
        # plt.subplots_adjust(
        #         wspace=-0.0)


        
        




#%% Time scales of CUTI

time_scales = 0
if time_scales == 1 :
    
    # SST data
    ostia = xr.load_dataset("../satellite_data/ostia_all_2000_2023.nc", engine="netcdf4")  # all coordinates close to California coast
    
    
    
        
    def large_spectrum(data, freq, channels, time_series=0, plot = 1, total_bins = 40):
        '''
        use:
        spectrum(data_spectrum,fs,['u_E', 'v_E', 'w_E'])
        
        provide frequency in cycles per day
        '''
        freq = float(freq)
        if data.index.dtype == 'datetime64[ns]':
            start = data.index[0].strftime("%Y-%m-%d")
            end   = data.index[-1].strftime("%Y-%m-%d")
        else:
            try:
                start = data.GPStime.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
                end   = data.GPStime.iloc[-1].strftime("%H:%M:%S") 
            except:
                start = 'start'
                end = 'end'
                
        if time_series==1:
            fig, (ax1, ax) = plt.subplots(1, 2, figsize=(15,7))
        else:
            fig, ax = plt.subplots(figsize=(12,6))
            
        
        i=0
        for v in channels: 
            i = i+1
            
            if v == "None":  # Skip the item with value 3
                ax.loglog(np.nan, np.nan, linewidth=0.5,label='',alpha=0.0)    
                continue
            
            # Define signal
            fftsig = data[v].interpolate(limit=2).dropna()
            
            if time_series==1:
                times = fftsig.index
                #fig1, ax1 = plt.subplots()
                #ax1.plot(times,fftsig.values, label ='original')
            
            # Detrend
            fftsig = sp.signal.detrend(fftsig)
    
            
            if time_series==1:
                ax1.plot(times,fftsig, ",", label =v+' detrend', lw=1, alpha=0.5)
                ax1.grid(True)
                ax1.set_xlabel("Time (s)")
                #ax1.set_ylabel("Wind speed "+v+' m/s', fontsize='x-large')
                ax1.legend(loc='best')  
                ax1.set_title('Time series')
                fig.autofmt_xdate()
            
            N = len(fftsig) 
            nyq = freq/2
            
            # frequency axis
            X = fftfreq(N, 1/freq)
            X = X[1:int(N/2)]
            
            # FFT
            Y = fft(fftsig)
            
            # PSD
            Y = 2 * (abs(Y[1:int(N/2)])  / N)**2 /(X[2] - X[1])
      
            # 5/3 line    
            y2 = X**(-5/3.) / 1e3
            
            # Smothing over logarithmic aequidistant bins
            start_freq = X[2] # 0.01         # Smoothing starts at 0.1 Hz
            # total_bins = 200                  # Time series is divided in N bins 
            
            start_log = np.log10(start_freq)
            stop_log  = np.log10(nyq)
            bins = np.logspace(start_log,stop_log, num = total_bins, endpoint = True)   # define bin boundaries
            idx  = np.digitize(X,bins)  # sort X-values into bins
            bins     = (bins[1:] + bins[:-1]) / 2   # center of bins, right edge of bins would be  bins[0:total_bins-1]       
            smooth        = [np.average(Y[idx==k]) for k in range(total_bins)]                   # average Y-values for each bin                                                # remove 1st point - it's a NaN
            smooth        = smooth[1:total_bins]                                                      # remove 1st point - it's a NaN
    
            if v == channels[0]: 
                #fig, ax = plt.subplots(figsize=(10,7))
                ax.grid(True) 
               # ax.loglog(X,y2, label ='$f^{-5/3}$', color = 'black')
                ax.set_xlabel("Frequency f (cycles per day)", fontsize='large')
                ax.set_ylabel("PSD$\cdot f$, normalized", fontsize='large')
                plt.tight_layout()
    
            #first, = ax.loglog(X,X*Y, linewidth=0.5,label='',alpha=0.3)
            first, = ax.loglog(np.nan, np.nan, linewidth=0.5,label='',alpha=0.0)
            #ax.loglog(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = '.',label=v,color = first.get_color(), zorder=10)
            #ax.legend(loc='best', fontsize='large')  
            #ax.set_title('Power spectral density (PSD) {} to {}'.format(start, end), fontsize='large')
     
    
     
        #ax.set_ylim(10**-9, 10**-1)
        ax2 = ax.twiny()
        plt.xscale('log')
        mn, mx = ax.get_xlim()
        ax2.set_xlim(1/mn/60/60/24, 1/mx/60/60/24)
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax2.grid()
        plt.xlabel('Time (days)', fontsize='large')
        plt.yscale('linear')
        plt.tight_layout()  
        
        if plot == 1:
            return fig, ax, ax2
        else:   
            plt.close()
            return bins, smooth
        

    
    
    
    lat_range = np.arange(31,45)   # [40,41]
    
    plt.rcParams.update({'font.size': 14})
    
    fig, ax, ax2 = large_spectrum(CUTI,freq=1,channels=['41N'])
 
    for lat in np.arange(35, 45): #[41]: # 
    
            if (lat == 36) or (lat == 41):
                lw = 2
            else:
                lw = 1
                
            max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0] 
            min_lon = max_lon + lon_difference(offshore_diff, max_lon)
            

            # Average satellite data over 1 deg in N-S direction and up to Xkm offshore
            sat_local = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 
            ostia_local = ostia.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 
        
            # Merge model CUTI and satellite area average and SST local area
            all = CUTI.merge(
                    sat_local[["CUTI_Oscar", "CUTI_SLA", "U_Ek", 
                               "U_Geo", "U_Geo_Oscar", "CUTI_Cop", 'U_Geo_cop']].to_dataframe(), 
                    left_index=True, right_index=True, how="outer")
            all = all.resample("D").median().merge(
                    ostia_local[["analysed_sst"]].to_dataframe().resample("D").median(), 
                    left_index=True, right_index=True, how="outer")
            all = all.replace([np.inf, -np.inf], np.nan)

        
        
            for param, color in zip(["U_Ek",'U_Geo_cop', str(int(lat))+"N",  "CUTI_Cop", "analysed_sst"], ["C1","C3","C0", "C2","grey"]):
                bins, smooth = large_spectrum(all,freq=1,channels=[param], plot = 0) # 
                if param == "analysed_sst":
                    smooth = [x/10 for x in smooth]
                ax.semilogx(bins,bins*smooth / np.nanmedian(bins*smooth),'-',marker = '.',label=param, color = color, alpha = 0.9, lw = lw, zorder = 10)
                #ax.legend()
    mn, mx = ax.get_xlim()
    ax2.set_xlim(1/mn, 1/mx)
    plt.title("Time scales in upwelling, and Ekman and geostrophic transport at latitudes 35$^\circ$N - 45$^\circ$N")
    plt.tight_layout()
    
    plt.ylim(-1, 23)
    
































