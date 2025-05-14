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
import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})



sys.path.append("../satellite_data")
sys.path.append("../buoy_data")

from Read_satellites import read_ssa
from Functions_Oracle import *


Humboldt_coords = [40.9708, - 124.5901]
Morro_coords    = [35.71074, - 121.84606]






#%%  Read datasets


year_code = "*"


# Open NBS sea wind data
print("Reading NBS data ...")
save_path = "../satellite_data/NBS_winds"
# files = sorted(glob.glob(save_path +'/NBS*wind_stress*'+year_code+'*.nc'))[:]
# nbs = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 365})
# nbs = nbs.where(nbs>-10)
# nbs = nbs.sel(zlev=10)[['mask', 'x_tau', 'y_tau']].drop(['zlev'])



# files = sorted(glob.glob(save_path +'/NBS*wind_6hourly*'+year_code+'*.nc'))[:]
# nbs_winds = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 365})
# nbs_winds = nbs_winds.sel(zlev=10)[['mask', 'u_wind', 'v_wind']].drop(['zlev'])

# # Make daily averages (all other datasets are daily averages)   
# nbs = nbs.drop_duplicates(dim='time')
# nbs = nbs.chunk({'time': 365})
# nbs = nbs.resample(time='1D').median()

# nbs_winds = nbs_winds.drop_duplicates(dim='time')
# nbs_winds = nbs_winds.resample(time='1D').median()

years = np.arange(1993,2025)


# NBS wind stress
processed_datasets = []

for year in years:
    print(f"{year}...")
    
    year_files = sorted(glob.glob(save_path + f'/NBSv02_wind_stress_6hourly_{year}*.nc'))
    
    if not year_files:
        continue
    
    nbs = xr.open_mfdataset(year_files, concat_dim='time', combine='nested', chunks={'time': 40})
    
    nbs = nbs.where(nbs > -10)
    nbs = nbs.sel(zlev=10)[['mask', 'x_tau', 'y_tau']].drop(['zlev'])

    nbs = nbs.drop_duplicates(dim='time').sortby('time')
    nbs = nbs.resample(time='1D').median()
    
    processed_datasets.append(nbs)

nbs = xr.concat(processed_datasets, dim='time')
nbs = nbs.drop_duplicates(dim='time')


# plt.figure()
# ax = plt.subplot(1, 1, 1)  
# nbs.isel(time=-1).x_tau.plot()
# coastline = np.load("coastline.npy")
# ax.plot(coastline[:, 0]+360, coastline[:, 1], linewidth=1, color="gray") 
# ax.plot(Humboldt_coords[1]+360, Humboldt_coords[0], marker='o', ms=10, color='black',  alpha=0.9)


# nbs_humb = nbs.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest")


# plt.figure()
# plt.plot(nbs_humb.time, nbs_humb.x_tau)    

# NBS winds
processed_datasets = []

for year in years:
    print(f"{year}...")
    
    year_files = sorted(glob.glob(save_path + f'/NBSv02_wind_6hourly_{year}*.nc'))
        
    print (len(year_files))
    
    if not year_files:
        continue
    
    nbs_winds = xr.open_mfdataset(year_files, concat_dim='time', combine='nested', chunks={'time': 1000})
    
    nbs_winds = nbs_winds.sel(zlev=10)[['mask', 'u_wind', 'v_wind']].drop(['zlev'])

    nbs_winds = nbs_winds.drop_duplicates(dim='time').sortby('time')
    nbs_winds = nbs_winds.resample(time='1D').median()   # .resample(time='6H').first()
    
    processed_datasets.append(nbs_winds)

nbs_winds = xr.concat(processed_datasets, dim='time')
nbs_winds = nbs_winds.drop_duplicates(dim='time')


# compression_settings = dict(zlib=True, complevel=5)
# encoding = {var: compression_settings for var in nbs_winds.data_vars}
# nbs_winds.to_netcdf('../satellite_data/NBS_winds/nbs_winds_1993_2023_6h.nc', encoding=encoding)





# SSH/ SSA 
print("Reading SLA data ...")
save_path = "../satellite_data/SSA"
files = sorted(glob.glob(save_path +'/*sla*'+year_code+'*.nc'))[:]
sla = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})

# Select data at the Humboldt location
#sla_humb = sla.sel(latitude=Humboldt_coords[0], longitude=Humboldt_coords[1], method = "nearest")
    


# Oscar current data
print("Reading Oscar data ...")
save_path = 'C:/Users/uegerer/Desktop/Oracle/satellite_data/Oscar/'
files = sorted(glob.glob(save_path +'*Oscar_*'+year_code+'*.nc'))[:-1]
oscar = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
oscar = oscar.sel(time=slice(nbs.time[0], nbs.time[-1]))
oscar = oscar.rename({'latitude': 'lat', 'longitude': 'lon' })

# Select data at the Humboldt location
#oscar_humb = oscar.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1]+360, method = "nearest")


# CUTI data
print("Reading CUTI data ...")
CUTI = pd.read_csv('C:/Users/uegerer/Desktop/Oracle/CUTI_data/CUTI_daily.csv', header = 0, index_col=None).apply(pd.to_numeric, errors='coerce')
CUTI.index = pd.to_datetime(CUTI[['year', 'month', 'day']])



# MLD from Argo floats (interpolated data)
mld = xr.open_dataset("MLD_data", engine = "netcdf4")
mld = mld.drop_vars(['lat', 'lon', 'month']).rename({'iLAT': 'lat', 'iLON': 'lon', 'iMONTH': 'month'})
# mld_humb = mld.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest")
# mld_morro = mld.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest")


# MLD from ARMOR3 (sate and in-situ)
# files = ['../satellite_data/Copernicus_MLD/dataset-armor-3d-rep-weekly_1727224995342.nc']+ ['../satellite_data/Copernicus_MLD/dataset-armor-3d-nrt-weekly_1727224831658.nc']
# mld_sat = xr.open_mfdataset(files, combine='nested', compat ='override', chunks={'time': 100}, parallel=True)   
# mld_sat.to_netcdf('../satellite_data/Copernicus_MLD/mld_sat_1993_2024_mld.nc')
mld_sat = xr.open_dataset('../satellite_data/Copernicus_MLD/mld_sat_1993_2024_mld.nc')









#%% Costline and coast angle coordinates

calculate_smooth_coastline = 0
if calculate_smooth_coastline == 1  :
    
    # Load coastline data
    coastline_feature = NaturalEarthFeature(category='physical', scale='110m',
                                            facecolor='none', name='coastline')
    
    # Extract coastline coordinates
    coastline_geoms = list(coastline_feature.geometries())
    coastline_coords = []
    for geom in coastline_geoms:
        if isinstance(geom, LineString) :
            coastline_coords.extend(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                coastline_coords.extend(line.coords)
                    
    coastline_coords = [(lon, lat) for lon, lat in coastline_coords if lat >= 20 and lat <= 48 and lon >= -128 and lon <= -115]
                    
    
    
    # Convert coordinates to numpy array for easy computation
    coastline_array = np.array(coastline_coords)
    
    # Interpolate to 0.25-degree grid
    new_lat = np.arange(20,50.25,0.25)
    new_lon = np.interp(new_lat, coastline_array[:, 1], coastline_array[:, 0], left = np.nan, right = np.nan)
    coastline_array = np.column_stack((new_lon, new_lat))
    
    # Smooth longitude coordinate using a moving average with window size 3
    padded_lon = np.pad(coastline_array[:, 0], (1, 1), mode='edge')
    coastline_array[:, 0] = np.convolve(padded_lon, np.ones(3)/3, mode='valid')
    
    # Calculate differences in x and y coordinates
    dx = np.diff(coastline_array[:, 0])
    dy = np.diff(coastline_array[:, 1])
    
    # Calculate angle at each latitude
    angles_rad = np.arctan2(dx, dy)
    angles_deg = np.degrees(angles_rad)
    angles_deg = np.append(angles_deg, np.nan)
    
    # Append angles to the coastline_array
    coastline_array = np.column_stack((coastline_array, angles_deg))
    
    # plt.figure()
    # plt.plot(coastline_array[:, 0], coastline_array[:, 1], linewidth=2) 
    
    #np.save("coastline", coastline_array)

#%% Calculate CUTI for Humboldt location

test_with_Humb_loc = 0

if test_with_Humb_loc ==1 :
    
    # Calculate along-and cross shore wind stress
    nbs_humb['tau_along'], nbs_humb['tau_cross'] = calculate_along_cross_shore_components(nbs_humb.x_tau, nbs_humb.y_tau, coastline_angle=0)
    
    
    # Coriolis parameter
    f = calculate_coriolis_parameter(Humboldt_coords[0])
    
    # sea water density
    rho_ocean = 1025 # kg/m3
    
    
    # Ekman transport
    nbs_humb['U_Ek'] = nbs_humb['tau_along'] / rho_ocean / f
    
    
    # Calculate along-and cross shore geostrophic velocities
    sla_humb['gos_along'], sla_humb['gos_cross'] = calculate_along_cross_shore_components(sla_humb.ugos, sla_humb.vgos, coastline_angle=0)
    
    # Calculate along-and cross shore geostrophic velocities
    oscar_humb['gos_along'], oscar_humb['gos_cross'] = calculate_along_cross_shore_components(oscar_humb.ug, oscar_humb.vg, coastline_angle=0)
    oscar_humb['curr_along'], oscar_humb['curr_cross'] = calculate_along_cross_shore_components(oscar_humb.u, oscar_humb.v, coastline_angle=0)
    
    
    
    # GEostrophic transport (positive when offshore)
    MLD = 10
    sla_humb["U_Geo"] = sla_humb['gos_cross'] * MLD
    
    fig = plt.figure(figsize = (15,5))
    #plt.plot(CUTI['36N'],".", label = "Morro daily")
    plt.plot(CUTI['41N'],".", label = "CUTI Humboldt", color="black")
    plt.plot(nbs_humb.time, nbs_humb.U_Ek , label = "U_Ek from satellite")
    #plt.plot(sla_humb.time, sla_humb.gos_along , label = "U_Geo along from satellite")
    plt.plot(sla_humb.time, sla_humb.gos_cross , label = "U_Geo cross from satellite")
    plt.plot(sla_humb.time, sla_humb.U_Geo , label = "U_Geo cross * MLD from satellite")
    plt.plot(oscar_humb.time, oscar_humb.gos_cross , label = "U_Geo cross from Oscar")
    plt.plot(oscar_humb.time, oscar_humb.curr_cross , label = "Total cross current from Oscar")
    plt.grid(True)
    plt.legend()
    plt.ylabel('CUTI (m$^2$/s)')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.xlim(pd.to_datetime('2020-01-01'),pd.to_datetime('2021-01-01'))
    
    
    
    fig = plt.figure(figsize = (15,5))
    plt.plot(sla_humb.time, sla_humb.ugos , label = "SLA ugos")
    plt.plot(sla_humb.time, sla_humb.vgos , label = "SLA vgos")
    plt.plot(oscar_humb.time, oscar_humb.ug , label = "Oscar ug")
    plt.plot(oscar_humb.time, oscar_humb.vg , label = "Oscar vg")
    plt.plot(oscar_humb.time, oscar_humb.u , label = "Oscar u")
    plt.plot(oscar_humb.time, oscar_humb.v , label = "Oscar v")
    plt.legend()
    plt.ylabel('current velocity (m/s)')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.xlim(pd.to_datetime('2020-01-01'),pd.to_datetime('2021-01-01'))
    
        




#%% Calculate CUTI for all locations

print("Calculating CUTI ...")


# Interpolate sla grid onto nbs grid
if nbs['lon'].values.max()>180:
    nbs['lon'] = nbs['lon']-360
nbs = nbs.where(nbs['mask'] == 1) # filter mask 1 value (only satellite data over ocean)
if nbs_winds['lon'].values.max()>180:
    nbs_winds['lon'] = nbs_winds['lon']-360
nbs_winds = nbs_winds.where(nbs_winds['mask'] == 1) # filter mask 1 value (only satellite data over ocean)

sla_interp = sla.interp(latitude=nbs['lat'], longitude=nbs['lon'], time=nbs["time"], method="linear")

if oscar['lon'].values.max()>180:
    oscar['lon'] = oscar['lon']-360
oscar_interp = oscar.interp(lat=nbs['lat'], lon=nbs['lon'], time=nbs["time"], method="linear")
    

# Concatenate datasets along the time dimension
sat = xr.merge([  nbs, 
                  nbs_winds[['u_wind', 'v_wind']], 
                  sla_interp[['sla', 'ugos', 'vgos']].drop(['latitude', 'longitude']),
                  oscar_interp
                  ])
# sat = sat.sel(time=slice("2020-01-01", "2022-07-01"))  # this is to where I have all datasets currently.
    
# Add the coastline angle values to the sat dataset
sat = add_costline_angle_to_sat(sat, plot = 0)


# Coriolis parameter
sat["f"] = calculate_coriolis_parameter(sat.lat)  

# sea water density
rho_ocean = 1025 # kg/m3

# Calculate along-and cross shore wind stress
sat['tau_along'], sat['tau_cross'] = calculate_along_cross_shore_components(
                                        sat.x_tau, sat.y_tau, coastline_angle=sat.coastline_angle)

# Ekman transport
sat['U_Ek'] = sat['tau_along'] / rho_ocean / sat["f"]
# remove outliers
sat['U_Ek'] = sat['U_Ek'].where(sat['U_Ek']>-10)
sat['U_Ek'] = sat['U_Ek'].where(sat['U_Ek']<20)
   
# Calculate along-and cross shore geostrophic ocean velocities from SLA
sat['gos_along'], sat['gos_cross'] = calculate_along_cross_shore_components(
                                        sat.ugos, sat.vgos, coastline_angle=sat.coastline_angle)

# Calculate along-and cross shore geostrophic velocities from Oscar
sat['go_along'], sat['go_cross'] = calculate_along_cross_shore_components(
                                        sat.ug, sat.vg, coastline_angle=sat.coastline_angle)
sat['curr_along'], sat['curr_cross'] = calculate_along_cross_shore_components(
                                            sat.u, sat.v, coastline_angle=sat.coastline_angle)

# Mixed layer depth (fixed depth assumtopn - first guess)
MLD = 10

# GEostrophic transport (positive when offshore) with mixed layer depth
sat["U_Geo"] = sat['gos_cross'] * MLD
sat["U_Geo_Oscar"] = sat['go_cross'] * 30

# Calculate CUTI from satellite data
sat['CUTI_SLA_fix_mld'] = (sat['U_Ek'] + sat['U_Geo'])
sat['CUTI_Oscar_fix_mld'] = (sat['U_Ek'] + sat['U_Geo_Oscar'])       


# Add geostrophic currents from Copernicus
print("Adding new geostrophic currents ...")
save_path = "../satellite_data/Copernicus_SLA"
files = sorted(glob.glob(save_path +'/*sliced*'+year_code+'*.nc'))[:]
cop = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
cop = cop.interp(lat=sat['lat'], lon=sat['lon'], time=sat["time"])[["cop_ugos", "cop_vgos", "err_sla"]]
sat = xr.merge([  sat,
                  cop
                  ])
sat['cop_gos_along'], sat['cop_gos_cross'] = calculate_along_cross_shore_components(
                                        sat.cop_ugos, sat.cop_vgos, coastline_angle=sat.coastline_angle)




### Improve CUTI calc with new MLD

# Add MLD to sat dataset - Argo
mld_interpolated = mld.interp(lat=sat['lat'], lon=sat['lon'])
sat['mld_argo'] =  mld_interpolated['mld_da_median'].sel(month=sat['time'].dt.month, lat=sat['lat'], lon=sat['lon'])


# Add MLD to sat dataset - ARMOR3 (sat and in-situ)
mld_sat = mld_sat.rename({"latitude": "lat", "longitude": "lon"})
mld_interpolated = mld_sat.interp(lat=sat['lat'], lon=sat['lon'], time=sat['time'], method='nearest')
sat['mld_sat'] =  mld_interpolated.sel(time=sat['time'], lat=sat['lat'], lon=sat['lon']).mlotst


# Add year and month
sat['month'] =  sat['time'].dt.month
sat['year'] =  sat['time'].dt.year



# Geostrophic transport (positive when offshore) with new mixed layer depth
sat["U_Geo"] = sat['gos_cross'] *  sat.mld_sat
sat["U_Geo_Oscar"] = sat['go_cross'] * sat.mld_sat
sat["U_Geo_cop"] = sat['cop_gos_cross'] *  sat.mld_sat

# Calculate CUTI from satellite data
sat['CUTI_SLA'] = (sat['U_Ek'] + sat['U_Geo'])
sat['CUTI_Oscar'] = (sat['U_Ek'] + sat['U_Geo_Oscar'])
sat['CUTI_Cop'] = (sat['U_Ek'] + sat['U_Geo_cop'])



### Test length of all datasets

sat_humb = sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest")

nbs_humb = nbs.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest")

plt.figure()
plt.plot(sat_humb.time, sat_humb.mld_sat, label = "MLD")
plt.plot(sat_humb.time, sat_humb.ug, label = "Oscar")
plt.plot(sat_humb.time, sat_humb.u_wind, label = "NBS winds")
plt.plot(sat_humb.time, sat_humb.x_tau, label = "NBS wind stress")
plt.plot(nbs_humb.time, nbs_humb.x_tau, label = "NBS wind stress")
plt.plot(sat_humb.time, sat_humb.CUTI_Oscar, label = "Oscar CUTI")
plt.grid()
plt.legend()





# Save combined sat dataset
print("Saving file ...")

sat = sat.sel(time=slice("1993-01-05", "2024-01-01"))  # current data availability

sat.to_netcdf("Sat_data_combined_final_not_chunked2.nc", engine = "netcdf4") 









#%% Plot parameter maps



plot_parameter_maps = 0 

if plot_parameter_maps == 1:
    
    
   # Coastline
   coastline = np.load("coastline.npy")
   
   
   ostia = xr.load_dataset("../satellite_data/ostia_all_2000_2023.nc", engine="netcdf4")  # all coordinates close to California coast
  
   
   sat_masked = ostia # sat


   # days to plot
   day = '2021-08-30'
   parameters = ["y_tau",   "tau_along"]  
   parameters = ["analysed_sst", "analysed_sst"]
   
   
   # Number of columns and rows
   num_days = len(parameters)
   num_rows = int(num_days**0.5)
   num_cols = (num_days + num_rows - 1) // num_rows
   

   
   

   # Plot
   fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 6 * num_rows), sharex=True, sharey=True)
   axes = axes.flatten()
   
   for i, p in enumerate(axes):
        ax = axes[i]
        ax.axis('off')         
    
 
   
   # Iterate over days and create subplots
   for i, parameter in enumerate(parameters):
       

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
       df = sat_masked.sel(time= day, method='nearest')[parameter]     
       
       #Plot color map     
       if parameter == 'analysed_sst':
           cmap = "coolwarm"
       else:
            cmap = 'viridis'
       c = ax.pcolormesh(df['lon'], df['lat'], df[ :, :],
                         cmap=cmap, transform=ccrs.PlateCarree(), shading='auto', 
                         vmin = df.quantile(0.05).values,
                         vmax = df.quantile(0.95).values
                         
                         )
       
       sel = sat_masked.sel(time= day, method='nearest')

       
       # Add arrows for wind vectors 
       try:
           u =  sel['u_wind'].values
           v =  sel['v_wind'].values
           scale_factor = 500  # Adjust the scale factor for arrow length
           ax.quiver(sel['lon'], sel['lat'],
                     u[ :, :], v[ :, :], scale=scale_factor, color='white', transform=ccrs.PlateCarree())
       except:
           pass
       
       
       # Add colorbar
       #if i%num_cols ==1:
       cbar = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.03, shrink=0.7, label= parameter)
                   
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


       
       





test_Humb_Morro = 0

if  test_Humb_Morro == 1:

    
    # Select Humboldt location and Morro 
    humb = sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest") # point location at Humboldt
    morro = sat.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest")   
    humb =  sat.sel(lat=slice(40.5, 41.5), lon=slice(-125.06, -124.15)).mean(dim=['lat', 'lon']) # average over the Humboldt CUTI area
    morro = sat.sel(lat=slice(35.5, 36.5), lon=slice(-122.14, -121.3)).mean(dim=['lat', 'lon']) # average over the Morro CUTI area
    
    
    morro.to_netcdf("Sat_data_combinded_Morro", engine = "netcdf4")
    humb.to_netcdf("Sat_data_combinded_Humb", engine = "netcdf4")
    
    
    fig = plt.figure(figsize = (15,5))
    plt.plot(CUTI['41N'],".", label = "CUTI Humboldt", color="black", zorder=100)
    plt.plot(humb.time, humb.U_Ek , label = "U_Ek from satellite")
    #plt.plot(sla_humb.time, sla_humb.gos_along , label = "U_Geo along from satellite")
    #plt.plot(humb.time, humb.gos_cross , label = "U_Geo cross from satellite")
    plt.plot(humb.time, humb.CUTI_SLA, label = "CUTI from satellite, SLA")
    plt.plot(humb.time, humb.CUTI_Oscar, label = "CUTI from satellite, Oscar")
    #plt.plot(humb.time, humb.U_Geo , label = "U_Geo * MLD cross from satellite")
    plt.grid(True)
    plt.legend()
    plt.ylabel('CUTI (m$^2$/s)')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.xlim(pd.to_datetime('2020-01-01'),pd.to_datetime('2021-01-01'))
    plt.ylim(-10,12)
    
    
    
    fig = plt.figure(figsize = (15,5))
    plt.plot(CUTI['36N'],".", label = "CUTI Morro", color="black", zorder=100)
    plt.plot(morro.time, morro.U_Ek , label = "U_Ek from satellite")
    #plt.plot(sla_humb.time, sla_humb.gos_along , label = "U_Geo along from satellite")
    # plt.plot(morro.time, morro.gos_cross , label = "U_Geo cross from satellite")
    # plt.plot(morro.time, morro.ugos , label = "U_geo")
    # plt.plot(morro.time, morro.vgos , label = "V_Geo")
    # plt.plot(morro.time, morro.gos_cross , label = "U_Geo cross from satellite")
    # plt.plot(morro.time, morro.ugos , label = "U_geo")
    # plt.plot(morro.time, morro.vgos , label = "V_Geo")
    plt.plot(morro.time, morro.CUTI_SLA, label = "CUTI from satellite, SLA")
    plt.plot(morro.time, morro.CUTI_Oscar, label = "CUTI from satellite, Oscar")
    #plt.plot(morro.time, morro.U_Geo , label = "U_Geo * MLD cross from satellite")
    plt.grid(True)
    plt.legend()
    plt.ylabel('CUTI (m$^2$/s)')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.xlim(pd.to_datetime('2020-01-01'),pd.to_datetime('2021-01-01'))
    plt.ylim(-10,12)
    





















