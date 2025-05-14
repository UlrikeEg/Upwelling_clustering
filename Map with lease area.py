import pandas as pd
import s3fs
import numpy as np
import pyarrow
from windrose import WindroseAxes
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import scipy as sp
import scipy.signal
import sys
import time
import glob
import pickle
import datetime
import matplotlib.cm as cm
from matplotlib import rcParams, rcParamsDefault
import glob
import xarray
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
np.seterr(divide='ignore', invalid='ignore')

sys.path.append("../satellite_data")
sys.path.append("../buoy_data")

from Read_satellites import *
from read_buoy_data import *

def lon_difference(zonal_distance_km, latitude):
    radius_earth = 6371.0
    phi = np.radians(latitude)
    delta_lon = zonal_distance_km / (radius_earth * np.cos(phi))
    return np.degrees(delta_lon)


Humboldt_coords = [40.9708, - 124.5901]
Morro_coords    = [35.71074, - 121.84606]
Buoy_coords    = [40.748, - 124.527]




#%% Read data


## Satellite data


## GOES data from NOAA coastwatch
goes_noaa = read_goes_noaa()


## Ostia SST measuremnets
ostia = xr.load_dataset("../satellite_data/ostia_all.nc", engine="netcdf4")  # all coordinates close to California coast

# test delta lon caclulation
delta_longitude = lon_difference(150, Humboldt_coords[0])


coast_dist = 40 # approx the same for both buoys (perpendicular to coast line)
offshore_dist = 60 #km
onshore_dist  = 10 #km
offshore_lon_humb = Humboldt_coords[1] - lon_difference(offshore_dist - coast_dist, Humboldt_coords[0])
onshore_lon_humb  = Humboldt_coords[1] - lon_difference(onshore_dist  - coast_dist, Humboldt_coords[0])
offshore_lon_morr = Morro_coords[1] - lon_difference(offshore_dist - coast_dist, Morro_coords[0])
onshore_lon_morr  = Morro_coords[1] - lon_difference(onshore_dist  - coast_dist, Morro_coords[0])

ostia_sst = pd.DataFrame()
ostia_sst['humb']          = ostia.sel(lat= Humboldt_coords[0], lon= Humboldt_coords[1], method = 'nearest').to_dataframe().analysed_sst
ostia_sst['humb_onshore']  = ostia.sel(lat= Humboldt_coords[0], lon= onshore_lon_humb, method = 'nearest').to_dataframe().analysed_sst
ostia_sst['humb_offshore'] = ostia.sel(lat= Humboldt_coords[0], lon= offshore_lon_humb, method = 'nearest').to_dataframe().analysed_sst
ostia_sst['humb_on_offshore'] = ostia_sst['humb_onshore'] - ostia_sst['humb_offshore']
ostia_sst['morro']          = ostia.sel(lat= Morro_coords[0], lon= Morro_coords[1], method = 'nearest').to_dataframe().analysed_sst
ostia_sst['morro_onshore']  = ostia.sel(lat= Morro_coords[0], lon= onshore_lon_morr, method = 'nearest').to_dataframe().analysed_sst
ostia_sst['morro_offshore'] = ostia.sel(lat= Morro_coords[0], lon= offshore_lon_morr, method = 'nearest').to_dataframe().analysed_sst
ostia_sst['morro_on_offshore'] = ostia_sst['morro_onshore'] - ostia_sst['morro_offshore']

# plt.figure()
# plt.plot(ostia_sst['morro_onshore'])
# plt.plot(ostia_sst['morro_offshore'])
# plt.plot(ostia_sst['morro_on_offshore'])
# plt.plot(ostia_sst['humb_onshore'])
# plt.plot(ostia_sst['humb_offshore'])
# plt.plot(ostia_sst['humb_on_offshore'])


# Sentinel from A2E
#sent_a2e = read_sentinel_a2e(filt_string = "*sent*202*.nc", location = Humboldt_coords, dist_location=0.5)


# SSH/ SSA 
ssa, ssa_humb, ssa_morro = read_ssa()  # defined in Read_satellites

ssa_humb = ssa_humb.sla.to_dataframe().sla
ssa_humb.index = pd.to_datetime(ssa_humb.index)

ssa_morro = ssa_morro.sla.to_dataframe().sla
ssa_morro.index = pd.to_datetime(ssa_morro.index)

# Read NBS satellite wind data
files = sorted(glob.glob('C:/Users/uegerer/Desktop/Oracle/satellite_data/NBS_winds/NBS*202*.nc'))[:]
nbs = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
nbs = nbs.sel(zlev=10)
# Select data at the Humboldt location
nbs_humb = nbs.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1]+360, method = "nearest").to_dataframe()
nbs_morro = nbs.sel(lat=Morro_coords[0], lon=Morro_coords[1]+360, method = "nearest").to_dataframe()
    






# Read buoy data
    
humb = read_CTD("Humb")
humb_curr = read_currents("Humb")
# humb_curr.index.get_level_values('depth').unique()
humb_meteo = read_meteo("humb")

morro = read_CTD("Morro")
morro_curr = read_currents("Morro")
morro_meteo = read_meteo("Morro")

buoy = read_buoy()

    
resample = "30D"
humb["SST_anomaly"] = humb.SST - humb.SST.rolling(resample, center=True).mean()
morro["SST_anomaly"] = morro.SST - morro.SST.rolling(resample, center=True).mean()

# plt.figure()
# plt.hist(humb_meteo.wind_direction,bins=200, density=True, alpha=0.5)
# plt.hist(humb_meteo.resample("1h").median().wind_direction,bins=200, density=True, alpha=0.5)
# plt.hist(humb_meteo.resample("1h").first().wind_direction,bins=200, density=True, alpha=0.5)
# # ---> use first!!




# Read CUTI data

CUTI = pd.read_csv('C:/Users/uegerer/Desktop/Oracle/CUTI_data/CUTI_daily.csv', header = 0, index_col=None).apply(pd.to_numeric, errors='coerce')
CUTI.index = pd.to_datetime(CUTI[['year', 'month', 'day']])
# CUTI_mon = pd.read_csv('C:/Users/uegerer/Desktop/Oracle/CUTI_data/CUTI_monthly.csv', header = 0, index_col=None).apply(pd.to_numeric, errors='coerce')
# CUTI_mon['day'] = 15
# CUTI_mon.index = pd.to_datetime(CUTI_mon[['year', 'month', 'day']])



        
        
#%% SST map

plot =1
if plot == 1:
    
    # Plot map for a day with high upwelling
    
    upwelling_days    = CUTI[ (CUTI['36N']>2.3) & (CUTI['41N'] >2.3) ]["2020-01-01":].index
    downwelling_days  = CUTI[ (CUTI['36N']<-0.5) & (CUTI['41N'] <-0.5) ]["2020-01-01":].index
    
    days = ['2020-10-22', '2020-11-07', '2022-04-01', '2022-04-10']
    days = ['2021-07-20', '2021-07-21', '2021-07-22', '2021-07-23', '2021-07-24', '2021-07-25', '2021-07-26', '2021-07-27']
    days = ['2021-07-23', '2021-07-24']
    # from datetime import datetime, timedelta
    # days = [datetime(2021, 8, 1) + timedelta(days=i*2) 
    #         for i in range((datetime(2021, 8, 31) - datetime(2021, 8, 1)).days // 2 + 1)]

    #days = downwelling_days[::]
    #days = upwelling_days[::]
    #days = []
    
    
    
    kw = dict(central_latitude=40, central_longitude=-125)
    
    num_days = len(days)
    num_rows = int(num_days**0.5)
    num_cols = (num_days + num_rows - 1) // num_rows
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 3 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, day in enumerate(axes):
        ax = axes[i]
        ax.axis('off')         
    
    # Iterate over days and create subplots
    for i, day in enumerate(days):
        
        #fig = plt.figure() # figsize=(10, 3)
        # ax = plt.axes(projection=ccrs.Stereographic(**kw))
    
        ax = axes[i]
        ax = plt.subplot(num_rows, num_cols, i + 1, projection=ccrs.Stereographic(**kw))
    
         
        
        # stations
        ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        #ax.plot(Buoy_coords[1], Buoy_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='grey',  alpha=0.9)
        
        # 200km lines
        # line = sgeom.LineString([(offshore_lon_humb, Humboldt_coords[0]), (onshore_lon_humb, Humboldt_coords[0])])
        # ax.plot(line.xy[0], line.xy[1], transform=ccrs.PlateCarree(), color='black')
        # line = sgeom.LineString([(offshore_lon_morr, Morro_coords[0]), (onshore_lon_morr, Morro_coords[0])])
        # ax.plot(line.xy[0], line.xy[1], transform=ccrs.PlateCarree(), color='black')
        
        # map
        ax.add_feature(cartopy.feature.BORDERS)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.LAND)
        cs = ax.coastlines(resolution='50m', linewidth=1)
        states_provinces = cartopy.feature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
        ax.add_feature(states_provinces, edgecolor='black', linestyle=':')
        from cartopy.feature import ShapelyFeature
        from cartopy.io.shapereader import Reader
        filename = r'../BOEM_Renewable_Energy_Shapefiles_8/Wind_Lease_Outlines_11_16_2023.shp'
        shape_feature = ShapelyFeature(Reader(filename).geometries(), ccrs.PlateCarree(), lw = 2,  facecolor='None',edgecolor = 'black')
        ax.add_feature(shape_feature)
        
        # Plot wind map
        df = ostia.sel(time= day, method='nearest').analysed_sst
        Temp_Humb = df.sel(lat= Humboldt_coords[0], lon= Humboldt_coords[1], method = 'nearest').values
        Temp_Morro = df.sel(lat= Morro_coords[0], lon= Morro_coords[1], method = 'nearest').values
        min_temp = (Temp_Humb + Temp_Morro ) / 2  - 1
        max_temp = (Temp_Humb + Temp_Morro ) / 2  + 3
        min_temp = 11
        max_temp = 18     
        df.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', cbar_kwargs={'label': 'SST (°C)'}, vmin=min_temp, vmax=max_temp)
      
        gl = ax.gridlines(draw_labels=True)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False
        #gl.xlocator = mticker.FixedLocator([-113])
        #gl.ylocator = mticker.FixedLocator([48.5, 48.7, 49, 49.5])
        #gl.xformatter = LONGITUDE_FORMATTER
        #gl.yformatter = LATITUDE_FORMATTER
        ax.set_extent([-128,-120, 35, 42], crs=ccrs.PlateCarree())  
        #ax.set_extent([-128,-123, 40.5, 41.5], crs=ccrs.PlateCarree())  
        
        # ax.set_title('{:%Y-%m-%d}, \nC$_H$={}, C$_M$={}'.format(day,
        #     round(CUTI.loc[pd.to_datetime(day), '41N'], 2), 
        #     round(CUTI.loc[pd.to_datetime(day), '36N'], 2))  , fontsize=10)   
        ax.set_title(day)   
        plt.tight_layout()
        
        
    
    
        
#%% Plot wind map for several days

    wind_map_more_days = 0
    if wind_map_more_days == 1:
   

       # # Read NBS satellite wind data
       # files = sorted(glob.glob('C:/Users/uegerer/Desktop/Oracle/satellite_data/NBS_winds/NBS*202*.nc'))[:]
       # nbs = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
       # nbs = nbs.sel(zlev=10)
       
       # Plot
       kw = dict(central_latitude=40, central_longitude=-125)
       
       num_days = len(days)
       num_rows = int(num_days**0.5)
       num_cols = (num_days + num_rows - 1) // num_rows
       
       fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 3 * num_rows), sharex=True, sharey=True)
       axes = axes.flatten()
       
       for i, day in enumerate(axes):
           ax = axes[i]
           ax.axis('off')         
       
       # Iterate over days and create subplots
       for i, day in enumerate(days):
           
            nbs_sel = nbs.sel(time=day.date(), method = "nearest")# .isel(time=2)
           
            #fig = plt.figure() # figsize=(10, 3)
            # ax = plt.axes(projection=ccrs.Stereographic(**kw))
            
            ax = axes[i]
            ax = plt.subplot(num_rows, num_cols, i + 1, projection=ccrs.Stereographic(**kw))
            
             
            
            # stations
            ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
            ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
            #ax.plot(Buoy_coords[1], Buoy_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='grey',  alpha=0.9)
            
            # 200km lines
            line = sgeom.LineString([(offshore_lon_humb, Humboldt_coords[0]), (onshore_lon_humb, Humboldt_coords[0])])
            ax.plot(line.xy[0], line.xy[1], transform=ccrs.PlateCarree(), color='black')
            line = sgeom.LineString([(offshore_lon_morr, Morro_coords[0]), (onshore_lon_morr, Morro_coords[0])])
            ax.plot(line.xy[0], line.xy[1], transform=ccrs.PlateCarree(), color='black')
            
            # map
            ax.add_feature(cartopy.feature.BORDERS)
            ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
            ax.add_feature(cartopy.feature.RIVERS)
            ax.add_feature(cartopy.feature.LAND)
            cs = ax.coastlines(resolution='50m', linewidth=1)
            states_provinces = cartopy.feature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
            ax.add_feature(states_provinces, edgecolor='black', linestyle=':')
              
            # Extract data variables
            u = nbs_sel['u_wind'].values
            v = nbs_sel['v_wind'].values
            
            # Calculate current speed
            speed = np.sqrt(u**2 + v**2)
            
            # Plot the map with current speed as color code
            c = ax.pcolormesh(nbs_sel['lon'], nbs_sel['lat'], speed[ :, :],
                              cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', vmin=0, vmax=15)
            
            # Add arrows for wind direction 
            scale_factor = 100  # Adjust the scale factor for arrow length
            ax.quiver(nbs_sel['lon'], nbs_sel['lat'],
                      u[ :, :], v[ :, :], scale=scale_factor, color='white', transform=ccrs.PlateCarree())
            
            # Add colorbar
            cbar = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.1)
            cbar.set_label('wind Speed (m/s)')
            
         
            gl = ax.gridlines(draw_labels=False)
            gl.bottom_labels = False
            gl.left_labels = False
            gl.right_labels = False
            gl.top_labels = False
            gl.frame_on = False
           #gl.xlocator = mticker.FixedLocator([-113])
           #gl.ylocator = mticker.FixedLocator([48.5, 48.7, 49, 49.5])
           #gl.xformatter = LONGITUDE_FORMATTER
           #gl.yformatter = LATITUDE_FORMATTER
            ax.set_extent([-128,-120, 35, 42], crs=ccrs.PlateCarree())  
           #ax.set_extent([-128,-123, 40.5, 41.5], crs=ccrs.PlateCarree())  
           
            ax.set_title('{:%Y-%m-%d}, \nC$_H$={}, C$_M$={}'.format(day,
               round(CUTI.loc[pd.to_datetime(day), '41N'], 2), 
               round(CUTI.loc[pd.to_datetime(day), '36N'], 2))  , fontsize=10)   
            plt.tight_layout()
           
    
    
    

































