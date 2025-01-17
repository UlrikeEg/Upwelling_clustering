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
from scipy.spatial.distance import cdist
from minisom import MiniSom    
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import calendar
sys.path.append("../../NSO/NSO_data_processing")
from Functions_general import *
#from sompy import SOMFactory


# Setting the print options to display float numbers without scientific notation
np.set_printoptions(suppress=True, precision=3, floatmode='fixed')





from Functions_Oracle import *


#%%  Read datasets



# Combines satellite dataset
sat = xr.open_dataset("Sat_data_combined_final_not_chunked2.nc", engine = "netcdf4") # "Sat_data_combinded_ARGO_mld"
list(sat.variables.keys())

# CUTI data
CUTI = pd.read_csv('C:/Users/uegerer/Desktop/Oracle/CUTI_data/CUTI_daily.csv', header = 0, index_col=None).apply(pd.to_numeric, errors='coerce')
CUTI.index = pd.to_datetime(CUTI[['year', 'month', 'day']])


# SST data
ostia = xr.load_dataset("../satellite_data/Ostia/ostia_all_1993_2023_4.nc", engine="netcdf4", chunks={'time': 100, 'lat': 100, 'lon': 100})  # all coordinates close to California coast
ostia = ostia.sel(lat=slice(35, 45), lon=slice(-128, -120))  # this is the extent I pulled from eagle
ostia = ostia.dropna(dim="lon", how = "all").dropna(dim="lat", how = "all").dropna(dim="time", how = "all")


#  Chlorophyll data
chl = xr.open_dataset('../satellite_data/Chlorophyll/CHL_1997_2023.nc')



#%% Prepare data


# Calculate the longitude of the 100km offshore line based on latitude
coastline = np.load("coastline.npy")
offshore_diff = 75 # 100
lon_diff = get_offshore_parallel_coastline(coastline, offshore_diff)


sat_masked = sat.copy()

# Loop through each latitude value and remove latitudes outside of the 100 km coastline for CUTI
for i, lat_sat in enumerate(sat_masked.lat.values):
    lon_sat = lon_diff[i] 
    sat_masked[[ "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop" ]].loc[dict(lat=lat_sat, lon=slice(-150, lon_sat-0.25 ) )] =np.nan
        



# Select parameters of interest
data_to_use = sat_masked[['U_Ek' ,  'u_wind',
  'v_wind','cop_ugos',
   'cop_vgos',  'U_Geo_cop', 'U_Geo_Oscar', "CUTI_Oscar",
    'mld_argo','CUTI_Cop', 'tau_cross','tau_along', 'coastline_angle'
   ]].drop(['month'])





# Add SST and CHL data
data_to_use = data_to_use.sel(time=slice(None, '2023-01-31')) # Ostia data only available on eagle till 2023-01-31, but starting 2000
ostia = ostia.chunk({'lat': 100, 'lon': 100, 'time': 100})
ostia_interp = ostia.interp(lat=data_to_use['lat'], 
                            lon=data_to_use['lon'], 
                            time=data_to_use["time"], method="nearest")[['analysed_sst']]
data_to_use = xr.merge([   data_to_use, ostia_interp
                  ])
chl = chl.chunk({'lat': 100, 'lon': 100, 'time': 100})   
chl_interp = chl.interp(lat=data_to_use['lat'], 
                            lon=data_to_use['lon'], 
                            time=data_to_use["time"], method="nearest")[['CHL']]
data_to_use = xr.merge([   data_to_use, chl_interp
                  ])   


# Add SST anomaly
climatology = data_to_use['analysed_sst'].groupby('time.dayofyear').mean(dim=['time', "lat", "lon"]) # 
data_to_use["sst_anomaly"] = data_to_use['analysed_sst'] - climatology.sel(dayofyear=data_to_use['analysed_sst'].time.dt.dayofyear)
data_to_use["sst_anomaly"] = data_to_use["sst_anomaly"].where(data_to_use["analysed_sst"] )

ostia["sst_anomaly"] = ostia['analysed_sst'] - climatology.sel(dayofyear=ostia['analysed_sst'].time.dt.dayofyear)
ostia["sst_anomaly"] = ostia["sst_anomaly"].where(ostia["analysed_sst"] )

ostia_short = ostia.sel(time=ostia.time> pd.to_datetime("2019-12-31T12:00:00"))  # 


# # Add wind data
# files = sorted(glob.glob(save_path +'/NBS*wind_6hourly_*.nc'))[:5]
# nbs_winds = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
# nbs_winds = nbs_winds.sel(zlev=10)[['mask', 'u_wind', 'v_wind']].drop(['zlev'])
# nbs_winds['tau_along'], nbs_winds['tau_cross'] = calculate_along_cross_shore_components(
#                                             nbs_winds.x_tau, nbs_winds.y_tau, coastline_angle=sat.coastline_angle)


# Add CUTI to sat
CUTI = CUTI.drop(['year', 'month', 'day'], axis=1)
lats = [float(col[:-1]) for col in CUTI.columns[:]]
lat_values = coastline[:, 1]
lon_values = coastline[:, 0]
lat_to_lon = {lat: lon for lat, lon in zip(lat_values, lon_values) if not np.isnan(lon)}
lons = [lat_to_lon.get(lat, np.nan)  - lon_difference(50, lat) for lat in lats]





# Cut to area of interest and drop nans
data_to_use = data_to_use.sel(lat=slice(34, 45), lon=slice(-128, -120))
data_to_use = data_to_use.where(np.isfinite(data_to_use), np.nan)
data_to_use = data_to_use.dropna(dim="lon", how = "all").dropna(dim="lat", how = "all").dropna(dim="time", how = "all").fillna(0)




# Filter high CUTI days

# test = data_to_use.where(data_to_use.mean(dim=['lat', 'lon'])['CUTI_Cop'] >= 0.5, drop=True)




#%% Prepare the data for clustering and plotting

#!!!!!!! Test modifying the contribution weight of U_geo
# data_to_use['CUTI_Cop'] = (data_to_use['U_Ek'] + data_to_use['U_Geo_cop'] * 1)
# for i, lat_sat in enumerate(sat_masked.lat.values):
#     lon_sat = lon_diff[i] 
#     try:
#         data_to_use[  "CUTI_Cop" ].loc[dict(lat=lat_sat, lon=slice(-150, lon_sat-0.25 ) )] =np.nan
#     except:
#         pass
# data_to_use = data_to_use.where(np.isfinite(data_to_use), np.nan)
# data_to_use = data_to_use.dropna(dim="lon", how = "all").dropna(dim="lat", how = "all").dropna(dim="time", how = "all").fillna(0)




# Extracting the variable of interest for clustering
param =   "CUTI_Oscar"  # "tau_cross" 'U_Ek' # 'analysed_sst'   'CUTI_Cop' CUTI_model
u_wind = data_to_use[param]  # .dropna(dim="lon", how = "all").dropna(dim="lat", how = "all").dropna(dim="time", how = "all").fillna(0)


# Reshape the u_wind data into a 2D array (time as rows, lat*lon as columns)
u_wind_2d = u_wind.values.reshape(u_wind.shape[0], -1)

# Normalize the data to have values between 0 and 1 - scaling not necessary here since data are at similar values
u_wind_normalized =u_wind_2d #  (u_wind_2d - u_wind_2d.min()) / (u_wind_2d.max() - u_wind_2d.min())  #  

#u_wind_normalized = u_wind_normalized[~np.isnan(u_wind_normalized).all(axis=0)]


# Create a meshgrid of lon and lat coordinates for plotting
lon_mesh, lat_mesh = np.meshgrid(u_wind['lon'], u_wind['lat'])





#% Cluster algorithm

# select cluster method
method = "k-means"


## K-means clustering for CUTI and other variables
if method == "k-means" :

    ### This modification calculates the distances between each time step and each cluster centroid and assigns each time step to the cluster with the minimum distance (most similar spatial pattern).
    
    # Define the number of clusters (patterns) you want to identify
    num_clusters = 4
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10,  random_state=42)
    cluster_labels = kmeans.fit_predict(u_wind_normalized)
    
    
    # Calculate the distance between each time step and each cluster centroid - most metrics provide very similar results default: euclidean
    distances = cdist(u_wind_normalized, kmeans.cluster_centers_, metric =  'cosine')
    
    # Assign each time step to the cluster with the minimum distance (most similar spatial pattern)
    cluster_assignments = np.argmin(distances, axis=1)
    
    # Count the number of cases in each cluster
    cluster_counts = [np.sum(cluster_assignments == i) for i in range(num_clusters)]
    
    
    
    
## Self-organizing maps    
elif method == "SOM":

    # Define the SOM grid dimensions (e.g., 2x2 for 4 clusters)
    som_shape = (2, 2)
    som = MiniSom(som_shape[0], som_shape[1], u_wind_normalized.shape[1], sigma=1.0, learning_rate=0.5)
    
    # Train the SOM
    som.random_weights_init(u_wind_normalized)  # initialize the weights by picking random samples from the data.
    iterations = 1000
    som.train_random(u_wind_normalized, iterations)
    
    # Get the winner node for each data point
    win_map = som.win_map(u_wind_normalized)
    
    # Assign each time step to the nearest SOM node (cluster)
    cluster_assignments = np.zeros(u_wind_normalized.shape[0], dtype=int)
    for i in range(u_wind_normalized.shape[0]):
        cluster_assignments[i] = np.ravel_multi_index(som.winner(u_wind_normalized[i]), som_shape)
    
    # Count the number of cases in each cluster
    num_clusters = som_shape[0] * som_shape[1]
    cluster_counts = [np.sum(cluster_assignments == i) for i in range(num_clusters)]


sorted_clusters = np.argsort(cluster_counts)


#%% Visualize the clusters (patterns)

additional_params = ['sst_anomaly', "U_Geo_Oscar", 'U_Ek' ] # , 'CHL']  

data_to_use['U_Geo_Oscar'] = data_to_use['U_Geo_Oscar'].transpose('time', 'lat', 'lon')

min_cuti = -3
max_cuti = 3   

    
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(10, 9))
plt.suptitle(f"Satellite CUTI "+method+f" clusters {data_to_use.time[0].values.astype('datetime64[Y]').astype(int) + 1970} to {data_to_use.time[-1].values.astype('datetime64[Y]').astype(int) + 1970}")


for pos, i in enumerate(sorted_clusters):
#for i in range(num_clusters):
    
    main_colormap = 'viridis'
    
    ### Indices of this cluster
    cluster_indices = np.where(cluster_assignments == i)[0]
    cluster_time_indeces = data_to_use.time.values.reshape(data_to_use.time.shape[0], -1)[cluster_indices]
    
    
    ### Plot parameter of cluster variable (CUTI)
    # Calculate mean of each pattern
    mean_pattern = np.mean(u_wind_normalized[cluster_indices], axis=0)
    mean_pattern = mean_pattern.reshape(u_wind['lat'].shape[0], u_wind['lon'].shape[0])
    mean_pattern[mean_pattern == 0] = np.nan 
    
    ax = plt.subplot(len(additional_params)+2, num_clusters, pos+1, projection=ccrs.PlateCarree())
    plt.pcolormesh(lon_mesh, lat_mesh, mean_pattern, cmap=main_colormap, transform=ccrs.PlateCarree(),
                   vmin = min_cuti, 
                   vmax = max_cuti) 
    plt.title(f'Cluster {i+1} (N={cluster_counts[i]})') 
    
    # Model CUTI   
    cuti = CUTI[CUTI.index.isin(cluster_time_indeces.flatten())].mean()
    plt.scatter(lons, lats, c=cuti, cmap=main_colormap, transform=ccrs.PlateCarree(),edgecolors="black", 
               vmin = min_cuti, 
               vmax = max_cuti)

    # Stations
    ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
    ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)

    # Map
    ax.add_feature(cartopy.feature.BORDERS)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)
    ax.add_feature(cartopy.feature.LAND)
    ax.coastlines(resolution='10m', linewidth=1)
    gl = ax.gridlines(draw_labels=True)
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.top_labels = False
    ax.set_extent([-128,-120, 35, 45], crs=ccrs.PlateCarree())  
    
    ### Histograms
    ax = plt.subplot(len(additional_params)+2, num_clusters, pos+len(additional_params)*num_clusters+1 + num_clusters)
    
    ## histogram for cuti over latitude
    # latitudes = np.arange(35, 45)
    # cmap = plt.get_cmap('viridis')
    # norm = plt.Normalize(vmin=min(latitudes), vmax=max(latitudes))
    # colors = cmap(norm(latitudes))    
    # for n, lat in enumerate(latitudes): 
    #     max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0]
    #     min_lon = max_lon + lon_difference(offshore_diff, max_lon)
    #     sat_local = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon'])      
    #     hist_data = sat_local[param].values.reshape(u_wind.shape[0], -1)[cluster_indices]

    #     #plt.hist(cuti[f"{lat}N"], bins=80, density=True, alpha = 0.5, color=colors[n], range=(-4, 4), label = f"{lat}N")
    #     plt.hist(hist_data.flatten(), bins=40, density=True,  alpha = 0.5, color=colors[n], range=(-4, 4), label = f"{lat}N")
        
    #     if i==num_clusters-1:
    #         handles, labels = plt.gca().get_legend_handles_labels()
    #         plt.legend(handles[::3], labels[::3], loc="upper right")

    ## histogram for cuti over latitude
   # plt.hist(data_to_use['time'].dt.month.values, density=True, bins=12)
    out_hist = plt.hist(cluster_time_indeces.flatten().astype('datetime64[M]').astype(int) % 12 + 1, density=True, bins=12, alpha = 0.5, color="b")
    ax.yaxis.set_ticks([])
    plt.xticks(out_hist[1].astype(int), [calendar.month_abbr[m] for m in out_hist[1].astype(int)]) 
    ticks = out_hist[1].astype(int)
    labels = [calendar.month_abbr[m] for m in ticks]
    plt.xticks(ticks[::3], labels[::3])

    
    if pos==num_clusters-1:
        cbar = plt.colorbar(label="CUTI (m$^2$/s)" )
        cbar.ax.yaxis.label.set_size(14)
        
    ### Plot additional variables:
    for j,par in enumerate(additional_params, start=1):
        
        if par == "sst_anomaly":
            label = "SST anomaly ($^\circ$C)"
        elif par == "U_Ek":
            label = "U$^{Ek}$ (m$^2$/s)"
        elif par == "U_Geo_Oscar":
            label = "U$^{Geo}$ (m$^2$/s)"
        elif par == "wind_speed":
            label = "Wind speed (m/s)"
        elif par == "U_Geo_Oscar":
            label = "Geostrophic current (m/s)"
        elif par == "geostr_curr":
            label = "Geostrophic current (m/s)"
        elif par == "mld_sat":
            label = "MLD (m)"  
        elif par == "CHL":
             label = "Chl (mg/m$^3$)"
        else:
            label = par
        
        # Calculate mean of each pattern  
        if (par == "analysed_sst") or (par == "sst_anomaly"):
            
            cluster_sst_time_indeces = cluster_time_indeces[cluster_time_indeces >= ostia_short.time.values[0]] 
            
            par_2d = ostia_short[par].sel(time=cluster_sst_time_indeces.flatten()+pd.to_timedelta(12,"h")) 
        

            # Create a meshgrid of lon and lat coordinates for plotting
            lon_mesh_T, lat_mesh_T = np.meshgrid(ostia_short[par]['lon'], ostia_short[par]['lat'])   
    
            mean_pattern = np.nanmean(par_2d, axis=0)
            mean_pattern = mean_pattern.reshape(ostia_short[par].shape[1], ostia_short[par].shape[2])
            mean_pattern[mean_pattern == 0] = np.nan 
        
        else:  
            par_2d = data_to_use[par].values.reshape(data_to_use[par].shape[0], -1)  # Reshape the data_to_use data into a 2D array (time as rows, lat*lon as columns)
            
            mean_pattern = np.nanmean(par_2d[cluster_indices], axis=0) 
            mean_pattern = mean_pattern.reshape(u_wind.shape[1], u_wind.shape[2])    
            mean_pattern[mean_pattern == 0] = np.nan 
            
            

        
        ax = plt.subplot(len(additional_params)+2, num_clusters, pos+j*num_clusters+1, projection=ccrs.PlateCarree())
        if (par == "analysed_sst") or (par == "sst_anomaly"):
            plt.pcolormesh(lon_mesh_T, lat_mesh_T, mean_pattern, cmap='coolwarm', transform=ccrs.PlateCarree(),
                        vmin = -3, 
                        vmax = 3) 
        elif (par == "CHL"):
            plt.pcolormesh(lon_mesh, lat_mesh, mean_pattern, cmap='BuGn', transform=ccrs.PlateCarree(),
                       vmin = np.nanpercentile(data_to_use[par].values.flatten()[data_to_use[par].values.flatten() != 0], 0), 
                       vmax = np.nanpercentile(data_to_use[par].values.flatten()[data_to_use[par].values.flatten() != 0], 98 )) 
        else:
            plt.pcolormesh(lon_mesh, lat_mesh, mean_pattern, cmap=main_colormap, transform=ccrs.PlateCarree(),
                       vmin = min_cuti, 
                       vmax = max_cuti) 
        # Stations
        ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
    
        # Map
        ax.add_feature(cartopy.feature.BORDERS)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines(resolution='10m', linewidth=1)
        gl = ax.gridlines(draw_labels=True)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False

        
        if pos==num_clusters-1:
            cbar = plt.colorbar(label=label)
            cbar.ax.yaxis.label.set_size(14)

plt.tight_layout()

# fig.savefig("paper_plots/SOM_maps.png", dpi=300)



#%% Plot ERA 5 data for cluster assignments


plot_ERA_5 = 0

if plot_ERA_5 == 1:
    
     # Read ERA 5 data
     era5_path = "../ERA5/"
     file_list = sorted(glob.glob(era5_path +'*geopotential*.nc'))
     era5 = xr.open_mfdataset(file_list[:], concat_dim='time', combine='nested', chunks= 'auto')
     era5["h_geop_500hP"] = era5.z / 9.80665   # geopotential height from geopotential, unit: m
    
     file_list =  sorted(glob.glob(era5_path +'*surface*.nc'))
     era5_surf = xr.open_mfdataset(file_list[:], concat_dim='time', combine='nested', chunks= 'auto')
     
     # assert set(era5.dims) == set(era5_surf.dims)
     era5 = era5.update(era5_surf)
       
       
    

    
     # era5 = era5.sel(latitude= slice(50, 20)  , longitude=slice(-140, -110)  ).isel(time=[1])
    
    
     lon_mesh_era, lat_mesh_era = np.meshgrid(era5['longitude'], era5['latitude'])

    
     additional_params = ['h_geop_500hP', "msl" ] # "sp",
    
     min_cuti = -3
     max_cuti = 3   
     
     max_pres = 5750
     min_pres = 5450
        
    
     fig = plt.figure(figsize=(15, 9))
     plt.suptitle(f"Satellite CUTI "+method+f" clusters {data_to_use.time[0].values.astype('datetime64[Y]').astype(int) + 1970} to {data_to_use.time[-1].values.astype('datetime64[Y]').astype(int) + 1970}")
    
    
     for pos, i in enumerate(sorted_clusters):
     #for i in range(num_clusters):
         
         main_colormap = 'viridis'
         
         ### Indices of this cluster
         cluster_indices = np.where(cluster_assignments == i)[0]
         cluster_time_indeces = data_to_use.time.values.reshape(data_to_use.time.shape[0], -1)[cluster_indices]
         cluster_time_indeces = cluster_time_indeces[cluster_time_indeces >= np.datetime64('2000-01-01T00:00:00')]   # ERA5 currently only available after 2000
         
         
         ### Plot parameter of cluster variable (CUTI)
         # Calculate mean of each pattern
         mean_pattern = np.mean(u_wind_normalized[cluster_indices], axis=0)
         mean_pattern = mean_pattern.reshape(u_wind['lat'].shape[0], u_wind['lon'].shape[0])
         mean_pattern[mean_pattern == 0] = np.nan 
         
         ax = plt.subplot(len(additional_params)+1, num_clusters, pos+1, projection=ccrs.PlateCarree())
         plt.pcolormesh(lon_mesh, lat_mesh, mean_pattern, cmap=main_colormap, transform=ccrs.PlateCarree(),
                        vmin = min_cuti, 
                        vmax = max_cuti) 
         plt.title(f'Cluster {i+1} (N={cluster_counts[i]})') 
         
         # Model CUTI   
         cuti = CUTI[CUTI.index.isin(cluster_time_indeces.flatten())].mean()
         plt.scatter(lons, lats, c=cuti, cmap=main_colormap, transform=ccrs.PlateCarree(),edgecolors="black", 
                    vmin = min_cuti, 
                    vmax = max_cuti)
    
         # Stations
         ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
         ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
    
         # Map
         ax.add_feature(cartopy.feature.BORDERS)
         ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
         ax.add_feature(cartopy.feature.RIVERS)
         ax.add_feature(cartopy.feature.LAND)
         ax.coastlines(resolution='10m', linewidth=1)
         gl = ax.gridlines(draw_labels=True)
         gl.bottom_labels = True
         gl.left_labels = True
         gl.right_labels = False
         gl.top_labels = False
         ax.set_extent([-128,-120, 35, 45], crs=ccrs.PlateCarree())  
    
         
         if pos==num_clusters-1:
             plt.colorbar(label=param)
             
         ### Plot additional variables:
         for j,par in enumerate(additional_params, start=1):
             
             # Calculate mean of each pattern  
             par_2d = era5.sel(time=cluster_time_indeces.flatten()) #  era5[par].values.reshape(era5[par].shape[0], -1)
             mean_pattern = par_2d[par].mean(dim='time')       
             mean_pattern = mean_pattern.where(mean_pattern != 0, np.nan)

             
             ax = plt.subplot(len(additional_params)+1, num_clusters, pos+j*num_clusters+1, projection=ccrs.PlateCarree())
             c = plt.pcolormesh(lon_mesh_era, lat_mesh_era, mean_pattern, cmap='Spectral', transform=ccrs.PlateCarree(),
                            vmin = np.nanpercentile(era5[par].values.flatten()[era5[par].values.flatten() != 0], 5), # min_pres,# 
                            vmax = np.nanpercentile(era5[par].values.flatten()[era5[par].values.flatten() != 0], 95) # max_pres #
                            ) 
             levels = np.linspace(np.nanmin(mean_pattern), np.nanmax(mean_pattern), 10)  # Define contour levels
             plt.contour(lon_mesh_era, lat_mesh_era, mean_pattern, levels=levels, colors='black',
                        transform=ccrs.PlateCarree(), linewidths=0.5)
             
             # Add arrows for wind vectors 
             if par == "msl":
                 u = par_2d["u10"].mean(dim='time').values
                 v = par_2d["v10"].mean(dim='time').values
                 scale_factor = 100  # Adjust the scale factor for arrow length
                 ax.quiver(par_2d['longitude'][::10], par_2d['latitude'][::10],
                          u[ ::10, ::10], v[ ::10, ::10], scale=scale_factor, color='white', transform=ccrs.PlateCarree())
            

             # Stations
             ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
             ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
         
             # Map
             ax.add_feature(cartopy.feature.BORDERS)
             ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
             ax.add_feature(cartopy.feature.RIVERS)
             ax.add_feature(cartopy.feature.LAND)
             ax.coastlines(resolution='10m', linewidth=1)
             gl = ax.gridlines(draw_labels=True)
             gl.bottom_labels = True
             gl.left_labels = True
             gl.right_labels = False
             gl.top_labels = False
    
             
             if pos==num_clusters-1:
                 plt.colorbar(c, label=par)
      
    
     plt.tight_layout()






#%% Plot only the cluster variable


single = 0
if single == 1:
    
    # Visualize the clusters (patterns)
    fig = plt.figure(figsize=(17, 4))
    for i in range(num_clusters):
        ax = plt.subplot(1, num_clusters, i+1, projection=ccrs.PlateCarree())
        cluster_indices = np.where(cluster_assignments == i)[0]
        cluster_time_indeces = data_to_use.time.values.reshape(data_to_use.time.shape[0], -1)[cluster_indices]
        
        mean_pattern = np.mean(u_wind_normalized[cluster_indices], axis=0)
        mean_pattern = mean_pattern.reshape(u_wind.shape[1], u_wind.shape[2])
        mean_pattern[mean_pattern == 0] = np.nan 
        plt.pcolormesh(lon_mesh, lat_mesh, mean_pattern, cmap='coolwarm_r', transform=ccrs.PlateCarree(),
                       vmin = min_cuti, 
                       vmax = max_cuti) 
        plt.title(f'Cluster {i+1} (N={cluster_counts[i]})') 
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Stations
        ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
    
        # Model CUTI
        cuti = CUTI[CUTI.index.isin(cluster_time_indeces.flatten())].mean()
        plt.scatter(lons, lats, c=cuti, cmap=main_colormap, transform=ccrs.PlateCarree(),edgecolors="black", s=80,
                   vmin = min_cuti, 
                   vmax = max_cuti)
    
        # Map
        ax.add_feature(cartopy.feature.BORDERS)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines(resolution='10m', linewidth=1)
        
        gl = ax.gridlines(draw_labels=True)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False
        ax.set_extent([-128,-120, 35, 45], crs=ccrs.PlateCarree())  
        
    plt.colorbar(label=param)
    plt.tight_layout()
    plt.subplots_adjust( wspace=0.15)

    


#%% Histogram

histogram = 0
if histogram == 1:
    
    
    # Morro
    offshore_diff = 75 # 75 is the distance used in Jacox 2018
    lat = round(Morro_coords[0])
    max_lon = coastline[np.where(coastline[:, 1] ==lat)[0][0], 0]
    min_lon = max_lon + lon_difference(offshore_diff, max_lon)
    sat_morro = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 
    
    # Humboldt
    lat = round(Humboldt_coords[0])
    max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0]
    min_lon = max_lon + lon_difference(offshore_diff, max_lon)
    sat_humb = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon'])     
    
    
    plt.figure()
    plt.title(f"Histograms: CUTI model {CUTI.index[0].year}-{CUTI.index[-1].year}, sat {sat.time[0].values.astype('datetime64[Y]').astype(int) + 1970}-{sat.time[-1].values.astype('datetime64[Y]').astype(int) + 1970}")
    plt.hist(CUTI["36N"], bins=150, density=True, label = "Model Morro", alpha = 0.5, color = 'brown')
    plt.hist(sat_morro["CUTI_Cop"].values, bins=150, density=True, label = "Sat Morro", alpha = 0.5, color = 'coral')
    plt.hist(CUTI["41N"], bins=150, density=True, label = "Model Humboldt", alpha = 0.5, color = 'darkblue')
    plt.hist(sat_humb["CUTI_Cop"].values, bins=150, density=True, label = "Sat Humboldt", alpha = 0.5, color = 'violet')
    plt.legend()
    
    # plt.hist(data_to_use.sel(lat=40, lon=-125, method="nearest")["analysed_sst.values"], bins=100)
    





#%% K-means for SST

SST_clustering = 0

if SST_clustering == 1:

    
    ostia_short = ostia# .sel(time=ostia.time> pd.to_datetime("2019-12-31T12:00:00"))  # 
    
    # PArameter on which clustering is based
    param =   "sst_anomaly"  # "tau_cross" 'U_Ek' # 'analysed_sst'   'CUTI_Cop' CUTI_model
    
    
    # Get the coastline and 75km offshore line from the ostia coordinates
    target_latitudes = ostia_short.lat.values # Define the target latitudes (ostia_short.lat.values)
    interpolated_coastline = np.empty((len(target_latitudes), 3))# Initialize the output array to store interpolated columns
    for col in range(3): # Iterate through each column to interpolate
        # Extract valid data points for the latitude reference (second column)
        valid_mask = ~np.isnan(coastline[:, col])
        valid_latitudes = coastline[valid_mask, 1]  # Latitude column
        valid_values = coastline[valid_mask, col]
        interpolated_values = np.interp(target_latitudes, valid_latitudes, valid_values)# Interpolate the values for the current column
        interpolated_coastline[:, col] = interpolated_values# Store the results in the corresponding column of the output array
    interpolated_coastline = np.round(interpolated_coastline, 3)
        
    
    
    
    
    # make the clustering based on the near-shore pattern
    lon_diff = get_offshore_parallel_coastline(interpolated_coastline, offshore_diff=75) # Calculate the longitude of the 100km offshore line based on latitude
    ostia_short_masked = ostia_short.copy(deep=True)
    
    for i, lat_sat in enumerate(ostia_short_masked.lat.values):   # Loop through each latitude value and remove latitudes outside of the 100 km coastline
        lon_sat = lon_diff[i] 
        lon_coast = interpolated_coastline[i, 0]
        ostia_short_masked[param].loc[dict(lat=lat_sat, lon=slice(-150, lon_sat) )] =np.nan
        ostia_short_masked[param].loc[dict(lat=lat_sat, lon=slice(lon_coast,  -110) )] =np.nan
        
    
    
    
    # plt.figure()
    # ax = plt.subplot(1, 1,  1,projection=ccrs.PlateCarree())
    # plt.grid()
    # plt.plot(coastline[:, 0], coastline[:, 1],".")
    # plt.plot(interpolated_coastline[:, 0], interpolated_coastline[:, 1],".")
    # plt.plot(lon_diff, interpolated_coastline[:, 1],".")
    # df  = ostia_short.sel(time= '2021-07-23', method='nearest')
    # ax.pcolormesh(df['lon'], df['lat'], df["sst_anomaly"][ :, :],cmap='coolwarm', transform=ccrs.PlateCarree(), shading='auto' )
    
    
    

    

    # Extracting the variable of interest for clustering
    u_wind = ostia_short_masked[param].dropna(dim="lon", how = "all").dropna(dim="lat", how = "all").dropna(dim="time", how = "all").fillna(0)

    
    # Reshape the u_wind data into a 2D array (time as rows, lat*lon as columns)
    u_wind_2d = u_wind.values.reshape(u_wind.shape[0], -1)
    
    # Normalize the data to have values between 0 and 1 - scaling not necessary here since data are at similar values
    u_wind_normalized =u_wind_2d #  (u_wind_2d - u_wind_2d.min()) / (u_wind_2d.max() - u_wind_2d.min())  #  
    

    
    
    
    #% Cluster algorithm
    
    # select cluster method
    method = "k-means"
    
    
    ## K-means clustering for CUTI and other variables
    if method == "k-means" :
    
        ### This modification calculates the distances between each time step and each cluster centroid and assigns each time step to the cluster with the minimum distance (most similar spatial pattern).
        
        # Define the number of clusters (patterns) you want to identify
        num_clusters = 4
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init=10) #,  random_state=42)
        cluster_labels = kmeans.fit_predict(u_wind_normalized)
        
        
        # Calculate the distance between each time step and each cluster centroid - most metrics provide very similar results default: euclidean
        distances = cdist(u_wind_normalized, kmeans.cluster_centers_, metric =  'cosine')
        
        # Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
        print(kmeans.inertia_)
        
        # Assign each time step to the cluster with the minimum distance (most similar spatial pattern)
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Count the number of cases in each cluster
        cluster_counts = [np.sum(cluster_assignments == i) for i in range(num_clusters)]
        
    sorted_clusters = np.argsort(cluster_counts)
    
 
    
    # %
    fig = plt.figure(figsize=(18, 4), layout='constrained')
    for i in  sorted_clusters:

        ax = plt.subplot(1, num_clusters, i+1, projection=ccrs.PlateCarree())
        cluster_indices = np.where(cluster_assignments == i)[0]
        cluster_time_indeces = ostia_short_masked.time.values.reshape(ostia_short_masked.time.shape[0], -1)[cluster_indices] - pd.to_timedelta("12h")
 
 

        
        # Re-do the flattening with the non-masked array
        u_wind = ostia_short[param].dropna(dim="lon", how = "all").dropna(dim="lat", how = "all").dropna(dim="time", how = "all").fillna(0)
        u_wind_2d = u_wind.values.reshape(u_wind.shape[0], -1)
        u_wind_normalized =u_wind_2d 
        
        # Create a meshgrid of lon and lat coordinates for plotting
        lon_mesh, lat_mesh = np.meshgrid(u_wind['lon'], u_wind['lat'])         
         
               
        mean_pattern = np.nanmean(u_wind_normalized[cluster_indices], axis=0)
        mean_pattern = mean_pattern.reshape(u_wind.shape[1], u_wind.shape[2])
        mean_pattern[mean_pattern == 0] = np.nan 
        
        
        plt.pcolormesh(lon_mesh, lat_mesh, mean_pattern, cmap='coolwarm', transform=ccrs.PlateCarree(),
                   vmin = -3, 
                   vmax = 3) 
        if i==3:
               
            if param == "sst_anomaly":
                label = "SST anomaly ($^\circ$C)"
            else:
                label = param
            plt.colorbar(label=label, pad=0.1)
        plt.title(f'Cluster {i+1} (N={cluster_counts[i]})') 
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Stations
        ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
    
        # Model CUTI
        cuti = CUTI[CUTI.index.isin(cluster_time_indeces.flatten())].mean()
        plt.scatter(lons, lats, c=cuti, cmap="viridis", transform=ccrs.PlateCarree(),edgecolors="black", s=80,
                   vmin = min_cuti, 
                   vmax = max_cuti)
        if i == 3:
            cbar = plt.colorbar(label="CUTI (m$^2$/s)" )
    
        # Map
        ax.add_feature(cartopy.feature.BORDERS)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines(resolution='10m', linewidth=1)
        
        ax.plot(lon_diff, interpolated_coastline[:, 1],color="grey")
        
        gl = ax.gridlines(draw_labels=True)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False
        ax.set_extent([-128,-120, 35, 45], crs=ccrs.PlateCarree())  
    
        
    # plt.tight_layout()
    # plt.subplots_adjust( wspace=0.15)







