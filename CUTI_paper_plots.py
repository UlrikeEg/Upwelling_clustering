import pandas as pd
import numpy as np
import earthaccess
import matplotlib
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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
import calendar
sys.path.append("../../NSO/NSO_data_processing")
from Functions_general import *
from scipy import signal
import matplotlib.colors as mcolors



sys.path.append("../satellite_data")
sys.path.append("../buoy_data")

from Read_satellites import read_ssa, read_sentinel_a2e


from Functions_Oracle import *






#%%  Read datasets


sat = xr.open_dataset("Sat_data_combined_final_not_chunked2.nc", engine = "netcdf4")
#sat = sat.chunk({'time': 100, 'lat': 40, 'lon': 40})
sat[['U_Geo_Oscar',"ug", "vg"]] = sat[['U_Geo_Oscar',"ug", "vg"]].transpose(*('time','lat', 'lon'))

# CUTI data
CUTI = pd.read_csv('C:/Users/uegerer/Desktop/Oracle/CUTI_data/CUTI_daily.csv', header = 0, index_col=None).apply(pd.to_numeric, errors='coerce')
CUTI.index = pd.to_datetime(CUTI[['year', 'month', 'day']])


# Coastline
coastline = np.load("coastline.npy")



#  Chlorophyll data
# files = glob.glob('../satellite_data/Chlorophyll/cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D_*.nc')
# chl = xr.open_mfdataset(files, combine='nested', compat ='override', chunks='auto', parallel=True)   
# chl = chl.rename({'latitude': 'lat', 'longitude': 'lon' })
# chl.to_netcdf('../satellite_data/Chlorophyll/CHL_1997_2023.nc')
chl = xr.open_dataset('../satellite_data/Chlorophyll/CHL_1997_2023.nc')

# SST data
ostia = xr.load_dataset("../satellite_data/Ostia/ostia_all_1993_2023_4.nc", engine="netcdf4", chunks={'time': 100, 'lat': 100, 'lon': 100})  # all coordinates close to California coast


  





# sat_humb = sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest")

# plt.figure()
# plt.plot(sat_humb.time, sat_humb.U_Geo,".-")
# plt.plot(sat_humb.time, sat_humb.U_Geo_Oscar,".-")
# plt.plot(sat_humb.time, sat_humb.U_Geo_cop,".-")
# plt.plot(sat_humb.time, sat_humb.U_Ek,".-")

# # Add day of year
# sat = sat.assign_coords(DOY=pd.to_datetime(sat['time'].values).dayofyear)



#%% re-calculate CUTI with parameterized wind stress

calc_tau = 0
if calc_tau == 1:
      
    
    
    # tau calculkation (Charnock parameterization)

    sat["wind_speed"] = (sat.u_wind**2 + sat.v_wind**2 )**0.5
    
    def calculate_drag_coefficient(V):
        """
        based on Trenberth 1990, https://doi.org/10.1175/1520-0485(1990)020<1742:TMACIG>2.0.CO;2
        
        Calculate the neutral drag coefficient (C_N) based on wind speed (V).
        
        Parameters:
            V (float): Wind speed in m/s.
        
        Returns:
            float: Drag coefficient (C_N) as a dimensionless value.
        """
        if V > 10:
            return (0.49 + 0.065 * V) * 1e-3
        elif 3 <= V <= 10:
            return 1.14 * 1e-3
        elif 0 < V < 3:
            return (0.62 + 1.56 / V) * 1e-3
        elif V <=0:
            return np.nan
    
    
    
    sat = sat.chunk({'time': 10})
    sat['C_D'] = xr.apply_ufunc(
        calculate_drag_coefficient,  # Apply the function directly
        sat['wind_speed'],           # Apply to the wind_speed variable
        dask="parallelized",         # Enable parallelization for large datasets
        output_dtypes=[float],       # Ensure the output is a float
        vectorize=True               # This will handle vectorization internally
        )
    
    # plt.hist(sat.isel(time=no).C_D.values.flatten(), bins=100)
    
    #sat['C_D'] =0.018*  (sat["wind_speed"])/10 # Chatgpt (???)
    
    sat['C_D'] = 4.4 * 1e-4  *sat['wind_speed']**0.55   # Stull, p. 266, based on Charnack
    
    sat['x_tau_param'] = 1.225* sat['C_D'] *sat['u_wind'] * abs(sat['u_wind'])
    sat['y_tau_param'] = 1.225* sat['C_D'] *sat['v_wind'] * abs(sat['v_wind'])
    
    CD_10 = 4.4 * 1e-4  *10**0.55   # Stull, p. 266, based on Charnack
    CUTI_10ms = 1.225 * CD_10 * 10**2 / rho_ocean / calculate_coriolis_parameter(40)  
    
    CD_8 = 4.4 * 1e-4  *8**0.55   # Stull, p. 266, based on Charnack
    CUTI_8ms = 1.225 * CD_8 * 8**2 / rho_ocean / calculate_coriolis_parameter(40)  
    
    
    ## compare with NBS tau 
    
    # plt.figure()
    # no=slice(0,100)
    # plt.plot(sat.isel(time=no)['y_tau'].values.flatten(), sat.isel(time=no)['y_tau'].values.flatten(),".")
    
    # R = pd.Series(sat.isel(time=no)['y_tau'].values.flatten()).corr(pd.Series(sat.isel(time=no)['y_tau_param'].values.flatten()))
    # plt.plot(sat.isel(time=no)['y_tau'].values.flatten(), sat.isel(time=no)['y_tau_param'].values.flatten(),".", label = "tau_y, R={:.2f}".format(R))
    
    # R = pd.Series(sat.isel(time=no)['x_tau'].values.flatten()).corr(pd.Series(sat.isel(time=no)['x_tau_param'].values.flatten()))
    # plt.plot(sat.isel(time=no)['x_tau'].values.flatten(), sat.isel(time=no)['x_tau_param'].values.flatten(),".", label = "tau_x, R={:.2f}".format(R))
    
    # plt.legend()
    
    
    # Add the coastline angle values to the sat dataset
    sat = add_costline_angle_to_sat(sat, plot = 0)


    # Coriolis parameter
    sat["f"] = calculate_coriolis_parameter(sat.lat)  

    # sea water density
    rho_ocean = 1025 # kg/m3

    # Calculate along-and cross shore wind stress
    sat['tau_along_param'], sat['tau_cross_param'] = calculate_along_cross_shore_components(
                                            sat.x_tau_param, sat.y_tau_param, coastline_angle=sat.coastline_angle)
    
    plt.figure()
    no=slice(0,100)
    day = '2021-07-23'
    plt.plot(sat.sel(time=pd.to_datetime(day), method='nearest')['y_tau'].values.flatten(), sat.sel(time=pd.to_datetime(day), method='nearest')['y_tau'].values.flatten(),".")
    
    R = pd.Series(sat.sel(time=pd.to_datetime(day), method='nearest')['y_tau'].values.flatten()).corr(pd.Series(sat.sel(time=pd.to_datetime(day), method='nearest')['y_tau_param'].values.flatten()))
    plt.plot(sat.sel(time=pd.to_datetime(day), method='nearest')['y_tau'].values.flatten(), sat.sel(time=pd.to_datetime(day))['y_tau_param'].values.flatten(),".", label = "tau_y, R={:.2f}".format(R))
    
    R = pd.Series(sat.sel(time=pd.to_datetime(day), method='nearest')['x_tau'].values.flatten()).corr(pd.Series(sat.sel(time=pd.to_datetime(day), method='nearest')['x_tau_param'].values.flatten()))
    plt.plot(sat.sel(time=pd.to_datetime(day), method='nearest')['x_tau'].values.flatten(), sat.sel(time=pd.to_datetime(day), method='nearest')['x_tau_param'].values.flatten(),".", label = "tau_x, R={:.2f}".format(R))
    
    R = pd.Series(sat.sel(time=pd.to_datetime(day), method='nearest')['tau_along'].values.flatten()).corr(pd.Series(sat.sel(time=pd.to_datetime(day), method='nearest')['tau_along_param'].values.flatten()))
    plt.plot(sat.sel(time=pd.to_datetime(day), method='nearest')['tau_along_param'].values.flatten(), sat.sel(time=pd.to_datetime(day), method='nearest')['tau_along_param'].values.flatten(),".", label = "tau_along, R={:.2f}".format(R))
    
    plt.grid()    
    
    plt.xlabel("tau from NBS")
    plt.ylabel("tau parameterized from NBS wind speed")
    
    
    plt.legend()
    
    plt.figure()
    #sat.sel(time=pd.to_datetime(day), method='nearest')['tau_along'].plot(vmin=-0.3, vmax=0.3)
    sat.sel(time=pd.to_datetime(day), method='nearest')['tau_along'].plot(vmin=-0.3, vmax=0.3)
        

    # Ekman transport
    sat['U_Ek_param'] = sat['tau_along_param'] / rho_ocean / sat["f"]
    # remove outliers
    sat['U_Ek_param'] = sat['U_Ek_param'].where(sat['U_Ek_param']>-10)
    sat['U_Ek_param'] = sat['U_Ek_param'].where(sat['U_Ek_param']<20)
       

    # Calculate CUTI from satellite data
    sat['CUTI_Oscar_param'] = (sat['U_Ek_param'] + sat['U_Geo_Oscar'])
    sat['CUTI_Oscar_param_neg_geo'] = (sat['U_Ek_param'] - sat['U_Geo_Oscar'])


#%%  Curl-driven component



test_curl = 0
if test_curl == 1: 

    
    # Constants
    R = 6371000  # Earth's radius in meters
    
    # Convert degrees to radians for latitude and longitude
    lat_rad = np.deg2rad(sat['lat'])
    lon_rad = np.deg2rad(sat['lon'])
    
    # Calculate grid spacing in meters
    dlat = np.gradient(lat_rad) * R
    dlon = np.gradient(lon_rad) * R * np.cos(lat_rad)
    
    # # Set a random grid point to zero to see impact of curl  
    # sat['y_tau_param'].loc[{'lat': slice(38, 38.5), 'lon': slice(-124.5, -124)}] = 0 
    # sat['x_tau_param'].loc[{'lat': slice(38, 38.5), 'lon': slice(-124.5, -124)}] = 0 
   
    
    # sat['x_tau'] = sat['x_tau'].where(~sat['U_Geo_Oscar'].isnull())
    # sat['y_tau'] = sat['y_tau'].where(~sat['U_Geo_Oscar'].isnull())
    
    
    # Compute derivatives and curl
    dvdx = sat['y_tau_param'].differentiate('lon') / dlon  # Partial derivative of vg with respect to x
    dudy = sat['x_tau_param'].differentiate('lat') / dlat  # Partial derivative of ug with respect to y
    
    sat["curl"] = dvdx - dudy
    
    sat["w_curl"] = 1/(calculate_coriolis_parameter(sat.lat)*1025) * sat.curl 
    
    sat["CUTI_curl"] = sat["w_curl"] * dlon
    
    sat['CUTI_Oscar_curl'] = (sat['U_Ek'] + sat['U_Geo_Oscar'] + sat["CUTI_curl"])
    
    

    
    

    
    
  #    params = ['x_tau_param', 'y_tau_param', "CUTI_curl"]  # "x_tau", "x_tau_diff" ] # "CUTI_Oscar"]
    params = ['CUTI_Oscar', 'CUTI_Oscar_param', "CUTI_curl", "CUTI_Oscar_curl"] 
    no =10426
       
    plt.figure()
    plt.suptitle(day)
    
    for i, param in enumerate(params):
    
          ax = plt.subplot(1, len(params),  i+1, projection=ccrs.PlateCarree())
          ax.coastlines(resolution='10m', linewidth=1)
          if param=="CUTI_curl_test":
              sat.isel(time=no)[param].plot(vmin=-3, vmax=3, cmap='viridis') #
          else:
              sat.isel(time=no)[param].plot(
                  vmin=-3, vmax=3,
                  cmap='viridis',
                  cbar_kwargs={'label': f"{param} ($m^2/s$)"}) # 
          ax.set_extent([-128,-120, 35, 42], crs=ccrs.PlateCarree())  
          
          if param=="CUTI_Oscar":
              title = "CUTI, original"
          elif param=="CUTI_Oscar_param":
                title = "CUTI, with wind stress parameterized"
          elif param=="CUTI_curl":
                title = "curl-driven CUTI"
          elif param=="CUTI_Oscar_curl":
                title = "CUTI, incl. curl component"
          else: 
                title = param
          ax.set_title(title)
     
        
          u =   sat.isel(time=no)['x_tau_param'].values
          v =   sat.isel(time=no)['y_tau_param'].values
          scale_factor = 2  # Adjust the scale factor for arrow length
          ax.quiver(sat['lon'], sat['lat'],
                    u[ :, :], v[ :, :], scale=scale_factor, color='black', transform=ccrs.PlateCarree())
                
        

    
    # params = [ 'ug', 'vg', "U_Geo_Oscar"]  # "x_tau", "x_tau_diff" ] # "CUTI_Oscar"]
       
    # plt.figure()
    
    # for i, param in enumerate(params):
    
    #     ax = plt.subplot(1, len(params),  i+1, projection=ccrs.PlateCarree())
    #     ax.coastlines(resolution='10m', linewidth=1)
    #     sat.isel(time=no)[param].plot( cmap='coolwarm') #vmin=-1, vmax=1,
    
    #     ax.set_extent([-128,-120, 35, 42], crs=ccrs.PlateCarree())  
     
    #     # if param=="curl":
    #     u =   sat.isel(time=no)['ug'].values
    #     v =   sat.isel(time=no)['vg'].values
    #     scale_factor = 2  # Adjust the scale factor for arrow length
    #     ax.quiver(sat['lon'], sat['lat'],
    #               u[ :, :], v[ :, :], scale=scale_factor, color='black', transform=ccrs.PlateCarree())
                
        








#%%  Error calculation



error_calc = 0
if error_calc == 1: 
    
    # Define the variables
    U = np.sqrt(sat.u_wind**2 + sat.v_wind**2) # (m/s)
    U_along, U_cross = calculate_along_cross_shore_components(
                                             sat.u_wind, sat.v_wind, coastline_angle=sat.coastline_angle)
    C_D = 4.4 * 1e-4  *U**0.55    # 0.0026  # from Bakun 1973 as noted in Jacox 2018
    
    # tau_u = 1.225* C_D *sat['u_wind'] * abs(sat['u_wind'])
    
    
    # plt.figure()
    # plt.hist(sat["x_tau"].values.flatten(), bins=60, density=True, range=(-0.5, 0.5))
    # plt.hist(tau_u.values.flatten(), bins=60, density=True, range=(-0.5, 0.5))
   
    sigma_U = 2  # Uncertainty in wind speed (m/s)
  
    
    f = calculate_coriolis_parameter(sat.lat)    # Coriolis parameter
    
    MLD = sat.mld_sat  # Mixed Layer Depth in meters 
    sigma_u_geo_cross = 0.082  # Uncertainty in geostrophic velocity (cross-component), m/s
    u_geo_cross = sat.go_cross  # Geostrophic velocity (cross-component)
    sigma_MLD = 29.7  # Uncertainty in MLD, m
    
    # Calculate the CUTI uncertainty
    sat["CUTI_error"] = np.sqrt(
        (2 * C_D * abs(U_along) * sigma_U * 1.225/1025 / f)**2 
         + (MLD * sigma_u_geo_cross)**2 
      + (u_geo_cross * sigma_MLD)**2
    )
    
    
    # Calculate the longitude of the 100km offshore line based on latitude
    lon_diff = get_offshore_parallel_coastline(coastline, offshore_diff=75)
    
    sat_masked = sat.copy()

    # Loop through each latitude value and remove latitudes outside of the 75 km coastline
    for i, lat_sat in enumerate(sat_masked.lat.values):
        lon_sat = lon_diff[i] 
        sat_masked[[ "CUTI_error", "CUTI_Oscar"]].loc[dict(lat=lat_sat, lon=slice(-150, lon_sat-0.25 ) )] =np.nan
        

    
    
    # plt.figure()
    # plt.hist(sat_masked["CUTI_error"].values.flatten(), bins=60, density=True)
    
    

    # plt.figure()
    # plt.plot(sat_masked.CUTI_Oscar.values.flatten(), sat_masked.CUTI_error.values.flatten(),".", ms=1, alpha=0.1)

    # plt.figure()
    # plt.hist(sat_masked["CUTI_Oscar"].values.flatten(), bins=60, density=True)
    
    
    # Define the range and bins for CUTI_Oscar
    cuti_bins = np.arange(-5, 6, 1)  # Steps of 1 from -5 to 5
    cuti_labels = [(cuti_bins[i], cuti_bins[i+1]) for i in range(len(cuti_bins)-1)]
    
    # Generate a colormap from coolwarm centered at 0
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=-5, vmax=5)
    
    # # relative error
    # sat_masked["CUTI_error"] = abs(sat_masked["CUTI_error"]/sat_masked["CUTI_Oscar"])
    
    
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.rcParams.update({'font.size': 15})
    
    
    # Initialize an array to keep track of the cumulative base for each bin
    cumulative_base = np.zeros(60)  # Assuming 60 bins for the histogram
    
    # Define bin edges for the histogram
    bin_edges = np.linspace(0, 10, 61)  # 60 bins between 0 and 10
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
    # Loop through the bins and plot histograms
    for i, (low, high) in enumerate(cuti_labels):
        # Filter the dataset for the current range
        mask = (sat_masked["CUTI_Oscar"] >= low) & (sat_masked["CUTI_Oscar"] < high)
        cuti_error_values = sat_masked["CUTI_error"].where(mask, drop=True).values.flatten()
        
        #cuti_error_values = cuti_error_values[(cuti_error_values >= 0) & (cuti_error_values <= 10)]
        
        # Remove NaN values from the data
        cuti_error_values = cuti_error_values[~np.isnan(cuti_error_values)]
        
        # Skip empty data
        if len(cuti_error_values) == 0:
            continue
        
        # Calculate histogram counts for the current range
        counts, _ = np.histogram(cuti_error_values, bins=bin_edges, density=True)
        
        bar_width = abs((low + high) / 2 - 5)/60
        
        # Plot the histogram with stacking
        plt.bar(
            bin_centers, counts, 
            width=np.diff(bin_edges), #bar_width, #
            align='center', 
            alpha=0.4, 
            label=f'{low} ≤ CUTI < {high}',
            color=cmap(norm((low + high) / 2)),  # Color based on the center of the bin
            edgecolor=cmap(norm((low + high) / 2)),
            bottom=0#cumulative_base
        )
        
        # Update the cumulative base for the next histogram
        cumulative_base += counts
        

        
    # Add legend, labels, and title
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable with colorbar
    cbar = fig.colorbar(sm, ax=ax)  
    cbar.set_label("CUTI (m$^2$/s)")
    plt.xlabel("Propagated RMS uncertainty of CUTI (m$^2$/s)")
    plt.ylabel("Probability density")  # (Staggered)
    plt.grid(True)
    plt.tight_layout()

    
   # plt.savefig("paper_plots/CUTI_error.png", dpi=300)
     
    
    
    
    
    
    
#%% Time series for GEng



time_series_geng = 0

if time_series_geng == 1:
    
    # Calculate wind speed and direction
    sat['wind_speed'] = (sat.u_wind**2 + sat.v_wind**2)**0.5
    sat['wdir'] = np.degrees(np.arctan2(sat.u_wind, sat.v_wind)) + 180
    
    # Isolate the Morro and Humboldt location
    Buoy_coords = [37.356, -122.881]
    sat_humb = sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").to_dataframe()
    sat_morro = sat.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest").to_dataframe()  
    
    
    ostia_humb = ostia.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").to_dataframe()
    ostia_morro = ostia.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest").to_dataframe()    
    
    
    start = "2021-01-14"
    end = "2021-01-24"
    
    weekly_avg_humb = sat_humb[start:end]
    weekly_avg_morro = sat_morro[start:end]
    weekly_avg_ostia_humb = ostia_humb[start:end]
    weekly_avg_ostia_morro = ostia_morro[start:end]
    
    cuti = CUTI[start:end]
    
    #%
    
    x = 0.01
    y = 0.85
    
    # Plot
    num_rows=4
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure( figsize=(16, 9.5))
    
    # Wind speed
    ax0 = plt.subplot(num_rows, 2, 1) 
    plt.title("Buoy")
    plt.ylabel("Wind speed (m/s)")
    plt.grid(True)
    plt.plot(weekly_avg_humb.wind_speed, color="black", lw=1, alpha = 1)  # Average over all years
    ax0.set_xticklabels([])
    plt.ylim(0,14)
    ax0.text(x,y, "(a)", transform=ax0.transAxes)
       
    
    ax = plt.subplot(num_rows, 2, 2, sharex = ax0) 
    plt.title("Morro Bay")
    plt.grid(True)
    plt.plot(weekly_avg_morro.wind_speed, color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax0.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,y, "(b)", transform=ax.transAxes)
    
    # Wind direcion
    ax1 = plt.subplot(num_rows, 2, 3, sharex = ax0) 
    plt.ylabel("Wind dir ($^\circ$)")
    plt.grid(True)
    plt.plot(weekly_avg_humb.wdir,".", color="black", ms=5, alpha = 1)  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticklabels([])
    ax1.text(x,y, "(c)", transform=ax1.transAxes)
    
    ax = plt.subplot(num_rows, 2, 4, sharex = ax0) 
    plt.grid(True)
    plt.plot(weekly_avg_morro.wdir,".", color="black", ms=5, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,0.6, "(d)", transform=ax.transAxes)
    
        
    # SST
    ax1 = plt.subplot(num_rows, 2, 5, sharex = ax0) 
    plt.ylabel("SST ($^\circ$C)")
    plt.grid(True)
    plt.plot(weekly_avg_ostia_humb.analysed_sst, color="black", lw=1, alpha = 1)  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticklabels([])
    ax1.text(x,y, "(e)", transform=ax1.transAxes)
       
    
    ax = plt.subplot(num_rows, 2, 6, sharex = ax0) 
    plt.grid(True)
    plt.plot(weekly_avg_ostia_morro.analysed_sst,color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    # ax.set_ylim(ax1.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,y, "(f)", transform=ax.transAxes)
    
    # CUTI
    ax1 = plt.subplot(num_rows, 2,7) 
    plt.ylabel("CUTI (m$^2$/s)")
    plt.grid(True)
    plt.plot(weekly_avg_humb.CUTI_Oscar, color="black", lw=1, alpha = 1, label = "satellite")  # Average over all years
    plt.plot(cuti["41N"], ".", ms=4, label = "model")
    ax1.set_xlim(ax0.get_xlim())
    
    plt.ylim(-4,4)
    ax1.text(x,y, "(g)", transform=ax1.transAxes)
    
    
    ax = plt.subplot(num_rows, 2,8, sharey=ax1) 
    plt.grid(True)
    plt.plot(weekly_avg_morro.CUTI_Oscar,color="black", lw=1, alpha = 1)  # Average over all years
    plt.plot(cuti["36N"], ".", ms=4, label = "model")
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    ax.text(x,y, "(h)", transform=ax.transAxes)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
     
    fig.autofmt_xdate()
    

#%% Time series short version



time_series = 0

if time_series == 1:
    
    # Calculate wind speed and direction
    sat['wind_speed'] = (sat.u_wind**2 + sat.v_wind**2)**0.5
    sat['wdir'] = np.degrees(np.arctan2(sat.u_wind, sat.v_wind)) + 180
    
    # Isolate the Morro and Humboldt location
    sat_humb = sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").to_dataframe()
    sat_morro = sat.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest").to_dataframe()  
    
    sat_humb.CUTI_Oscar.median()
    sat_morro.CUTI_Oscar.median()
    
    ostia_humb = ostia.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").to_dataframe()
    ostia_morro = ostia.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest").to_dataframe()    
    
    
    # resample to weeks and re-calculate wind dir  (for week of year use "W" nand %U, for day pf year use "D" and %j)
    # this is called weekly average, but we use daily instead.
    sat_humb = sat_humb.resample("D").median()
    sat_humb["week"] = [int(x) for x in sat_humb.index.strftime("%j")] # sat_humb.index.isocalendar().week
    sat_humb['wdir'] = np.degrees(np.arctan2(sat_humb.u_wind, sat_humb.v_wind)) + 180
    weekly_avg_humb = sat_humb.groupby('week').median()
    weekly_avg_humb['wdir'] = np.degrees(np.arctan2(weekly_avg_humb.u_wind, weekly_avg_humb.v_wind)) + 180
    
    sat_morro = sat_morro.resample("D").median()
    sat_morro["week"] = [int(x) for x in sat_morro.index.strftime("%j")]
    sat_morro['wdir'] = np.degrees(np.arctan2(sat_morro.u_wind, sat_morro.v_wind)) + 180
    weekly_avg_morro = sat_morro.groupby('week').median()
    weekly_avg_morro['wdir'] = np.degrees(np.arctan2(weekly_avg_morro.u_wind, weekly_avg_morro.v_wind)) + 180
    
    ostia_humb = ostia_humb.resample("D").median()
    ostia_humb["week"] = [int(x) for x in ostia_humb.index.strftime("%j")] # sat_humb.index.isocalendar().week
    weekly_avg_ostia_humb = ostia_humb.groupby('week').median()

    ostia_morro = ostia_morro.resample("D").median()
    ostia_morro["week"] = [int(x) for x in ostia_morro.index.strftime("%j")] # sat_morro.index.isocalendar().week
    weekly_avg_ostia_morro = ostia_morro.groupby('week').median()   
    
    start = "2021-01-14"
    end = "2021-01-24"
    
    weekly_avg_humb = sat_humb[start:end]
    weekly_avg_morro = sat_morro[start:end]
    weekly_avg_ostia_humb = ostia_humb[start:end]
    weekly_avg_ostia_morro = ostia_morro[start:end]
    
    cuti = CUTI[start:end]
    
    #%%
    
    x = 0.01
    y = 0.85
    
    # Plot
    num_rows=4
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure( figsize=(16, 9.5))
    
    # Wind speed
    ax0 = plt.subplot(num_rows, 2, 1) 
    plt.title("Humboldt")
    plt.ylabel("Wind speed (m/s)")
    plt.grid(True)
    # for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.wind_speed, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_humb.wind_speed, color="black", lw=1, alpha = 1)  # Average over all years
    ax0.set_xticklabels([])
    plt.ylim(0,14)
    ax0.text(x,y, "(a)", transform=ax0.transAxes)

    
    ax = plt.subplot(num_rows, 2, 2, sharex = ax0) 
    plt.title("Morro")
    plt.grid(True)
    # for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.wind_speed, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_morro.wind_speed, color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax0.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,y, "(b)", transform=ax.transAxes)
    
    # Wind direcion
    ax1 = plt.subplot(num_rows, 2, 3, sharex = ax0) 
    plt.ylabel("Wind dir ($^\circ$)")
    plt.grid(True)
    # for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.wdir,".", color="C0", ms=0.5, alpha = 0.5)
    plt.plot(weekly_avg_humb.wdir,".", color="black", ms=5, alpha = 1)  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticklabels([])
    ax1.text(x,y, "(c)", transform=ax1.transAxes)
    
    ax = plt.subplot(num_rows, 2, 4, sharex = ax0) 
    plt.grid(True)
    # for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.wdir,".", color="C0", ms=0.5, alpha = 0.5)
    plt.plot(weekly_avg_morro.wdir,".", color="black", ms=5, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,0.6, "(d)", transform=ax.transAxes)
    
    
    # # Offshore grostrophic current
    # ax1 = plt.subplot(num_rows, 2, 5, sharex = ax0) 
    # plt.ylabel("$u_{geo, cross}$ (m/s)")
    # plt.grid(True)
    # for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.go_cross, color="C0", lw=0.5, alpha = 0.5)
    # plt.plot(weekly_avg_humb.go_cross, color="black", lw=1, alpha = 1)  # Average over all years
    # ax1.set_xlim(ax0.get_xlim())
    # ax1.set_xticklabels([])
    # ax1.text(x,y, "(e)", transform=ax1.transAxes)
    
    # ax = plt.subplot(num_rows, 2, 6, sharex = ax0) 
    # plt.grid(True)
    # for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.go_cross, color="C0", lw=0.5, alpha = 0.5)
    # plt.plot(weekly_avg_morro.go_cross,color="black", lw=1, alpha = 1)  # Average over all years
    # ax.set_xlim(ax0.get_xlim())
    # ax.set_ylim(ax1.get_ylim())
    # ax.set_xticklabels([])
    # ax.text(x,y, "(f)", transform=ax.transAxes)
    
        
    # SST
    ax1 = plt.subplot(num_rows, 2, 5, sharex = ax0) 
    plt.ylabel("SST ($^\circ$C)")
    plt.grid(True)
    # for year, group in ostia_humb.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.analysed_sst, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_ostia_humb.analysed_sst, color="black", lw=1, alpha = 1)  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticklabels([])
    ax1.text(x,y, "(g)", transform=ax1.transAxes)

    
    ax = plt.subplot(num_rows, 2, 6, sharex = ax0) 
    plt.grid(True)
    for year, group in ostia_morro.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.analysed_sst, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_ostia_morro.analysed_sst,color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    # ax.set_ylim(ax1.get_ylim())
    ax.set_xticklabels([])
    # plt.ylim(10,20)
    ax.text(x,y, "(h)", transform=ax.transAxes)
    
    
    # # MLD
    # ax1 = plt.subplot(num_rows, 2, 9, sharex = ax0) 
    # plt.ylabel("MLD (m)")
    # plt.grid(True)
    # for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.mld_sat, color="C0", lw=0.5, alpha = 0.5)
    # plt.plot(weekly_avg_humb.mld_sat, color="black", lw=1, alpha = 1)  # Average over all years
    # ax1.set_xlim(ax0.get_xlim())
    # ax1.set_xticklabels([])
    # ax1.text(x,y, "(i)", transform=ax1.transAxes)
    # plt.xlim(0,365)
   
    # ax = plt.subplot(num_rows, 2, 10, sharex = ax0) 
    # plt.grid(True)
    # for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.mld_sat, color="C0", lw=0.5, alpha = 0.5)
    # plt.plot(weekly_avg_morro.mld_sat,color="black", lw=1, alpha = 1)  # Average over all years
    # ax.set_xlim(ax0.get_xlim())
    # ax.set_ylim(ax1.get_ylim())
    # ax.set_xticklabels([])
    # ax.text(x,y, "(j)", transform=ax.transAxes)
    # plt.xlim(0,365)
    
    # CUTI
    ax1 = plt.subplot(num_rows, 2,7) 
    plt.ylabel("CUTI (m$^2$/s)")
    plt.grid(True)
    # for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.CUTI_Oscar, color="C0", lw=0.5, alpha = 0.5)
    #plt.plot(weekly_avg_humb.CUTI_Oscar, color="black", lw=1, alpha = 1, label = "satellite")  # Average over all years
    plt.plot(cuti["41N"], ".", ms=4, label = "model")
    ax1.set_xlim(ax0.get_xlim())
   # plt.xlim(0,365)
    plt.ylim(-4,4)
    ax1.text(x,y, "(k)", transform=ax1.transAxes)
    #plt.xlabel("Day of Year")
    
    ax = plt.subplot(num_rows, 2,8, sharey=ax1) 
    plt.grid(True)
    # for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
    #     plt.plot(group.week, group.CUTI_Oscar, color="C0", lw=0.5, alpha = 0.5)
    #plt.plot(weekly_avg_morro.CUTI_Oscar,color="black", lw=1, alpha = 1)  # Average over all years
    plt.plot(cuti["36N"], ".", ms=4, label = "model")
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    ax.text(x,y, "(l)", transform=ax.transAxes)
    #plt.xlabel("Day of Year")

    #plt.xlim(0,365)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
 
    fig.autofmt_xdate()
    
    # fig.savefig("paper_plots/time_series.png", dpi=300)
     
    
    

 
    

#%% Time series long version



time_series_long = 0

if time_series_long == 1:
    
    # Calculate wind speed and direction
    sat['wind_speed'] = (sat.u_wind**2 + sat.v_wind**2)**0.5
    sat['wdir'] = np.degrees(np.arctan2(sat.u_wind, sat.v_wind)) + 180
    
    # Isolate the Morro and Humboldt location
    sat_humb = sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").to_dataframe()
    sat_morro = sat.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest").to_dataframe()    
    
    ostia_humb = ostia.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").to_dataframe()
    ostia_morro = ostia.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest").to_dataframe()   
    
    # PErcentage of negative/ positive CUTI valeus
    len(sat_humb.CUTI_Oscar.where(sat_humb.CUTI_Oscar>0).dropna()) / (
        len(sat_humb.CUTI_Oscar.where(sat_humb.CUTI_Oscar<0).dropna()) + 
        len(sat_humb.CUTI_Oscar.where(sat_humb.CUTI_Oscar>0).dropna()))
    
    len(sat_morro.CUTI_Oscar.where(sat_morro.CUTI_Oscar>0).dropna()) / (
        len(sat_morro.CUTI_Oscar.where(sat_morro.CUTI_Oscar<0).dropna()) + 
        len(sat_morro.CUTI_Oscar.where(sat_morro.CUTI_Oscar>0).dropna()))
    
    sat_humb.wind_speed.mean()
    
    
    # resample to weeks and re-calculate wind dir  (for week of year use "W" nand %U, for day pf year use "D" and %j)
    # this is called weekly average, but we use daily instead.
    sat_humb = sat_humb.resample("D").median()
    sat_humb["week"] = [int(x) for x in sat_humb.index.strftime("%j")] # sat_humb.index.isocalendar().week
    sat_humb['wdir'] = np.degrees(np.arctan2(sat_humb.u_wind, sat_humb.v_wind)) + 180
    weekly_avg_humb = sat_humb.groupby('week').median()
    weekly_avg_humb['wdir'] = np.degrees(np.arctan2(weekly_avg_humb.u_wind, weekly_avg_humb.v_wind)) + 180
    
    sat_morro = sat_morro.resample("D").median()
    sat_morro["week"] = [int(x) for x in sat_morro.index.strftime("%j")]
    sat_morro['wdir'] = np.degrees(np.arctan2(sat_morro.u_wind, sat_morro.v_wind)) + 180
    weekly_avg_morro = sat_morro.groupby('week').median()
    weekly_avg_morro['wdir'] = np.degrees(np.arctan2(weekly_avg_morro.u_wind, weekly_avg_morro.v_wind)) + 180
    
    ostia_humb = ostia_humb.resample("D").median()
    ostia_humb["week"] = [int(x) for x in ostia_humb.index.strftime("%j")] # sat_humb.index.isocalendar().week
    weekly_avg_ostia_humb = ostia_humb.groupby('week').median()

    ostia_morro = ostia_morro.resample("D").median()
    ostia_morro["week"] = [int(x) for x in ostia_morro.index.strftime("%j")] # sat_morro.index.isocalendar().week
    weekly_avg_ostia_morro = ostia_morro.groupby('week').median()   
    
    #%
    
    x = 0.01
    y = 0.85
    
    # Plot
    num_rows=6
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure( figsize=(16, 9.5))
    
    # Wind speed
    ax0 = plt.subplot(num_rows, 2, 1) 
    plt.title("Humboldt")
    plt.ylabel("Wind speed (m/s)")
    plt.grid(True)
    for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.wind_speed, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_humb.wind_speed, color="black", lw=1, alpha = 1)  # Average over all years
    ax0.set_xticklabels([])
    plt.ylim(0,14)
    ax0.text(x,y, "(a)", transform=ax0.transAxes)

    
    ax = plt.subplot(num_rows, 2, 2, sharex = ax0) 
    plt.title("Morro")
    plt.grid(True)
    for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.wind_speed, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_morro.wind_speed, color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax0.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,y, "(b)", transform=ax.transAxes)
    
    # Wind direcion
    ax1 = plt.subplot(num_rows, 2, 3, sharex = ax0) 
    plt.ylabel("Wind dir ($^\circ$)")
    plt.grid(True)
    for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.wdir,".", color="C0", ms=0.5, alpha = 0.5)
    plt.plot(weekly_avg_humb.wdir,".", color="black", ms=5, alpha = 1)  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticklabels([])
    ax1.text(x,y, "(c)", transform=ax1.transAxes)
    
    ax = plt.subplot(num_rows, 2, 4, sharex = ax0) 
    plt.grid(True)
    for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.wdir,".", color="C0", ms=0.5, alpha = 0.5)
    plt.plot(weekly_avg_morro.wdir,".", color="black", ms=5, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,0.6, "(d)", transform=ax.transAxes)
    
    
    # Offshore grostrophic current
    ax1 = plt.subplot(num_rows, 2, 5, sharex = ax0) 
    plt.ylabel("$u_{geo, cross}$ (m/s)")
    plt.grid(True)
    for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.go_cross, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_humb.go_cross, color="black", lw=1, alpha = 1)  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticklabels([])
    ax1.text(x,y, "(e)", transform=ax1.transAxes)
    
    ax = plt.subplot(num_rows, 2, 6, sharex = ax0) 
    plt.grid(True)
    for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.go_cross, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_morro.go_cross,color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,y, "(f)", transform=ax.transAxes)
    
        
    # SST
    ax1 = plt.subplot(num_rows, 2, 7, sharex = ax0) 
    plt.ylabel("SST ($^\circ$C)")
    plt.grid(True)
    for year, group in ostia_humb.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.analysed_sst, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_ostia_humb.analysed_sst, color="black", lw=1, alpha = 1)  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticklabels([])
    ax1.text(x,y, "(g)", transform=ax1.transAxes)

    
    ax = plt.subplot(num_rows, 2, 8, sharex = ax0) 
    plt.grid(True)
    for year, group in ostia_morro.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.analysed_sst, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_ostia_morro.analysed_sst,color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    # ax.set_ylim(ax1.get_ylim())
    ax.set_xticklabels([])
    # plt.ylim(10,20)
    ax.text(x,y, "(h)", transform=ax.transAxes)
    
    
    # MLD
    ax1 = plt.subplot(num_rows, 2, 9, sharex = ax0) 
    plt.ylabel("MLD (m)")
    plt.grid(True)
    for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.mld_sat, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_humb.mld_sat, color="black", lw=1, alpha = 1)  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xticklabels([])
    ax1.text(x,y, "(i)", transform=ax1.transAxes)
    plt.xlim(0,365)
   
    ax = plt.subplot(num_rows, 2, 10, sharex = ax0) 
    plt.grid(True)
    for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.mld_sat, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_morro.mld_sat,color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    ax.set_xticklabels([])
    ax.text(x,y, "(j)", transform=ax.transAxes)
    plt.xlim(0,365)
    
    # CUTI
    ax1 = plt.subplot(num_rows, 2,11) 
    plt.ylabel("CUTI (m$^2$/s)")
    plt.grid(True)
    for year, group in sat_humb.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.CUTI_Oscar, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_humb.CUTI_Oscar, color="black", lw=1, alpha = 1, label = "satellite")  # Average over all years
    ax1.set_xlim(ax0.get_xlim())
    plt.xlim(0,365)
    ax1.set_xticks(np.arange(0, 366, 60))
    plt.ylim(-5,5)
    ax1.text(x,y, "(k)", transform=ax1.transAxes)
    plt.xlabel("Day of Year")
    
    ax = plt.subplot(num_rows, 2,12, sharey=ax1) 
    plt.grid(True)
    for year, group in sat_morro.groupby(pd.Grouper(freq='Y')): # Single years
        plt.plot(group.week, group.CUTI_Oscar, color="C0", lw=0.5, alpha = 0.5)
    plt.plot(weekly_avg_morro.CUTI_Oscar,color="black", lw=1, alpha = 1)  # Average over all years
    ax.set_xlim(ax0.get_xlim())
    ax.set_xticks(np.arange(0, 366, 60))
    ax.set_ylim(ax1.get_ylim())
    ax.text(x,y, "(l)", transform=ax.transAxes)
    plt.xlabel("Day of Year")

    plt.xlim(0,365)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    
    # fig.savefig("paper_plots/time_series.png", dpi=300)
     
    
    
    
    
    
#%% Histograms at Humboldt and Morro


hists = 0

if hists == 1:
    
    # Calculate wind speed and direction
    sat['wind_speed'] = (sat.u_wind**2 + sat.v_wind**2)**0.5
    sat['wdir'] = np.degrees(np.arctan2(sat.u_wind, sat.v_wind)) + 180
    
    # Isolate the Morro and Humboldt location
    sat_humb = sat.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").to_dataframe()
    sat_morro = sat.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest").to_dataframe()    
    
    ostia_humb = ostia.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1], method = "nearest").to_dataframe()
    ostia_morro = ostia.sel(lat=Morro_coords[0], lon=Morro_coords[1], method = "nearest").to_dataframe() 
    
    sat_humb = pd.merge(sat_humb, ostia_humb.resample("D").first(),left_index=True, right_index=True, how="inner")
    sat_morro = pd.merge(sat_morro, ostia_morro.resample("D").first(),left_index=True, right_index=True, how="inner")
    
    # PErcentile of CUTI
    from scipy.stats import percentileofscore
    
    
    percentileofscore(sat_humb.CUTI_Oscar.dropna(), 1.9, kind='mean')
    percentileofscore(sat_morro.CUTI_Oscar.dropna(), 2.2, kind='mean')

    
    #%% Plot
    
    # Histogram of high-wind duration
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure( figsize=(10, 5))
    
    # Wind speed
    ax0 = plt.subplot(1, 2, 1) 
    plt.title("(a) Humboldt")
    plt.ylabel("Probability density")
    plt.xlabel("Wind speed event duration (days)")
    plt.grid(True)
    plt.xlim(-1, 22) 
    plt.ylim(0, 0.92)    
    
    ax = plt.subplot(1, 2, 2) 
    plt.title("(b) Morro Bay")
    plt.xlabel("Wind speed event duration (days)")
    plt.grid(True)  
    plt.xlim(-1, 22)
    plt.ylim(0, 0.92)
    
    for threshold in [ 5,7,9,11]:
        

    

        # Calculate N (how many rows since wind_speed was above threshold)
        sat_humb['N'] = (
            sat_humb.index.to_series()  # Use the index for row positions
            .where(sat_humb['wind_speed'] < threshold)  # Keep indices where condition is met
            .ffill()  # Forward fill to propagate last valid occurrence
            .pipe(lambda x: (sat_humb.index - x))  # Calculate time difference in rows
        ).dt.total_seconds() / (24 * 3600)
        
        sat_morro['N'] = (
            sat_humb.index.to_series()  # Use the index for row positions
            .where(sat_morro['wind_speed'] < threshold)  # Keep indices where condition is met
            .ffill()  # Forward fill to propagate last valid occurrence
            .pipe(lambda x: (sat_morro.index - x))  # Calculate time difference in rows
        ).dt.total_seconds() / (24 * 3600)
        
     
        ax0.hist(sat_humb.N.values, width = 5/threshold, density=True, bins=np.arange(0, 25, 1), alpha=0.5, label = f"wind speed>{threshold}m/s") 
        ax.hist(sat_morro.N.values, width = 5/threshold, density=True, bins=np.arange(0, 25, 1), alpha=0.5)  
        
    ax0.legend()
    
    # fig.savefig("paper_plots/Histogram time duration.png", dpi=300)
    
    
    #%
    
    ## CUTI change with wind change
    ax2 = plt.subplot(2, 2, 3, sharex = ax0) 
    plt.grid()
    ax3 = plt.subplot(2, 2, 4, sharex = ax2, sharey=ax2) 
    plt.grid()
    
    for threshold in [ 5,7,9,11]:
        

    

        # Calculate N (how many rows since wind_speed was above threshold)
        sat_humb['N'] = (
            sat_humb.index.to_series()  # Use the index for row positions
            .where(sat_humb['wind_speed'] < threshold)  # Keep indices where condition is met
            .ffill()  # Forward fill to propagate last valid occurrence
            .pipe(lambda x: (sat_humb.index - x))  # Calculate time difference in rows
        ).dt.total_seconds() / (24 * 3600)
        
        sat_morro['N'] = (
            sat_humb.index.to_series()  # Use the index for row positions
            .where(sat_morro['wind_speed'] < threshold)  # Keep indices where condition is met
            .ffill()  # Forward fill to propagate last valid occurrence
            .pipe(lambda x: (sat_morro.index - x))  # Calculate time difference in rows
        ).dt.total_seconds() / (24 * 3600)
        
        sat_humb = sat_humb.where((sat_humb.month<5) | (sat_humb.month>10)).dropna(subset=['month'])
        sat_morro = sat_morro.where((sat_morro.month<5) | (sat_morro.month>10)).dropna(subset=['month'])
        
        # Ensure N is an integer (fill NaNs with 0 or other appropriate value)
        sat_humb['N'] = sat_humb['N'].fillna(0).astype(int)
        
        # Compute the difference dynamically
        sat_humb['CUTI_Diff'] = sat_humb.apply(lambda row: row['analysed_sst'] - sat_humb['analysed_sst'].shift(-int(row['N'])).loc[row.name] 
                                               if pd.notna(row['analysed_sst']) and row['N'] > 0 else None, axis=1)



        line, = ax2.plot(sat_humb.N, sat_humb.CUTI_Diff,".", label = f"threshold wind speed {threshold}m/s", alpha = 0.7, ms=2)
        
        
        # Remove NaNs for regression
        valid_data = sat_humb.dropna(subset=['N', 'CUTI_Diff'])
        x = valid_data.N.values
        y = valid_data.CUTI_Diff.values
        
        # Perform linear regression (y = mx + b)
        m, b = np.polyfit(x, y, 1)  # 1st-degree polynomial fit (linear)
        
        # Generate regression line
        x_line = np.linspace(x.min(), x.max(), 100)  # Smooth x values for plotting
        y_line = m * x_line + b  # Compute y values
        
        # Plot regression line
        ax2.plot(x_line, y_line, color = line.get_color(), label="")       
        
        
        
        # Ensure N is an integer (fill NaNs with 0 or other appropriate value)
        sat_morro['N'] = sat_morro['N'].fillna(0).astype(int)
        
        # Compute the difference dynamically
        sat_morro['CUTI_Diff'] = sat_morro.apply(lambda row: row['analysed_sst'] - sat_morro['analysed_sst'].shift(-int(row['N'])).loc[row.name] 
                                               if pd.notna(row['analysed_sst']) and row['N'] > 0 else None, axis=1)



        line, = ax3.plot(sat_morro.N, sat_morro.CUTI_Diff,".", label = f"threshold wind speed {threshold}m/s", alpha = 0.7, ms=2)
        
        
        # Remove NaNs for regression
        valid_data = sat_morro.dropna(subset=['N', 'CUTI_Diff'])
        x = valid_data.N.values
        y = valid_data.CUTI_Diff.values
        
        # Perform linear regression (y = mx + b)
        m, b = np.polyfit(x, y, 1)  # 1st-degree polynomial fit (linear)
        
        # Generate regression line
        x_line = np.linspace(x.min(), x.max(), 100)  # Smooth x values for plotting
        y_line = m * x_line + b  # Compute y values
        
        # Plot regression line
        ax3.plot(x_line, y_line, color = line.get_color(), label="")       
                
                
    ax2.set_ylabel("SST difference across $N$ days ($^\circ$C)")
    ax2.set_xlabel("Wind speed event duration $N$ (days)")
        
   # ax3.legend(markerscale = 3)
    ax3.set_xlabel("Wind speed event duration $N$ (days)")
    
    
    #%%
    
    wind_speed_threshold = 7
    day_threshold = 6 
    
    sat_humb['N'] = (
        sat_humb.index.to_series()  # Use the index for row positions
        .where(sat_humb['wind_speed'] < wind_speed_threshold)  # Keep indices where condition is met
        .ffill()  # Forward fill to propagate last valid occurrence
        .pipe(lambda x: (sat_humb.index - x))  # Calculate time difference in rows
    ).dt.total_seconds() / (24 * 3600)
    
    sat_morro['N'] = (
        sat_humb.index.to_series()  # Use the index for row positions
        .where(sat_morro['wind_speed'] < wind_speed_threshold)  # Keep indices where condition is met
        .ffill()  # Forward fill to propagate last valid occurrence
        .pipe(lambda x: (sat_morro.index - x))  # Calculate time difference in rows
    ).dt.total_seconds() / (24 * 3600)

    summer_months = [ ]
    winter_months = [ ]
    condition_humb = ((sat_humb.wind_speed>wind_speed_threshold) & (sat_humb.N>day_threshold))
    condition_morro = ((sat_morro.wind_speed>wind_speed_threshold) & (sat_morro.N>day_threshold))


    
    num_rows=7
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure( figsize=(16, 12))
    
    # Wind speed
    ax0 = plt.subplot(num_rows, 2, 1) 
    plt.title("Humboldt")
    plt.xlabel("Wind speed (m/s)")
    plt.grid(True)
    plt.hist(sat_humb.wind_speed, color="black", density=True, bins=100)  
    plt.hist(sat_humb[sat_humb.index.month.isin(summer_months)].wind_speed, color="red", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[sat_humb.index.month.isin(winter_months)].wind_speed, color="blue", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[condition_humb].wind_speed, color="orange", density=True, bins=100, alpha = 0.5)  

    
    ax = plt.subplot(num_rows, 2, 2) 
    plt.title("Morro")
    plt.xlabel("Wind speed (m/s)")
    plt.grid(True)
    plt.hist(sat_morro.wind_speed, color="black", density=True, bins=100)  
    plt.hist(sat_morro[sat_morro.index.month.isin(summer_months)].wind_speed, color="red", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_morro[sat_morro.index.month.isin(winter_months)].wind_speed, color="blue", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_morro[condition_morro].wind_speed, color="orange", density=True, bins=100, alpha = 0.5)  
    
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax0.get_ylim())
    
    # Wind direcion
    ax1 = plt.subplot(num_rows, 2, 3) 
    plt.xlabel("Wind dir ($^\circ$)")
    plt.grid(True)
    plt.hist(sat_humb.wdir, color="black", density=True, bins=100)  
    plt.hist(sat_humb[sat_humb.index.month.isin(summer_months)].wdir, color="red", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[sat_humb.index.month.isin(winter_months)].wdir, color="blue", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[condition_humb].wdir, color="orange", density=True, bins=100, alpha = 0.5)  

    ax = plt.subplot(num_rows, 2, 4) 
    plt.xlabel("Wind dir ($^\circ$)")
    plt.grid(True)
    plt.hist(sat_morro.wdir, color="black", density=True, bins=100) 
    plt.hist(sat_morro[sat_morro.index.month.isin(summer_months)].wdir, color="red", density=True, bins=100, alpha = 0.5) 
    plt.hist(sat_morro[sat_morro.index.month.isin(winter_months)].wdir, color="blue", density=True, bins=100, alpha = 0.5) 
    plt.hist(sat_morro[condition_morro].wdir, color="orange", density=True, bins=100, alpha = 0.5)  
    ax.set_xlim(ax1.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    
    
    # Offshore grostrophic current
    ax1 = plt.subplot(num_rows, 2, 5) 
    plt.xlabel("$u_{geo, cross}$ (m/s)")
    plt.grid(True)
    plt.hist(sat_humb.U_Geo_Oscar, color="black", density=True, bins=100)  
    plt.hist(sat_humb[sat_humb.index.month.isin(summer_months)].U_Geo_Oscar, color="red", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[sat_humb.index.month.isin(winter_months)].U_Geo_Oscar, color="blue", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[condition_humb].U_Geo_Oscar, color="orange", density=True, bins=100, alpha = 0.5)  


    ax = plt.subplot(num_rows, 2, 6) 
    plt.xlabel("$u_{geo, cross}$ (m/s)")
    plt.grid(True)
    plt.hist(sat_morro.U_Geo_Oscar,color="black", density=True, bins=100)  
    plt.hist(sat_morro[sat_morro.index.month.isin(summer_months)].U_Geo_Oscar, color="red", density=True, bins=100, alpha = 0.5) 
    plt.hist(sat_morro[sat_morro.index.month.isin(winter_months)].U_Geo_Oscar, color="blue", density=True, bins=100, alpha = 0.5) 
    plt.hist(sat_morro[condition_morro].U_Geo_Oscar, color="orange", density=True, bins=100, alpha = 0.5)  
    ax.set_xlim(ax1.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    
    # SST
    ax1 = plt.subplot(num_rows, 2, 7) 
    plt.xlabel("SST ($^\circ$C)")
    plt.grid(True)
    plt.hist(ostia_humb.analysed_sst, color="black", density=True, bins=100)  
    plt.hist(ostia_humb[ostia_humb.index.month.isin(summer_months)].analysed_sst, color="red", density=True, bins=100, alpha = 0.5) 
    plt.hist(ostia_humb[ostia_humb.index.month.isin(winter_months)].analysed_sst, color="blue", density=True, bins=100, alpha = 0.5) 

    
    ax = plt.subplot(num_rows, 2, 8) 
    plt.xlabel("SST ($^\circ$C)")
    plt.grid(True)
    plt.hist(ostia_morro.analysed_sst,color="black", density=True, bins=100)  
    plt.hist(ostia_morro[ostia_morro.index.month.isin(summer_months)].analysed_sst, color="red", density=True, bins=100, alpha = 0.5) 
    plt.hist(ostia_morro[ostia_morro.index.month.isin(winter_months)].analysed_sst, color="blue", density=True, bins=100, alpha = 0.5) 
    ax.set_xlim(ax1.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    
    
    # MLD
    ax1 = plt.subplot(num_rows, 2, 9) 
    plt.xlabel("MLD (m)")
    plt.grid(True)
    plt.hist(sat_humb.mld_sat, color="black", density=True, bins=100)  
    plt.hist(sat_humb[sat_humb.index.month.isin(summer_months)].mld_sat, color="red", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[sat_humb.index.month.isin(winter_months)].mld_sat, color="blue", density=True, bins=100, alpha = 0.5) 
    plt.hist(sat_humb[condition_humb].mld_sat, color="orange", density=True, bins=100, alpha = 0.5)  


    ax = plt.subplot(num_rows, 2, 10)
    plt.xlabel("MLD (m)")
    plt.grid(True)
    plt.hist(sat_morro.mld_sat,color="black", density=True, bins=100)  
    plt.hist(sat_morro[sat_morro.index.month.isin(summer_months)].mld_sat, color="red", density=True, bins=100, alpha = 0.5) 
    plt.hist(sat_morro[sat_morro.index.month.isin(winter_months)].mld_sat, color="blue", density=True, bins=100, alpha = 0.5) 
    plt.hist(sat_morro[condition_morro].mld_sat, color="orange", density=True, bins=100, alpha = 0.5)  
    ax.set_xlim(ax1.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    
    # CUTI
    ax1 = plt.subplot(num_rows, 2,11) 
    plt.xlabel("CUTI (m$^2$/s)")
    plt.grid(True)
    plt.hist(sat_humb.CUTI_Oscar, color="black", density=True, bins=100) 
    plt.hist(sat_humb[sat_humb.index.month.isin(summer_months)].CUTI_Oscar, color="red", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[sat_humb.index.month.isin(summer_months)].CUTI_Oscar, color="red", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_humb[sat_humb.index.month.isin(winter_months)].CUTI_Oscar, color="blue", density=True, bins=100, alpha = 0.5)     
    plt.hist(sat_humb[condition_humb].CUTI_Oscar, color="orange", density=True, bins=100, alpha = 0.5)  
    plt.xlim(-5,5)
    
    ax = plt.subplot(num_rows, 2, 12) 
    plt.xlabel("CUTI (m$^2$/s)")
    plt.grid(True)
    plt.hist(sat_morro.CUTI_Oscar,color="black", density=True, bins=100)  
    plt.hist(sat_morro[sat_morro.index.month.isin(summer_months)].CUTI_Oscar, color="red", density=True, bins=100, alpha = 0.5)  
    plt.hist(sat_morro[sat_morro.index.month.isin(winter_months)].CUTI_Oscar, color="blue", density=True, bins=100, alpha = 0.5)     
    plt.hist(sat_morro[condition_morro].CUTI_Oscar, color="orange", density=True, bins=100, alpha = 0.5)  
    ax.set_xlim(ax1.get_xlim())
    ax.set_ylim(ax1.get_ylim())
    
    plt.tight_layout()
   # plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    
    # plt.figure()
    # plt.plot(sat_humb.U_Ek, sat_humb.CUTI)

    
    







#%% Compare CUTI from satellite and model for latitude bins


lat_range = np.arange(33,46)   #np.arange(33,45) #  [40,41]
offshore_diff = 75 # 75 is the distance used in Jacox 2018





compare_CUTI_sat = 0

if compare_CUTI_sat == 1:
    
    
    params_to_test = ["CUTI_Oscar",  "U_Ek", "CUTI_Oscar_param", "U_Ek_param", "CUTI_Oscar_param_neg_geo", "CUTI_Oscar_curl"] # "CUTI_SLA", "CUTI_Cop",
    params_to_test = ["U_Ek", "CUTI_Oscar"] 
    
    # Create a DataFrame with latitude values as rows and multiple columns for each value
    MAE = pd.DataFrame(index=lat_range, columns = params_to_test) 
    
    
    num_lats = len(lat_range)-1
    num_rows = int(num_lats**0.5)
    num_cols = (num_lats + num_rows - 1) // num_rows
    
    plt.rcParams.update({'font.size': 13})
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i in np.arange(0, num_lats): 
        
    
        ax = axes[i]
        
        row_idx = ax.get_subplotspec().rowspan.start
        col_idx = ax.get_subplotspec().colspan.start
        
        lat = lat_range[i]
        
        print (lat)
        
        max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0]
        min_lon = max_lon + lon_difference(offshore_diff, max_lon)
        
        
        
        # Average satellite data over 1 deg in N-S direction and up to Xkm offshore
        sat_local = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 
        
        
        # Xarray with all binned latitudes and sat values and time steps
        sat_local_xr = sat_local.copy(deep=True).expand_dims('lat').assign_coords(lat=[lat])
        if i == 0:
            #CUTI_xr = sat_local.expand_dims(dim={"lat": [lat]})
            CUTI_xr = sat_local_xr
        else:
            CUTI_xr = xr.concat(
                [CUTI_xr.drop_dims('concat_dim', errors='ignore')  , 
                 sat_local_xr.drop_dims('concat_dim', errors='ignore')
                 ], 
                dim='lat')
            
    
        # Merge model CUTI and satellite area average
        all = CUTI.merge(
                sat_local[params_to_test].to_dataframe(), # "CUTI_SLA", "CUTI_Cop",  "U_Geo", "U_Geo_Oscar",
                left_index=True, right_index=True, how="inner")
        all = all.replace([np.inf, -np.inf], np.nan)

        ax.set_title("{}$^\circ$N".format(lat)) 
        ax.plot(all[str(lat)+"N"], all[str(lat)+"N"], "--", color="grey", zorder=-10, label = "")
        for parameter in params_to_test:  # 
            mae = round(root_mean_squared_error(all[[str(lat)+"N", parameter]].dropna()[parameter], 
                                    all[[str(lat)+"N", parameter]].dropna()[str(lat)+"N"]),2)
            R = all[[str(lat)+"N", parameter]].dropna()[parameter].corr(
                                    all[[str(lat)+"N", parameter]].dropna()[str(lat)+"N"]).round(2)
            if (parameter == "CUTI_Oscar") or (parameter == "CUTI_Oscar_curl"):
                label_name = "CUTI"
            elif (parameter == "U_Ek") or (parameter == "U_Ek_param"):
                label_name = "U$_{Ek}$"            
            else:
                label_name = parameter
            #if (parameter == "CUTI_Oscar") or (parameter == "U_Ek") or (parameter == "CUTI_Oscar_param_test"):
            ax.plot(all[str(lat)+"N"], all[parameter], ".", ms=0.5,
              label = label_name + ", RMSE=" + 
              str( mae )  
              # + ", R=" + 
              # str( R ) 
              )
            MAE[parameter][lat] = mae
        if row_idx == num_rows - 1:
            ax.set_xlabel("CUTI model (m$^2$/s)")
        if col_idx == 0:
            ax.set_ylabel("CUTI satellite (m$^2$/s)")
        ax.grid()
        ax.legend(loc="upper left", markerscale=30)
        
    plt.ylim(-6, 8.9)
    plt.xlim(-6, 6)
        
    plt.tight_layout()
    
   # fig.savefig("paper_plots/CUTI_sat_vs_model_RMSE.png", dpi=300)
    
    


#%% Error contributions for CUTI

error_to_CUTI = 0

if error_to_CUTI == 1:    
    
        
        # plt.figure()
        # for parameter in ["U_Ek", "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop"]:
        #     plt.plot(MAE.index, MAE[parameter], label = parameter)
        # plt.ylabel("MAE of CUTI")
        # plt.xlabel("latitude")
        # plt.legend()
        # plt.grid()
        
        
        # First need to run the cell "Compare CUTI from satellite and model for latitude bins" !!
        

        # Add CUTI to the xarray
        CUTI_df = CUTI.drop(columns=['year', 'month', 'day'])
        CUTI_df['time'] = CUTI_df.index
        CUTI_df = CUTI_df.reset_index(drop=True) 
        CUTI_df = CUTI_df.rename(columns=lambda col: int(col.replace('N', '')) if 'N' in col else col)  # Rename the columns to match latitudes
        CUTI_long = CUTI_df.melt(id_vars='time', var_name='lat', value_name='CUTI')  # Convert to long format to create 'lat' and 'time' dimensions    
        CUTI_xarray = CUTI_long.set_index(['time', 'lat']).to_xarray()  # Convert to xarray, reshaping it to have 'lat' and 'time' as dimensions
        CUTI_xarray = CUTI_xarray.sel(lat=CUTI_xr.lat)  #  Ensure the 'lat' dimension aligns with CUTI_xr latitudes      
        CUTI_xr['CUTI_model'] = CUTI_xarray['CUTI']   # Add CUTI data to CUTI_xr as a new variable
        
        # Difference between model and sat CUTI
        CUTI_xr["CUTI_diff"] = (CUTI_xr.CUTI_Oscar - CUTI_xr.CUTI_model) /  CUTI_xr.CUTI_model
        
        CUTI_xr["wind_speed"] = (CUTI_xr.u_wind**2 + CUTI_xr.v_wind**2)**0.5
        CUTI_xr['wdir'] = np.degrees(np.arctan2(CUTI_xr.u_wind, CUTI_xr.v_wind)) + 180
        
       
        # list(CUTI_xr.variables)
        
        
        
        for x in [ "CUTI_model", "U_Ek", "wind_speed",  'mld_sat']:
            
            CUTI_xr[x+"_round"] = CUTI_xr[x].round(0)
        
        for x in [ 'wdir']:
            
            CUTI_xr[x+"_round"] =(CUTI_xr[x] / 20).round() * 20

            
        parameter = "CUTI_diff"
        
        variables = [ "month", "lat","wind_speed_round", 'mld_sat_round', "wdir_round", "year"]  # Add your variables here
        
        # Number of columns and rows
        num_days = len(variables)
        num_rows = int(num_days**0.5)
        num_cols = (num_days + num_rows - 1) // num_rows

        

        # Plot
        plt.rcParams.update({'font.size': 13})
        
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows) , sharey=True)
        axs = axs.flatten()

        for i, var in enumerate(variables):
            
            CUTI_xr[var] = CUTI_xr[var].compute()
            
            # Histogram
            var_values = CUTI_xr[var].values.flatten()
            var_values = var_values[~np.isnan(var_values)]
            var_values = var_values[np.isfinite(var_values)]
            # axs[i].hist(var_values, bins=50, zorder=10,color="gray", density=True)
            counts, bins, _ = axs[i].hist(var_values, bins=50, zorder=-100, alpha=0, color="gray", density=True)
            if var == "wdir_round":
                factor = 100
            elif var == "mld_sat_round":
                factor = 20                
            else: 
                factor = 10
            
            axs[i].hist(bins[:-1], bins, weights=counts * factor, color="lightblue", zorder=-10)
            
            ax2 = axs[i].twinx()
            ax2.set_yticklabels([])
            # ax2.hist(var_values, bins=50, zorder=-10,color="lightblue", density=True)
            # if var == "wdir_round":
            #     ax2.set_ylim(-0.224, 0.28)
            # elif var == "mld_sat_round":
            #     ax2.set_ylim(-0.224, 0.28)    
            # elif var == "year":
            #     ax2.set_ylim(-0.224, 0.28) 
            # else: 
            #     ax2.set_ylim(-0.4, 0.5)          
            # axs[i].set_zorder(ax2.get_zorder() + 1)
            # axs[i].patch.set_visible(False)  
            # counts, bins, _ = axs[i].hist(var_values, bins=50, zorder=-100, alpha=0, color="gray", density=True)

            
            # Loop over each latitude and extract data
            value_list = np.unique(CUTI_xr[var].values.flatten())
            value_list = value_list[~np.isnan(value_list)]
            for value in value_list:
                
                width = (max(value_list) - min(value_list) ) / 30
                if var == "mld_sat_round":
                    width = width / 3     
                elif var == "year":
                    width = width / 2 
                
                lat_values = CUTI_xr.where(CUTI_xr[var]==value, drop=True)[parameter].values
                lat_values = lat_values[~np.isnan(lat_values)]

                axs[i].boxplot(
                    lat_values, 
                    positions=[int(value)],  # Set the latitudes on the x-axis
                    widths=width,  # Adjust the width of the boxes
                    patch_artist=False,  # Fill the boxes with color
                    showfliers=False  # Optionally hide outliers
                )
                    
            
            axs[i].set_xticklabels(axs[i].get_xticks(), rotation=45)
            
            if var == "mld_sat_round":
              
                # Show only every third tick label on the x-axis
                x_ticks = axs[i].get_xticks()  # Get current x-ticks
                axs[i].set_xticks(x_ticks[::5])  # Set x-ticks to every third one
                axs[i].set_xticklabels(x_ticks[::5], rotation=45)  # Rotate labels for better visibility
                
                axs[i].set_xlim(10,35)    
                
                xlabel = "MLD (m)"
                axs[i].set_ylabel("(CUTI$_{sat}$ - CUTI$_{model}$) / CUTI$_{model}$")
                
            elif var == "wind_speed_round":
              
                # Show only every third tick label on the x-axis
                x_ticks = axs[i].get_xticks()  # Get current x-ticks
                axs[i].set_xticks(x_ticks[::2])  # Set x-ticks to every third one
                axs[i].set_xticklabels(x_ticks[::2], rotation=45)  # Rotate labels for better visibility
                xlabel = "Wind speed (m/s)"
                
                axs[i].set_xlim(-0.5,15)     
                ax2.set_ylabel('Probability density (a.u.)', color='steelblue')
                
            elif  var == "wdir_round":
                
                # Show only every third tick label on the x-axis
                x_ticks = axs[i].get_xticks()  # Get current x-ticks
                axs[i].set_xticks(x_ticks[::2])  # Set x-ticks to every third one
                axs[i].set_xticklabels(x_ticks[::2], rotation=45)  # Rotate labels for better visibility
                
                axs[i].set_xlim(-10,370)     
                
                xlabel = "Wind direction ($^\circ$)"
                
            elif  var == "year":
                
                # Show only every third tick label on the x-axis
                x_ticks = axs[i].get_xticks()  # Get current x-ticks
                axs[i].set_xticks(x_ticks[::4])  # Set x-ticks to every third one
                axs[i].set_xticklabels(x_ticks[::4], rotation=45)  # Rotate labels for better visibility
                
                xlabel = "Year"
                ax2.set_ylabel('Probability density (a.u.)', color='steelblue')
                
            elif var == "month":
                
                xlabel = "Month of year"
                axs[i].set_ylabel("(CUTI$_{sat}$ - CUTI$_{model}$) / CUTI$_{model}$")
                
            elif  var == "lat":
                
                # Show only every third tick label on the x-axis
                x_ticks = axs[i].get_xticks()  # Get current x-ticks
                axs[i].set_xticks(x_ticks[::2])  # Set x-ticks to every third one
                axs[i].set_xticklabels(x_ticks[::2], rotation=45)  # Rotate labels for better visibility
                
                xlabel = "Latitude ($^\circ$N)"
                
            axs[i].set_xlabel(xlabel)
            axs[i].grid()
        
            #plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: int(x)))
            
            axs[i].set_ylim(-4, 5)
            
            plt.tight_layout()
            
            
            fig.savefig("paper_plots/Model_vs_sat_factors.png", dpi=300)
            
            
        # # Temporal trend
        # CUTI_xr_humb = CUTI_xr.sel(lat=Humboldt_coords[0], method = "nearest").resample(time="Y").median()
            
        # plt.figure()
        # plt.plot(CUTI_xr_humb.time, CUTI_xr_humb.CUTI_model, label = "model")
        # plt.plot(CUTI_xr_humb.time, CUTI_xr_humb.CUTI_Oscar, label = "satellite")
        # plt.legend()
        

 

    
#%% Map of U_Ek and CUTI in 23 July 2021 upwelling period

plot_UEk_CUTI = 0
if plot_UEk_CUTI == 1:
    
    # Calculate the longitude of the 100km offshore line based on latitude
    lon_diff = get_offshore_parallel_coastline(coastline, offshore_diff=75)
    
    
    sat_masked = sat.copy()
    sat_masked["wind_speed"] = (sat_masked.u_wind**2 + sat_masked.v_wind**2)**0.5
    
    sat_masked["U_Ek_percent"] = sat_masked.U_Ek /  sat_masked.CUTI_Oscar *100

    # Loop through each latitude value and remove latitudes outside of the 100 km coastline
    for i, lat_sat in enumerate(sat_masked.lat.values):
        lon_sat = lon_diff[i] 
        sat_masked[[ "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop" ]].loc[dict(lat=lat_sat, lon=slice(-150, lon_sat-0.25 ) )] =np.nan
        

    # days to plot
    # days = ['2020-10-22', '2020-11-07', '2022-04-01', '2022-04-10']
    # days = ['2021-07-20', '2021-07-21', '2021-07-22', '2021-07-23', '2021-07-24', '2021-07-25', '2021-07-26', '2021-07-27']
    day = '2021-07-23'
    # days = ['2021-08-30', '2021-08-31']
    # days = ['2020-11-01', '2020-11-02', '2020-11-03', '2020-11-04', '2020-11-05', '2020-11-06',
    #         '2020-11-07', '2020-11-08', '2020-11-09', '2020-11-10', '2020-11-11', '2020-11-12']

 #   day = '2022-12-10'
 
    day = '2021-01-20'


    parameters = ["U_Ek", "CUTI_Oscar", "U_Ek_percent"]  # "CUTI_Cop"  "U_Geo_cop"  "U_Ek"  'analysed_sst' "wind_speed"
    min_cuti = -4
    max_cuti = 4   
    
    # Number of columns and rows
    num_days = len(parameters)
    num_rows = int(num_days**0.5)
    num_cols = (num_days + num_rows - 1) // num_rows

    

    # Plot
    plt.rcParams.update({'font.size': 13})
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(17.2, 5 * num_rows), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.flatten()
    
    for i, p in enumerate(axes):
         ax = axes[i]
         ax.axis('off')     
     
    
    # Iterate over days and create subplots
    for i, parameter in enumerate(parameters):
        

        #ax = axes[i]
        ax = plt.subplot(num_rows, num_cols, i + 1, projection=ccrs.PlateCarree())  # Stereographic(dict(central_latitude=40, central_longitude=-125)))
        
        #fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 8))  
        
        # # stations
        # ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        # ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        # #ax.plot(Buoy_coords[1], Buoy_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='grey',  alpha=0.9)

        # map
        ax.add_feature(cartopy.feature.BORDERS)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.LAND)
        cs = ax.coastlines(resolution='10m', linewidth=1)
        
        


                
        # Plotting the extracted coastline and a line 75km offshore
        ax.plot(coastline[:, 0], coastline[:, 1], linewidth=1, color="gray") 
        ax.plot(lon_diff, coastline[:, 1], linewidth=1, color="gray")
                
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
        df = sat_masked.sel(time= pd.to_datetime(day), method='nearest')
        
        #Plot color map  
        if parameter== "U_Ek_percent":
            cm = "cividis"
            vmin = 0
            vmax = 100
        else:
            cm = "viridis"
            vmin = min_cuti
            vmax = max_cuti            
             
        c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap=cm, transform=ccrs.PlateCarree(), shading='auto', vmin=vmin, vmax=vmax)
        
        
        # Add arrows for wind vectors 
        if parameter == "U_Ek":
            u =   df['u_wind'].values
            v =   df['v_wind'].values
            scale_factor = 400  # Adjust the scale factor for arrow length
            ax.quiver(df['lon'], df['lat'],
                      u[ :, :], v[ :, :], scale=scale_factor, color='white', transform=ccrs.PlateCarree())
        
        
        # Add colorbar
        
        if parameter == "CUTI_Oscar":
            title = "(b) CUTI = U$_{Ek}$ + U$_{Geo}$"
            cb_label = "CUTI   or   U$_{Ek}$ (m$^2$/s)" 
        elif parameter == "U_Ek":
            title = "(a) U$_{Ek}$"
            cb_label = "U$^{Ek}$ (m$^2$/s)" 
        elif parameter == "U_Ek_percent":
            title = "(c) U$_{Ek}$ / CUTI"
            cb_label = "U$_{Ek}$ / CUTI (%)" 
            
        gl = ax.gridlines(draw_labels=True)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False
            
            
        if i%num_cols !=0:
            gl.left_labels = False
            cbar = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.03, shrink=0.99, label= cb_label ) #
                    

        
        
        # 1 degree bins for sat averaging and model
        if parameter == "CUTI_Oscar":
            for lat in np.arange(30, 50): 
                
                max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0]
                min_lon = max_lon + lon_difference(offshore_diff, max_lon)
                
                lat_min = lat - 0.5
                lat_max = lat + 0.5
                
                # Create a rectangle for this latitude-longitude bin
                rect = matplotlib.patches.Rectangle((min_lon, lat_min), width=(max_lon - min_lon), height=1.0,
                                         edgecolor='blue', facecolor='none', lw=2)
                ax.add_patch(rect)
                
            # Model CUTI 
            lats = [float(col[:-1]) for col in CUTI.drop(['year', 'month', 'day'], axis=1).columns[:]]
            lat_values = coastline[:, 1]
            lon_values = coastline[:, 0]
            lat_to_lon = {lat: lon for lat, lon in zip(lat_values, lon_values) if not np.isnan(lon)}
            lons = [lat_to_lon.get(lat, np.nan)  - lon_difference(75/2, lat) for lat in lats]
    
            cuti = CUTI.drop(['year', 'month', 'day'], axis=1).loc[pd.to_datetime(day)]
            plt.scatter(lons, lats, c=cuti, s=80, cmap="viridis", transform=ccrs.PlateCarree(),edgecolors="red", 
                       vmin = min_cuti, 
                       vmax = max_cuti)

        ax.set_extent([-128,-120, 34.5, 42], crs=ccrs.PlateCarree())  
        
        # ax.set_title('{:%Y-%m-%d}, \nC$_H$={}, C$_M$={}'.format(day,
        #     round(CUTI.loc[pd.to_datetime(day), '41N'], 2), 
        #     round(CUTI.loc[pd.to_datetime(day), '36N'], 2))  , fontsize=10)   
        ax.set_title(title)   
    #    plt.tight_layout()

        # fig.savefig(f"paper_plots/CUTI_and_UEk_one_day_{day}.png", dpi=300)
        


    
    
    




#%% Map of variables during 23th July 2021 upwelling period

plot =0
if plot == 1:
    
    
    try_sent_data = 0
    if try_sent_data==1:
    
        
        sent_a2e = read_sentinel_a2e_map(filt_string = "*sent*2021072*.nc")
        
        
        times = sent_a2e.timestamp.unique()
        times = sent_a2e.groupby(sent_a2e['timestamp'].dt.date).first().timestamp.values
        
        num_days = len(times)
        num_rows = int(num_days**0.5)
        num_cols = (num_days + num_rows - 1) // num_rows
    
        plt.figure()
        
        for i, time in enumerate(times):
            
            ax = plt.subplot(num_cols, num_rows,  i+1, projection=ccrs.PlateCarree())
            plt.title(time)
            df = sent_a2e[  ( sent_a2e.timestamp == time) ]
            plt.scatter(df.Lon, df.Lat, c=df.WindSpeed, cmap='viridis', s=0.5)
            
            # map
            ax.add_feature(cartopy.feature.BORDERS)
            ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
            ax.add_feature(cartopy.feature.RIVERS)
            ax.add_feature(cartopy.feature.LAND)
            cs = ax.coastlines(resolution='10m', linewidth=1)
            
        plt.colorbar(label="Wind speed (m/s)")
        plt.xlabel('Longitude ($^\circ$)')
        plt.ylabel('Latitude ($^\circ$)')
        plt.grid(False)
    

    
    



        
    
    
    # Calculate the longitude of the 100km offshore line based on latitude
    
    lon_diff = get_offshore_parallel_coastline(coastline, offshore_diff=75)
    
    
    sat_masked = sat.copy()

    # Loop through each latitude value and remove latitudes outside of the 100 km coastline
    for i, lat_sat in enumerate(sat_masked.lat.values):
        lon_sat = lon_diff[i] 
        sat_masked[[ "CUTI_SLA", "CUTI_Oscar", "CUTI_Cop" ]].loc[dict(lat=lat_sat, lon=slice(-150, lon_sat-0.25 ) )] =np.nan
        

    # days to plot
    '2021-07-23' #day = '2021-11-01' # day = 
    day = '2022-12-10'
    day = '2021-10-21'
    day = '2021-01-19'
    
    # SST data
    # ostia = xr.load_dataset("../satellite_data/Ostia/ostia_all_2000_2023.nc", engine="netcdf4")  # all coordinates close to California coast
    

    sat_masked["wind_speed"] = (sat_masked.u_wind**2 + sat_masked.v_wind**2)**0.5
    sat_masked["geostr_curr"] = (sat_masked.ug**2 + sat_masked.vg**2)**0.5
    
    parameters = ["CUTI_model", "wind_speed", "analysed_sst", "geostr_curr", "mld_sat", "CHL"] #]  #   "U_Geo_Oscar"  "U_Ek" 
    min_cuti = -4
    max_cuti = 4   
        
    
    # Number of columns and rows
    num_days = len(parameters)
    num_rows = int(num_days**0.5)
    num_cols = (num_days + num_rows - 1) // num_rows
    

    

    # Plot
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(18, 5 * num_rows))
    
    plt.suptitle(day)

   
    # Iterate over days and create subplots
    for i, parameter in enumerate(parameters):

        ax = plt.subplot(num_rows, num_cols, i + 1,projection=ccrs.PlateCarree())

        ax.set_title(f"({chr(97 + i)})") # chr(97) is 'a', chr(98) is 'b', etc.

        # # stations
        # ax.plot(Humboldt_coords[1], Humboldt_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        # ax.plot(Morro_coords[1], Morro_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)
        # #ax.plot(Buoy_coords[1], Buoy_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='grey',  alpha=0.9)

        # map
        ax.add_feature(cartopy.feature.BORDERS)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        ax.add_feature(cartopy.feature.RIVERS)
        ax.add_feature(cartopy.feature.LAND, zorder=10)
        cs = ax.coastlines(resolution='10m', linewidth=1, zorder=10)

                
        # Plotting the extracted coastline and a line 75km offshore
       # ax.plot(coastline[:, 0], coastline[:, 1], linewidth=1, color="r") 
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
        
        # Select parameter values
        if parameter == 'analysed_sst':
            df  = ostia.sel(time= day, method='nearest')
        elif parameter == 'CHL':
            df  = chl.sel(time= day, method='nearest')
        else:
            df = sat_masked.sel(time= day, method='nearest')
            
            
        #Plot color map               
        if parameter == "analysed_sst":
            
            # test = df.U_Geo_Oscar   # df.U_Ek +  (df['go_cross'] *100 * df.mld_sat)
            
            # c = ax.pcolormesh(df['lon'], df['lat'], test[ :, :]  ,   #df[parameter][ :, :],
            #                   cmap='coolwarm', transform=ccrs.PlateCarree(), shading='auto', 
            #                   )
            # label = "test"
            
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='coolwarm', transform=ccrs.PlateCarree(), shading='auto', 
                              vmin = np.nanpercentile(df[parameter].values.flatten()[df[parameter].values.flatten() != 0], 5), 
                              vmax = np.nanpercentile(df[parameter].values.flatten()[df[parameter].values.flatten() != 0], 95)) 
            label = "SST ($^\circ$C)"
            
        elif parameter == "wind_speed":
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', 
                              vmin = 0, 
                              vmax = 16) 
            label = "Wind speed (m/s)"
        elif parameter == "U_Geo_Oscar":
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', 
                              vmin = -0.6, 
                              vmax = 0.6) 
            label = "Geostrophic current (m/s)"
        elif parameter == "geostr_curr":
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', 
                              vmin = 0, 
                              vmax = 0.6) 
            label = "Geostrophic current (m/s)"
        elif parameter == "mld_sat":
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='Blues', transform=ccrs.PlateCarree(), shading='auto', 
                              vmin = 0, 
                              vmax = 50) 
            label = "MLD (m)"
            
        elif parameter == "CHL":
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='BuGn', transform=ccrs.PlateCarree(), shading='auto', 
                               vmin = 0, 
                               vmax = 50
                              ) 
            label = "Chl mass concentration (mg/m$^3$)"
        elif parameter == "CUTI_Oscar":
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', 
                               vmin = min_cuti, 
                               vmax = max_cuti
                              ) 
            label = "CUTI (m$^2$/s)" 
        elif parameter == "CUTI_model":
            try:
                CUTI = CUTI.drop(['year', 'month', 'day'], axis=1)
            except:
                pass
            lats = [float(col[:-1]) for col in CUTI.columns[:]]
            lat_values = coastline[:, 1]
            lon_values = coastline[:, 0]
            lat_to_lon = {lat: lon for lat, lon in zip(lat_values, lon_values) if not np.isnan(lon)}
            lons = [lat_to_lon.get(lat, np.nan)  - lon_difference(50, lat) for lat in lats]


            cuti = CUTI.loc[day]
            c = plt.scatter(lons, lats, c=cuti, s=170, cmap='viridis', transform=ccrs.PlateCarree(),edgecolors="black", 
                       vmin = min_cuti, 
                       vmax = max_cuti)
            label = "CUTI (m$^2$/s)" 
        else:
            c = ax.pcolormesh(df['lon'], df['lat'], df[parameter][ :, :],
                              cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', vmin=min_cuti, vmax=max_cuti)
            label = parameter
        
        
        # Add arrows for wind vectors 
        if parameter == "wind_speed":
            u =   df['u_wind'].values
            v =   df['v_wind'].values
            scale_factor = 200  # Adjust the scale factor for arrow length
            ax.quiver(df['lon'], df['lat'],
                      u[ :, :], v[ :, :], scale=scale_factor, color='white', transform=ccrs.PlateCarree())
            
        if parameter == "geostr_curr":
            u =   df['ug'].values
            v =   df['vg'].values
            scale_factor = 10  # Adjust the scale factor for arrow length
            ax.quiver(df['lon'], df['lat'],
                      u[ :, :], v[ :, :], scale=scale_factor, color='white', transform=ccrs.PlateCarree())
        
        
        # Add colorbar
        cbar = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.03, shrink=0.7, label= label ) #'CUTI (m$^2$/s)')
                    
        gl = ax.gridlines(draw_labels=True)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False
        
        for spine in ax.spines.values():
            spine.set_zorder(15) 

        ax.set_extent([-128,-120, 35, 42], crs=ccrs.PlateCarree())  
        
        # ax.set_title('{:%Y-%m-%d}, \nC$_H$={}, C$_M$={}'.format(day,
        #     round(CUTI.loc[pd.to_datetime(day), '41N'], 2), 
        #     round(CUTI.loc[pd.to_datetime(day), '36N'], 2))  , fontsize=10)   
       # fig.suptitle(day)   
        plt.tight_layout()
        # plt.subplots_adjust(
        #         hspace=0.6, wspace=0.1)
        
        
        
        # fig.savefig(f"paper_plots/Upwelling_case_{day}.png", dpi=300)


    


        
        
    
    
    
    
#%% Make lat - month - CUTI plot   - plot takes very long!!

lat_month_CUTI = 0
if  lat_month_CUTI == 1:
    
    lat_step = 0.25
    lat_range = np.arange(31,45, lat_step)

    
    
    # Create a DataFrame with latitude values as rows and months as columns
    months = np.arange(1,13)
    Av_CUTI = pd.DataFrame(index=lat_range, columns=months )
    parameter = "CUTI_Oscar"
    
    for lat in lat_range: 
    
        max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0] 
        min_lon = max_lon + lon_difference(offshore_diff, max_lon)
      
        # Average satellite data over 1 deg in N-S direction and up to Xkm offshore
        sat_local = sat.sel(lat=slice(lat-lat_step/2, lat+lat_step/2), lon=slice(min_lon, max_lon))
    
        for month in months:
        
            av = sat_local.where(sat_local.month==month, drop=True)[parameter].median()  # .mean(dim=['lat', 'lon']) 
     
            Av_CUTI[month][lat] = av
    
    
    # Define the custom colormap
    colors = [(0, 0, 0, 1)]  # Black for values between -0.5 and 0.5
    bounds = [-2, -0.1, 0.1, 2]  # Define value ranges
    cmap = mcolors.ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)])  # Transparent-Black-Transparent
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

        
    from matplotlib.path import Path
    from matplotlib.markers import MarkerStyle
    
    # Define a custom rectangle marker (width > height)
    rectangle_vertices = np.array([
        [-0.6, -0.1],  # Bottom-left
        [0.6, -0.1],   # Bottom-right
        [0.6, 0.1],    # Top-right
        [-0.6, 0.1],   # Top-left
        [-0.6, -0.1]   # Closing point
    ])
    
    # Create a path for the custom rectangle
    rectangle_marker = Path(rectangle_vertices)

    
    # Scatter plot   
    
         
    fig = plt.figure()
    sc = plt.scatter(*np.meshgrid(Av_CUTI.columns, Av_CUTI.index), c=Av_CUTI.values, cmap='viridis', s=600,
                     marker = MarkerStyle(rectangle_marker), vmin=-2, vmax = 2)
    cbar = plt.colorbar(label="CUTI (m$^{2}$/s)")
    cbar.ax.scatter(0.5, 0, color='black', s=50, clip_on=False, zorder=3)
    plt.scatter(*np.meshgrid(Av_CUTI.columns, Av_CUTI.index), c=Av_CUTI.values, cmap=cmap, norm=norm, s=100, marker = ".")

    plt.xlabel('Month')
    plt.ylabel('Latitude ($^\circ$)')
    plt.title('CUTI across latitude and seasons')
    plt.grid(False)
    plt.xticks(Av_CUTI.columns.astype(int).values, [calendar.month_abbr[m] for m in Av_CUTI.columns.astype(int).values]) 
    plt.tight_layout()
    
    # fig.savefig("paper_plots/CUTI_month_lat.png", dpi=300)    






#%% Time scales of CUTI

time_scales =0
if time_scales == 1 :
    
  
    # Wind data (6-hourly)
    # nbs_winds = xr.open_dataset('../satellite_data/NBS_winds/nbs_winds_1993_2023_6h.nc')
    # nbs_winds["wind_speed"] = np.sqrt(nbs_winds.u_wind**2 + nbs_winds.v_wind**2) 
    # nbs_winds["wind_dir"] = np.degrees(np.arctan2(nbs_winds.u_wind, nbs_winds.v_wind)) + 180
    
        
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
                ax.set_xlabel("Frequency $f$ (cycles per day)", fontsize='large')
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
        plt.xlabel('Time period $T$ (days)', fontsize='large')
        plt.yscale('linear')
        plt.tight_layout()  
        
        if plot == 1:
            return fig, ax, ax2
        else:   
            plt.close()
            return bins, smooth
        
        

        

    
    # Make the plot
    
    lat_range = np.arange(35,45)   # [40]  #  
    
    plt.rcParams.update({'font.size': 14})
    
    fig, ax, ax2 = large_spectrum(CUTI,freq=1,channels=['41N'])
 
    for lat in lat_range: 
    
            if (lat == 36) or (lat == 41):
                lw = 2
            else:
                lw = 1
                
            max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0] 
            min_lon = max_lon + lon_difference(offshore_diff, max_lon)
            

            # Average satellite data over 1 deg in N-S direction and up to Xkm offshore
            sat_local = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']).to_dataframe()
            ostia_local = ostia.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']).to_dataframe() 
            chl_local = chl.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']).to_dataframe()
            # nbs_local = nbs_winds.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 

        
        
            # Merge model CUTI and satellite area average and SST local area
            all = CUTI.merge(
                    sat_local[["CUTI_Oscar", "CUTI_SLA", "U_Ek", 
                               "U_Geo", "U_Geo_Oscar", "CUTI_Cop", 'U_Geo_cop']],   
                    left_index=True, right_index=True, how="outer")
            all = all.merge(
                    ostia_local[["analysed_sst"]].resample("D").median(), 
                    left_index=True, right_index=True, how="outer")
            all = all.replace([np.inf, -np.inf], np.nan)
            

                
        
            for param, color in zip(["U_Geo_Oscar"], ["C1"]):
            # for param, color in zip(["U_Ek",'U_Geo_Oscar', "CUTI_Oscar", "analysed_sst"], ["C1","C3","C0", "grey"]):
            # for param, color in zip([str(int(lat))+"N",  "CUTI_Oscar"], ["C2","C0"]):
                
                            
                if param == "U_Ek":
                    label = "U$_{Ek}$"
                elif param == "U_Geo_Oscar":
                    label = "U$_{Geo}$"
                elif param == str(int(lat))+"N":
                    label = "CUTI model"
                elif param == "CUTI_Oscar":
                    label = "CUTI"                
                elif param == "analysed_sst":
                    label = "SST"  
                else:
                    label = param
                
                total_bins = 60
                
                bins, smooth = large_spectrum(all,freq=1,channels=[param], plot = 0, total_bins=total_bins) # 
                if param == "analysed_sst":
                    bins, smooth = large_spectrum(ostia_local,freq=1,channels=[param], plot = 0, total_bins=total_bins) # 
                    smooth = [x/10 for x in smooth]
                col=cmap(norm(lat))
                ax.semilogx(bins,bins*smooth / np.nanmedian(bins*smooth),'-',marker = '.',label=label, color = col, alpha = 0.9, lw = lw, zorder = 10)
            
            # param =  "wind_speed"
            # color = "black"
            # bins, smooth = large_spectrum(nbs_winds_local,freq=1/4,channels=[param], plot = 0) # 
            # ax.semilogx(bins,bins*smooth / np.nanmedian(bins*smooth),'-',marker = '.',label=label, color = color, alpha = 0.9, lw = lw, zorder = 10)
 
            # param =  "wind_dir"
            # color = "blue"
            # bins, smooth = large_spectrum(nbs_winds_local,freq=1/4,channels=[param], plot = 0) # 
            # ax.semilogx(bins,bins*smooth / np.nanmedian(bins*smooth),'-',marker = '.',label=label, color = color, alpha = 0.9, lw = lw, zorder = 10)
                
                
            # param =  "CHL"
            # color = "green"
            # label = "Chlorophyll"  
            # bins, smooth = large_spectrum(chl_local,freq=1,channels=[param], plot = 0, total_bins=total_bins) # 
            # ax.semilogx(bins,bins*smooth / np.nanmedian(bins*smooth),'-',marker = '.',label=label, color = color, alpha = 0.9, lw = lw, zorder = 10)
                          
                
            if lat == lat_range[0]:
                ax.legend()
    mn, mx = ax.get_xlim()
    ax2.set_xlim(1/mn, 1/mx)
    #plt.title("Time scales in upwelling, and Ekman and geostrophic transport at latitudes 35$^\circ$N - 45$^\circ$N")
    plt.tight_layout()

    
    plt.ylim(-1, 23)
    
    #fig.savefig("paper_plots/Time_scales.png", dpi=300)
    






#%% Correlation CUTI vs Chl

corr_CUTI_Chl = 0
if corr_CUTI_Chl == 1 :
    
    # Make the plot
    
    lat_range = np.arange(35,45)   # 
    
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure( figsize=(15,7))
 
    for lat in lat_range: 
    
            if (lat == 36) or (lat == 41):
                lw = 2
            else:
                lw = 1
                
            max_lon = coastline[np.where(coastline[:, 1] == lat)[0][0], 0] 
            min_lon = max_lon + lon_difference(offshore_diff, max_lon)
            

            # Average satellite data over 1 deg in N-S direction and up to Xkm offshore
            sat_local = sat.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 
            chl_local = chl.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(min_lon, max_lon)).mean(dim=['lat', 'lon']) 

        
            # Merge model CUTI and satellite area average and SST local area
            all = pd.merge(sat_local[["CUTI_Oscar"]].to_dataframe(), 
                           chl_local[["CHL"]].to_dataframe(), 
                    left_index=True, right_index=True, how="outer")
            all = all.replace([np.inf, -np.inf], np.nan)
            all = all.dropna()
            
            
            # Cross-correlation
            x_signal = all.CUTI_Oscar
            y_signal = all.CHL
            
            x_signal[x_signal.isna()==False] = scipy.signal.detrend(x_signal.dropna())             
            y_signal[y_signal.isna()==False] = scipy.signal.detrend(y_signal.dropna()) 
            
            correlation = signal.correlate(x_signal, y_signal, mode="full")
           
            # Calculate the standard deviations of both series
            std_x = np.std(x_signal.dropna())
            std_y = np.std(y_signal.dropna())
            
            # Normalize the cross-correlation result
            correlation /= (std_x * std_y)
    

            # Lages for the correlation
            lags = signal.correlation_lags(x_signal.size, y_signal.size, mode="full")
            
            lag = lags[np.argmax(correlation)]
            
            # plt.figure()
            plt.plot(lags, correlation)
            # plt.plot(x_signal, y_signal,".")
                
                
            if lat == lat_range[0]:
                ax.legend()
                
    plt.ylabel("Cross-correlation between CUTI and Chlorophyll")
    plt.xlabel("Time lage (days)")
    mn, mx = ax.get_xlim()
    ax2.set_xlim(1/mn, 1/mx)
    plt.title("Cross-correlation between CUTI and Chl at latitudes 35$^\circ$N - 45$^\circ$N")
    plt.tight_layout()
    plt.grid()

    

    


























