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

sys.path.append("../../NSO/NSO_data_processing")
from Functions_general import *


from Functions_Oracle import *




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
        fig, ax = plt.subplots(figsize=(8,6))
        
    
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
            ax.set_ylabel("PSD$\cdot$ f ($\sigma^2)$, normalized", fontsize='large')
            plt.tight_layout()

        #first, = ax.loglog(X,X*Y, linewidth=0.5,label='',alpha=0.3)
        first, = ax.loglog(np.nan, np.nan, linewidth=0.5,label='',alpha=0.0)
        ax.loglog(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = '.',label=v,color = first.get_color(), zorder=10)
        ax.legend(loc='best', fontsize='large')  
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



#%%  Read data

# Read buoy data
    
humb = read_CTD("Humb")
humb_curr = read_currents("Humb")
# humb_curr.index.get_level_values('depth').unique()
humb_meteo = read_meteo("humb")

morro = read_CTD("Morro")
morro_curr = read_currents("Morro")
morro_meteo = read_meteo("Morro")

current_depth = 16
humb = humb.merge(humb_curr.xs(current_depth, level='depth'), left_index=True, right_index=True, how="outer")
humb = humb.merge(humb_meteo, left_index=True, right_index=True, how="outer")
morro = morro.merge(morro_curr.xs(current_depth, level='depth'), left_index=True, right_index=True, how="outer")
morro = morro.merge(morro_meteo, left_index=True, right_index=True, how="outer")

buoy = read_buoy()

    
resample = "30D"
humb["SST_anomaly"] = humb.SST - humb.SST.rolling(resample, center=True).mean()
morro["SST_anomaly"] = morro.SST - morro.SST.rolling(resample, center=True).mean()



# satellite data

## Ostia SST measuremnets
ostia = xr.load_dataset("../satellite_data/ostia_all.nc", engine="netcdf4")  # all coordinates close to California coast
ostia_sst = pd.DataFrame()
ostia_sst['humb']  = ostia.sel(lat= Humboldt_coords[0], lon= Humboldt_coords[1], method = 'nearest').to_dataframe().analysed_sst


# Read NBS satellite wind data
files = sorted(glob.glob('C:/Users/uegerer/Desktop/Oracle/satellite_data/NBS_winds/NBS*202*.nc'))[:]
nbs = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
nbs = nbs.sel(zlev=10)
nbs_humb = nbs.sel(lat=Humboldt_coords[0], lon=Humboldt_coords[1]+360, method = "nearest").to_dataframe()
nbs_morro = nbs.sel(lat=Morro_coords[0], lon=Morro_coords[1]+360, method = "nearest").to_dataframe()
    



# Oscar data
files = sorted(glob.glob('C:/Users/uegerer/Desktop/Oracle/satellite_data/Oscar/' +'*Oscar_*.nc'))[:-1]
oscar = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'time': 1})
oscar_humb = oscar.sel(latitude=Humboldt_coords[0], longitude=Humboldt_coords[1]+360, method = "nearest")
oscar_humb["curr_dir"] = np.degrees(np.arctan2( oscar_humb.u,  oscar_humb.v) )+180


# Read CUTI data

CUTI = pd.read_csv('C:/Users/uegerer/Desktop/Oracle/CUTI_data/CUTI_daily.csv', header = 0, index_col=None).apply(pd.to_numeric, errors='coerce')
CUTI.index = pd.to_datetime(CUTI[['year', 'month', 'day']])


# Calculate wind components
humb["along_shore_wind"], humb["cross_shore_wind"] = calculate_wind_components(
    wind_direction=humb.wind_direction, wind_speed = humb.wind_speed, coastline_angle=180) # angle is 180 for Humboldt and 145 for Morro
humb["along_shore_current"], humb["cross_shore_current"] = calculate_wind_components(
    wind_direction=humb.current_direction, wind_speed = humb.current_speed, coastline_angle=180) # angle is 180 for Humboldt and 145 for Morro
morro["along_shore_wind"], morro["cross_shore_wind"] = calculate_wind_components(
    wind_direction=morro.wind_direction, wind_speed = morro.wind_speed, coastline_angle=145) # angle is 180 for morrooldt and 145 for Morro
morro["along_shore_current"], morro["cross_shore_current"] = calculate_wind_components(
    wind_direction=morro.current_direction, wind_speed = morro.current_speed, coastline_angle=145) # angle is 180 for morrooldt and 145 for Morro
buoy["along_shore_wind"], buoy["cross_shore_wind"] = calculate_wind_components(
    wind_direction=buoy.wind_direction, wind_speed = buoy.wind_speed, coastline_angle=180) # angle is 180 for buoyoldt and 145 for Morro





#%% Calculate time scales - plots






res = 40

## Humboldt spectrum
fig, ax, ax2 = large_spectrum(humb,freq=144,channels=['SST',
                                                         'along_shore_wind','cross_shore_wind',
                                                         'along_shore_current', 'cross_shore_current',
                                                         #'pressure', 
                                                         'air_temperature'], time_series = 0, total_bins = res) # 
bins, smooth = large_spectrum(CUTI,freq=1,channels=['41N'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = 'x',label="CUTI", color="black")
ax.legend()
mn, mx = ax.get_xlim()
ax2.set_xlim(1/mn, 1/mx)
plt.title("Humboldt buoy")
plt.tight_layout()


## Morro spectrum
fig, ax, ax2 = large_spectrum(morro,freq=144,channels=['SST',
                                                         'along_shore_wind','cross_shore_wind',
                                                         'along_shore_current', 'cross_shore_current',
                                                         #'pressure', 
                                                         'air_temperature'], time_series = 0, total_bins = res)
bins, smooth = large_spectrum(CUTI,freq=1,channels=['36N'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = 'x',label="CUTI", color="black")
ax.legend()
mn, mx = ax.get_xlim()
ax2.set_xlim(1/mn, 1/mx)
plt.title("Morro buoy")
plt.tight_layout()


# Buoy spectrum
fig, ax, ax2 = large_spectrum(buoy,freq=144,channels=['SST',
                                                        'along_shore_wind','cross_shore_wind',
                                                        "None", "None",
                                                        #'sea_level_pressure', 
                                                        'air_temp'
                                                        #'significant_wave_height',
                                                ], time_series = 0, total_bins = res)
bins, smooth = large_spectrum(CUTI,freq=1,channels=['41N'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = 'x',label="CUTI", color="black")
ax.legend()
mn, mx = ax.get_xlim()
ax2.set_xlim(1/mn, 1/mx)
plt.title("NDBC buoy near Humboldt")
plt.tight_layout()

# Satellite spectrum
ostia_sst = ostia_sst.rename(columns={"humb": "SST"})
oscar = oscar_humb.to_dataframe()

fig, ax, ax2 = large_spectrum(ostia_sst,freq=1,channels=['SST'], time_series = 0, total_bins = res)
bins, smooth = large_spectrum(nbs_humb,freq=4,channels=['v_wind'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = '.',label="along wind")
bins, smooth = large_spectrum(nbs_humb,freq=4,channels=['u_wind'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = '.',label="cross wind")
bins, smooth = large_spectrum(oscar,freq=1,channels=['v'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = '.',label="along current")
bins, smooth = large_spectrum(oscar,freq=1,channels=['u'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = '.',label="cross current")
bins, smooth = large_spectrum(CUTI,freq=1,channels=['41N'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = 'x',label="CUTI", color="black")
ax.legend()
mn, mx = ax.get_xlim()
ax2.set_xlim(1/mn, 1/mx)
plt.title("Satellite data near Humboldt")
plt.tight_layout()




#CCS ROMS model at Humboldt (read data first)
## Humboldt spectrum
df= read_CCS_ROMS_model_single_loc_csv(
    "../CCS_model_data/fmrc_CCSRA_2016a_Phys_ROMS_z-level_(depth)_Aggregation_best_Humb_2011-2023.csv") # _15km_offshore

fig, ax, ax2 = large_spectrum(df[df.alt==-2],freq=1,channels=['temp',
                                                         'None','None',
                                                         ], time_series = 0, total_bins = res) #

bins, smooth = large_spectrum(df[df.alt==-20],freq=1,channels=['v'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = '.',label="along-shore corrent (20m depth)") 
bins, smooth = large_spectrum(df[df.alt==-20],freq=1,channels=["u"], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = '.',label="across-shore current (20m depth)") 
bins, smooth = large_spectrum(df[df.alt==-20],freq=1,channels=['w'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = 'x',label="w (20m depth)", color="green")
bins, smooth = large_spectrum(df[df.alt==-20].resample("10D").mean(),freq=1/(60*60*24*10),channels=['w'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth)/10,'-',marker = 'x',label="w (20m depth), 10-day average", color="lightgreen")
bins, smooth = large_spectrum(CUTI,freq=1,channels=['41N'], plot = 0)
ax.semilogx(bins,bins*smooth / np.nanmean(bins*smooth),'-',marker = 'x',label="CUTI", color="black")
ax.legend()
mn, mx = ax.get_xlim()
ax2.set_xlim(1/mn, 1/mx)
plt.title("CCS-ROMS model data near Humboldt (daily res)")
plt.tight_layout()









































