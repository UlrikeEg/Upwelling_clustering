import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr



Humboldt_coords = [40.9708, - 124.5901]
Morro_coords    = [35.71074, - 121.84606]
Buoy_coords    = [40.748, - 124.527]








def calculate_coriolis_parameter(latitude):
    # Earth's angular velocity (radians per second)
    omega = 7.2921159e-5
    
    # Convert latitude from degrees to radians
    latitude_rad = np.radians(latitude)
    
    # Calculate Coriolis parameter
    f = 2 * omega * np.sin(latitude_rad)
    
    return f







def lon_difference(zonal_distance_km, latitude):
    radius_earth = 6371.0
    phi = np.radians(latitude)
    delta_lon = zonal_distance_km / (radius_earth * np.cos(phi))
    return np.degrees(delta_lon)



def calculate_along_cross_shore_components(eastward_wind, northward_wind, coastline_angle):
    
    """
    Definitions:
        coastline angle: 0deg when aligned north-south
                         -30deg when looking more south when looking offshore
                         +30deg when looking more north when looking offshore
        along and cross shore defined positive when upwelling supported (along is more northward, cross is more westward)
    """
   
    # plt.figure()
    # ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())  
    # ax.pcolormesh(sat.isel(time= 100)['lon'], sat.isel(time= 100)['lat'],
    #               sat.isel(time= 100).coastline_angle[ :, :].T, cmap='viridis', transform=ccrs.PlateCarree(), shading='auto')
    # gl = ax.gridlines(draw_labels=True)
    # gl.bottom_labels = True
    # gl.left_labels = True
    # gl.right_labels = False
    # gl.top_labels = False  
    
    # Convert coastline angle to radians
    coastline_angle_rad = np.radians(coastline_angle)
    
    # Calculate along-shore wind component
    along_shore_wind = - (eastward_wind * np.sin(coastline_angle_rad) + northward_wind * np.cos(coastline_angle_rad))
    
    # Calculate cross-shore wind component
    cross_shore_wind = (- eastward_wind * np.cos(coastline_angle_rad) + northward_wind * np.sin(coastline_angle_rad))
    
    # eastward_wind = np.array([0, 0, 0, 10, 10, 10,0,0,0,0,0,0])  # Eastward wind component
    # northward_wind = np.array([10,10,10,10,10,10,0, 0, 0, -10, -10, -10])  # Northward wind component
    # along_shore_wind, cross_shore_wind = calculate_along_cross_shore_components(eastward_wind, northward_wind, coastline_angle=-30)

    # plt.figure()
    # plt.plot(eastward_wind,".", label = "east")
    # plt.plot(northward_wind,".", label = "north")
    # plt.plot(along_shore_wind,"x", label = "along")
    # plt.plot(cross_shore_wind,"x", label = "cross")
    # plt.legend()
    
    return along_shore_wind, cross_shore_wind



# Along- and cross shore components
def calculate_wind_components(wind_direction, wind_speed, coastline_angle):
    # Convert coastline angle to radians
    coastline_angle_rad = np.radians(coastline_angle)
    
    # Convert wind direction to radians
    wind_direction_rad = np.radians(wind_direction)
    
    # Calculate along-shore wind component
    cross_shore_wind = wind_speed * np.sin(wind_direction_rad - coastline_angle_rad)
    
    # Calculate cross-shore wind component
    along_shore_wind = wind_speed * np.cos(wind_direction_rad - coastline_angle_rad)
    
    return round(along_shore_wind, 3), round(cross_shore_wind,3)


def calculate_coriolis_parameter(latitude):
    # Earth's angular velocity (radians per second)
    omega = 7.2921159e-5
    
    # Convert latitude from degrees to radians
    latitude_rad = np.radians(latitude)
    
    # Calculate Coriolis parameter
    f = 2 * omega * np.sin(latitude_rad)
    
    return f



def add_costline_angle_to_sat(sat, plot = 0):
    
    # Get coastline array
    coastline_array = np.load("coastline.npy")
    
    ## This is the old way (same coastline angle at same latitude)
    # coastline_ds = xr.Dataset({'coastline_angle': (('lat',), coastline_array[:, 2])},
    #                           coords={'lat': sat['lat']})
    # coastline_ds = coastline_ds.expand_dims(time=sat['time'], lon=sat['lon'] )  # Broadcast the coastline angle values across lon and time dimensions
    # sat['coastline_angle'] = coastline_ds['coastline_angle']

    
    coastline_array = np.apply_along_axis(lambda col: pd.Series(col).ffill().bfill(), 0, coastline_array)
    
    
    # Function to calculate Euclidean distance
    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Calculate distance between each grid point and all coastline points
    distances = np.zeros((len(sat['lat']), len(sat['lon']), len(coastline_array)))
    for i, lat in enumerate(sat['lat']):
        for j, lon in enumerate(sat['lon']):
            distances[i, j] = euclidean_distance(lon.values, lat.values, coastline_array[:, 0], coastline_array[:, 1])
    
    # Find index of minimum distance for each grid point
    min_distance_index = np.argmin(distances, axis=2)
    
    # Retrieve coastline angle based on minimum distance index
    coastline_angles = coastline_array[min_distance_index, 2]
    
    # Create coastline angle dataset
    coastline_ds = xr.Dataset({'coastline_angle': (('lat', 'lon'), coastline_angles)},
                              coords={'lat': sat['lat'], 'lon': sat['lon']})
    
    # Assign coastline angle to the satellite dataset
    sat['coastline_angle'] = coastline_ds['coastline_angle']
    
    
    if plot == 1:
        plt.figure()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())  
        c = ax.pcolormesh(sat.isel(time= 100)['lon'], sat.isel(time= 100)['lat'],
                      sat.isel(time= 100).coastline_angle[ :, :], cmap='viridis', transform=ccrs.PlateCarree(), shading='auto')
        gl = ax.gridlines(draw_labels=True)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False  
        # map
        ax.add_feature(cartopy.feature.LAND, zorder = 100)
        ax.add_feature(cartopy.feature.BORDERS, zorder = 100)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5, zorder = 100)
        ax.add_feature(cartopy.feature.RIVERS, zorder = 100)
        cs = ax.coastlines(resolution='10m', linewidth=1, zorder = 101)
        ax.plot(coastline_array[:, 0], coastline_array[:, 1], linewidth=2, color="r") 
        plt.colorbar(c, ax=ax, orientation='vertical', label= 'coastline angle, based on closest distance to smoothed coastline')
                
    return sat



def get_offshore_parallel_coastline(coastline, offshore_diff):
    
    lon_diff =  [elem1 - elem2 / elem3 for elem1, elem2, elem3 in zip(coastline[:, 0], 
                                               lon_difference(offshore_diff,  coastline[:, 1]) ,  
                                               np.cos(np.radians(coastline[:, 2]))    
                                               )]

    return lon_diff

