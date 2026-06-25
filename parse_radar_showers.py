'''

Parse file for radar data from orbits new 29

Each file has hundereds of lines; each one being a meteor echo that was detected
This script collects clean echos by checking if timeof flight speed and the corrected speed agree to within 5%
Plotting will be done in a seperate function or file from this one

Script written by Zach Aubry on May 11th, 2026
The first step to collecting radar data that will go into a master's level project with the Western Meteor Physics Group

Filtering criteria:
- Velocities should agree to within 5%
- interferometry error to be less than 2 degrees
- radiant solid angle error to be less than 5 degrees
- Will impliment to look for orbits with more than 4 station measurements

'''

import os
import sys
# print('script running here', sys.executable) # used this for conda installing troubleshoot
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm # removed from matplotlib
from matplotlib import patches
import sklearn as sk
from pathlib import Path
from scipy.spatial import ConvexHull
import plotly.graph_objects as go


# My imported functions
from cel2hel2cel import *


# This function is used to go through each meteor file for a chose year; the raw orbit data must exist in the file already
    # Might make this a class to do multiple tests at once for different filtering criteria; need to review classes first

# Global path which is used in several functions
home = Path.home() / 'Desktop/radar'

# Source locations by ecliptic coordinates organized by [long, lat]
helion = [np.arange(360, 320, -1), np.arange(-15, 15)]

helion_plus = [np.arange(360, 320, -1), np.arange(0, 15)]
helion_minus = [np.arange(360, 320, -1), np.arange(-15, 0)]

antihelion = [np.arange(220, 180, -1), np.arange(-15, 15)]
north_apex = [np.arange(320, 220, -1), np.arange(0, 50)]
south_apex = [np.arange(320, 220, -1), np.arange(-25, 0)]
north_toroidal = [np.arange(360, 180, -1), np.arange(50, 75)]

# print(antihelion)


def Parse(folder, filename, method='all', sources=[False, 'AH']):

    '''
    Goes through the specified radar file and discards any events we deem to be too noisy
    This is based on the percent difference between the time of flight velocity and the pre-t0 velocity; 
        if they differ by more than 5% we discard the event as a noisy echo

    The clean echoes are stored in a parent dictionary that is keyed by the observed date and time, and contains the velocities, percent difference, and ecliptic coordinates for each event
    
    Being used on my MacBook Air, so the directories will have different names than the ones on my Linux Desktop Script
    '''

    # home= '/home/zaubs/Desktop/radar/'
    # home = Path.home() / 'Desktop/radar'

    isolate, source = sources # boolean value and the name of the soruce

    sl = filename[9:12] # solar longitude

    path = os.path.join(home, folder, filename)
    # print(path)
    # print(os.path.exists(path)) # True if the directory exists
    with open(path, 'r') as data:
        # print(data)
        parent_dict = {} # will organize event parameters here that are keyed by date and time ; dates are mostly the same for this file

        # skips first two lines
        for r in range(2):
            next(data, None)

        for row in data:
            # print(row) # works
            line = row.strip()
            params = line.split()
            # print(params)
           

            # Indexing each column; no influence on the code, I just used this for reference
            if params[0] == "#":
                if params[1] == "date":
                    for i, item in enumerate(params[1:-1]):
                        # print(i, item)
                        continue # works
            
            elif line == "":
                continue

            # defining variables
            else:
                # print(params) # works
                date = params[0]
                # print(date)
                time = params[1] # EST? UTC?
                # print(time)

                # Distance from the Zehr
                R0 = params[12]

                # zenithal and azimuthal angles
                theta = params[13]
                phi = params[14]

                # ecliptic coordinates of the geocentric radiant
                
                ecl_lon = params[44]
                ecl_lat = params[45]

                # Equatorial radiant position (J2000)

                alpha_p = params[24]
                delta_p = params[25]

                # Geocentric radiant position (J2000)
                alpha_g = params[26] # right ascension
                delta_g = params[27] # declination
                del_rad_g = params[28] # uncertainty in the radiant position

                # solar longitude
                slon = params[8]    

                # solar centered longitude
                corrected_lon = str(round(float(ecl_lon) - float(slon), 2)) # causes longitude to shift left
                
                
                # apsidal direction of the geocentric radiant
                ecl_lon_aps = params[54]
                ecl_lat_aps = params[55]

                # velocity parameters
                vel_TimeofFlight = params[20] # in atmosphere time of flight speed
                del_vel_TimeofFlight = params[21] # tof speed uncertainty

                # want to check if the two uncertainties overlap

                vel_PTN0 = params[124] # pre-t0 speed
                del_vel_PTN0 = params[125] # pre-t0 speed uncertainty

                vel_geo = params[39] # geocentric velocity; put in convex hull function and include a check if the speed is defined or not
                del_vel_geo = params[40] # uncertainty in geocentric velocity

                # interferometry error
                int_error = params[15]

                # radiant solid angle error
                solid_angle_error = params[19]

                # number of stations
                num_stations = params[2]

                # station measurement error
                sdel = params[123]
                
                # turning radiant into heliocentric coordinates - I want to use this to get locations of the showers in ecl lon/lat
                v_cel      = getvec(float(alpha_g), float(delta_g))
                v_hel      = cel2hel(v_cel, float(slon))
                l_comp, b_comp = getangle(v_hel) # run through solar longitude correction

                # solar longitude correction for c2h coordinates : this mirrors the distribution so might be applied one too many times
                # alpha, delta = slon_corr(float(slon), l_comp, b_comp)

                # v_cel      = getvec(float(alpha), float(delta))
                # v_hel      = cel2hel(v_cel, float(slon))
                # l_comp, b_comp = getangle(v_hel) # add these to dict, pull through grab_coords and call when using echo_plot
                
                # Margaret said her function should be converting to corrected longitude already
                # corrected_l_comp = str(round(float(l_comp) - float(slon), 2))
                corrected_l_comp = str(round(float(l_comp) - 90, 2))

                # filtering function calls
                percent_check, overlap, percent_diff = vel_check(vel_TimeofFlight, vel_PTN0, del_vel_TimeofFlight, del_vel_PTN0, vel_geo) # using the function to check if the velocities agree within 5%
                # list of boolean value, and percentage

                del_int = int_check(int_error) # using the function to check if the interferometry error is less than 2 degrees

                del_radiant = solid_angle_check(solid_angle_error) # using the function to check if the radiant solid angle error is less than 5 degrees

                del_stations = station_check(num_stations, sdel) # using the function to check if the event has more than 4 station measurements with an error of less than 3 degrees


                # want to check what the numbers are for each criteria individually met, and for combinations for each criteria, then finally for all four criteria met
                # might create a few branches here to track each case

                # use this if you only want to plot one particular source
                if isolate == True:
                    # CASE: plotting only the antihelion
                    if source == 'AH':
                        if (int(float(corrected_lon)) % 360) not in antihelion[0] or int(float(ecl_lat)) not in antihelion[1]:
                            # print('not in source')
                            continue # should skip this event if it is not in the source

                    # CASE: plotting only the helion
                    elif source == 'H':
                        if (int(float(corrected_lon)) % 360) not in helion[0] or int(float(ecl_lat)) not in helion[1]:
                            # print('not in source')
                            continue # should skip this event if it is not in the source
                    
                    # SUBCASE: Only positive latitudes in the helion
                    elif source == 'H+':
                        if (int(float(corrected_lon)) % 360) not in helion_plus[0] or int(float(ecl_lat)) not in helion_plus[1]:
                            # print('not in source')
                            continue # should skip this event if it is not in the source
                    
                    # SUBCLASS: Only negative latitudes in the helion
                    elif source == 'H-':
                        if (int(float(corrected_lon)) % 360) not in helion_minus[0] or int(float(ecl_lat)) not in helion_minus[1]:
                            # print('not in source')
                            continue # should skip this event if it is not in the source

                    # CASE: plotting only north apex
                    elif source == 'NA':
                        if (int(float(corrected_lon)) % 360) not in north_apex[0] or int(float(ecl_lat)) not in north_apex[1]:
                            # print('not in source')
                            continue # should skip this event if it is not in the source
                    
                    # CASE: plotting only south apex
                    elif source == 'SA':
                        if (int(float(corrected_lon)) % 360) not in south_apex[0] or int(float(ecl_lat)) not in south_apex[1]:
                            # print('not in source')
                            continue # should skip this event if it is not in the source
                    
                    # CASE: plotting only north toroidal
                    elif source == 'NT':
                        if (int(float(corrected_lon)) % 360) not in north_toroidal[0] or int(float(ecl_lat)) not in north_toroidal[1]:
                            # print('not in source')
                            continue # should skip this event if it is not in the source

                # CASE: Raw data only
                if method == 'raw':
                    # want scaled longitudes for plotting
                    parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo,
                                          "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0,
                                          "Cel2Hel Longitude" : l_comp, "Cel2Hel Latitude" : b_comp}
                
                else:
                    # CASE: All four filters + Duplicate check applied
                    if method == 'all':
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_check != False and overlap != False) and del_int != False and del_radiant != False and del_stations != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, 
                                                    "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff, "Interferometry Error": int_error, "Solid Angle Error": solid_angle_error, 
                                                    "Station Measurement Error": sdel, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0,
                                                    "Cel2Hel Longitude" : l_comp, "Cel2Hel Latitude" : b_comp}

                    # # CASE: velocity check only
                    elif method == 'vel': # for 2025, 652,372 events are returned from AND condition below
                                          # for 2025, 758,183 events are returned from OR condition below
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if percent_check != False and overlap != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0,
                                                    "Cel2Hel Longitude" : l_comp, "Cel2Hel Latitude" : b_comp}
                        
                    # CASE: Interferometry check only
                    elif method == 'int':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_int != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Interferometry Error": int_error,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0,
                                                "Cel2Hel Longitude" : l_comp, "Cel2Hel Latitude" : b_comp}                
                    
                    # CASE: Radiant Location Check only
                    elif method == 'angle':
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_radiant != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Solid Angle Error": solid_angle_error,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0,
                                                    "Cel2Hel Longitude" : l_comp, "Cel2Hel Latitude" : b_comp}                
                    
                    #  CASE: Station Measurement Check only
                    elif method == 'station':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_stations != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Station Measurement Error": sdel,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0,
                                                    "Cel2Hel Longitude" : l_comp, "Cel2Hel Latitude" : b_comp}
                    

                    # CASE: velocity and interferometry check only
                    elif method == 'vel and int' or method.upper() == 'VI':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_check != False and overlap != False) and del_int != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                "Interferometry Error": int_error, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0,
                                                    "Cel2Hel Longitude" : l_comp, "Cel2Hel Latitude" : b_comp}
                        
                    # CASE: velocity, interferometry and solid angle check only
                    elif method == 'vel and int and angle' or method.upper() == 'VIA':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_check != False and overlap != False) and del_int != False and del_radiant != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                    "Interferometry Error": int_error, "Solid Angle Error": solid_angle_error, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                        "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0,
                                                        "Cel2Hel Longitude" : l_comp, "Cel2Hel Latitude" : b_comp}

        clean_data = len(parent_dict)

        # print(f'File {filename} has {clean_data} clean echoes')

    return parent_dict, clean_data # dictionary of clean echo data, number of clean files (with and without defined coordinates)


# Filtering methods below

# def vel_check(vel1, vel2, dvel1, dvel2, velg):
#     '''
#     This function will be used to check if the time of flight velocity and the pre-t0 velocity agree to within 5%
#     take vel m uncertainty too (velm - d_velm)
#     '''
#     if vel1[0] == '.' or vel2[0] == '.' or dvel1[0] == '.' or dvel2[0] == '.':
#         return False, False, 0 # skip rows with missing speed data; keep return type consistent

#     # changing string format to float numbers
#     vel1, vel2, dvel1, dvel2 = float(vel1), float(vel2), float(dvel1), float(dvel2)
#     velg = float(velg)

#     # Should skip this step for any meteor with a velocity of 40 km/s or greater

#     # if velg >= 40:
#     #     return True, True # this should add the high speed echoes to the clean file data, assuming they satisfy the other restrictions


#     difference = vel1 - vel2
#     average = (vel1 + vel2) / 2
#     percent_diff = np.abs(difference / average) * 100

#     # consider a more restrictive filter on faster meteors, but not one strong enough as the percent difference test?
#     if velg >= 48:
#         return True, True,  percent_diff # 48 km/s is roughly in between the low and high velocity distributions
#     # the check is currently if percent_diff is True AND <10. For these meteors it cannot be both

#     # uncertainty overlap check

#     lower1, upper1 = vel1 - dvel1, vel1 + dvel1
#     lower2, upper2 = vel2 - dvel2, vel2 + dvel2
#     overlap = max(lower1, lower2) <= min(upper1, upper2)

#     if percent_diff <= 10:

#         percent_check = True
    
#     else:

#         percent_check = False
    
#     return percent_check, overlap, percent_diff

def vel_check(vel1, vel2, dvel1, dvel2, velg):
    '''
    This function will be used to check if the time of flight velocity and the pre-t0 velocity agree to within 5%
    take vel m uncertainty too (velm - d_velm)
    '''
    if vel1[0] == '.' or vel2[0] == '.' or dvel1[0] == '.' or dvel2[0] == '.':
        return False, False, 0 # skip rows with missing speed data; keep return type consistent

    # changing string format to float numbers
    vel1, vel2, dvel1, dvel2 = float(vel1), float(vel2), float(dvel1), float(dvel2)
    velg = float(velg)

    # Should skip this step for any meteor with a velocity of 40 km/s or greater

    # if velg >= 40:
    #     return True, True # this should add the high speed echoes to the clean file data, assuming they satisfy the other restrictions


    # consider a more restrictive filter on faster meteors, but not one strong enough as the percent difference test?
    
    difference = vel1 - vel2
    average = (vel1 + vel2) / 2
    percent_diff = np.abs(difference / average) * 100

    lower1, upper1 = vel1 - dvel1, vel1 + dvel1
    lower2, upper2 = vel2 - dvel2, vel2 + dvel2
    overlap = max(lower1, lower2) <= min(upper1, upper2)

    # leaving out high velocity meteors to cap our radiant space
    if velg > 80 or velg < 10: # might include this step somewhere in the voxel plotting to only reduce the radiant space of the shower removal
        return False, False, percent_diff

    if vel1 >= 48:
        return True, True, percent_diff # 48 km/s is roughly in between the low and high velocity distributions

    else:
        return percent_diff <= 10, overlap, percent_diff



def int_check(int_error):
    '''
    This function will be used to check if the interferometry error is less than 2 degrees
    '''
    if int_error[0] == '.':
        return False # skip rows with missing interferometry error data

    # Error within 2 degrees is acceptable
    if float(int_error) <= 2:
        return True
    else:
        return False
    

def solid_angle_check(solid_angle_error):
    '''
    This function will be used to check if the radiant solid angle error is less than 5 degrees
    '''
    if solid_angle_error[0] == '.':
        return False # skip rows with missing solid angle error data

    # Error within 5 degrees is acceptable
    if float(solid_angle_error) <= 5:
        return True
    else:
        return False


def station_check(num_stations, sdel):
    '''
    This function will be used to check if the event has more than 4 station measurements
    '''
    if num_stations[0] == '.':
        return False # skip rows with missing station measurement data

    # looking for orbits with measurements coming from more than 4 stations (3 is the minimum that determines a good orbit)
    if int(num_stations) > 3:

        if sdel[0] == '.': 
            return False # skip rows with missing station measurement data
        
        if float(sdel) <= 3: # station measurement error within 5 degrees is acceptable
            return True
        else:
            return False
    
    else:
        return False 


def voxel_map(lmda, beta, vels, year, name=None, map_mode='shower', threshold=0, bounds=None, shower_info=None):
    '''
    put meteor distribution into a voxel (cubic space)
    voxels defined by buffer (5 degrees solar longitude) - I used copliot to do this step, I've never created voxels in a 3d plot before
        subtraction (5/10 solar longitude?)
    convex hull created as a boundary around meteors left after subtraction
    '''

    figure = plt.figure(figsize=(10,5))
    ax = figure.add_subplot(projection='3d')

    # converting to 1d arrays
    lmda = np.asarray(lmda)
    beta = np.asarray(beta)
    vels = np.asarray(vels)
    # dates = np.asarray(dates)

    
    if bounds != None:
        lon_bounds, lat_bounds, vel_bounds = bounds[0], bounds[1], bounds[2]
        

        # this works; will have to check if I am computing the average matrix properly after then apply this to the 3d voxel step
        mask = (
            (lmda >= lon_bounds[0]) & (lmda <= lon_bounds[1]) &
            (beta >= lat_bounds[0]) & (beta <= lat_bounds[1]) &
            (vels >= vel_bounds[0]) & (vels <= vel_bounds[1])
        )
        
        lmda = lmda[mask]
        beta  = beta[mask]
        vels = vels[mask]
        # dates = dates[mask]

    # avoid zero-range bins
    eps = 1e-6
    if lmda.max() == lmda.min(): lmda = lmda + np.linspace(0, eps, lmda.size)
    if beta.max() == beta.min(): beta = beta + np.linspace(0, eps, beta.size)
    if vels.max() == vels.min(): vels = vels + np.linspace(0, eps, vels.size)

    # 3D histogram into 8x8x8 voxels
    bins = (8, 8, 8)
    H, edges = np.histogramdd(np.vstack((lmda, beta, vels)).T, bins=bins) # count per voxel

    print(edges)
    voxels = H > threshold

    # print(voxels) # array of boolean values
    # print(H) # 3d array of bin counts, do subtraction with this

    cmap = plt.colormaps['plasma']

    norm = plt.Normalize(vmin=H[voxels].min() if voxels.any() else 0,
                     vmax=H[voxels].max() if voxels.any() else 1)
    facecolors = np.zeros(voxels.shape + (4,), dtype=float)
    facecolors[voxels] = cmap(norm(H[voxels])) # maps counts to the voxels

    # norm_alpha = plt.Normalize(vmin=H[voxels].min(), vmax=H[voxels].max())
    # alpha_min = 0.5  # lowest opacity for sparse voxels (0 = invisible)
    # alpha_max = 0.9  # highest opacity for dense voxels
    # facecolors[voxels, 3] = alpha_min + (alpha_max - alpha_min) * norm_alpha(H[voxels])

    # build grid of voxel edges (shape (nx+1, ny+1, nz+1))
    X, Y, Z = np.meshgrid(edges[0], edges[1], edges[2], indexing='ij')

    # print(X, Y, Z)

    # loop here to group which events are in which voxel?
    new_folder_name = f'{year} {name}'
    print(new_folder_name)

    active_folder_name = f'{home}/clean shower data/{new_folder_name} clean events/{new_folder_name} active'
    os.makedirs(active_folder_name, exist_ok=True)

    active_folder = os.listdir(active_folder_name)

    # for filename in active_folder:
    
    #     file_path = os.path.join(active_folder_name, filename)

    #     voxel_meteors = {} # each voxel generated is a key to the meteors found within its borders

    #     kept_lines = []

    #     with open(file_path, 'r') as f:

    #         for line in f:

    #             line = line.strip()
    #             params = line.split()

    #             # keep non-data/header lines
    #             if params == [] or params[0][0] != '2':
    #                 kept_lines.append(line)
    #                 continue

    #             date = params[0]
    #             time = params[1]

    #             active_lmda = float(params[3])
    #             active_beta = float(params[4])
    #             active_vel = float(params[11])

    #             # find which voxel bin this meteor falls into (1-indexed)
    #             i = np.digitize(active_lmda, edges[0]) - 1
    #             j = np.digitize(active_beta, edges[1]) - 1
    #             k = np.digitize(active_vel,  edges[2]) - 1

    #             # print('i', i, 'j', j, 'k', k)

    #             # clamp to valid range [0, bins-1] — digitize can return out-of-range for edge values
    #             i = np.clip(i, 0, bins[0] - 1)
    #             j = np.clip(j, 0, bins[1] - 1)
    #             k = np.clip(k, 0, bins[2] - 1)
    #             # print('i', i, 'j', j, 'k', k)

    #             voxel_key = (i, j, k)

    #             # only keep meteors that fall in an occupied voxel
    #             if voxels[i, j, k]:
    #                 if voxel_key not in voxel_meteors:
    #                     voxel_meteors[voxel_key] = []
    #                 voxel_meteors[voxel_key].append({
    #                     'lmda': active_lmda,
    #                     'beta': active_beta,
    #                     'vel':  active_vel,
    #                     'Meteor': date + time
    #                 })

    # # inspect results
    # for voxel_key, meteors in voxel_meteors.items():
    #     i, j, k = voxel_key
    #     print(f'Voxel {voxel_key} | lmda:[{edges[0][i]:.1f}-{edges[0][i+1]:.1f}] '
    #         f'beta:[{edges[1][j]:.1f}-{edges[1][j+1]:.1f}] '
    #         f'vel:[{edges[2][k]:.1f}-{edges[2][k+1]:.1f}] '
    #         f'| {len(meteors)} meteors')

    # draw voxels in real-world coordinate space
    ax.voxels(X, Y, Z, voxels, facecolors=facecolors, edgecolor='k')

    # add colorbar tied to the same cmap/norm
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    figure.colorbar(mappable, ax=ax, shrink=0.6, label='Number of Meteors') # might have to confirm this

    # Should perform this for a single file of orbit data (echo data for a single solar longitude - a day)

    # Define 3d plot of ecliptic latitude, longitude and geocentric velocities


    # ax.scatter(lmda, beta, vels, marker='o')
    ax.set_xlabel('Ecliptic Longitude')
    ax.set_ylabel('Ecliptic Latitude', ha='right', va='center')
    ax.set_zlabel('Geocentric Velocity', ha='left', va='bottom')

    ax.set_title(f'Clean Meteor Sources scaled by geocentric velocity - measurements from {year}', fontsize=14)

    # limits to where the DSX meteors are (reproducing Kipreos et al (2022)), no data seen here for me though; might have an issue with defining my coordinates in my scale or relabel function
    ax.set_xlim(min(lmda)-20,max(lmda)+20)
    ax.set_ylim(min(beta)-10, max(beta)+10)
    ax.set_zlim(0,80)
    # maybe manually define limits of ecliptic coordinates for each shower to only include echoes in this region

    # ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
    ax.invert_xaxis()

    # ax.yaxis.get_label().set_alignment('right')
    # ax.zaxis.get_label().set_alignment('left')

    plt.grid()
    plt.tight_layout()
    plt.show()

    # thinking of using this in two instances: once before the convex hull to show the area of interest, and once after to show the isolated meteor shower

    if map_mode == 'shower':

        return H, edges, lmda, beta, vels # edges contains three arrays of coordinates by bin (lon, lat, vel)

def voxel_map2(counts, edges, active_lmda, active_beta, active_vels, threshold=0):
    
    '''
    This voxel map function takes the counts per voxel and the voxel's positions (edges) 
    following background number density subtraction and a 3 sigma test of shower counts (from the voxel subtract function)
    '''

    figure = plt.figure(figsize=(10,5))
    ax = figure.add_subplot(projection='3d')

    new_voxels = counts > threshold # increase the threshold to make the plot a bit more strict to the shower

    cmap = plt.colormaps['plasma']

    norm = plt.Normalize(vmin=counts[new_voxels].min() if new_voxels.any() else 0,
    vmax=counts[new_voxels].max() if new_voxels.any() else 1)
    facecolors = np.zeros(new_voxels.shape + (4,), dtype=float)
    facecolors[new_voxels] = cmap(norm(counts[new_voxels])) # maps counts to the voxels


    # build grid of voxel edges (shape (nx+1, ny+1, nz+1))
    X, Y, Z = np.meshgrid(edges[0], edges[1], edges[2], indexing='ij')


    # draw voxels in real-world coordinate space
    ax.voxels(X, Y, Z, new_voxels, facecolors=facecolors, edgecolor='k', alpha=0.9)

    # add colorbar tied to the same cmap/norm
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    figure.colorbar(mappable, ax=ax, shrink=0.6, label='Number of Meteors') # might have to confirm this

    # Should perform this for a single file of orbit data (echo data for a single solar longitude - a day)

    # Define 3d plot of ecliptic latitude, longitude and geocentric velocities


    # ax.scatter(lmda, beta, vels, marker='o')
    ax.set_xlabel('Ecliptic Longitude')
    ax.set_ylabel('Ecliptic Latitude')
    ax.set_zlabel('Geocentric Velocity')

    ax.set_title(f'Clean Meteor Sources scaled by geocentric velocity - measurements from {year}', fontsize=14)

    # limits to where the DSX meteors are (reproducing Kipreos et al (2022)), no data seen here for me though; might have an issue with defining my coordinates in my scale or relabel function
    ax.set_xlim(min(active_lmda)-20, max(active_lmda)+20)
    ax.set_ylim(min(active_beta)-10, max(active_beta)+10)
    ax.set_zlim(0,80)
    # maybe manually define limits of ecliptic coordinates for each shower to only include echoes in this region

    # ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
    ax.invert_xaxis()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return counts, edges

def voxel_with_hull(counts, edges, active_lmda, active_beta, active_vels, threshold=0):

    figure = plt.figure(figsize=(10, 5))
    ax = figure.add_subplot(projection='3d')

    new_voxels = counts > threshold
    cmap = plt.colormaps['plasma']
    norm = plt.Normalize(vmin=counts[new_voxels].min() if new_voxels.any() else 0,
                        vmax=counts[new_voxels].max() if new_voxels.any() else 1)
    facecolors = np.zeros(new_voxels.shape + (4,), dtype=float)
    facecolors[new_voxels] = cmap(norm(counts[new_voxels]))

    X, Y, Z = np.meshgrid(edges[0], edges[1], edges[2], indexing='ij')
    ax.voxels(X, Y, Z, new_voxels, facecolors=facecolors, edgecolor='k', alpha=0.9)

    # CONVEX HULL #
    # --- voxel centres of occupied voxels ---
    ix, iy, iz = np.where(new_voxels)
    cx = (edges[0][ix] + edges[0][ix + 1]) / 2
    cy = (edges[1][iy] + edges[1][iy + 1]) / 2
    cz = (edges[2][iz] + edges[2][iz + 1]) / 2
    points = np.column_stack((cx, cy, cz))

    fig = go.Figure()

    # --- voxel centres as scatter points coloured by count ---
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode='markers',
        marker=dict(
            size=6,
            color=counts[new_voxels],
            colorscale='Plasma',
            colorbar=dict(title='Number of Meteors'),
            opacity=0.9
        ),
        name='Voxels'
    ))

    # --- convex hull ---
    if len(points) >= 4:
        hull = ConvexHull(points)
        # hull.simplices gives triangular faces as index triples
        i, j, k = hull.simplices[:, 0], hull.simplices[:, 1], hull.simplices[:, 2]
        fig.add_trace(go.Mesh3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            i=i, j=j, k=k,
            opacity=0.15,
            color='cyan',
            name='Convex Hull'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Ecliptic Longitude',
            yaxis_title='Ecliptic Latitude',
            zaxis_title='Geocentric Velocity',
            zaxis=dict(range=[0, 80]),
        ),
        title=f'Clean Meteor Sources - {year}',
    )

    fig.show()


def voxel_map_parse(year, name, edges):
    '''
    The hope with this function is to take each file in the active showers, parse its lat/lon/vel as needed and do a voxel plot this way while storing file info with whatever voxel it can be found in. Then do the background subtraction and 3 sigma tests after for each meteor and remove the ones that fail.
    This would do all the things that the voxel functions already do above with the capability of tracking which specific meteors are being removed from the dataset.
    '''

    new_folder_name = f'{year} {name}'

    active_folder_name = f'{home}/clean shower data/{new_folder_name} clean events/{new_folder_name} active'
    os.makedirs(active_folder_name, exist_ok=True)

    active_folder = os.listdir(active_folder_name)

    # contains nested dictionaries of coordinates and date/time for each file from shower days
    voxel_meteors = {}

    # this loop is solely for grouping observation time to coordinates being plotted
    for filename in active_folder:

        file_path = os.path.join(active_folder_name, filename)

        with open(filename, 'r') as active_data:

            for line in active_data:

                line = line.strip()
                params = line.split()

                date = params[0]
                time = params[1]

                lmda = float(params[3])
                beta = float(params[4])
                velg = float(params[11])

                voxel_meteors[filename].append({
                        'lmda': lmda,
                        'beta': beta,
                        'vel':  velg,
                    })

    # voxel dimensions
    


def heat_map(lmda, beta, year, path, method, month=None, meteor_source=None, shower_name=None, background=False, bounds=None): # include month mode at some point for labelling
    '''
    This function generates a heat map of the user specified orbit file, based on meteor counts per bin
    Month and source modes may be worked individually or simultaneously - in terms of saving distinct files of data
    Shower mode is best worked on its own - mainly using this mode to collect the number density matrices for days before/after shower activity
    '''

    figure, ax = plt.subplots(figsize=(10,5))

    lmda = np.asarray(lmda, dtype=float)
    beta  = np.asarray(beta, dtype=float)

    if bounds != None:
        lon_bounds, lat_bounds = bounds[0], bounds[1]

        # this works; will have to check if I am computing the average matrix properly after then apply this to the 3d voxel step
        mask = (
            (lmda >= lon_bounds[0]) & (lmda <= lon_bounds[1]) &
            (beta >= lat_bounds[0]) & (beta <= lat_bounds[1])
        )
        
        lmda = lmda[mask]
        beta  = beta[mask]
        ax.set_xlim(lon_bounds[0], lon_bounds[1])
        ax.set_ylim(lat_bounds[0], lat_bounds[1])

    else:
        ax.set_ylim(-60, 90)
        ax.set_xlim(-150, 150)
        

    h = ax.hist2d(lmda, beta, bins=200, cmap='plasma') # should save files by bin size now for different runs

    binsize = len(h[0])
    # print(binsize)

    

    # print(h[0]) # counts per bin 

    # for i in range(len(h[0])):
    #     h[i] = h[i].strip('[]')
    #     count_list = h[i].split()
    #     print(count_list) # this is the list of counts per bin; need to convert to a 2d array for the colorbar

    # print(h[1]) # average longitude per bin
    # print(h[2]) # average latitude per bin
    # print(h[3]) 

    figure.colorbar(h[3], ax=ax, label='Number of meteors per bin')

    ax.set_xlabel('Ecliptic Longitude (Lambda)')
    ax.set_ylabel('Ecliptic Latitude (Beta)')

    # aiming to have something like this, but I want to mask data points outside these bounds instead and set the limits to the regular bounds
    
    # ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees

    ax.set_facecolor("#0D0F81")

    ax.invert_xaxis()
    # plt.grid()
    # plt.legend()
    counts_path = f'{home}/clean file data/0602/{method} events' # should change this save directory at some point
    os.makedirs(counts_path, exist_ok=True)

    if month == None and meteor_source == None and shower_name == None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')
        plt.savefig(f'{path}/{year}_{method}Filter_radiantColorDist{binsize}.png')

        counts_file = os.path.join(counts_path, f"{method}-counts-{year}-{binsize}-29.txt")
    
    elif month == None and meteor_source != None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from the {meteor_source} in {year}')
        plt.savefig(f'{path}/{year}_{method}Filter_{meteor_source}_radiantColorDist{binsize}.png')

        counts_file = os.path.join(counts_path, f"{method}-counts-{meteor_source}-{year}-{binsize}-29.txt")

    elif month != None and meteor_source == None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {month}{year}')
        plt.savefig(f'{path}/{year}{month}_{method}Filter_radiantColorDist{binsize}.png')

        counts_file = os.path.join(counts_path, f"{method}-counts-{month}-{year}-{binsize}-29.txt")
    
    elif month != None and meteor_source != None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from the {meteor_source} in {month}/{year}')
        plt.savefig(f'{path}/{year}{month}_{method}Filter_{meteor_source}_radiantColorDist{binsize}.png')

        counts_file = os.path.join(counts_path, f"{method}-counts-{meteor_source}-{year}{month}-{binsize}-29.txt")

    elif shower_name != None:
        if background == False:
            ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from the {shower_name} in {year}')
            plt.savefig(f'{path}/{year}_{method}Filter_{shower_name}_radiantColorDist{binsize}.png')

            counts_file = os.path.join(counts_path, f"{method}-counts-{shower_name}-{year}-{binsize}-29.txt")
        else:
            ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - background sporadic measurements around {shower_name} in {year}')
            plt.savefig(f'{path}/{year}_{method}Filter_{shower_name}_backgroundColorDist{binsize}.png')

            counts_file = os.path.join(counts_path, f"{method}-counts-{shower_name}-background-{year}-{binsize}-29.txt")

    with open(counts_file, 'w') as meteor_counts:
        meteor_counts.write('Ecl Lon, Ecl Lat, Counts per bin\n\n')
        for i in range(len(h[0])):
        
        # writing to file: Velocity bin value, counts in that bin
            meteor_counts.write(f'{h[1][i]} {h[2][i]} [{",".join([str(val) for val in h[0][i]])}]\n')
            # meteor_counts.write(f' {h[0][i]}')

            # for j in range(len(h[0][i])):
            #     meteor_counts.write(f' {h[0][i][j]}') # writing the count value for each bin in the row
            # meteor_counts.write('\n')
             

    plt.show()

    # calculating bin width here
    d_lmda = np.diff(h[1])[0] # longitude width per bin
    d_beta = np.diff(h[2])[0] # latitude width per bin

    # print(d_lmda, d_beta)

    bin_area = d_lmda * d_beta

    # print(bin_area, len(lmda))

    num_density = h[0] / (len(lmda) * bin_area)

    # h[0] is the matrix of counts per bin
    # dividing this by the number of longitude coordinates * area per bin

    # print(type(num_density)) # write to file to see if this is a number density matrix that would work?

    return h # should contain the counts per bin, the average ecliptic coordinates constructing the bins, and the color bar
# instead of creating a new density matrix with num_density, will try to subtract h[0]'s first -  as these are the counts and might be considered as number densities in a sense
# If i return num_density to echo plot, will need to either work with it within that function, or return it again using echo plot



def vel_map(lmda, beta, vels, year, path, method, month=None, meteor_source=None):
    '''
    This function generates a heat map of the user specified orbit file, based on average geocentric velocity per bin
    '''
    # velocity method 1: using hexbins
    figure, ax = plt.subplots(figsize=(10,5))

    hb = ax.hexbin(lmda, beta, C=vels, reduce_C_function=np.mean, 
        gridsize=50, cmap='plasma')
    
    # ax.set_xlabel('Ecliptic Longitude (Lambda)')
    # ax.set_ylabel('Ecliptic Latitude (Beta)')
    figure.colorbar(hb, ax=ax, label='Geocentric Velocity (km/s)')

    # ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')

    ax.set_xlabel('Ecliptic Longitude (Lambda)')
    ax.set_ylabel('Ecliptic Latitude (Beta)')

    # background figure's color - other than white
    ax.set_facecolor("#0D0F81")

    ax.invert_xaxis()
    # plt.grid()
    # plt.legend()

    ax.set_ylim(-60, 90)
    ax.set_xlim(-150, 150)

    if month == None and meteor_source == None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')
        plt.savefig(f'{path}/{year}_{method}Filter_velColorDist.png')
    
    elif month != None and meteor_source == None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year} ({month})')
        plt.savefig(f'{path}/{year}{month}_{method}Filter_velColorDist.png')

    elif month == None and meteor_source != None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from the {meteor_source} in {year}')
        plt.savefig(f'{path}/{year}_{method}Filter_{meteor_source}_velColorDist.png')

    elif month != None and meteor_source != None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from the {meteor_source} in {month}/{year}')
        plt.savefig(f'{path}/{year}{month}_{method}Filter_{meteor_source}_velColorDist.png')

    plt.show()



def scatter_map(lmda, beta, year, path, method, month=None, meteor_source=None):
    '''
    This function generates distribution of observed meteors as a scatter plot of ecliptic coordinates
    '''

    figure, ax = plt.subplots(figsize=(10,5))

    ax.scatter(lmda, beta, s=10, alpha=0.5, color='blue')

    ax.set_xlabel('Ecliptic Longitude (Lambda)')
    ax.set_ylabel('Ecliptic Latitude (Beta)')

    ax.set_ylim(-100, 100)
    ax.set_xlim(-180, 180)
    ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
    ax.invert_xaxis()
    plt.grid()
    # plt.legend()

    if month == None and meteor_source == None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')
        plt.savefig(f'{path}/{year}_{method}Filter_radiantDist.png')
    elif month != None and meteor_source == None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {month}/{year}')
        plt.savefig(f'{path}/{year}{month}_{method}Filter_radiantDist.png')
    elif month == None and meteor_source != None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from the {meteor_source} in {year}')
        plt.savefig(f'{path}/{year}_{method}Filter_{meteor_source}_radiantDist.png')
    elif month != None and meteor_source != None:
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from the {meteor_source} in {month}/{year}')
        plt.savefig(f'{path}/{year}{month}_{method}Filter_{meteor_source}_radiantDist.png')

    plt.show()



def duplicate(parent, times, dists, angles, dr_count, da_count):
    '''
    This function goes through all stored clean events, and checks if any two events have similar times (within 0.01 seconds from eachother)
    This will have to be called following the Parse function, but before the clean_echoes call, since we want the clean data to be written to our files
    '''

    # counts the events that are deleted for being a copy of another event
    dr = 0
    da = 0

    # creating a copy of the dictionary and updating this with non-duplicate events. Using the old parent dictionary to iterate through events
    parent_new = parent.copy()

    try:
    # runs if the number of clean events seen are even
   
        for i, key in enumerate(parent):
            # in range(0, len(times), 2): # each iteration goes through two events stored in the parent dictionary
            # print(key)
            time1 = times[i].strip()
            time2 = times[i+1].strip()

            dist1 = dists[i]
            dist2 = dists[i+1]

            theta1, phi1 = angles[i]
            theta2, phi2 = angles[i+1]

            hour1 = time1[0:2]
            hour2 = time2[0:2]

            minutes1 = time1[3:5]
            minutes2 = time2[3:5]

            seconds1 = float(time1[6:12])
            seconds2 = float(time2[6:12])

            # same hour
            if hour1 == hour2:
                # print('hour')
                # same minute
                if minutes1 == minutes2:
                    # print(seconds1, seconds2)
                    # we assume this is close enough for the two events to be a duplicate, which also depends on their position in the sky
                    if abs(seconds1 - seconds2) < 0.010:

                    # We only delete the event if the two happen at similar times AND they're close together in the sky
                        # angle condition
                        if abs(float(theta1) - float(theta2)) <=5 and abs(float(phi1) - float(phi2)) <=5:
                          
                            del parent_new[key]
                            # print('deleted (angle)')
                            da += 1
                            # print(da_count)
                        
                        # range condition
                        elif abs(float(dist1) - float(dist2)) <= 6: # if the difference between the meteor's distance from the Zehr is within 6 km of each other
                            del parent_new[key]
                            # print('deleted (range)')
                            dr += 1


        # print(f'deleted ranges: {dr}, deleted angles: {da}')
        return parent_new, dr, da
    
    # should only happen at the end of the list, in which case I'll have to check the last event to see if it is a duplicate
    except IndexError:
        return parent_new, dr, da

    # angular position test (simlilar theta/phi)

  
# Organizational function that saves clean echo data

def clean_echoes(parent, num_locs_init, date, folder, filename, num_dr, num_da, method='all'):

    '''
    This follows the parse function above, and writes the echo info for each clean event to an organized text file
    '''

    # defining a path to create a new folder containing saved info of clean echo data
    sub_folder = f'{home}/clean file data/{date[0:4]} clean events'
    os.makedirs(sub_folder, exist_ok=True)

    path = os.path.join(sub_folder, f"clean-{date}-29.txt") # 29 MHz is the frequency used by CMOR

    num_clean = len(parent) # number of clean echoes that satisfy the set velocity condition in Parse

    # print(f'File {filename} has {num_clean} clean echoes, {num_locs} of which have defined coordinates that can be plotted')
    # will add to this line later the more filtering I include, meaning I need to return more objects from my functions above

    with open(path, "w") as clean_data:


        if method == 'raw' or method == 'r':
            clean_data.write(f'This file {filename} has {num_locs_init} detected events. No filtering methods were applied to reject events, this is the raw data that was seen.\n')
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')


        if method == 'vel' or method == 'v':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12} {'Percent difference':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} ")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12} {value['Percent difference']:>10.2f}\n')
       

        elif method == 'int' or method == 'i':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")

            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12} {'Int Error':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} ")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12} {value['Interferometry Error']:>10}\n')


        elif method == 'angle' or method == 'a':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12} {'Radiant Error':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} ")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12} {value['Solid Angle Error']:>10}\n')


        elif method == 'station' or method == 's':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12} {'Station Error':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} ")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12} {value['Station Measurement Error']:>10}\n')


        elif method == 'vel and int' or method.upper() == 'VI':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12} {'Percent difference':>10} {'Int Error':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} ")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12} {value['Percent difference']:>10.2f} {value['Interferometry Error']:>10} \n')


        elif method == 'vel and int and angle' or method.upper() == 'VIA':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12} {'Percent difference':>10} {'Int Error':>10} {'Radiant Error':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} ")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}  {value['Percent difference']:>10.2f} {value['Interferometry Error']} {value['Solid Angle Error']:>10}\n')


        elif method == 'all':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")

            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12} {'Percent difference':>10} {'Int Error':>10} {'Radiant Error':>10} {'Station Error':>10}\n\n")

            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12} {value['Percent difference']:>10.2f} {value['Interferometry Error']:>10} {value['Solid Angle Error']:>10} {value['Station Measurement Error']:>10}\n')

    # could also write to a new txt file the total year data; total events seen no removed 

    return num_clean # adding on to counter to track number of clean echoes per year that I can work with (for plotting and for other purposes)


# Need ecliptic coordinates for plotting, which is what this funciton is for
def grab_coords(parent):

    '''
    This function will be used to grab the ecliptic coordinates of the clean echoes for plotting
    '''

    # storing defined coordinates here to be used for plotting
    latitudes = []
    longitudes = []
    geo_vels = []

    ptn0_vels = []
    del_ptn0_vels = []

    # lists for the function duplicate; checking for overlapping events
    times = []
    dists = []
    angles = []

    c2h_lmda = []
    c2h_beta = []

    loc_count = 0 # to keep track of files that contain defined coordinates
    keys_to_delete = [] # collect keys during iteration

    # v are nested dictionaries containing important info for each meteor
    for k, v in sorted(parent.items(), key=lambda kv: (kv[1]['date'], kv[1]['time'])):

        time = v['time']

        dist = v['R0']

        theta = v['Theta']
        phi = v['Phi']

        l_comp = v['Cel2Hel Longitude']
        b_comp = v['Cel2Hel Latitude']

        times.append(time)
        dists.append(dist)
        angles.append([theta, phi])


        beta = v['Ecliptic latitude']
        lmda = v['Ecliptic longitude'] # some longitude coordinates are negative, we want them from 0-360 for plotting

        vel_ptn0 = v['Pre-t0 velocity']
        del_vel_ptn0 = v['Uncertainty in Pre-t0 velocity']
        vel_g = v['Geocentric velocity']

        # checking for a defined ptn0 calculation
        if vel_ptn0[0] == '.' or del_vel_ptn0[0] == '.':
            keys_to_delete.append(k)
            # print('deleted for having undefined coordinates')
            continue

        # checking for a defined set of ecliptic coordinates
        if beta == '0.00' or lmda == '0.00':
                keys_to_delete.append(k)
                # print('deleted for having undefined coordinates')
                continue # skip rows with missing speed data; does not seem to change anything, as any file without these coordinates already lacks pre-to velocity
        
        # counting the meteors we can make distributions with
        loc_count += 1
        # lmda = long_transform(float(lmda)) # transforming longitude to 0-360 scale for plotting
        # print(lmda)

        # Using these three for 3d plots
        latitudes.append(float(beta))
        longitudes.append(float(lmda))
        geo_vels.append(float(vel_g))

        ptn0_vels.append(float(vel_ptn0))
        del_ptn0_vels.append(float(del_vel_ptn0))

        c2h_lmda.append(float(l_comp))
        c2h_beta.append(float(b_comp))

    # Delete collected keys after iteration completes
    for k in keys_to_delete:
        del parent[k]

    # print(times)
    return latitudes, longitudes, ptn0_vels, del_ptn0_vels, geo_vels, keys_to_delete, loc_count, times, dists, angles, c2h_lmda, c2h_beta # also returning the number of files with defined coordinates for more precise tracking purposes


def echo_plot(lmda, beta, vels, year, method, month=None, shower=None, source=None, mode='year', map_mode='scatter', bounds=None):
    '''
    This function takes the ecliptic coordinates of clean echoes that satisfy set restrictions and maps them to a 2 dimensional grid representing a celestial 'sphere'
        A goal is to create elliptical figures, but currently rectangular until I figure out how to do that
    There are two modes:
        year - plots the yearly echo data and saves the figure to a directory outside of the txt files
        month - plots the echo data for each month and saves those figures in appropriate folders with the corresponding monthly txt files
    '''

    if bounds != None:
        lon_bounds, lat_bounds = bounds[0], bounds[1] # each is a list of mmin, max values
    else:
        lon_bounds, lat_bounds = None, None
    if mode == 'year':
    
        plot_path = f'{home}/clean file data/{year} clean figures' # new directory made
        os.makedirs(plot_path, exist_ok=True)

        if map_mode == 'density':

            h = heat_map(lmda, beta, year, plot_path, method)
        
        elif map_mode == 'velocity':

            vel_map(lmda, beta, vels, year, plot_path, method)

        else:

            scatter_map(lmda, beta, year, plot_path, method)

    
    # monthly mode has a save option to the the clean file direcotry
    elif mode == 'month':

        plot_folder = f'{home}/clean file data/{year} clean events by month/figures'
        os.makedirs(plot_folder, exist_ok=True)

    
        if map_mode == 'density':

            h = heat_map(lmda, beta, year, plot_folder, method)

        elif map_mode == 'velocity':

            vel_map(lmda, beta, vels, year, plot_folder, method)
        
        else:
            scatter_map(lmda, beta, year, plot_folder, method, month=month)

        # ax.set_xlabel('Ecliptic Longitude (Lambda)')
        # ax.set_ylabel('Ecliptic Latitude (Beta)')
        # ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {month}/{year}')
        # ax.set_ylim(-100, 100)
        # ax.set_xlim(180, -180)

        # ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        # ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        # ax.invert_xaxis()
        # plt.grid()

        # plt.savefig(f'{plot_folder}/{year}{month}_radiantDist.png') # save the plot to the same folder as the data for that month
        # plt.show()

    elif mode == 'shower':

        plot_folder = f'{home}/clean shower data/{year} {shower} clean events/figures'
        os.makedirs(plot_folder, exist_ok=True)

        if map_mode == 'density':
            
            h = heat_map(lmda, beta, year, plot_folder, method, shower_name=shower) # , bounds=[lon_bounds, lat_bounds]

            return h

        elif map_mode == 'velocity':

            vel_map(lmda, beta, vels, year, plot_folder, method)
        
        else:

            scatter_map(lmda, beta, year, plot_folder, method)

        # ax.set_xlabel('Ecliptic Longitude (Lambda)')
        # ax.set_ylabel('Ecliptic Latitude (Beta)')
        # ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')
        # ax.set_ylim(-100, 100)
        # ax.set_xlim(180, -180)

        # ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        # ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        # ax.invert_xaxis()
        # plt.grid()

        # plt.savefig(f'{plot_folder}/{year}{shower}_radiantDist.png') # save the plot to the same folder as the data for that month
        # plt.show()

    elif mode == 'background':

        plot_folder = f'{home}/clean shower data/{year} {shower} clean events/figures'
        os.makedirs(plot_folder, exist_ok=True)

        if map_mode == 'density':
            
            h = heat_map(lmda, beta, year, plot_folder, method, shower_name=shower)

            return h

        elif map_mode == 'velocity':

            vel_map(lmda, beta, vels, year, plot_folder, method)
        
        else:

            scatter_map(lmda, beta, year, plot_folder, method)


    elif mode == 'source':
        
        # use the below code to overplot clean data overtop of raw data, which is specific to an existing raw file
        # overplotting new clean data over raw data
        # raw_folder = home / f'clean file data/source plotting/{source}/raw test/{year} clean events'

        # os.makedirs(raw_folder, exist_ok=True)

        # raw_lons = []
        # raw_lats = []
        # raw_vels = []

        # raw_file = raw_folder / 'clean-2022-350-29.txt'

        # with open(raw_file, 'r') as raw_data:

        #     for r in range(2):
        #             next(raw_data, None)

        #     for line in raw_data:

        #         line = line.strip()
        #         params = line.split()

        #         ecl_lon = float(params[3])
        #         ecl_lat = float(params[4])
        #         vel_geo = float(params[11])

        #         scaled_lon = scale(ecl_lon)

        #         raw_lons.append(scaled_lon)
        #         raw_lats.append(ecl_lat)
        #         raw_vels.append(vel_geo)

        plot_path = f'{home}/clean source data/{year} {source} clean figures' # new directory made
        os.makedirs(plot_path, exist_ok=True)


        if map_mode == 'density':
            
            # heat_map(raw_lons, raw_lats, year, plot_path)

            h = heat_map(lmda, beta, year, plot_path, method, meteor_source=source)
        
        elif map_mode == 'velocity':

            # vel_map(raw_lons, raw_lats, vels, year, plot_path)

            vel_map(lmda, beta, vels, year, plot_path, method)
        
        # ax_raw.set_xlabel('Ecliptic Longitude (Lambda)')
        # ax_raw.set_ylabel('Ecliptic Latitude (Beta)')
        # ax_raw.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')

        # ax_raw.set_ylim(-100, 100)
        # ax_raw.set_xlim(-180, 180)
        # # ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        # # ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        # ax_raw.invert_xaxis()
        # plt.grid()
        # plt.legend()

        # plt.savefig(f'{plot_path}/{year}_radiantDist.png')
        # plt.show()


        # ax_clean.set_xlabel('Ecliptic Longitude (Lambda)')
        # ax_clean.set_ylabel('Ecliptic Latitude (Beta)')
        # ax_clean.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')

        # ax_clean.set_ylim(-100, 100)
        # ax_clean.set_xlim(-180, 180)
        # # ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        # # ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        # ax_clean.invert_xaxis()
        # plt.grid()
        # plt.legend()

        # plt.savefig(f'{plot_path}/{year}_radiantDist.png')
        # plt.show()


        else:
            
            # scatter_map(raw_lons, raw_lats, year, plot_path)

            scatter_map(lmda, beta, year, plot_path, method)

    # add a successive plot here; showing distribution after each successive filter applied


def echo_3d_plot(lmda, beta, vels, year, month=None, shower=None, source=None, mode='month', bounds=None):
    '''
    This function takes the ecliptic coordinates and the geocentric velocities of clean echoes that satisfy set restrictions and maps them to a 3 dimensional grid representing a density map
        The density is based on geocentric speeds of meteors; will also be used when creating convex hulls to remove significant meteor showers
    There are two modes:
        year - plots the yearly echo data and saves the figure to a directory outside of the txt files; 
        this is a lot of data to include in one plot, so it is recommened to only use the month plotting mode for this function

        month - plots the echo data for each month and saves those figures in appropriate folders with the corresponding monthly txt files
    '''

    if mode == 'year':
        figure = plt.figure(figsize=(10,5))
        ax = figure.add_subplot(projection='3d')

        ax.scatter(lmda, beta, vels, marker='o')
        ax.set_xlabel('Ecliptic Longitude')
        ax.set_ylabel('Ecliptic Latitude')
        ax.set_zlabel('Geocentric Velocity')

        ax.set_title(f'Clean Meteor Sources scaled by geocentric velocity - measurements from {year}', fontsize=14)

        # limits to where the gemenids are (reproducing Kipreos et al (2022)), no data seen here for me though; might have an issue with defining my coordinates in my scale or relabel function
        # ax.set_xlim(320, 340)
        # ax.set_ylim(-30, 0)

        ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        ax.invert_xaxis()
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    elif mode == 'month':

        plot_folder_3D = f'{home}/clean file data/{year} clean events by month/figures 3D'
        os.makedirs(plot_folder_3D, exist_ok=True)

        figure = plt.figure(figsize=(10,5))
        ax = figure.add_subplot(projection='3d')

        ax.scatter(lmda, beta, vels, marker='o')
        ax.set_xlabel('Ecliptic Longitude')
        ax.set_ylabel('Ecliptic Latitude')
        ax.set_zlabel('Geocentric Velocity')

        ax.set_title(f'Clean Meteor Sources scaled by geocentric velocity - measurements from {month}/{year}', fontsize=14)

        # limits to where the gemenids are (reproducing Kipreos et al (2022)), no data seen here for me though; might have an issue with defining my coordinates in my scale or relabel function
        # ax.set_xlim(320, 340)
        # ax.set_ylim(-30, 0)

        ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        ax.invert_xaxis()
        plt.grid()

        plt.savefig(f'{plot_folder_3D}/{year}{month}_radiantDist_3D.png')
        plt.tight_layout()
        plt.show()

    elif mode == 'shower':

        plot_folder_3D = f'{home}/clean shower data/{year} {shower} clean events/figures 3D'
        os.makedirs(plot_folder_3D, exist_ok=True)

        lmda = np.asarray(lmda, dtype=float)
        beta  = np.asarray(beta, dtype=float)
        vels = np.asarray(vels, dtype=float)
        
        # restricts the boundary of sporadics to the space created by shower_parser
        if bounds != None:
            lon_bounds, lat_bounds, vel_bounds = bounds[0], bounds[1], bounds[2]
            

            # this works; will have to check if I am computing the average matrix properly after then apply this to the 3d voxel step
            mask = (
                (lmda >= lon_bounds[0]) & (lmda <= lon_bounds[1]) &
                (beta >= lat_bounds[0]) & (beta <= lat_bounds[1]) &
                (vels >= vel_bounds[0]) & (vels <= vel_bounds[1])
            )
            
            lmda = lmda[mask]
            beta  = beta[mask]
            vels = vels[mask]

        figure = plt.figure(figsize=(10,5))
        ax = figure.add_subplot(projection='3d')

        ax.scatter(lmda, beta, vels, marker='o')
        ax.set_xlabel('Ecliptic Longitude')
        ax.set_ylabel('Ecliptic Latitude')
        ax.set_zlabel('Geocentric Velocity')

        ax.set_title(f'Clean Meteor Sources scaled by geocentric velocity - measurements from {year}', fontsize=14)

        # limits to where the gemenids are (reproducing Kipreos et al (2022)), no data seen here for me though; might have an issue with defining my coordinates in my scale or relabel function
        # ax.set_xlim(320, 340)
        # ax.set_ylim(-30, 0)

        ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        ax.invert_xaxis()
        plt.grid()

        plt.savefig(f'{plot_folder_3D}/{year}{shower}_radiantDist_3D.png')
        plt.tight_layout()
        plt.show()

        return lmda, beta, vels


def vel_histo(vels, year, method, month=None, shower=None, source=None, mode='year'):
    '''
    This function takes the geocentric velocities and plots histograms to show the distribution of observed meteor speeds
    There are two modes:
        year - plots the yearly echo data and saves the figure to a directory outside of the txt files
        month - plots the echo data for each month and saves those figures in appropriate folders with the corresponding monthly txt files
    Something to potentially include is calulating the area under the curve to estimate how many meteors fall within FWHM
    '''

    mean = round(np.mean(vels), 2)
    median = round(np.median(vels), 2)
    std = round(np.std(vels), 2)
    rms = round(np.sqrt(np.mean(np.square(vels))), 2)

    print('Mean Velocity:', mean)
    print('Median Velocity:', median)
    print('Root Mean Square Velocity', rms)
    print('Standard Deviation:', std)
    
    if mode == 'year':

        plot_path = f'{home}/clean file data/{year} clean figures' # new directory made
        os.makedirs(plot_path, exist_ok=True)

        figure = plt.figure(figsize=(10,5))

        n, bins, patches = plt.hist(vels, bins=200)
        # print(n, bins, patches) # counts, mean vel per bin, object type? only worry about first two
        bin_index = np.digitize(mean, bins) - 1
        bin_index = np.clip(bin_index, 0, len(n) - 1)

        print('Distribution Peak:', n[bin_index])
        print('Distribution Width:', 2*std)

        # will ask which width to go with, but here are a few options

        plt.axvline(mean, color='red', label='Mean Velocity') 
        plt.axvline(mean - std, color='red', linestyle='--')
        plt.axvline(mean + std, color='red', linestyle='--')

        plt.axvline(median, color='orange', label='Median Velocity')
        plt.axvline(median - std, color='orange', linestyle='-')
        plt.axvline(median + std, color='orange', linestyle='-')

        plt.axvline(rms, color='green', label='RMS Velocity')
        plt.axvline(rms - std, color='green', linestyle='-.')
        plt.axvline(rms + std, color='green', linestyle='-.')

        plt.axhline(n[bin_index]/2, color='black', label='Full Width Half Maximum', linestyle='--')

        plt.xlabel('Geocentric Velocities (km/s)')
        plt.ylabel('Number of Events')
        plt.title(f'Geocentric Velocities of clean meteor echoes - measurements from {year}', fontsize=14)

        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f'{plot_path}/{year}_velocities.png')
        plt.show()

        num_bins = len(n)

        # Writing the histogram data to a txt file
        data_path = f'{home}/clean file data/{year} clean events'
        data_file = os.path.join(data_path, f"FULL-{year}-29.txt")

        with open(data_file, 'a') as vel_data:
        
            vel_data.write(f'\nMean Velocity: {mean} km/s\n') # should include uncertanties at some point too
            vel_data.write(f'Median Velocity: {median} km/s\n')
            vel_data.write(f'Root Mean Square Velocity: {rms} km/s\n')
            vel_data.write(f'Standard Deviation: +/-{std} km/s\n')
            vel_data.write(f'Distribution Peak (Mean Index): {n[bin_index]}\n')
            vel_data.write(f'Distribution Width (mean +/- std): {2*std} km/s\n')
        
        

        # use this to keep track of how many events per bin are making it through the filtering
        case_vels_path = f'{home}/clean file data/0602_2/{method} events'
        os.makedirs(case_vels_path, exist_ok=True)

        vels_file = os.path.join(case_vels_path, f"{method}-velocities-{year}-{num_bins}-29.txt")

        

        with open(vels_file, 'w') as vel_counts:
            vel_counts.write('Velocities, Counts\n\n')
            for i in range(len(n)):
            
            # writing to file: Velocity bin value, counts in that bin
                vel_counts.write(f'{bins[i]} {n[i]}\n')

     
    elif mode == 'month':

        plot_folder_vel = f'{home}/clean file data/{year} clean events by month/velocity histograms'
        os.makedirs(plot_folder_vel, exist_ok=True)

        figure = plt.figure(figsize=(10,5))

        n, bins, patches = plt.hist(vels, bins=50)
        bin_index = np.digitize(mean, bins) - 1
        bin_index = np.clip(bin_index, 0, len(n) - 1)

        print('Distribution Peak:', n[bin_index])
        print('Distribution Width:', 2*std)

        plt.axvline(mean, color='red', label='Mean Velocity') 
        plt.axvline(mean - std, color='red', linestyle='--')
        plt.axvline(mean + std, color='red', linestyle='--')

        plt.axvline(median, color='orange', label='Median Velocity')
        plt.axvline(median - std, color='orange', linestyle='-')
        plt.axvline(median + std, color='orange', linestyle='-')

        plt.axvline(rms, color='green', label='RMS Velocity')
        plt.axvline(rms - std, color='green', linestyle='-.')
        plt.axvline(rms + std, color='green', linestyle='-.')

        plt.axhline(n[bin_index]/2, color='black', label='Full Width Half Maximum', linestyle='--')

        plt.xlabel('Geocentric Velocities (km/s)')
        plt.ylabel('Number of Events')
        plt.title(f'Geocentric Velocities of clean meteor echoes - measurements from {month}/{year}', fontsize=14)

        plt.grid(alpha=0.3)
        plt.savefig(f'{plot_folder_vel}/{year}{month}_velocities.png')
        plt.show()

        num_bins = len(n)

        data_path = f'{home}/clean file data/{year} clean events by month/{year} {month} clean echoes'
        data_file = os.path.join(data_path, f"FULL-{year}{month}-29.txt")

        with open(data_file, 'a') as vel_data: # should not exist yet for months, but working on it (change w to a when it does)
        
            vel_data.write(f'\nMean Velocity: {mean} km/s\n') # should include uncertanties at some point too
            vel_data.write(f'Median Velocity: {median} km/s\n')
            vel_data.write(f'Root Mean Square Velocity: {rms} km/s\n')
            vel_data.write(f'Standard Deviation: +/-{std:.2f} km/s\n')
            vel_data.write(f'Distribution Peak (Mean Index): {n[bin_index]}\n')
            vel_data.write(f'Distribution Width (mean +/- std): {2*std} km/s\n')


    elif mode == 'shower':

        plot_path = f'{home}/clean shower data/{year} {shower} clean figures' # new directory made
        os.makedirs(plot_path, exist_ok=True)

        figure = plt.figure(figsize=(10,5))

        n, bins, patches = plt.hist(vels, bins=50)
        bin_index = np.digitize(mean, bins) - 1
        bin_index = np.clip(bin_index, 0, len(n) - 1)

        print('Distribution Peak:', n[bin_index])
        print('Distribution Width:', 2*std)

        # will ask which width to go with, but here are a few options

        plt.axvline(mean, color='red', label='Mean Velocity') 
        plt.axvline(mean - std, color='red', linestyle='--')
        plt.axvline(mean + std, color='red', linestyle='--')

        plt.axvline(median, color='orange', label='Median Velocity')
        plt.axvline(median - std, color='orange', linestyle='-')
        plt.axvline(median + std, color='orange', linestyle='-')

        plt.axvline(rms, color='green', label='RMS Velocity')
        plt.axvline(rms - std, color='green', linestyle='-.')
        plt.axvline(rms + std, color='green', linestyle='-.')

        plt.axhline(n[bin_index]/2, color='black', label='Full Width Half Maximum', linestyle='--')

        plt.xlabel('Geocentric Velocities (km/s)')
        plt.ylabel('Number of Events')
        plt.title(f'Geocentric Velocities of clean meteor echoes - measurements from {year}', fontsize=14)

        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f'{plot_path}/{year}{shower}_velocities.png')
        plt.show()

        num_bins = len(n)

        # Writing the histogram data to a txt file
        data_path = f'{home}/clean shower data/{year} {shower} clean events'
        data_file = os.path.join(data_path, f"FULL-{year}{shower}-29.txt")

        os.makedirs(data_path, exist_ok=True)

        with open(data_file, 'w') as vel_data:
        
            vel_data.write(f'\nMean Velocity: {mean} km/s\n') # should include uncertanties at some point too
            vel_data.write(f'Median Velocity: {median} km/s\n')
            vel_data.write(f'Root Mean Square Velocity: {rms} km/s\n')
            vel_data.write(f'Standard Deviation: +/-{std} km/s\n')
            vel_data.write(f'Distribution Peak (Mean Index): {n[bin_index]}\n')
            vel_data.write(f'Distribution Width (mean +/- std): {2*std} km/s\n')

    elif mode == 'source':

        plot_path = f'{home}/clean source data/{year} {source} clean figures' # new directory made
        os.makedirs(plot_path, exist_ok=True)

        figure = plt.figure(figsize=(10,5))

        n, bins, patches = plt.hist(vels, bins=50)
        bin_index = np.digitize(mean, bins) - 1
        bin_index = np.clip(bin_index, 0, len(n) - 1)

        print('Distribution Peak:', n[bin_index])
        print('Distribution Width:', 2*std)

        # will ask which width to go with, but here are a few options

        plt.axvline(mean, color='red', label='Mean Velocity') 
        plt.axvline(mean - std, color='red', linestyle='--')
        plt.axvline(mean + std, color='red', linestyle='--')

        plt.axvline(median, color='orange', label='Median Velocity')
        plt.axvline(median - std, color='orange', linestyle='-')
        plt.axvline(median + std, color='orange', linestyle='-')

        plt.axvline(rms, color='green', label='RMS Velocity')
        plt.axvline(rms - std, color='green', linestyle='-.')
        plt.axvline(rms + std, color='green', linestyle='-.')

        plt.axhline(n[bin_index]/2, color='black', label='Full Width Half Maximum', linestyle='--')

        plt.xlabel('Geocentric Velocities (km/s)')
        plt.ylabel('Number of Events')
        plt.title(f'Geocentric Velocities of clean meteor echoes - measurements from {year}', fontsize=14)

        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f'{plot_path}/{year}{source}_velocities.png')
        plt.show()

        num_bins = len(n)

        # Writing the histogram data to a txt file
        data_path = f'{home}/clean source data/{year} {source} clean events'
        os.makedirs(data_path, exist_ok=True)

        data_file = os.path.join(data_path, f"FULL-{year}{source}-{num_bins}-29.txt")

        
        with open(data_file, 'w') as vel_data:
        
            vel_data.write(f'\nMean Velocity: {mean} km/s\n') # should include uncertanties at some point too
            vel_data.write(f'Median Velocity: {median} km/s\n')
            vel_data.write(f'Root Mean Square Velocity: {rms} km/s\n')
            vel_data.write(f'Standard Deviation: +/-{std} km/s\n')
            vel_data.write(f'Distribution Peak (Mean Index): {n[bin_index]}\n')
            vel_data.write(f'Distribution Width (mean +/- std): {2*std} km/s\n')


# Essentially another Parse function that organizes clean echo data by month for a given year; requires clean echo data to exist first
    # should integrate this with Parse in a new file
def monthly_echoes(year, folder_name, file):
    '''
    This will be used to organize the files for a given year by month in new folders within that year's clean events folder
    The folder will be the one put in by the user, and it's year will point to the location of its associated clean folder 
    This function call will be following the execution of clean_echoes so the clean data will be filed as needed
    '''
    
    # pointing to location of the folder the user entered

    path = os.path.join(home, folder_name, file)
    # contains all txt files; I want to organize them by month, which is part of their first entry

    # creating new folder to store monthly organized data
    sub_folder = f'{home}/clean file data/{year} clean events by month'
    os.makedirs(sub_folder, exist_ok=True)

    header=""


    with open(path, 'r') as clean_data:
        for line in clean_data:
            
            # print(line) # works

            line = line.strip()
            params = line.split()

            # print(params)

            # skipping empty lines
            if params == []:
                continue # skip lines that don't start with a date; works

            else:

                # skip lines that don't start with a date; works
                if params[0][0] != '2': 
                    header += line
                    header += '\n'
                    # print(header)
                    continue

                date = params[0]
                month = date[4:6] # extract month from date string


                # monthly folder nested in the yearly folder
                month_folder = f'{home}/clean file data/{year} clean events by month/{year} {month} clean echoes'

                os.makedirs(month_folder, exist_ok=True)

                month_path = os.path.join(month_folder, f'clean-{date}-29.txt')

                with open(month_path, 'w') as month_file: # using append mode to add to the file for each echo in that month
                    
                    month_file.write(f'{header}\n')
                    month_file.write(clean_data.read()) # write the clean echo data for that month to the new file; works but need to check that it's writing the correct data to the correct file; will do later


def monthly_plotter(year, month, folder, file, method, map_method='scatter'):
    '''
    This will be used to create plots for each month of data for a specified year by the user
    Follows the organization function and uses the copied data that has been sorted into monthly folders
    '''

    echo_count = 0

    # relabels the month to fit the stlye being used in previous functions
    if month < 10:
        month = f'0{month}' # add leading zero to month for file path 

    try:
        
        clean_month_name = f'{home}/clean file data/{year} clean events by month/{year} {month} clean echoes'
        # print(os.path.exists(clean_month_name)) # True if the directory exists

        month_files = sorted(os.listdir(clean_month_name))

        # lists for plotting
        longitudes = []
        latitudes = []
        velocities = []

        for month_file in month_files:
            month_file_path = os.path.join(clean_month_name, month_file)
            with open(month_file_path, 'r') as month_data:
                for line in month_data:
                    # print(line)

                    line = line.strip()
                    params = line.split()


                    if params == []:
                        continue # skip empty lines

                    else:

                        if params[0][0] != '2': 
                            continue # skip lines that don't start with a date; works
                        
                        echo_count += 1
                        
                        date = params[0]
                        year = date[0:4]

                        velg = params[11] # same index for all three cases

                        lmda = params[3] # ecliptic longitude
                        beta = params[4] # ecliptic latitude

                        # print(lmda, beta)

                        longitudes.append(float(lmda))
                        latitudes.append(float(beta))
                        velocities.append(float(velg))

        print(f'Number of Echoes seen in {month}/{year}: {echo_count}')

        data_path = f'{home}/clean file data/{year} clean events by month/{year} {month} clean echoes'
        data_file = os.path.join(data_path, f"FULL-{year}{month}-29.txt")

        message1 = f'\n\nMONTHLY DATA:\n\nThe total number of clean echoes observed during {month}/{year} is {echo_count}.\n'
        # message2 = f'For all events within 10 milliseconds of eachother, {delranges} events were deleted for being within 6km of each other, {delangles} events were deleted for having zenith/azimuth angles within 5 degrees.\n'
        # message3 = f'There were {num_without_coords} events deleted for not having a defined set of ecliptic coordinates.\n'

        # will work on latter two another time

        # 'w' creates the file for each month, and vel dist function appends to this created file for each monthly distribution
        with open(data_file, 'w') as vel_data:
            vel_data.write(message1)

        scaled_longitudes = scale(longitudes) # scaling longitudes to be centered at 270 degrees

        ## 2d plot
        echo_plot(scaled_longitudes, latitudes, velocities, year, method, month, mode='month', map_mode=map_method)

        # 3d plot
        echo_3d_plot(scaled_longitudes, latitudes, velocities, year, month)

        # velocities histogram
        vel_histo(velocities, year, method, month=month, mode='month')
        
    
    except FileNotFoundError:
        print(f"No data found for {month}/{year}. Please check that the month and year are correct and that the data has been organized by month using the monthly_echoes function.")


def shower_parser(year, folder, file, file_slon, slon_peak, radiants, radiant_drifts, name, method, slon_status='active', boundaries=None):
    '''
    I'd want to take each file for a specific shower and make a lat/long plot for each to spot where the shower is
    once spotted, this area will be localized so that I can make the convex hull and take the shower out from sporadic data
    '''

    new_folder_name = f'{year} {name}'
    # print(name, new_folder_name) # want this to be the directory name of the shower only meteors

    path = os.path.join(home, folder, file)
    # contains all txt files; I want to organize them by month, which is part of their first entry
    print(path)

    # creating new folder to store monthly organized data
    sub_folder = f'{home}/clean shower data/{new_folder_name} clean events'
    os.makedirs(sub_folder, exist_ok=True)


    header=""

    date_time = []
    
    shower_lons = []
    shower_lats = []
    shower_vels = []
    

    day_gap = file_slon - slon_peak # negative difference for days before, positive difference for days after
    print(file_slon, day_gap)

    corrected_coordinates = {}

    meteor_count = 0


    # parses and saves file data of shower data
    if slon_status == 'active':

        active_folder = f'{sub_folder}/{year} {name} active'
        os.makedirs(active_folder, exist_ok=True)

        active_path = os.path.join(active_folder, f'clean-{file_slon}-29.txt')


        with open(path, 'r') as shower_data, open(active_path, 'w') as active_file:
            for line in shower_data:

                line = line.strip()
                params = line.split()

                # print(params)

                # skipping empty lines
                if params == []:
                    continue # skip lines that don't start with a date; works

                else:

                    # skip lines that don't start with a date; works
                    if params[0][0] != '2': 
                        active_file.write(f'{line}\n') # this lists the number of events in the TOTAL file, not the shower file; could write something to do this in the future - would probably have to be after all shower parsing is complete for a given day
                        # header += line
                        # header += '\n'
                        # # print(header)
                        continue

                    date = params[0]
                    time = params[1]
                    month = date[4:6] # extract month from date string

                    velg = params[11]

                    lmda = params[3]
                    beta = params[4]
                    slon = params[5]
                    alpha = params[12]
                    delta = params[13]
                    del_rad = params[14]


                    lmda, beta = float(lmda), float(beta)
                    alpha, delta = float(alpha), float(delta)
                    velg = float(velg)

                    # unpacking the radiant of the shower everytime the loop is ran so the shower starts with the same coordinates each time
                    # solar_longitudes = slons[name] revisit if I want to access all solar longitudes in this function at some point
                    shower_alpha, shower_delta = radiants[name] # coordinates on day of the peak
                    alpha_drift, delta_drift = radiant_drifts[name]

                    # C2H conversion for the center of the shower on it's peak day
                    # is i want to impliment drift per day, having the c2h function calls here resets the shower location for each new day 
                    # this way the computed drift does not diverge
                    shower_cel = getvec(shower_alpha, shower_delta)
                    shower_hel = cel2hel(shower_cel, slon_peak)
                    shower_lmda, shower_beta = getangle(shower_hel)
                    # this step is done for each shower before the function call; make dict and pass to this function if not using drifts

                    diff_lmda = np.abs(shower_lmda - lmda)
                    diff_beta = np.abs(shower_beta - beta)
                    # C2H conversion for the current meteor
                    # or use its given lat/lon and compare to the location of the shower
                    # if diff_lmda <= 5 and diff_beta <= 5:
                    # if boundaries is not None:
                        
                    #     # two lists
                    #     min_lmda_bounds, max_lmda_bounds = boundaries[0][0], boundaries[0][1]
                    #     min_beta_bounds, max_beta_bounds = boundaries[1][0], boundaries[1][1]

                    #     shower_mask = ((lmda >= min_lmda_bounds) & (lmda <= max_lmda_bounds) &
                    #                    (beta >= min_beta_bounds) & (beta <= max_beta_bounds)
                    #                    )
                    #     lmda_filtered = lmda[shower_mask]
                    #     beta_filtered = beta[shower_mask]
                    #     vels_filtered = velg[shower_mask]

                    date_time.append(f'{date} {time}')
                

                    shower_lons.append(lmda)
                    shower_lats.append(beta)
                    shower_vels.append(velg)

                    active_file.write(f'{line}\n') # copies the file to a new directory of shower meteors

                    # only saves meteors close to the shower on active days
                    # corrected_coordinates[date + time] = {"lmda_h" : lmda, "beta_h" : beta}

                    meteor_count += 1

                    # meteor_hel = cel2hel(file_slon, alpha, delta)
                    # meteor_beta, meteor_lmda = getangle(meteor_hel)
                    

                    # alpha, delta correction for ra/dec drift per solar longitude
                    # active_slon = file_slon

                    # corrects ra/dec if the shower solar longitude is not at its peak
                    # if file_slon != slon_peak:

                    #     alpha += day_gap*alpha_drift
                    #     delta += day_gap*delta_drift

                    #     shower_alpha += day_gap*alpha_drift
                    #     shower_delta += day_gap*delta_drift

                    #     # if file_slon < slon_peak: # days before the peak of the shower
                    #     # elift file_slon > slon_peak: # days after the peak of the shower
                    
                    
                    

                    # # only plotting the echoes that fall within 10 degrees of the shower radiant
                    # # will check if this value is a good range by next meeting
                    # # if abs(shower_alpha - alpha) <= 5 and abs (shower_delta - delta) <= 5: # this gives cartesian boundary
                    

                    # # if abs(shower_lmda - lmda) <= 5 and abs(shower_beta - beta) <= 5:
                    # # circular distance from the source
                    # if np.sqrt((shower_alpha - alpha)**2 + (shower_delta - delta)**2) <= 5:
                    #     # print('Computed lon: ', shower_lmda, 'Computed lat: ', shower_beta)
                    #     date_time.append(f'{date} {time}')
                    #     shower_lons.append(lmda)
                    #     shower_lats.append(beta)
                    #     shower_vels.append(velg)

                    #     # shower_lons.append(meteor_lmda)
                    #     # shower_lats.append(meteor_beta)
                    #     # shower_vels.append(velg)

                    #     active_file.write(f'{line}\n') # copies the file to a new directory of shower meteors

                    #     # only saves meteors close to the shower on active days
                    #     corrected_coordinates[date + time] = {"RA" : alpha, "Dec" : delta}

                    #     meteor_count += 1

                    #     # counter here for header?
    
        return shower_lons, shower_lats, shower_vels, corrected_coordinates, meteor_count, date_time

    # parses and saves file data from solar longitudes 5 days before and after the shower days
    elif slon_status == 'outer':
 
        outer_folder = f'{sub_folder}/{year} {name} outer'
        os.makedirs(outer_folder, exist_ok=True)

        outer_path = os.path.join(outer_folder, f'clean-{file_slon}-29.txt')


        with open(path, 'r') as shower_data, open(outer_path, 'w') as outer_file:
            for line in shower_data:

                line = line.strip()
                params = line.split()

                # print(params)

                # skipping empty lines
                if params == []:
                    continue # skip lines that don't start with a date; works

                else:

                    # skip lines that don't start with a date; works
                    if params[0][0] != '2': 
                        outer_file.write(f'{line}\n')
                        # header += line
                        # header += '\n'
                        # # print(header)
                        continue

                    date = params[0]
                    time = params[1]
                    month = date[4:6] # extract month from date string

                    velg = params[11]

                    lmda = params[3]
                    beta = params[4]
                    slon = params[5]
                    alpha = params[12]
                    delta = params[13]
                    del_rad = params[14]

                    # unpacking the radiant of the shower
                    # solar_longitudes = slons[name]
                    shower_alpha, shower_delta = radiants[name]
                    alpha_drift, delta_drift = radiant_drifts[name]

                    lmda, beta = float(lmda), float(beta)
                    alpha, delta = float(alpha), float(delta)
                    velg = float(velg)

                    # unpacking the radiant of the shower everytime the loop is ran so the shower starts with the same coordinates each time
                    # solar_longitudes = slons[name] revisit if I want to access all solar longitudes in this function at some point
                    shower_alpha, shower_delta = radiants[name] # coordinates on day of the peak
                    alpha_drift, delta_drift = radiant_drifts[name]

                    # ra/dec correction step
                    # day_gap = file_slon - slon_peak # negative difference for days before, positive difference for days after

                    alpha += day_gap*alpha_drift
                    delta += day_gap*delta_drift # might not matter as much for these days, as we do not look for shower meteors necessarily and just want these ones for background subtraction
                    # since these take place a while before and after the peak of the shower, the drift would be much greater on these meteors than on the active day meteors

                    shower_alpha += day_gap*alpha_drift
                    shower_delta += day_gap*delta_drift

                    # radiant to ecliptic coordinate conversion
                    meteor_cel = getvec(alpha, delta)
                    shower_cel = getvec(shower_alpha, shower_delta)

                    meteor_hel = cel2hel(meteor_cel, file_slon)
                    shower_hel = cel2hel(shower_cel, slon_peak)

                    meteor_lmda, meteor_beta = getangle(meteor_hel)
                    shower_lmda, shower_beta = getangle(shower_hel)

                    # if file_slon < slon_peak: # days before the peak of the shower
                    # elift file_slon > slon_peak: # days after the peak of the shower

                    # only plotting the echoes that fall within 10 degrees of the shower radiant
                    # will check if this value is a good range by next meeting
                    # if abs(shower_alpha - alpha) <= 5 and abs (shower_delta - delta) <= 5: 
                    # if np.sqrt((shower_alpha - alpha)**2 + (shower_delta - delta)**2) <= 5: Without this in place, all meteors in the background days are tracked
                        
                    # if abs(shower_lmda - lmda) <= 5 and abs(shower_beta - beta) <= 5:
                    date_time.append(f'{date} {time}')
                    shower_lons.append(lmda)
                    shower_lats.append(beta)
                    shower_vels.append(velg)

                    outer_file.write(f'{line}\n') # copies the file to a new directory of shower meteors

                    corrected_coordinates[date + time] = {"RA" : alpha, "Dec" : delta}

                    meteor_count += 1

        return shower_lons, shower_lats, shower_vels, corrected_coordinates, meteor_count, date_time


def background_subtract(year, outer_lmda, outer_beta, outer_vel, name, method, slon_status='active'):

    '''
    This function goes over each file in the 'outer' folders which reads info on sporadic background meteors
    At the same time, the shower data is parsed and for each line in the shower files, the ecliptic coordinates and velocity is checked to be within 3*sigma of that of the background meteors
    where sigma = standard deviation of a chosen parameter. In this case, a sigma will be computed for lmda, beta and vel)
    '''

    # instead of subtracting the exact background point from the shower data, maybe take any bin out with counts > 0 within a given radius of any nonzero bin from the background
    # should write a new function for this
    # would have to go over each line in the outer folder, grab its coordinates and store to a list
    # then go through each file in the active folder, and check if its coordinates fall within a range of the stored coordinates
    # if so, remove this line from the file its in
    # plot up the remaining

    lmda_std = np.std(outer_lmda)
    beta_std = np.std(outer_beta)
    vel_std = np.std(outer_vel)


    new_folder_name = f'{year} {name}'

    # creating new folder to store monthly organized data

    active_folder_name = f'{home}/clean shower data/{new_folder_name} clean events/{new_folder_name} active'
    active_folder = os.listdir(active_folder_name)

    # store surviving data here
    shower_lmda = []
    shower_beta = []
    shower_vel = []

    import shutil # for backing up files that are changed based on shower subtraction methods

    for filename in active_folder:
        file_path = os.path.join(active_folder_name, filename)

        # # backup the original file before modifying
        # try:
        #     shutil.copy2(file_path, file_path + '.bak')
        #       check if this exists, if so do not make a new file
        # except Exception:
        #     pass

        kept_lines = []

        with open(file_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                params = stripped.split()

                # keep non-data/header lines
                if params == [] or params[0][0] != '2':
                    kept_lines.append(line)
                    continue

                # try to parse the relevant fields; on failure, keep the line
                try:
                    active_lmda = float(params[3])
                    active_beta = float(params[4])
                    active_vel = float(params[11])
                except (IndexError, ValueError):
                    kept_lines.append(line)
                    continue

                # if this active line matches any background coordinate, drop it
                remove_line = False
                for l, b, v in zip(outer_lmda, outer_beta, outer_vel):
                    if abs(active_lmda - l) >= 3*lmda_std and abs(active_beta - b) >= 3*beta_std: # also applied to number density in Kipreos et al (2022)
                        remove_line = True
                        break
                    # 3d plotting case; will work on soon
                    # if abs(active_lmda - l) <= 5 and abs(active_beta - b) <= 5 and abs(active_vel - v) <= 2:
                    #     pass

                if not remove_line:
                    kept_lines.append(line)
                    shower_lmda.append(active_lmda)
                    shower_beta.append(active_beta)
                    shower_vel.append(active_vel)

        # overwrite file with filtered contents
        with open(file_path, 'w') as f:
            f.writelines(kept_lines)

    return shower_lmda, shower_beta, shower_vel  

def voxel_organize(H, edges, shower_dict, year):
    '''
    Go over each voxel and log which specific meteors are in them
    Hope to use this in voxel subtract - any meteors in a voxel set to zero should be removed from the data set

    '''

    voxel_lmda = edges[0]
    voxel_beta = edges[1]
    voxel_vels = edges[2]

    voxel_dict = {} # store each meteor in here keyed by voxel boundaries
    # key = lon {n} lat {j} vel {i}
    # value = date of meteor

    # iterate over each meteor, check which voxel is in, do the voxel count test and if this fails, delete this meteor and clear the voxel
    # for d, nd in shower_dict.items():

    #     l, b, v = nd['lmda'], nd['beta'], nd['vel'] 

    #     for i, matrix in enumerate(H): 
            
    #         # stops from defining an index out of bounds of the velocity bins
    #         if i + 1 >= len(voxel_vels):
    #             continue 

    #         vel_coord, vel_coord1 = voxel_vels[i], voxel_vels[i+1]
    #         # print(vel_coord)

    #         # print(shower_counts_copy[i])

    #         for j in range(len(H[i])): # goes over each row of voxels in the i'th shower matrix
                
    #             if j+1 >= len(voxel_beta):
    #                 continue

    #             row = H[i][j]

    #             beta_coord, beta_coord1 = voxel_beta[j], voxel_beta[j+1]

    #             # print(beta_coord)

    #             for n in range(len(row)):
    #                 if n+1 >= len(voxel_lmda):
    #                     continue

    #                 lmda_coord, lmda_coord1 = voxel_lmda[n], voxel_lmda[n+1]
    #                 # print(lmda_coord)
    #                 counts = row[n]
    #                 # print(counts, 3*sigma) # this should be the specific voxel
    #                 # print(counts)

    #                 if lmda_coord < l < lmda_coord1 and beta_coord < b < beta_coord1 and vel_coord < v and vel_coord1:
    #                     key = f"Lon {n} Lat {j} Vel {i}"
    #                     if key not in voxel_dict.keys():

    #                         voxel_dict[key] = {'dates' : []}

    #                     voxel_dict[key]['dates'].append(d)
    
   
    for i, matrix in enumerate(H): 
        
        # stops from defining an index out of bounds of the velocity bins
        # if i + 1 >= len(voxel_vels):
        #     continue 

        vel_coord, vel_coord1 = voxel_vels[i], voxel_vels[i+1]
        # print(vel_coord)

        # print(shower_counts_copy[i])

        for j in range(len(H[i])): # goes over each row of voxels in the i'th shower matrix
            
            # if j+1 >= len(voxel_beta):
            #     continue

            row = H[i][j]

            beta_coord, beta_coord1 = voxel_beta[j], voxel_beta[j+1]

            # print(beta_coord)

            for n in range(len(row)):
                # if n+1 >= len(voxel_lmda):
                #     continue

                lmda_coord, lmda_coord1 = voxel_lmda[n], voxel_lmda[n+1]
                # print(lmda_coord)
                counts = row[n]
                # print(counts, 3*sigma) # this should be the specific voxel
                # print(counts)

                key = f"Lon {n} Lat {j} Vel {i}" # indicies for the iterated voxel

                if key not in voxel_dict.keys():

                    voxel_dict[key] = {'dates' : [], 'lon bounds' : [lmda_coord, lmda_coord1],
                                       'lat bounds' : [beta_coord, beta_coord1], 'vel bounds' : [vel_coord, vel_coord1]}


    # # # the lists for each voxel is created above, now time to iterate over each meteor and find out which voxel to put it in
    # for d, nd in shower_dict.items():

    #     l, b, v = float(nd['lmda']), float(nd['beta']), float(nd['vel'])
    #     print(l,b,v)

    #     for value in voxel_dict.values():

    #         dates = value['dates']
    #         lmdas = value['lon bounds']
    #         betas = value['lat bounds']
    #         vels = value['vel bounds']
    #         print(lmdas, betas, vels)

    #         # this does not work for some odd reason; will trouble shoot soon
    #         if float(lmdas[0]) < l < float(lmdas[1]) and float(betas[0]) < b < float(betas[1]) and float(vels[0]) < v < float(vels[1]):

    #             voxel_dict[key]['dates'].append(d)

    # for v in voxel_dict.values():
    #     dates = v['dates']
    #     print(dates)
    
    return voxel_dict # should contain keys of each voxel and which meteors can be found in its boundaries

    #                     voxel_dict[f'Boundaries: ({lmda_coord}, {lmda_coord1}), ({beta_coord}, {beta_coord}), ({vel_coord}, {vel_coord})'] = {'date' : d, 'lambda' : l, 'beta' : b, 'vel' : v}


def voxel_subtract(edges, lmda, beta, vels, days, year, shower_counts, background_counts, name, method, slon_status='active'):
    '''
    The 3d version of the above background subtraction function to be applied to voxel plots
    any shower voxel that has less counts than 3 times the st dev of the 2d plane it is in will be removed
    voxels with this little counts will be taken as sporadic sources, since shower sources are much higher in counts per voxel
    '''

    new_folder_name = f'{year} {name}'

    # creating new folder to store monthly organized data

    active_folder_name = f'{home}/clean shower data/{new_folder_name} clean events/{new_folder_name} active'
    active_folder = os.listdir(active_folder_name)

    # store surviving data here
    shower_lmda = []
    shower_beta = []
    shower_vel = []

    removed_voxels = []

    import shutil # for backing up files that are changed based on shower subtraction methods

    ## the hopes with this is to keep track of which specific events are being removed
    # this way I can have an explicit data set of meteors that are sporadics, removing the specific events that are in showers
    # another way I could do this is to use the convex hull in 3d space to isolate a region of coordinates and velocities, and any meteor with coordinates that fall in the convex hull are removed
    # will be focusing on the sigma test for now ##
    kept_lines = []

    # for filename in active_folder:
    #     file_path = os.path.join(active_folder_name, filename)

    #     with open(file_path, 'r') as f:
    #         for line in f:
    #             stripped = line.strip()
    #             params = stripped.split()

    #             # keep non-data/header lines
    #             if params == [] or params[0][0] != '2':
    #                 kept_lines.append(line)
    #                 continue

    # both shower counts and background counts are 3d arrays, with each 2d array being 9x9 matrices
    

    shower_counts_copy = shower_counts.copy()

    num_removed = 0
    num_left = 0
    # init_len = len(shower_dict)

    # lmda_list  = lmda.copy().tolist()
    # beta_list  = beta.copy().tolist()
    # vels_list  = vels.copy().tolist()
    # dates_list = dates.copy()

    # voxel coordinates

    voxel_lmda = edges[0]
    voxel_beta = edges[1]
    voxel_vels = edges[2]

    lmda = np.asarray(lmda)
    beta = np.asarray(beta)
    vels = np.asarray(vels)
    days = np.asarray(days)


    keys_to_remove = set()

    # iterate over each meteor, check which voxel is in, do the voxel count test and if this fails, delete this meteor and clear the voxel
    # for d, nd in shower_dict.items():

    #     l, b, v = nd['lmda'], nd['beta'], nd['vel']

    # goes over each background plane of lat/lon scaled by velocity - i corresponds to velocity index
    for i, matrix in enumerate(background_counts): 
        
        # stops from defining an index out of bounds of the velocity bins
        if i + 1 >= len(voxel_vels):
            continue # does this kill the loop? 

        mean = np.mean(matrix)
        sigma = np.std(matrix) # st dev of the slab of voxels' counts
        # print(sigma)

        vel_coord, vel_coord1 = voxel_vels[i], voxel_vels[i+1]
        # print(vel_coord)

        # print(shower_counts_copy[i])

        # iterates over each corresponding row inthe shower data
        for j in range(len(shower_counts_copy[i])): # goes over each row of voxels in the i'th shower matrix
            
            if j+1 >= len(voxel_beta):
                continue

            row = shower_counts_copy[i][j]

            beta_coord, beta_coord1 = voxel_beta[j], voxel_beta[j+1]

            # print(beta_coord)


            for n in range(len(row)):
                if n+1 >= len(voxel_lmda):
                    continue

                lmda_coord, lmda_coord1 = voxel_lmda[n], voxel_lmda[n+1]
                # print(lmda_coord, lmda_coord1)
                # print(beta_coord, beta_coord1)
                # print(vel_coord, vel_coord1)
                counts = row[n]
                # print(counts, 3*sigma) # this should be the specific voxel

                if 0.0 < counts < 3*sigma:
                    print(counts, sigma, f'found in bin lon {n} lat {j} vel {i}')
                    shower_counts_copy[i][j][n] = 0
                    num_removed += counts

                    # include meteors outside of the current removed voxel in each mask
                    count_mask = ((lmda_coord > lmda) & (lmda < lmda_coord1) &
                                  (beta_coord > beta) & (beta < beta_coord1) &
                                  (vel_coord > vels) & (vels < vel_coord1)) # use to only grab meteors found in the current voxel
                    
                    # might be too strong
                    lmda_count = lmda[count_mask]
                    beta_count = beta[count_mask]
                    vels_count = vels[count_mask]
                    days_count = days[count_mask]

                    # the main lists should contain where the removed meteors are roughly found in the defined radiant space
                    if [[lmda_coord, lmda_coord1], [beta_coord, beta_coord1], [vel_coord, vel_coord1]] not in removed_voxels:
                        removed_voxels.append([[lmda_coord, lmda_coord1], [beta_coord, beta_coord1], [vel_coord, vel_coord1]])
                    # shower_lmda.append([lmda_coord, lmda_coord1])
                    # shower_beta.append([beta_coord, beta_coord1])
                    # shower_vel.append([vel_coord, vel_coord1])

                    # if i add lists of the voxel coordinates to bigger lists and go over each one, I can find out in seperat loop which meteors fall there


                else:
                    num_left += counts

                # for d, nd in shower_dict.items(): # where the loop was when it was deleting every event
                # # i think this deletes everything because it goes over every meteor in the dictionary while set boundaries for voxels are defined
                # # so when new voxels are defined, this loop is ran over again and the coordinates that fit within them are deleted
                # # if this is done each time for every voxel it should remove everything; maybe move the dictionary loop outside the voxel check

                #     l, b, v = nd['lmda'], nd['beta'], nd['vel']

                #     # print('Boundaries: ', lmda_coord, lmda_coord1, beta_coord, beta_coord1, vel_coord, vel_coord1)
                #     # print('Current coordinates: ', l, b, v)
                    
                #     if lmda_coord < l < lmda_coord1 and beta_coord < b < beta_coord1 and vel_coord < v < vel_coord1:
                #         # shower_lmda.append(l)
                #         # if 0.0 < counts < sigma:
                #         #     # print(counts)
                #         #     shower_counts_copy[i][j][n] = 0
                #             # num_removed += counts
                #         print('Deleted')
                            
                #         # filtered_lmda.remove(l)
                #         # filtered_beta.remove(b)
                #         # filtered_vels.remove(v)
                #         # filtered_dates.remove(d)
                #         # print(l, b, v, d, 'removed')
                #         keys_to_remove.add(d) # maybe use i, j and n as coordinates instead? will have to group them differently to dates somehow - could build a dictionary before or within this function
                #         # if meteor falls in this voxel
                #             # remove meteor and day


    print('Meteors after 3 sigma count test: ', len(lmda_count))

    # pack the remaining meteors in a list and return this instead of four seperate lists
    shower_coords = []

    shower_coords.append(lmda_count)
    shower_coords.append(beta_count)
    shower_coords.append(vels_count)
    shower_coords.append(days_count)

    # filtered_lmda  = lmda_list
    # filtered_beta  = beta_list
    # filtered_vels  = vels_list
    # filtered_dates = dates_list
    # print(len(shower_lmda)) # this should not always be equal to or less than num_removed, since removed voxels might have a count of zero
    
    voxel_meteors = 0
    # print(len(shower_dict))

    # new_dict = shower_dict.copy()
    removed_dict = {}

    print('Num of voxels that have been emptied after the count check', len(removed_voxels)) # length is correct
    print('Num of counts removed: ', num_removed)
    
    # for sl, sb, sv in zip(shower_lmda, shower_beta, shower_vel):
    # print('Num of meteors before count check:', len(shower_dict))
    # goes over each stored voxel that was emptied
    # for vox in removed_voxels:

    #     sl, sb, sv = vox[0], vox[1], vox[2]

    #     l_min, l_max = sl[0], sl[1]
    #     b_min, b_max = sb[0], sb[1]
    #     v_min, v_max = sv[0], sv[1] # the very last entry of velocities ends up being an entry from latitudes for some reason


    #     # goes over each meteor and checks if it lies in the current voxel and removed if so
    #     for d, nd in shower_dict.items():

    #         l, b, v = nd['lmda'], nd['beta'], nd['vel']
    #         # print(l_min, l, l_max)
    #         # print(b_min, b, b_max)
    #         # print(v_min, v, v_max)

    #         if l_min < l < l_max and b_min < b < b_max and v_min < v < v_max and d not in keys_to_remove:
                
    #             voxel_meteors += 1 # keeps track of the number of meteors deleted
    #             # print(d, end=' ')
    #             removed_dict[d] = {'lmda' : l, 'beta' : b, 'vel' : v}
    #             del new_dict[d] # keyed by date and time, which is unique to each meteor
    

    # for key in keys_to_remove:
    #     print(key, end=' ')
    #     del shower_dict[key]
    
    # print(voxel_meteors)
    # print('\nNum of meteors after count check: ', len(new_dict))

    # there are less meteors removed from this step than there are counts per voxel
    # my voxel function only returns information of counts, not where each meteor is and which voxel it lies in
    # should update the function to see if I can include that step there, might make this function a bit easier to use/understand

    # num_left = init_len - num_removed

    print(voxel_lmda, voxel_beta, voxel_vels) # voxel boundaries
    
    print(f'\nAfter checking voxel counts, {num_removed} meteors were removed for being within voxel having too low counts relative to the shower', end=' ') 
    print(f'and {num_left} meteors remain as part of the shower radiant.\n')
    # print('Number of days remaining in shower region: ', len(shower_dict.keys()))
    return shower_counts_copy, int(num_left), shower_coords


def coord_sigma(shower_lmda, shower_beta, shower_vels, shower_days, helios, slons, year):

    '''
    After creating a 3D voxel map of an isolated shower, the background flux is taken 5 days before and after the shower is active and a 3 sigma test is done on each meteor's
    ecliptic coordinates and geocentric velocity.
    The first three parameters are for all ecliptic coordinates close to and within the active timeframe of the shower
    The following three parameters are for all ecliptic coordinates within the active timeframe of the shower
    Taking the computed heliocentric coordinates of the shower's center location using shower_helios and doing the 3 sigma test around here
    '''

    # next is to subtract the background from any showers; will need days/locations with shower data first
    
    # using these for a 3*std test to isolate shower only meteors
    lmda_mean = np.mean(shower_lmda)
    beta_mean = np.mean(shower_beta)
    vels_mean = np.mean(shower_vels)

    lmda_std = np.std(shower_lmda)
    beta_std = np.std(shower_beta)
    vels_std = np.std(shower_vels)

    # new_lmda = np.asarray(shower_lmda.copy())
    # new_beta = np.asarray(shower_beta.copy())
    # new_vels = np.asarray(shower_vels.copy())

    new_lmda = []
    new_beta = []
    new_vels = []
    new_days = []
    # Yung used st dev of a generated wavelet in her paper
    # I'll use an over all st dev for the shower region in the meantime while I figure out if I can use a wavelet as well


    # print(lmda_mean, beta_mean, vels_mean)
    # print(lmda_std, beta_std, vels_std)

    # next need to find a way to get the flux; might be a few steps ahead of where I am in the project though
    # will confirm and revisit in the future if needed

    mass_index = 1 # from MCB

    # 3 sigma boundaries
    lmda_bounds = [shower_helios[0] - 3*lmda_std, shower_helios[0] + 3*lmda_std]
    beta_bounds = [shower_helios[1] - 3*beta_std, shower_helios[1] + 3*beta_std]
    vels_bounds = [vels_mean - 3*vels_std, vels_mean - 3*vels_std]

    shower_lmda = np.asarray(shower_lmda)
    shower_beta = np.asarray(shower_beta)
    shower_vels = np.asarray(shower_vels)

    # a mask that only keeps meteors found within 3 std of the shower's mean on all three axes
    loc_mask = ((lmda_bounds[0] <= shower_lmda) & (shower_lmda <= lmda_bounds[1]) &
                (beta_bounds[0] <= shower_beta) & (shower_beta <= beta_bounds[1]))
                # (vels_bounds[0] <= shower_vels) & (shower_vels <= vels_bounds[1]))
    # Too strong
    
    # mask = (
    # (np.abs(shower_lmda - lmda_mean) <= 3 * lmda_std) &
    # (np.abs(shower_beta - beta_mean) <= 3 * beta_std)  &
    # (np.abs(shower_vels - vels_mean) <= 3 * vels_std)
    # )

    new_lmda = shower_lmda[loc_mask]
    new_beta  = shower_beta[loc_mask]
    new_vels  = shower_vels[loc_mask]
    new_days = shower_days[loc_mask]

    # for l, b, v, d in zip(shower_lmda, shower_beta, shower_vels, shower_days):

    #     if abs(l - lmda_mean) <= 3*lmda_std and abs(b - beta_mean) <= 3*beta_std and abs(v - vels_mean) <= 3*vels_std:

    #         new_lmda.append(l)
    #         new_beta.append(b)
    #         new_vels.append(v)
    #         new_days.append(d)

        
    return new_lmda, new_beta, new_vels, new_days


# # Supplimentary functions for plotting
def long_transform(lmda):

    '''
    This function will be used to transform the ecliptic longitude values to plot on a 0-360 degree scale
    '''

    return lmda % 360

    # if lmda < 0:
    #     return 360 + lmda
    # else:
    #     return lmda      


def scale(x):
    '''
    This function scales the x axis to be centered at 270 degrees longitude
    '''
    x = np.asarray(x) % 360
    res = (x - 270) % 360

    return np.where(res > 180, res - 360, res)

def scale2(x):
    '''
    This function scales the x axis to be centered at 270 degrees longitude
    '''
    x = np.asarray(x) % 360
    res = (x) % 360

    return np.where(res > 180, res - 360, res)

# Only use this for presentation-style plots
def relabel(x, pos):
    # pass
    '''
    This function will relabel the x axis to have labels of 90 degrees, goig down to zero, then going from 359 down to 91 degrees
    '''
    if x == 0:
        return '270°'
    elif x == -90:
        return '180°'
    elif x == -180:
        return '90°'
    elif x == 90:
        return '0°'
    elif x == 180:
        return '90°'
    else:
        return ''



# this will be used to define the path to the folder with radar data
folder_name = input("Enter folder name (or drag folder here): ").strip("'\"")  # Strip quotes that may be included when dragging from file explorer

# connects the input folder name to the folder that exists in the WD
    # should eventually write a function that creates a new folder just by inputting a filename, and then writes the clean echoes to that folder; will do later
 
 # this will be used to go through each file found in the folder
folder = os.listdir(os.path.join(folder_name))

num_echoes = 0 # before duplicate correction
num_echo_locs = 0 # function call for clean_echos and the number of echoes after corrections
num_without_coords = 0

# duplicate check results

delranges = 0
delangles = 0

ecliptics = []

# used for one plot including year's worth of echoes
plot_lons = []
plot_lats = []
plot_vels = []

vel_ptns = []
d_vel_ptns = []

# from the cel2hel function used on each meteor's radiant
c2h_lons = []
c2h_lats = []


# filter that is currently being used - change for testing

method = input('Enter the method you wish to do testing with (choose from: raw, all, vel, int, angle, station, vel and int, vel and int and angle): ')

method = method.lower().strip()


# prompt asking for monthly organization

monthly = input('Do you want to organize your data by month? (Y or N): ')


# Source to be plotted based on user input

source_use = input('Do you wish to work directly with the sporadic sources? (Y or N): ') # maybe after this, add a check to Parse that writes raw data to a seperate file for overplotting if true

if source_use.upper() == 'Y':
    source_isolate = True
    source = input('Enter the source you wish to work with (choose from AH, H, NA, SA, NT, ALL): ')
    source = source.upper().strip()
else:
    source_isolate = False
    source = None


# Shower to be plotted based on user input

shower_use = input('Do you wish to work directly with a specific shower? (Y or N): ')

if shower_use.upper().strip() == 'Y':
    shower_isolate = True
    # will ask for the name of the shower following the yearly plot for now
    # might be worth working on different modes - if input is month, the code does monthly plotting, if the input is shower, the code does shower plotting
else:
    shower_isolate = False

# Option for radiant heatmap, or a scatter plot
map = input('Do you wish to see your plot as a color map? (Y or N): ')

if map.upper() == 'Y':
    map_mode = input('Enter the mode to display the map in (choose between velocity or density): ')
else:
    map_mode = 'scatter'

# main loop to go through each file in the provided directory
if len(folder) == 0:
    print('Your folder is empty! Please check the folder name and try again.')
else:
    # goes through each event file in the folder
    for file in folder:

        orb_date = file[4:12] # year-solar longitude
        sl = file[9:12] # solar longitude

        # Parse function call
        parent_dict, num_clean = Parse(folder_name, file, method=method, sources=[source_isolate, source]) # num clean being written as pre duplicate number of events
        # print(len(parent_dict))
        
        # coordinate collection call
        lat, lon, vel_ptn0, del_vel_ptn0, vel_g, deleted_events, num_locs, times, dists, angles, c2h_lon, c2h_lat = grab_coords(parent_dict) # will be used to grab the ecliptic coordinates of the clean echoes for plotting; will do later
        # num locs is the number of events before duplicate correction below

        ecliptics.append([lat, lon])

        # lat and lon are lists themselves, so I am putting those elements into a bigger list for plotting
        plot_lons.extend(lon)
        plot_lats.extend(lat)
        plot_vels.extend(vel_g)

        vel_ptns.extend(vel_ptn0)
        d_vel_ptns.extend(del_vel_ptn0)

        num_without_coords += len(deleted_events)

        # testing if the cel2hel function translates to the correct ecliptic coordinates
        # v_cel      = getvec(radiants[0], radiants[1])
        # v_hel      = cel2hel(v_cel, sl)
        # l_comp, b_comp = getangle(v_hel)

        c2h_lons.extend(c2h_lon)
        c2h_lats.extend(c2h_lat)

        # print('nums', num_locs, num_clean)
        # function call to save located clean echo data
        num_echoes += num_clean # adding on to counter to track number of clean echoes per year; might not need this
        
        # If we only want raw data, this runs and no duplicate check is applied
        if method == 'raw':

            parent_new = parent_dict.copy()

            dr, da = 0, 0

        
        # this one is for when filters are being used, hence a duplicate check is necessary
        else:
            
            # removing any meteors we deem duplicates (check duplicate function for criteria)
            parent_new, dr, da = duplicate(parent_dict, times, dists, angles, delranges, delangles)
            
            delranges += dr
            delangles += da

            if dr != 0 or da != 0:
                print(f'For events within 10 milliseconds of eachother, {delranges} events were deleted for being within 6km of each other, {delangles} events were deleted for having zenith/azimuth angles within 5 degrees')

        
        # print(len(parent_dict), len(parent_new)) # same as num locs, the second number is number of events after duplicate check

        # running the writing function to only include data for clean echoes with a defined set of coordinates
        num_echo_locs += clean_echoes(parent_new, num_clean, orb_date, folder_name, file, dr, da, method=method)

year = orb_date[0:4] # this is only valid if the data set being ran is within the same calendar year; need something more universal eventually 

# Full Year Data: Printed to the terminal and saved to the same directory as the clean echo data
message1 = f'\n\nFULL YEAR DATA:\n\n'
message2 = f'The total number of clean echoes across all files is {num_echoes}, with {num_echo_locs} events observed at distinct times and that have defined ecliptic coordinate and can be plotted.\n'
message3 = f'For all events within 10 milliseconds of eachother, {delranges} events were deleted for being within 6km of each other, {delangles} events were deleted for having zenith/azimuth angles within 5 degrees.\n'
message4 = f'There were {num_without_coords} events deleted for not having a defined set of ecliptic coordinates.\n'

year_message = message1+message2+message3+message4

print(year_message)

 # writes data to this newly created file regardless of the mode chosen (yearly, monthly, shower, sources)
main_path = f'{home}/clean file data/{year} clean events'

write_path = os.path.join(main_path, f"FULL-{year}-29.txt")

with open(write_path, 'w') as full_data:
    full_data.write(year_message)


scaled_lons = scale(plot_lons) # scaling longitudes to be centered at 270 degrees

scaled_c2h_lons = scale2(c2h_lons)



# number of orbits with ptn0 speeds vs uncertainty / number of orbits eliminated from filter
# will make both, unsure which one they want

# fractional change in uncertainty in the ptn0 speed

# frac_vel_ptns = []

# # print(vel_ptns, d_vel_ptns)

# for i in range(len(vel_ptns)):
#     frac_ptn0 = d_vel_ptns[i] / vel_ptns[i]
#     frac_vel_ptns.append(frac_ptn0)

# figure = plt.figure(figsize=(10,5))

# # plt.hist2d(vel_ptns, d_vel_ptns, bins=50)
# plt.scatter(vel_ptns, frac_vel_ptns)

# plt.title('Pre-t0 velocities vs fractional change in pre-t0 velocities')
# plt.xlabel('Pre-t0 Velocities (km/s)', fontsize=12)
# plt.ylabel(r'Fractional Change in Pre-t0 Velocity $({\Delta v_{ptn0} / v_{ptn0} })$', fontsize=12)

# plt.show()


# 3d map of clean meteors, wtih additional axis of geocentric velocity; should use this for monthly plotting instead of yearly, getting too many events in one plot to see anything
# echo_3d_plot(scaled_lons, plot_lats, plot_vels, year)

# voxel map for convex hull
# voxel_map(scaled_lons, plot_lats, plot_vels, year)

# velocity histogram binning; want to do this for raw data, then once again for the filtered data; currently doing it with all filtering applied
# vel_histo(plot_vels, year, method)

gen_message = message2+message3+message4 # message to go into full files that are not yearly data (for monthly, shower and source files)

## MONTHLY ORGANIZATION STEP ##

# here, the code organizes the file data by month; might take away the option for user input and do this automatically

if monthly.upper().strip() == 'Y':

    clean_folder_name = input('Enter the folder name of clean echo data you wish to organize by month (or drag folder here): ').strip("'\"")  # Strip quotes that may be included when dragging from file explorer

    clean_folder = sorted(os.listdir(os.path.join(clean_folder_name)))


    for clean_file in clean_folder:
        # print(clean_file)

        monthly_echoes(year, clean_folder_name, clean_file)


        # next step is to write a function that will go through the monthly organized files and plot the coordinates for each month to compare meteor sources across months; will do later

    for m in range(1, 13):
        # plot the coordinates for each month

        monthly_plotter(year, m, clean_folder_name, clean_file, method, map_method=map_mode)


## SOURCE PLOTTING STEP ##

# creating a map of the clean meteor sources based on their coordinates of ecliptic latitude and longitude
# echo_plot(scaled_lons, plot_lats, plot_vels, year, method)


if source_isolate:
    echo_plot(scaled_lons, plot_lats, plot_vels, year, method, source=source, mode='source', map_mode=map_mode, bounds=[(-150, 150), (-60, 90)])
    vel_histo(plot_vels, year, method, source=source, mode='source')

else:
    echo_plot(scaled_lons, plot_lats, plot_vels, year, method, map_mode=map_mode, bounds=[(-150, 150), (-60, 90)])
    vel_histo(plot_vels, year, method)

    echo_plot(c2h_lons, c2h_lats, plot_vels, year, method, map_mode=map_mode)
    # echo_plot(scaled_c2h_lons, c2h_lats, plot_vels, year, method, map_mode=map_mode, bounds=[(min(c2h_lons), max(c2h_lons)), (min(c2h_lats), max(c2h_lats))]) # heliocentric coordinates not 1-1 with ecliptic coordinates

    


## SHOWER REMOVAL STEP ##

# Localizing and clearing out the meteor showers from the sporadic background

if shower_isolate:

    # shower_folder_name = input('Enter the folder name of the clean shower data (format of the ending folder should be \'YYYY clean events\'): ').strip("'\"")

    shower_folder_name = f'{home}/clean file data/{year} clean events'


    shower_folder = sorted(os.listdir(shower_folder_name))

    # shower_name = shower_folder[-1:-4] # should be last three letters of the user input
    # print(shower_name)

    shower_name = input('Enter the abbreviation of the shower you\'d like to work with - select from the abbreviations: ARI, DSX, ETA, GEM, ORI, QUA, SDA: ').upper().strip()
    # from here, the shower parser function should enter the clean file data folder and pick out the days the shower is active using specified solar longitude by a dictionary

    # list of important solar longitudes per shower from Brown et al. (2010)
    shower_slon0 = {'ARI' : np.arange(62,99), 'DSX': np.arange(174,197), 'ETA' : np.arange(30, 66),
                'GEM' : np.arange(240,273), 'QUA' : np.arange(232, 291), 'SDA' : np.arange(114, 164)}
    
    # list of important solar longitudes per strong shower from MCB with 5 day buffer both before and after the active shower days - Using this
    shower_slon = {'ARI' : np.arange(52, 110), 'DSX': np.arange(164, 208), 'ETA' : np.arange(20, 77),
                'GEM' : np.arange(230, 284), 'ORI' : np.arange(188, 238), 'QUA' : np.arange(265, 302), 'SDA' : np.arange(104, 175)}
    
    # list of shower peak solar longitudes from MCB
    shower_slon_peaks = {'ARI' : 81, 'DSX': 186, 'ETA' : 45,
                'GEM' : 261, 'ORI' : 208, 'QUA' : 283, 'SDA' : 126}

    # ordered lists of geocentric right ascension (alpha) and declination (delta) from Brown et al. (2010) and MCB (same numbers) - Using this
    shower_rads = {'ARI' : [45.7, 25], 'DSX': [154.3, -1], 'ETA' : [337.9, -0.9],
                'GEM' : [112.5, 32.1], 'ORI' : [95.5, 15.2], 'QUA' : [231.5, 48.5], 'SDA' : [340.8, -16.3]}
    
    shower_hels = {}
    
    for shower, rads in shower_rads.items():

        v_cel = getvec(rads[0], rads[1])

        hel = cel2hel(v_cel, shower_slon_peaks[shower]) # currently doing peaks only, should do each one and tie into the correction done in shower parser for different ra/decs from the peak

        lon, lat = getangle(hel) # store these

        # scaled_lon = scale(lon) # bring to a 0-360 scale

        shower_hels[shower] = [float(lon), float(lat)]
    
    print(shower_hels)
    shower_helios = shower_hels[shower_name] # two entry list

    # defining bounds for the specific shower here - put into shower parser and only include meteors within these bounds using a mask
    lcomp_bounds = [shower_helios[0] - 10, shower_helios[0] + 10]
    bcomp_bounds = [shower_helios[1] - 10, shower_helios[1] + 10]

    print(lcomp_bounds[0], lcomp_bounds[1], lcomp_bounds[0] < lcomp_bounds[1])  # should be True
    print(bcomp_bounds[0], bcomp_bounds[1], bcomp_bounds[0] < bcomp_bounds[1])

    print('shower bounds: ', lcomp_bounds, bcomp_bounds)
    
    shower_rad_drifts = {'ARI' : [0.86, 0.18], 'DSX' : [-1.00, 0.56], 'ETA' : [0.70, 0.33],
                         'GEM' : [1.12, -0.17], 'ORI' : [0.78, 0.02], 'QUA' : [0.78, -0.38], 'SDA' : [0.78, 0.30]} # 'KEY' : [alpha drift value, delta drift value] both in degrees
   
    # get the solar longitudes of the showers from the filenames

    # only from the files on days of when the shower is active
    active_days = []
    active_lons = []
    active_lats = []
    active_vels = []

    active_dict = {} # want to store the above three values, keyed by filename

    # only from the files 5 days before and after the shower is active; use this along with heat_map to generate a number density matrix for each day

    outer_days = []
    outer_lons = []
    outer_lats = []
    outer_vels = []

    # includes files of slon 5 days before and after the shower is active
    full_days = []
    full_lons = []
    full_lats = []
    full_vels = []

    # main dictionary keyed by solar longitude, nested dictionaries keyed by date+time with values of RA/Dec
    corr_radiants = {}

    # to keep track of how many meteors are in the shower days and in the background days
    shower_meteors = 0
    background_meteors = 0
    total_meteors = 0

    # array of solar longitudes of a shower based on user input
    input_slons = shower_slon[shower_name]
    slon_peak = shower_slon_peaks[shower_name]
    min_slon, max_slon = min(input_slons), max(input_slons)

    slons_before = min_slon - 10
    slons_after = max_slon + 10

    outer_slons = list(range(slons_before, min_slon)) +  list(range(max_slon + 1, slons_after + 1))
    
    full_slons = list(range(slons_before, slons_after + 1)) # copy of the list to be used as one containing echoes 5 days before and after the shower appears

    # print(input_slons)
    # print(outer_slons)
    # print(full_slons)

    num_active_meteors = 0
    num_outer_meteors = 0
    total_shower_meteors = 0

    # collecting the shower files here
    for shower_file in shower_folder:

        if shower_file[0:4].upper() == 'FULL':
            continue

        file_slon = shower_file[11:14]
        # print(file_slon)


        if file_slon[0:2] == '00':
            file_slon = int(file_slon[2]) # taking the leading zero out from solar longitude
            # print(file_slon)


        elif file_slon[0] == '0':
            file_slon = int(file_slon[1:]) # taking the leading zero out from solar longitude
            # print(file_slon)

        else:
            file_slon = int(file_slon)
            # print(file_slon)
        # if not satisfied, this step should be skipped for any file not within the specified range of solar longitudes based on the input shower
        # should include somewhere here the 5 days before and after this range found below to calculate the average background flux for shower subtraction
        if file_slon in full_slons:
            
            # using these files to plot the ec# for 2022 dsx, went from 2360 meteors to 99hoes seen from the shower
            if file_slon in input_slons:
                
                shower_lmda, shower_beta, shower_vel, corr_shower_radiants, shower_meteor_count, shower_days = shower_parser(year, shower_folder_name, shower_file, file_slon, slon_peak, shower_rads, shower_rad_drifts, shower_name, method, slon_status='active', boundaries=[lcomp_bounds, bcomp_bounds])
                
                shower_lmda = scale(shower_lmda) # this step is important; keep here
                # apply to outer and update the counter to track meteors near the shower

                # masking step to isolate location of the shower and radiants within
                shower_lmda = np.asarray(shower_lmda)
                shower_beta = np.asarray(shower_beta)
                shower_vel = np.asarray(shower_vel)
                shower_days = np.asarray(shower_days)

                shower_mask = ((shower_lmda >= lcomp_bounds[0]) & (shower_lmda <= lcomp_bounds[1]) &
                                       (shower_beta >= bcomp_bounds[0]) & (shower_beta <= bcomp_bounds[1])
                                       )
                
                print(f'Passing lmda condition: {np.sum((shower_lmda >= lcomp_bounds[0]) & (shower_lmda <= lcomp_bounds[1]))}')
                print(f'Passing beta condition:  {np.sum((shower_beta >= bcomp_bounds[0]) & (shower_beta <= bcomp_bounds[1]))}')
                print(f'Passing both:            {np.sum(shower_mask)}')
                                
                lmda_filtered = shower_lmda[shower_mask]
                beta_filtered = shower_beta[shower_mask]
                vels_filtered = shower_vel[shower_mask]
                days_filtered = shower_days[shower_mask]
        
                
                # collects coordinates of the shower files, along with velocities
                # using these lists to calculate flux, which will be used for shower subtraction
                # active_days.extend(shower_days)
                # active_lons.extend(shower_lmda)
                # active_lats.extend(shower_beta)
                # active_vels.extend(shower_vel)

                active_days.extend(days_filtered)
                active_lons.extend(lmda_filtered)
                active_lats.extend(beta_filtered)
                active_vels.extend(vels_filtered)

                # corr_radiants[corr_shower_radiants] = corr_shower_radiants[file_slon]
                # corr_radiants.extend(corr_shower_radiants.items())
                corr_radiants[file_slon] = corr_shower_radiants

                # active_dict[shower_file] = {"ecl lon" : shower_lmda, "ecl lat" : shower_beta, "geo vel" : shower_vel}
                active_dict[shower_file] = {"ecl lon" : lmda_filtered, "ecl lat" : beta_filtered, "geo vel" : vels_filtered}

                # BEFORE MASK keeping track of number of meteors deemed close enough to the shower for a given solar longitude
                shower_meteors += shower_meteor_count
                total_meteors += shower_meteor_count

                # AFTER MASK
                num_active_meteors += len(lmda_filtered)

                # also store the file in a new directory here
                # will also want a way to copy sporadics to another file, or remove the shower meteors from this file using the background subtraction and 3 sigma test
            
            # using these to construct average sporadic number density matrices to subtract from the shower's number density matrix (heat map)
            elif file_slon in outer_slons:

                shower_lmda, shower_beta, shower_vel, corr_background_radiants, background_meteor_count, shower_days = shower_parser(year, shower_folder_name, shower_file, file_slon, slon_peak, shower_rads, shower_rad_drifts, shower_name, method, slon_status='outer')

                shower_lmda = scale(shower_lmda)

                # masking step to isolate location of shower and radiants within
                shower_lmda = np.asarray(shower_lmda)
                shower_beta = np.asarray(shower_beta)
                shower_vel = np.asarray(shower_vel)
                shower_days = np.asarray(shower_days)

                shower_mask = ((shower_lmda >= lcomp_bounds[0]) & (shower_lmda <= lcomp_bounds[1]) &
                                       (shower_beta >= bcomp_bounds[0]) & (shower_beta <= bcomp_bounds[1])
                                       )
                
                print(f'Passing lmda condition: {np.sum((shower_lmda >= lcomp_bounds[0]) & (shower_lmda <= lcomp_bounds[1]))}')
                print(f'Passing beta condition:  {np.sum((shower_beta >= bcomp_bounds[0]) & (shower_beta <= bcomp_bounds[1]))}')
                print(f'Passing both:            {np.sum(shower_mask)}')
                                
                lmda_filtered = shower_lmda[shower_mask]
                beta_filtered = shower_beta[shower_mask]
                vels_filtered = shower_vel[shower_mask]
                days_filtered = shower_days[shower_mask]

                print(len(outer_lons))

                outer_days.extend(days_filtered)
                outer_lons.extend(lmda_filtered)
                outer_lats.extend(beta_filtered)
                outer_vels.extend(vels_filtered)

                corr_radiants[file_slon] = corr_background_radiants

                # BEFORE MASK: keeping track of number of meteors close enough to the source in background days of the shower
                background_meteors += background_meteor_count
                total_meteors += background_meteor_count
                
                # AFTER MASK

                num_outer_meteors += len(shower_lmda)

            # storing both active and outer file data to these lists
            full_days.extend(shower_days)
            full_lons.extend(shower_lmda)
            full_lats.extend(shower_beta)
            full_vels.extend(shower_vel)

            total_shower_meteors += num_active_meteors + num_outer_meteors
            
            
            # plots each day/solar longitude
            # echo_plot(shower_lmda, shower_beta, year)
        
        else:
            continue

    # print(active_dict)
    
    # scaling longitudes 
    scaled_active_lons = scale(active_lons) # use shower radiants and find lmda/beta using c2h 
                                            # and create bounds for each day of 5 degrees in between each computed value

    scaled_outer_lons = scale(outer_lons) # could split this list into two - one for days before, one for days after

    scaled_full_lons = scale(full_lons)


    # new_lons, new_lats, new_vel = background_subtract(year, outer_lons, outer_lats, outer_vels, shower_name, method, slon_status='active')

    # print(new_lons, new_lats, new_vel)

    # scaled_new_lons = scale(new_lons)

    # plot of the active shower
    h_shower = echo_plot(active_lons, active_lats, active_vels, year, method, shower=shower_name, mode='shower', map_mode=map_mode) # , bounds=[lcomp_bounds, bcomp_bounds]
    
    print('Number of shower meteors before masking: ', shower_meteors)
    print('Number of shower meteors after masking: ', num_active_meteors)

    shower_lon_bounds = [min(scaled_active_lons), max(scaled_active_lons)]
    shower_lat_bounds = [min(active_lats), max(active_lats)]
    shower_vel_bounds = [min(active_vels), max(active_vels)]

    # plot of the sporadic background - take the background matrix from here
    h_background = echo_plot(outer_lons, outer_lats, outer_vels, year, method, shower=shower_name, mode='background', map_mode=map_mode) # , bounds=[lcomp_bounds, bcomp_bounds]
    # print('mean:', np.mean(h_background))

    print('Number of background meteors before masking: ', background_meteors)
    print('Number of background meteors after masking: ', num_outer_meteors)

    

    # one density matrix for both 5 days before/after
    # two density matrices for 5 days before, 5 days after

    # print(corr_radiants)


    for sl, meteor  in corr_radiants.items(): # iterating over nested dictionaries keyed by date+time. parent dictionary has keys of solar longitude

        # skips empty dictionaries; shouldn't be empty, will have to go over filtering of ra/dec in shower parser
        if meteor == {}:
            continue

        for obs, cels in meteor.items():

            ra, dec = cels['RA'], cels['Dec']

            cel = getvec(ra, dec)

            hel = cel2hel(cel, sl) # not quite 1-1 in relation to observed coordinates from orbit files

            lmda, beta = getangle(hel)

            # print('\nSolar Longitude: ', sl, 'Date/Time: ', obs)
            # print('\nCelestial: ', ra, dec)
            # print('\nHeliocentric ', lmda, beta)

    # for i, array in enumerate(h_background):
    #     print(i)
    #     print('mean;', np.mean(h_background[0][i]))
    #     # print('shape:', h_background[i].shape)
    #     print(h_background)
    #     background_avg = np.mean(h_background[0][i])

    #     h_background[0][:] = background_avg
    
    # print(h_background) # currently only sets the last array's average to all elements in the parent array 
    # print('sizes:', h_shower[3].size, h_background[3].size)

    background_avgs = np.mean(h_background[0], axis=1, keepdims=True)

    new_background = np.broadcast_to(background_avgs, h_background[0].shape)
    print(new_background)

    # plot of radiants after new subtraction method
    # h_new = echo_plot(scaled_new_lons, new_lats, new_vel, year, method, shower=shower_name, mode='shower', map_mode=map_mode)

    # print(h_shower[0], h_background[0])
    # instead of doing the radiant check on background showers, take all meteors from that day, and subtract those only within the bounds where the active shower radiants are found in
    h_diff = h_shower[0] - new_background # need to remove shower days here somehow
    # make all elements of h_background an average of the number distribution then subtract
    h_diff = np.clip(h_diff, 0, None)

    h_lons, h_lats = h_shower[1], h_shower[2]

    # print(h_diff) # run this in echo plot? needs to be in the function
    # print(np.mean(h_diff)) # average is 0.029, check for negative numbers?

    plt.figure(figsize=(10,5))
    plt.imshow(h_diff.T, origin='lower', cmap='plasma', extent=[h_lons[0], h_lons[-1], h_lats[0], h_lats[-1]])

    plt.colorbar(label='Shower Count Difference')
    plt.xlabel('Ecliptic Longitude')
    plt.ylabel('Ecliptic Latitude')
    plt.gca().invert_xaxis()
    
    plt.title('Shower minus background density')
    plt.show()

    # effectively getting the same plot as before subtraction
    # applying to 3d plot to see if there is a difference

    # 3d plot of shower day echoes - create the convex hull around radiants that survive the 3 sigma test following background subtraction
    # echo_3d_plot(scaled_active_lons, active_lats, active_vels, year, shower=shower_name, mode='shower')

    # 3d plot of background day echoes
    # masked_lons, masked_lats, masked_vels = echo_3d_plot(scaled_outer_lons, outer_lats, outer_vels, year, shower=shower_name, mode='shower', bounds=[lcomp_bounds, bcomp_bounds, shower_vel_bounds])

    # next would be to plot active days with counts = (active counts - outer counts)

    # histogram of velocities
    # vel_histo(full_vels, year, method, shower=shower_name, mode='shower') 
    # vel_histo(active_vels, year, method, shower=shower_name, mode='shower') 
    # vel_histo(masked_vels, year, method, shower=shower_name, mode='shower') # shows only the background sporadics in the bounds of the shower made from shower_parser

    ## VOXEL PLOTTING AND CONVEX HULL STEPS BELOW ## 

    # no meteors should be removed from the two calls below
    H_shower, edges_shower, shower_lons, shower_lats, shower_vels = voxel_map(active_lons, active_lats, active_vels, year, name=shower_name)
    H_background, edges_background, background_lons, background_lats, background_vels = voxel_map(outer_lons, outer_lats, outer_vels, year, name=shower_name, bounds=[lcomp_bounds, bcomp_bounds, shower_vel_bounds])

    print('sizes:', edges_shower[0].size, edges_background[0].size, edges_shower[1].size, edges_background[1].size, edges_shower[2].size, edges_background[2].size)

    Background_avgs = np.mean(H_background, axis=1, keepdims=True)

    new_Background = np.broadcast_to(Background_avgs, H_background.shape)
    # print(new_Background)
    

    # after background number density subtraction
    H_diff = H_shower  - new_Background

    # print(H_shower)

    # for date in H_shower:
        # if date not in h_diff dates:
            # remove date from shower date list

    # over filtering, each coordinate is checked independently regardless if its corresponding coordinates by index do not match
    diff_lons, diff_lats, diff_vels, diff_days = shower_lons.copy(), shower_lats.copy(), shower_vels.copy(), active_days.copy()
    back_lons, back_lats, back_vels, back_days = shower_lons.copy(), shower_lats.copy(), shower_vels.copy(), active_days.copy()

    print(len(diff_lons), len(diff_lats), len(diff_vels), len(diff_days))


    
    # defining arrays for background masking
    diff_lons =np.asarray(diff_lons)
    diff_lats = np.asarray(diff_lats)
    diff_vels = np.asarray(diff_vels)
    diff_days = np.asarray(diff_days)

    back_lons = np.asarray(back_lons)
    back_lats = np.asarray(back_lats)
    back_vels = np.asarray(back_vels)
    back_days = np.asarray(back_days)

    active_days = np.asarray(active_days)
    outer_days = np.asarray(outer_days)

    # # doing background subtraction for 3d distribution here
    # shower_mask = (
    #     (~np.isin(diff_lons, background_lons)) &
    #     (~np.isin(diff_lats, background_lats)) &
    #     (~np.isin(diff_vels, background_vels)) &
    #     (~np.isin(diff_days, outer_days))
    # )

    # back_mask = (
    #     (np.isin(back_lons, background_lons)) &
    #     (np.isin(back_lats, background_lats)) &
    #     (np.isin(back_vels, background_vels)) &
    #     (np.isin(back_days, outer_days))
    # )


    # build a set of background triplets for fast lookup
    background_triplets = set(zip(
        np.round(background_lons, 6),  # round to avoid float precision issues
        np.round(background_lats, 6),
        np.round(background_vels, 6)
    ))

    # check each active meteor's triplet against the background set
    active_triplets = list(zip(
        np.round(diff_lons, 6),
        np.round(diff_lats, 6),
        np.round(diff_vels, 6)
    ))

    shower_mask = np.array([t not in background_triplets for t in active_triplets])
    back_mask   = np.array([t in background_triplets     for t in active_triplets])


    # # print(diff_lons, background_lons)

    # # background subtracted coordinates ; might not need H_shower - H_background step with this in place
    diff_lons = diff_lons[shower_mask]
    diff_lats = diff_lats[shower_mask]
    diff_vels = diff_vels[shower_mask]
    diff_days = diff_days[shower_mask] # should i be removing events from this background subtraction?

    back_lons = back_lons[back_mask]
    back_lats = back_lats[back_mask]
    back_vels = back_vels[back_mask]
    back_days = back_days[back_mask]
    # print(shower_mask, back_mask)
    # print('days kept vs days removed: ', len(diff_days), len(back_days)) # does nothing for 2025 dsx

    
    print(len(diff_lons), len(diff_lats), len(diff_vels), len(diff_days))

    # print(diff_days)

    # shower_dict = {}

    # for l, b, v, d in zip (shower_lons, shower_lats, shower_vels, active_days):

    #     if l in background_lons and b in background_lats and v in background_vels:
            
    #         # meteors in the shower after background subtraction
    #         diff_lons.remove(l)
    #         diff_lats.remove(b)
    #         diff_vels.remove(v)
    #         diff_days.remove(d)

    #         # meteors removed following background subtraction
    #         back_lons.append(l)
    #         back_lats.append(b)
    #         back_vels.append(v)
    #         back_days.append(d)

    #     # want the meteors not in the above regions to move on to further filtering
    #     else:
    #         shower_dict[d] = {'lmda' : l, 'beta' : b, 'vel' : v}
    
    # echo_3d_plot(diff_lons, diff_lats, diff_vels, year, shower=shower_name, mode='shower')
    # echo_3d_plot(back_lons, back_lats, back_vels, year, shower=shower_name, mode='shower')
   

    diff_count = len(shower_lons) - len(diff_lons) # or len(back_lons)
    print(diff_count, len(back_lons), 'these should be the same\n')

    H_diff, edges_diff, lons_diff, lats_diff, vels_diff = voxel_map(diff_lons, diff_lats, diff_vels, year, shower_name, threshold=10)
    # print(H_diff)
    # shower voxel bins (lon/lat/vel)
    # [array([36.73   , 42.77875, 48.8275 , 54.87625, 60.925  , 66.97375,
    #    73.0225 , 79.07125, 85.12   ]), array([-15.43 , -14.275, -13.12 , -11.965, -10.81 ,  -9.655,  -8.5  ,
    #     -7.345,  -6.19 ]), array([14.78   , 20.79625, 26.8125 , 32.82875, 38.845  , 44.86125,
    #    50.8775 , 56.89375, 62.91   ])]

    # subtracted shower voxel bins (lon/lat/vel)
    # [array([36.68, 42.76, 48.84, 54.92, 61.  , 67.08, 73.16, 79.24, 85.32]), array([-15.27   , -14.13125, -12.9925 , -11.85375, -10.715  ,  -9.57625,
    #     -8.4375 ,  -7.29875,  -6.16   ]), array([14.27  , 19.9075, 25.545 , 31.1825, 36.82  , 42.4575, 48.095 ,
    #    53.7325, 59.37  ])]

    # lon/lat stays mostly the same, though velocity shifts upward after using np.isin
   
    
    print(f'\nFrom voxel subtraction, {diff_count} meteors were removed from the shower for overlapping with background sporadics.\n')
    print(f'Currently, there are {len(diff_lons)} meteors contained within the shower\'s radiant\n')
    # voxel_map2(H_diff, edges_shower, scaled_active_lons, active_lats, threshold=5)

    # echo_plot(diff_lons2, diff_lats2, diff_vels2, year, method, shower=shower_name, mode='shower', map_mode=map_mode)
    # print(shower_dict)
    # voxel_dict = voxel_organize(H_diff, edges_diff, shower_dict, year)
    # after 3 sigma clip ; do date/time removal here next
    H_prime, high_count_meteors, shower_coords = voxel_subtract(edges_diff, diff_lons, diff_lats, diff_vels, diff_days, year, H_diff, new_Background, shower_name, method) # will need to get lats/lons from this step and feed to coord_sigma
    # print(H_prime)

    # unpacking 3 sigma count meteors
    prime_lons, prime_lats, prime_vels, prime_days = shower_coords

    # # remaining meteors after voxel subtract
    # lon1, lat1, vel1 = [], [], []

    # # removed meteors after voxel subtract
    # lon2, lat2, vel2 = [], [], []

    # for d in shower_meteor_dict.values():

    #     l, b, v = d['lmda'], d['beta'], d['vel']

    #     lon1.append(l)
    #     lat1.append(b)
    #     vel1.append(v)
    
    # for d in removed_dict.values():

    #     l, b, v = d['lmda'], d['beta'], d['vel']

    #     lon2.append(l)
    #     lat2.append(b)
    #     vel2.append(v)

    # voxel_map(lon1, lat1, vel1, year)
    # print('Num remaining meteors: ', len(shower_meteor_dict))
    # voxel_map(lon2, lat2, vel2, year)
    # print('Num removed meteors: ', len(removed_dict))

    # voxel_map2(H_diff, edges_shower, scaled_active_lons, active_lats, active_vels, threshold=0)

    voxel_map2(H_prime, edges_shower, prime_lons, prime_lats, prime_vels, threshold=0)
    voxel_with_hull(H_prime, edges_shower, diff_lons, diff_lats, diff_vels, threshold=0)

    # the counts are not being passed onto the coordinate test, which might be why we see little change in 3sigma test
    final_lons, final_lats, final_vels, final_days = coord_sigma(prime_lons, prime_lats, prime_vels, prime_days, shower_helios, full_slons, year)
    # for the convex hull call, I will need the set of meteors contained within the remaining voxels
    # saving these meteors to a new file is the next step

    print('\nBefore background subtraction:', len(active_lons), 'After background subtraction: ', len(diff_lons),'After 3 sigma test on voxel counts: ', high_count_meteors, 'After 3 sigma test on coordinates', len(final_lons)) # currently not changing in the coord_sigma test -> checking within 3 sigma which might be too high

    # echo_3d_plot(diff_lons, diff_lats, diff_vels, year, shower=shower_name, mode='shower')
    # echo_plot(final_lons, final_lats, final_vels, year, method, shower=shower_name, mode='shower', map_mode=map_mode)

    H_final, edges_final, final_lons2, final_lats2, final_vels2 = voxel_map(final_lons, final_lats, final_vels, year, name=shower_name, threshold=3)

    voxel_map2(H_final, edges_final, final_lons, final_lats, final_vels, threshold=3)
    print(f'After all checks, there are {len(final_lons2)} meteors that will be found within the convex hull (currently not accounting for set threshold)')
    voxel_with_hull(H_final, edges_final, final_lons2, final_lats2, final_vels2, threshold=3)