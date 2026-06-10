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
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib import patches
import sklearn as sk
from pathlib import Path


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
                                          "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}
                
                else:
                    # CASE: All four filters + Duplicate check applied
                    if method == 'all':
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_check != False and overlap != False) and del_int != False and del_radiant != False and del_stations != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, 
                                                    "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff, "Interferometry Error": int_error, "Solid Angle Error": solid_angle_error, 
                                                    "Station Measurement Error": sdel, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}

                    # # CASE: velocity check only
                    elif method == 'vel': # for 2025, 652,372 events are returned from AND condition below
                                          # for 2025, 758,183 events are returned from OR condition below
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if percent_check != False and overlap != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}
                        
                    # CASE: Interferometry check only
                    elif method == 'int':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_int != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Interferometry Error": int_error,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}                
                    
                    # CASE: Radiant Location Check only
                    elif method == 'angle':
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_radiant != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Solid Angle Error": solid_angle_error,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}                
                    
                    #  CASE: Station Measurement Check only
                    elif method == 'station':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_stations != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Station Measurement Error": sdel,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}
                    

                    # CASE: velocity and interferometry check only
                    elif method == 'vel and int':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_check != False and overlap != False) and del_int != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                "Interferometry Error": int_error, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}
                        
                    # CASE: velocity, interferometry and solid angle check only
                    elif method == 'vel and int and angle':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_check != False and overlap != False) and del_int != False and del_radiant != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                    "Interferometry Error": int_error, "Solid Angle Error": solid_angle_error, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                        "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}

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

    if vel1 >= 48:
        return True, True, percent_diff # 48 km/s is roughly in between the low and high velocity distributions
    # the check is currently if percent_diff is True AND <10. For these meteors it cannot be both

    # uncertainty overlap check

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
    if int(num_stations) > 4:

        if sdel[0] == '.': 
            return False # skip rows with missing station measurement data
        
        if float(sdel) <= 3: # station measurement error within 5 degrees is acceptable
            return True
        else:
            return False
    
    else:
        return False 


def voxel_map(lmda, beta, vels, year, map_mode='shower'):
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

    # avoid zero-range bins
    eps = 1e-6
    if lmda.max() == lmda.min(): lmda = lmda + np.linspace(0, eps, lmda.size)
    if beta.max() == beta.min(): beta = beta + np.linspace(0, eps, beta.size)
    if vels.max() == vels.min(): vels = vels + np.linspace(0, eps, vels.size)

    # 3D histogram into 8x8x8 voxels
    bins = (8, 8, 8)
    H, edges = np.histogramdd(np.vstack((lmda, beta, vels)).T, bins=bins) # count per voxel

    voxels = H > 0

    # print(voxels) # array of boolean values
    # print(H) # 3d array of bin counts, do subtraction with this

    cmap = cm.get_cmap('plasma')

    norm = plt.Normalize(vmin=H[voxels].min() if voxels.any() else 0,
                     vmax=H[voxels].max() if voxels.any() else 1)
    facecolors = np.zeros(voxels.shape + (4,), dtype=float)
    facecolors[voxels] = cmap(norm(H[voxels])) # maps counts to the voxels

    # build grid of voxel edges (shape (nx+1, ny+1, nz+1))
    X, Y, Z = np.meshgrid(edges[0], edges[1], edges[2], indexing='ij')

    # draw voxels in real-world coordinate space
    ax.voxels(X, Y, Z, voxels, facecolors=facecolors, edgecolor='k', alpha=0.9)

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
    # ax.set_xlim(-50,50)
    # ax.set_ylim(-30, 20)
    # ax.set_zlim(20,60)
    # maybe manually define limits of ecliptic coordinates for each shower to only include echoes in this region

    # ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
    ax.invert_xaxis()
    plt.grid()
    plt.show()

    # thinking of using this in two instances: once before the convex hull to show the area of interest, and once after to show the isolated meteor shower

    if map_mode == 'shower':

        return H, edges # edges contains three arrays of coordinates by bin (lon, lat, vel)

def background_matrix(lmda, beta, vel, slon, year, shower):

    '''
    This function will be used to generate a heat map of the sporadic meteors 5 days before and after a shower, 
    and will return a number density matrix of the days before, days after, and an average of the two to be subtracted from the matrix of the shower.
    The shower's matrix may be generated by this function following the background.
    '''

    # heat_map(lmda, beta, year, ) path?

def convex_hull(lmda, beta, vels, shower_lmda, shower_beta, shower_vels, slons, year):

    '''
    After creating a 3D voxel map of an isolated shower, the background flux is taken 5 days before and after the shower is active and a3 sigma test is done on each meteor's
    ecliptic coordinates and geocentric velocity.
    The first three parameters are for all ecliptic coordinates close to and within the active timeframe of the shower
    The following three parameters are for all ecliptic coordinates within the active timeframe of the showwer
    '''

    # next is to subtract the background from any showers; will need days/locations with shower data first
    
    # using these for a 3*std test to isolate shower only meteors
    lmda_std = np.std(shower_lmda)
    beta_std = np.std(shower_beta)
    vels_std = np.std(shower_vels)

    days_before = slons[0:5] # first five solar longitudes
    days_after = slons[-6:-1] # last five solar longitudes

    print(days_before, days_after)

    # next need to find a way to get the flux; might be a few steps ahead of where I am in the project though
    # will confirm and revisit in the future if needed

    mass_index = 1 # from MCB

    # need files of the before and after slons to take average number density matrix; how 'big' should this matrix be (dimension wise)







def heat_map(lmda, beta, year, path, method, month=None, meteor_source=None, shower_name=None): # include month mode at some point for labelling
    '''
    This function generates a heat map of the user specified orbit file, based on meteor counts per bin
    Month and source modes may be worked individually or simultaneously - in terms of saving distinct files of data
    Shower mode is best worked on its own - mainly using this mode to collect the number density matrices for days before/after shower activity
    '''
    figure, ax = plt.subplots(figsize=(10,5))

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

    ax.set_ylim(-60, 90)
    ax.set_xlim(-150, 150)
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
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from the {shower_name} in {year}')
        plt.savefig(f'{path}/{year}_{method}Filter_{shower_name}_radiantColorDist{binsize}.png')

        counts_file = os.path.join(counts_path, f"{method}-counts-{shower_name}-{year}-{binsize}-29.txt")

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


        if method == 'raw':
            clean_data.write(f'This file {filename} has {num_locs_init} detected events. No filtering methods were applied to reject events, this is the raw data that was seen.\n')
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')


        if method == 'vel':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Percent difference':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Percent difference']:>10.2f}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')
       

        elif method == 'int':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")

            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Int Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Interferometry Error']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')


        elif method == 'angle':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Radiant Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Solid Angle Error']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')


        elif method == 'station':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Station Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Station Measurement Error']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')


        elif method == 'vel and int':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Percent difference':>10} {'Int Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Percent difference']:>10.2f}")
                clean_data.write(f'{value['Interferometry Error']:>10} {value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')


        elif method == 'vel and int and angle':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Percent difference':>10} {'Int Error':>10} {'Radiant Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Percent difference']:>10.2f}")
                clean_data.write(f'{value['Interferometry Error']} {value['Solid Angle Error']:>10} {value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')


        elif method == 'all':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")

            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Percent difference':>10} {'Int Error':>10} {'Radiant Error':>10} {'Station Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")

            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Percent difference']:>10.2f} {value['Interferometry Error']:>10} {value['Solid Angle Error']:>10} {value['Station Measurement Error']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')

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

    loc_count = 0 # to keep track of files that contain defined coordinates
    keys_to_delete = [] # collect keys during iteration

    # v are nested dictionaries containing important info for each meteor
    for k, v in sorted(parent.items(), key=lambda kv: (kv[1]['date'], kv[1]['time'])):

        time = v['time']

        dist = v['R0']

        theta = v['Theta']
        phi = v['Phi']

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

    # Delete collected keys after iteration completes
    for k in keys_to_delete:
        del parent[k]

    # print(times)
    return latitudes, longitudes, ptn0_vels, del_ptn0_vels, geo_vels, keys_to_delete, loc_count, times, dists, angles # also returning the number of files with defined coordinates for more precise tracking purposes


def echo_plot(lmda, beta, vels, year, method, month=None, shower=None, source=None, mode='year', map_mode='scatter'):
    '''
    This function takes the ecliptic coordinates of clean echoes that satisfy set restrictions and maps them to a 2 dimensional grid representing a celestial 'sphere'
        A goal is to create elliptical figures, but currently rectangular until I figure out how to do that
    There are two modes:
        year - plots the yearly echo data and saves the figure to a directory outside of the txt files
        month - plots the echo data for each month and saves those figures in appropriate folders with the corresponding monthly txt files
    '''

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
            
            h = heat_map(lmda, beta, year, plot_folder, method, shower_name=shower)

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


def echo_3d_plot(lmda, beta, vels, year, month=None, shower=None, source=None, mode='month'):
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
        plt.show()

    elif mode == 'shower':

        plot_folder_3D = f'{home}/clean shower data/{year} {shower} clean events/figures 3D'
        os.makedirs(plot_folder_3D, exist_ok=True)

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
        plt.show()


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


def shower_parser(year, folder, file, slon, radiants, name, method, slon_status='active'):
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
    
    shower_lons = []
    shower_lats = []
    shower_vels = []

    # parses and saves file data of shower data
    if slon_status == 'active':

        active_folder = f'{sub_folder}/{2025} {name} active'
        os.makedirs(active_folder, exist_ok=True)

        active_path = os.path.join(active_folder, f'clean-{slon}-29.txt')


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
                    month = date[4:6] # extract month from date string

                    velg = params[11]

                    lmda = params[3]
                    beta = params[4]
                    slon = params[5]
                    alpha = params[16]
                    delta = params[17]
                    del_rad = params[18]

                    # unpacking the radiant of the shower
                    # solar_longitudes = slons[name] revisit if I want to access all solar longitudes in this function at some point
                    shower_alpha, shower_delta = radiants[name]

                    lmda, beta = float(lmda), float(beta)
                    alpha, delta = float(alpha), float(delta)
                    velg = float(velg)

                    # only plotting the echoes that fall within 10 degrees of the shower radiant
                    # will check if this value is a good range by next meeting
                    if abs(shower_alpha - alpha) <= 20 and abs (shower_delta - delta) <= 20: 
                        shower_lons.append(lmda)
                        shower_lats.append(beta)
                        shower_vels.append(velg)

                        active_file.write(f'{line}\n') # copies the file to a new directory of shower meteors

                        # counter here for header?
    
        return shower_lons, shower_lats, shower_vels

    # parses and saves file data from solar longitudes 5 days before and after the shower days
    elif slon_status == 'outer':
 
        outer_folder = f'{sub_folder}/{2025} {name} outer'
        os.makedirs(outer_folder, exist_ok=True)

        outer_path = os.path.join(outer_folder, f'clean-{slon}-29.txt')


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
                    month = date[4:6] # extract month from date string

                    velg = params[11]

                    lmda = params[3]
                    beta = params[4]
                    slon = params[5]
                    alpha = params[16]
                    delta = params[17]
                    del_rad = params[18]

                    # unpacking the radiant of the shower
                    # solar_longitudes = slons[name]
                    shower_alpha, shower_delta = radiants[name]

                    lmda, beta = float(lmda), float(beta)
                    alpha, delta = float(alpha), float(delta)
                    velg = float(velg)

                    # only plotting the echoes that fall within 10 degrees of the shower radiant
                    # will check if this value is a good range by next meeting
                    if abs(shower_alpha - alpha) <= 20 and abs (shower_delta - delta) <= 20: 
                        shower_lons.append(lmda)
                        shower_lats.append(beta)
                        shower_vels.append(velg)

                        # write file info here by copying each line of the file that satisfies the above condition

                        # if slon_status == 'active':

                        #     active_folder = f'{sub_folder}/{2025} {name} active'
                        #     os.makedirs(active_folder, exist_ok=True)

                        #     active_path = os.path.join(active_folder, f'clean-{date}-29.txt')

                        #     with open(active_path, 'a') as active_file: # using append mode to add to the file for each echo in that month
                                    
                        #         # active_file.write(f'{header}\n')
                        outer_file.write(f'{line}\n') # copies the file to a new directory of shower meteors

        return shower_lons, shower_lats, shower_vels


def background_subtract(year, outer_lmda, outer_beta, outer_vel, name, method, slon_status='active'):

    # instead of subtracting the exact background point from the shower data, maybe take any bin out with counts > 0 within a given radius of any nonzero bin from the background
    # should write a new function for this
    # would have to go over each line in the outer folder, grab its coordinates and store to a list
    # then go through each file in the active folder, and check if its coordinates fall within a range of the stored coordinates
    # if so, remove this line from the file its in
    # plot up the remaining

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

        # backup the original file before modifying
        try:
            shutil.copy2(file_path, file_path + '.bak')
        except Exception:
            pass

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
                    if abs(active_lmda - l) <= 5 and abs(active_beta - b) <= 5:
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
    

# Main function call

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

        # Parse function call
        parent_dict, num_clean = Parse(folder_name, file, method=method, sources=[source_isolate, source]) # num clean being written as pre duplicate number of events
        # print(len(parent_dict))
        
        # coordinate collection call
        lat, lon, vel_ptn0, del_vel_ptn0, vel_g, deleted_events, num_locs, times, dists, angles = grab_coords(parent_dict) # will be used to grab the ecliptic coordinates of the clean echoes for plotting; will do later
        # num locs is the number of events before duplicate correction below

        ecliptics.append([lat, lon])

        # lat and lon are lists themselves, so I am putting those elements into a bigger list for plotting
        plot_lons.extend(lon)
        plot_lats.extend(lat)
        plot_vels.extend(vel_g)

        vel_ptns.extend(vel_ptn0)
        d_vel_ptns.extend(del_vel_ptn0)

        num_without_coords += len(deleted_events)

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
    echo_plot(scaled_lons, plot_lats, plot_vels, year, method, source=source, mode='source', map_mode=map_mode)
    vel_histo(plot_vels, year, method, source=source, mode='source')

else:
    echo_plot(scaled_lons, plot_lats, plot_vels, year, method, map_mode=map_mode)
    vel_histo(plot_vels, year, method)

    

## SHOWER REMOVAL STEP ##

# Localizing and clearing out the meteor showers from the sporadic background

if shower_isolate:

    shower_folder_name = input('Enter the folder name of the clean shower data (format of the ending folder should be \'YYYY clean events\'): ').strip("'\"")


    shower_folder = sorted(os.listdir(os.path.join(home, shower_folder_name)))

    # shower_name = shower_folder[-1:-4] # should be last three letters of the user input
    # print(shower_name)

    shower_name = input('Enter the abbreviation of the shower you\'d like to work with - select from the abbreviations: ARI, DSX, ETA, GEM, QUA, SDA: ').upper().strip()
    # from here, the shower parser function should enter the clean file data folder and pick out the days the shower is active using specified solar longitude by a dictionary

    # list of important solar longitudes per shower from Brown et al. (2010)
    shower_slon0 = {'ARI' : np.arange(62,99), 'DSX': np.arange(174,197), 'ETA' : np.arange(30, 66),
                'GEM' : np.arange(240,273), 'QUA' : np.arange(232, 291), 'SDA' : np.arange(114, 164)}
    
    # list of important solar longitudes per strong shower from MCB with 5 day buffer both before and after the active shower days - Using this
    shower_slon = {'ARI' : np.arange(52, 110), 'DSX': np.arange(164, 208), 'ETA' : np.arange(20, 77),
                'GEM' : np.arange(230, 284), 'QUA' : np.arange(265, 302), 'SDA' : np.arange(104, 175)}
    
    # list of shower peak solar longitudes from MCB
    shower_slon_peaks = {'ARI' : 81, 'DSX': 186, 'ETA' : 45,
                'GEM' : 261, 'QUA' : 283, 'SDA' : 126}

    # ordered lists of geocentric right ascension (alpha) and declination (delta) from Brown et al. (2010) and MCB (same numbers) - Using this
    shower_rads = {'ARI' : [45.7, 25], 'DSX': [154.3, -1], 'ETA' : [337.9, -0.9],
                'GEM' : [112.5, 32.1],'QUA' : [231.5, 48.5], 'SDA' : [340.8, -16.3]}
   
    # get the solar longitudes of the showers from the filenames

    # only from the files on days of when the shower is active
    active_lons = []
    active_lats = []
    active_vels = []

    # only from the files 5 days before and after the shower is active; use this along with heat_map to generate a number density matrix for each day

    outer_lons = []
    outer_lats = []
    outer_vels = []

    # includes files of slon 5 days before and after the shower is active
    full_lons = []
    full_lats = []
    full_vels = []

    # array of solar longitudes of a shower based on user input
    input_slons = shower_slon[shower_name]
    min_slon, max_slon = min(input_slons), max(input_slons)

    slons_before = min_slon - 5
    slons_after = max_slon + 5

    outer_slons = list(range(slons_before, min_slon)) +  list(range(max_slon + 1, slons_after + 1))
    
    full_slons = list(range(slons_before, slons_after + 1)) # copy of the list to be used as one containing echoes 5 days before and after the shower appears

    print(input_slons)
    print(outer_slons)
    print(full_slons)

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
            
            # using these files to plot the echoes seen from the shower
            if file_slon in input_slons:
                
                shower_lmda, shower_beta, shower_vel = shower_parser(year, shower_folder_name, shower_file, file_slon, shower_rads, shower_name, method, slon_status='active')

                
                # collects coordinates of the shower files, along with velocities
                # using these lists to calculate flux, which will be used for shower subtraction
                active_lons.extend(shower_lmda)
                active_lats.extend(shower_beta)
                active_vels.extend(shower_vel)

                # also store the file in a new directory here
                # will also want a way to copy sporadics to another file, or remove the shower meteors from this file using the background subtraction and 3 sigma test
            
            # using these to construct average sporadic number density matrices to subtract from the shower's number density matrix (heat map)
            if file_slon in outer_slons:

                shower_lmda, shower_beta, shower_vel = shower_parser(year, shower_folder_name, shower_file, file_slon, shower_rads, shower_name, method, slon_status='outer')

                outer_lons.extend(shower_lmda)
                outer_lats.extend(shower_beta)
                outer_vels.extend(shower_vel)
                # also copy the file to a new shower directory here

            # storing both active and outer file data to these lists
            full_lons.extend(shower_lmda)
            full_lats.extend(shower_beta)
            full_vels.extend(shower_vel)
                
            # plots each day/solar longitude
            # echo_plot(shower_lmda, shower_beta, year)
        
        else:
            continue
    
    # scaling longitudes 
    scaled_active_lons = scale(active_lons)

    scaled_outer_lons = scale(outer_lons) # could split this list into two - one for days before, one for days after

    scaled_full_lons = scale(full_lons)

    # outer folder


    new_lons, new_lats, new_vel = background_subtract(year, outer_lons, outer_lats, outer_vels, shower_name, method, slon_status='active')

    # print(new_lons, new_lats, new_vel)

    scaled_new_lons = scale(new_lons)

    # plot of the active shower
    h_shower = echo_plot(scaled_active_lons, active_lats, active_vels, year, method, shower=shower_name, mode='shower', map_mode=map_mode)

    # plot of the sporadic background - take the background matrix from here
    h_background = echo_plot(scaled_outer_lons, outer_lats, outer_vels, year, method, shower=shower_name, mode='shower', map_mode=map_mode)
    # one density matrix for both 5 days before/after
    # two density matrices for 5 days before, 5 days after

    # plot of radiants after new subtraction method
    h_new = echo_plot(scaled_new_lons, new_lats, new_vel, year, method, shower=shower_name, mode='shower', map_mode=map_mode)

    print(h_shower[0], h_background[0])

    h_diff = h_shower[0] - h_background[0]
    h_diff = np.clip(h_diff, 0, None)

    h_lons, h_lats = h_shower[1], h_shower[2]

    print(h_diff) # run this in echo plot? needs to be in the function
    print(np.mean(h_diff)) # average is 0.029, check for negative numbers?

    plt.figure(figsize=(10,5))
    plt.imshow(h_diff.T, origin='lower', cmap='plasma', extent=[h_lons[0], h_lons[-1], h_lats[0], h_lats[-1]])

    plt.colorbar(label='Shower Count Difference')
    plt.xlabel('Ecliptic Longitude')
    plt.ylabel('Ecliptic Latitude')
    # plt.gca().invert_xaxis()
    
    plt.title('Shower minus background density')
    plt.show()

    # effectively getting the same plot as before subtraction
    # applying to 3d plot to see if there is a difference

    # 3d plot of shower day echoes - create the convex hull around radiants that survive the 3 sigma test following background subtraction
    echo_3d_plot(scaled_active_lons, active_lats, active_vels, year, shower=shower_name, mode='shower')

    # 3d plot of background day echoes
    echo_3d_plot(scaled_outer_lons, outer_lats, outer_vels, year, shower=shower_name, mode='shower')

    # next would be to plot active days with counts = (active counts - outer counts)

    # histogram of velocities
    # vel_histo(full_vels, year, method, shower=shower_name, mode='shower') 
    vel_histo(active_vels, year, method, shower=shower_name, mode='shower') 
    vel_histo(outer_vels, year, method, shower=shower_name, mode='shower') 

    # convex hull call here?

    H_shower = voxel_map(scaled_active_lons, active_lats, active_vels, year)
    H_background = voxel_map(scaled_outer_lons, outer_lats, outer_vels, year)


    # print(np.size(H_shower[0]), np.size(H_background[0]))

    H_diff = H_shower[0]  - H_background[0]

    # instead of subtracting the exact background point from the shower data, maybe take any bin out with counts > 0 within a given radius of any nonzero bin from the background
    # should write a new function for this
    # would have to go over each line in the outer folder, grab its coordinates and store to a list
    # then go through each file in the active folder, and check if its coordinates fall within a range of the stored coordinates
    # if so, remove this line from the file its in
    # plot up the remaining

    figure = plt.figure(figsize=(10,5))
    ax = figure.add_subplot(projection='3d')

    print(H_diff)

    threshold = 0

    new_voxels = H_diff > threshold # increase the threshold to make the plot a bit more strict to the shower

    normalized = (H_diff - H_diff.min()) / (H_diff.max() - H_diff.min())
    colors = plt.cm.plasma(normalized)

    ax.voxels(new_voxels, facecolors=colors, edgecolor='k', alpha=0.7)

    norm = plt.Normalize(vmin=H_diff.min(), vmax=H_diff.max())
    mappable = cm.ScalarMappable(norm=norm, cmap='plasma')
    figure.colorbar(mappable, ax=ax, shrink=0.6, label='Number of Meteors')

    ax.set_xlabel('Ecliptic Longitude')
    ax.set_ylabel('Ecliptic Latitude')
    ax.set_zlabel('Geocentric Velocity (km/s)')
    ax.set_title(f'Clean Meteor Sources scaled by geocentric velocity - measurements from {year}', fontsize=14)
    plt.show()
    

    background_matrix(outer_lons, outer_lats, outer_vels, outer_slons, year, shower_name)

    # convex_hull(scaled_full_lons, full_lats, full_vels, scaled_active_lons, active_lats, active_vels, full_slons, year)
    