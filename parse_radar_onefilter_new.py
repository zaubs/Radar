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
from matplotlib import cm
from matplotlib import patches
import sklearn as sk
from pathlib import Path


# This function is used to go through each meteor file for a chose year; the raw orbit data must exist in the file already
    # Might make this a class to do multiple tests at once for different filtering criteria; need to review classes first

# Global path which is used in several functions
home = Path.home() / 'Desktop/radar'


def Parse(folder, filename, method='all_checks'):

    '''
    Goes through the specified radar file and discards any events we deem to be too noisy
    This is based on the percent difference between the time of flight velocity and the pre-t0 velocity; 
        if they differ by more than 5% we discard the event as a noisy echo

    The clean echoes are stored in a parent dictionary that is keyed by the observed date and time, and contains the velocities, percent difference, and ecliptic coordinates for each event
    
    Being used on my MacBook Air, so the directories will have different names than the ones on my Linux Desktop Script
    '''

    # home= '/home/zaubs/Desktop/radar/'
    # home = Path.home() / 'Desktop/radar'

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
                corrected_lon = str(float(ecl_lon) - float(slon))
                
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
                percent_diff, overlap = vel_check(vel_TimeofFlight, vel_PTN0, del_vel_TimeofFlight, del_vel_PTN0, vel_geo) # using the function to check if the velocities agree within 5%
                # list of boolean value, and percentage

                del_int = int_check(int_error) # using the function to check if the interferometry error is less than 2 degrees

                del_radiant = solid_angle_check(solid_angle_error) # using the function to check if the radiant solid angle error is less than 5 degrees

                del_stations = station_check(num_stations, sdel) # using the function to check if the event has more than 4 station measurements with an error of less than 3 degrees


                # want to check what the numbers are for each criteria individually met, and for combinations for each criteria, then finally for all four criteria met
                # might create a few branches here to track each case
                
                # CASE: Raw data only
                if method == 'raw check':
                    parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo,
                                          "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}
                
                else:
                    # CASE: All four filters + Duplicate check applied
                    if method == 'all checks':
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_diff != False and percent_diff <= 10 and overlap != False) and del_int != False and del_radiant != False and del_stations != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, 
                                                    "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff, "Interferometry Error": int_error, "Solid Angle Error": solid_angle_error, 
                                                    "Station Measurement Error": sdel, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}

                    # # CASE: velocity check only
                    elif method == vel_check:
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if percent_diff != False and percent_diff <= 10 and overlap != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}
                        
                    # CASE: Interferometry check only
                    elif method == int_check:

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_int != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Interferometry Error": int_error,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}                
                    
                    # CASE: Radiant Location Check only
                    elif method == solid_angle_check:
                        
                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_radiant != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Solid Angle Error": solid_angle_error,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}                
                    
                    #  CASE: Station Measurement Check only
                    elif method == station_check:

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if del_stations != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Station Measurement Error": sdel,
                                                "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon,
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}
                    

                    # CASE: velocity and interferometry check only
                    elif method == 'vel_check and int_check':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_diff != False and percent_diff <= 10 and overlap != False) and del_int != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                "Interferometry Error": int_error, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                    "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}
                        
                    # CASE: velocity, interferometry and solid angle check only
                    elif method == 'vel_check and int_check and solid_angle_check':

                        # meteors satisfying this condition are marked as clean data and added to the parent dictionary
                        if (percent_diff != False and percent_diff <= 10 and overlap != False) and del_int != False and del_radiant != False:
                            parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                                    "Interferometry Error": int_error, "Solid Angle Error": solid_angle_error, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon, 
                                                        "Geocentric Right Ascension" : alpha_g, "Geocentric Declination" : delta_g, "Geocentric Radiant Uncertainty" : del_rad_g, 'Uncertainty in Pre-t0 velocity' : del_vel_PTN0}

        clean_data = len(parent_dict)

        # print(f'File {filename} has {clean_data} clean echoes')

    return parent_dict, clean_data # dictionary of clean echo data, number of clean files (with and without defined coordinates)


# Filtering methods below

def vel_check(vel1, vel2, dvel1, dvel2, velg):
    '''
    This function will be used to check if the time of flight velocity and the pre-t0 velocity agree to within 5%
    take vel m uncertainty too (velm - d_velm)
    '''
    if vel1[0] == '.' or vel2[0] == '.' or dvel1[0] == '.' or dvel2[0] == '.':
        return False, False # skip rows with missing speed data; keep return type consistent

    # changing string format to float numbers
    vel1, vel2, dvel1, dvel2 = float(vel1), float(vel2), float(dvel1), float(dvel2)
    velg = float(velg)

    # Should skip this step for any meteor with a velocity of 40 km/s or greater

    # if velg >= 40:
    #     return True, True # this should add the high speed echoes to the clean file data, assuming they satisfy the other restrictions

    if velg >= 48:
        return True, True # 48 km/s is roughly in between the low and high velocity distributions
   
    difference = vel1 - vel2
    average = (vel1 + vel2) / 2
    percent_diff = np.abs(difference / average) * 100

    # uncertainty overlap check

    lower1, upper1 = vel1 - dvel1, vel1 + dvel1
    lower2, upper2 = vel2 - dvel2, vel2 + dvel2
    overlap = max(lower1, lower2) <= min(upper1, upper2)
    
    return percent_diff, overlap


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


def convex_hull(lmda, beta, vels, year):
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

    # limits to where the gemenids are (reproducing Kipreos et al (2022)), no data seen here for me though; might have an issue with defining my coordinates in my scale or relabel function
    # ax.set_xlim(320, 340)
    # ax.set_ylim(-30, 0)

    ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
    ax.invert_xaxis()
    plt.grid()
    plt.show()

    # next is to subtract the background from any showers; will need days/locations with shower data first
    
    # using these for a 3*std test to isolate shower only meteors
    lmda_std = np.std(lmda)
    beta_std = np.std(beta)
    vels_std = np.std(vels)






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

def clean_echoes(parent, num_locs_init, date, folder, filename, num_dr, num_da, method=vel_check):

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

        if method == 'raw check':
            clean_data.write(f'This file {filename} has {num_locs_init} detected events. No filtering methods were applied to reject events, this is the raw data that was seen.\n')
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')

        if method == vel_check:
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Percent difference':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Percent difference']:>10.2f}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')
       
        elif method == int_check:
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")

            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Int Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Interferometry Error']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')

        elif method == solid_angle_check:
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Radiant Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Solid Angle Error']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')

        elif method == station_check:
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Station Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Station Measurement Error']:>10}")
                clean_data.write(f'{value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')

        elif method == 'vel_check and int_check':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Percent difference':>10} {'Int Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Percent difference']:>10.2f}")
                clean_data.write(f'{value['Interferometry Error']:>10} {value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')

        elif method == 'vel_check and int_check and solid_angle_check':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Num Stations':>12} {'Ecl Lambda':>10} {'Ecl Beta':>10} {'Solar Lambda':>10} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Percent difference':>10} {'Int Error':>10} {'Radiant Error':>10} {'alpha_g':>10} {'delta_g':>10} {'del_rad_g':>12}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Percent difference']:>10.2f}")
                clean_data.write(f'{value['Interferometry Error']} {value['Solid Angle Error']:>10} {value['Geocentric Right Ascension']:>10} {value['Geocentric Declination']:>10} {value['Geocentric Radiant Uncertainty']:>12}\n')


        elif method == 'all checks':
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
        lmda = long_transform(float(lmda)) # transforming longitude to 0-360 scale for plotting

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


# Supplimentary functions for plotting
def long_transform(lmda):

    '''
    This function will be used to transform the ecliptic longitude values to plot on a 0-360 degree scale
    '''

    if lmda < 0:
        return 360 + lmda
    else:
        return lmda

def echo_plot(lmda, beta, year, month=None, shower=None, mode='year'):
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

        figure, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(lmda, beta, s=10, color='blue', alpha=0.5)
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Ecliptic Longitude (Lambda)')
        ax.set_ylabel('Ecliptic Latitude (Beta)')
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')

        ax.set_ylim(-100, 100)
        ax.set_xlim(180, -180)
        ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        ax.invert_xaxis()
        plt.grid()

        plt.savefig(f'{plot_path}/{year}_radiantDist.png')
        plt.show()

    
    # monthly mode has a save option to the the clean file direcotry
    elif mode == 'month':

        plot_folder = f'{home}/clean file data/{year} clean events by month/figures'
        os.makedirs(plot_folder, exist_ok=True)

        figure, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(lmda, beta, s=10, alpha=0.5, color='blue')
        ax.set_xlabel('Ecliptic Longitude (Lambda)')
        ax.set_ylabel('Ecliptic Latitude (Beta)')
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {month}/{year}')
        ax.set_ylim(-100, 100)
        ax.set_xlim(180, -180)

        ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        ax.invert_xaxis()
        plt.grid()

        plt.savefig(f'{plot_folder}/{year}{month}_radiantDist.png') # save the plot to the same folder as the data for that month
        plt.show()

    elif mode == 'shower':

        plot_folder = f'{home}/clean shower data/{year} {shower} clean events/figures'
        os.makedirs(plot_folder, exist_ok=True)

        figure, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(lmda, beta, s=10, alpha=0.5, color='blue')
        ax.set_xlabel('Ecliptic Longitude (Lambda)')
        ax.set_ylabel('Ecliptic Latitude (Beta)')
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {year}')
        ax.set_ylim(-100, 100)
        ax.set_xlim(180, -180)

        ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        ax.invert_xaxis()
        plt.grid()

        plt.savefig(f'{plot_folder}/{year}{shower}_radiantDist.png') # save the plot to the same folder as the data for that month
        plt.show()

    # add a successive plot here; showing distribution after each successive filter applied


def echo_3d_plot(lmda, beta, vels, year, month=None, shower=None, mode='month'):
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


def vel_histo(vels, year, method, month=None, shower=None, mode='year'):
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
        print(n, bins, patches) # counts, mean vel per bin, object type? only worry about first two
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
        # case_vels_path = f'{home}/clean file data/0528_2/{method} events'
        # os.makedirs(case_vels_path, exist_ok=True)

        # vels_file = os.path.join(case_vels_path, f"{method}-velocities-{year}-29.txt")

        # with open(vels_file, 'w') as vel_counts:
        #     vel_counts.write('Velocities, Counts\n\n')
        #     for i in range(len(n)):
            
        #     # writing to file: Velocity bin value, counts in that bin
        #         vel_counts.write(f'{bins[i]} {n[i]}\n')

     
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

        data_path = f'{home}/clean file data/{year} clean events by month/{year} {month} clean echoes'
        data_file = os.path.join(data_path, f"FULL-{year}{month}-29.txt")

        with open(data_file, 'a') as vel_data: # should not exist yet for months, but working on it (change w to a when it does)
        
            vel_data.write(f'\nMean Velocity: {mean} km/s\n') # should include uncertanties at some point too
            vel_data.write(f'Median Velocity: {median} km/s\n')
            vel_data.write(f'Root Mean Square Velocity: {rms} km/s\n')
            vel_data.write(f'Standard Deviation: +/-{std, 2} km/s\n')
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

        # Writing the histogram data to a txt file
        data_path = f'{home}/clean shower data/{year} {shower} clean events'
        data_file = os.path.join(data_path, f"FULL-{year}{shower}-29.txt")

        with open(data_file, 'a') as vel_data:
        
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


def monthly_plotter(year, month, folder, file, method):
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
        echo_plot(scaled_longitudes, latitudes, year, month, mode='month')

        # 3d plot
        echo_3d_plot(scaled_longitudes, latitudes, velocities, year, month)

        # velocities histogram
        vel_histo(velocities, year, month, mode='month')
        
    
    except FileNotFoundError:
        print(f"No data found for {month}/{year}. Please check that the month and year are correct and that the data has been organized by month using the monthly_echoes function.")


def shower_parser(year, folder, file, slons, radiants, name, method):
    '''
    I'd want to take each file for a specific shower and make a lat/long plot for each to spot where the shower is
    once spotted, this area will be localized so that I can make the convex hull and take the shower out from sporadic data
    '''

    new_folder_name = f'{year} {name}'
    print(name, new_folder_name) # want this to be the directory name of the shower only meteors

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

    with open(path, 'r') as shower_data:
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
                    header += line
                    header += '\n'
                    # print(header)
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
                solar_longitudes = slons[name]
                shower_alpha, shower_delta = radiants[name]

                lmda, beta = float(lmda), float(beta)
                alpha, delta = float(alpha), float(delta)
                velg = float(velg)

                # only plotting the echoes that fall within 10 degrees of the shower radiant
                # will check if this value is a good range by next meeting
                if abs(shower_alpha - alpha) <= 25 and abs (shower_delta - delta) <= 25: 
                    shower_lons.append(lmda)
                    shower_lats.append(beta)
                    shower_vels.append(velg)
    
    return shower_lons, shower_lats, shower_vels
                
                


def scale(x):
    '''
    This function scales the x axis to be centered at 270 degrees longitude
    '''
    x = np.asarray(x) % 360
    res = (x - 270) % 360

    return np.where(res > 180, res - 360, res)


def relabel(x, pos):
    pass
    '''
    This function will relabel the x axis to have labels of 90 degrees, goig down to zero, then going from 359 down to 91 degrees
    '''
    if x == 0:
        return '270°'
    elif x == -90:
        return '0°'
    elif x == -180:
        return '90°'
    elif x == 90:
        return '180°'
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

# filter method currently being used - change for testing

# method = 'raw check'
method = 'all checks'
# method = vel_check
# method = 'vel_check and int_check'
# method = 'vel_check and int_check and solid_angle_check'

# Source locations by ecliptic coordinates
helion = []
antihelion = []
north_apex = []
south_apex = []
north_toroidal = []

if len(folder) == 0:
    print('Your folder is empty! Please check the folder name and try again.')
else:
    # goes through each event file in the folder
    for file in folder:

        orb_date = file[4:12] # year-solar longitude

        # Parse function call
        parent_dict, num_clean = Parse(folder_name, file, method='all checks') # num clean being written as pre duplicate number of events
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
        if method.lower().strip() == 'raw check':

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
message1 = f'\n\nFULL YEAR DATA:\n\nThe total number of clean echoes across all files is {num_echoes}, with {num_echo_locs} events observed at distinct times and that have defined ecliptic coordinate and can be plotted.\n'
message2 = f'For all events within 10 milliseconds of eachother, {delranges} events were deleted for being within 6km of each other, {delangles} events were deleted for having zenith/azimuth angles within 5 degrees.\n'
message3 = f'There were {num_without_coords} events deleted for not having a defined set of ecliptic coordinates.\n'

message = message1+message2+message3

print(message)

main_path = f'{home}/clean file data/{year} clean events'

write_path = os.path.join(main_path, f"FULL-{year}-29.txt")

with open(write_path, 'w') as full_data:
    full_data.write(message)


scaled_lons = scale(plot_lons) # scaling longitudes to be centered at 270 degrees


# creating a map of the clean meteor sources based on their coordinates of ecliptic latitude and longitude
echo_plot(scaled_lons, plot_lats, year)



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
# convex_hull(scaled_lons, plot_lats, plot_vels, year)

# velocity histogram binning; want to do this for raw data, then once again for the filtered data; currently doing it with all filtering applied
vel_histo(plot_vels, year, method)

## MONTHLY ORGANIZATION STEP ##

# here, the code organizes the file data by month; might take away the option for user input and do this automatically
# clean_folder_name = input('Enter the folder name of clean echo data you wish to organize by month (or drag folder here): ').strip("'\"")  # Strip quotes that may be included when dragging from file explorer

# clean_folder = sorted(os.listdir(os.path.join(clean_folder_name)))


# for clean_file in clean_folder:
#     # print(clean_file)

#     monthly_echoes(year, clean_folder_name, clean_file)


#     # next step is to write a function that will go through the monthly organized files and plot the coordinates for each month to compare meteor sources across months; will do later

# for m in range(1, 13):
#     # plot the coordinates for each month

#     monthly_plotter(year, m, clean_folder_name, clean_file, method)
    

## SHOWER REMOVAL STEP ##

# Localizing and clearing out the meteor showers from the sporadic background


shower_folder_name = input('Enter the folder name of the clean shower data (format of the ending folder should be \'YYYY clean events\'): ').strip("'\"")


shower_folder = sorted(os.listdir(os.path.join(home, shower_folder_name)))

# shower_name = shower_folder[-1:-4] # should be last three letters of the user input
# print(shower_name)

shower_name = input('Enter the abbreviation of the shower you\'d like to work with - select from the abbreviations: ARI, DSX, ETA, GEM, QUA, SDA: ').upper().strip()
# from here, the shower parser function should enter the clean file data folder and pick out the days the shower is active using specified solar longitude by a dictionary

# list of important solar longitudes per shower
shower_slon = {'ARI' : np.arange(62,99), 'DSX': np.arange(174,-197), 'ETA' : np.arange(30, 66),
               'GEM' : np.arange(240,273), 'QUA' : np.arange(232, 291), 'SDA' : np.arange(114, 164)}

# ordered lists of geocentric right ascension (alpha) and declination (delta)
shower_rads = {'ARI' : [45.7, 25], 'DSX': [154.3, -1], 'ETA' : [337.9, -0.9],
               'GEM' : [112.5, 32.1],'QUA' : [231.5, 48.5], 'SDA' : [340.8, -16.3]}

# get the solar longitudes of the showers from the filenames

full_lons = []
full_lats = []
full_vels = []

# array of solar longitudes of a shower based on user input
input_slons = shower_slon[shower_name]
print(input_slons)

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
    if file_slon in input_slons:
        
        # print(shower_file)
        shower_lmda, shower_beta, shower_vel = shower_parser(year, shower_folder_name, shower_file, shower_slon, shower_rads, shower_name, method)

        full_lons.extend(shower_lmda)
        full_lats.extend(shower_beta)
        full_vels.extend(shower_vel)

        # plots each day/solar longitude
        # echo_plot(shower_lmda, shower_beta, year)
    
    else:
        continue

scaled_full_lons = scale(full_lons)
print(full_lons, full_lats)
# plots full shower duration
echo_plot(scaled_full_lons, full_lats, year, shower=shower_name, mode='shower')

echo_3d_plot(scaled_full_lons, full_lats, full_vels, year, shower=shower_name, mode='shower')

vel_histo(full_vels, year, method, shower=shower_name, mode='shower')

# convex hull call here?