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
from matplotlib import patches
import sklearn as sk


# This function is used to go through each meteor file for a chose year; the raw orbit data must exist in the file already
    # Might make this a class to do multiple tests at once for different filtering criteria; need to review classes first


def Parse(folder, filename):

    '''
    Goes through the specified radar file and discards any events we deem to be too noisy
    This is based on the percent difference between the time of flight velocity and the pre-t0 velocity; 
        if they differ by more than 5% we discard the event as a noisy echo

    The clean echoes are stored in a parent dictionary that is keyed by the observed date and time, and contains the velocities, percent difference, and ecliptic coordinates for each event
    
    Being used on my MacBook Air, so the directories will have different names than the ones on my Linux Desktop Script
    '''

    home= '/Users/aubz/Desktop/radar/'

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

                # interferometry error
                int_error = params[15]

                # radiant solid angle error
                solid_angle_error = params[19]

                # number of stations
                num_stations = params[2]

                # station measurement error
                sdel = params[123]
                
                # filtering function calls
                percent_diff, overlap = vel_check(vel_TimeofFlight, vel_PTN0, del_vel_TimeofFlight, del_vel_PTN0) # using the function to check if the velocities agree within 5%
                # list of boolean value, and percentage

                del_int = int_check(int_error) # using the function to check if the interferometry error is less than 2 degrees

                del_radiant = solid_angle_check(solid_angle_error) # using the function to check if the radiant solid angle error is less than 5 degrees

                del_stations = station_check(num_stations, sdel) # using the function to check if the event has more than 4 station measurements with an error of less than 3 degrees


                # want to check what the numbers are for each criteria individually met, and for combinations for each criteria, then finally for all four criteria met
                # might create a few branches here to track each case

               
                # CASE: All four filters + Duplicate check applied
                # if (percent_diff != False and percent_diff <= 10 and overlap != False) and del_int != False and del_radiant != False and del_stations != False:
                #     parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, 
                #                               "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff, "Interferometry Error": int_error, "Solid Angle Error": solid_angle_error, 
                #                               "Station Measurement Error": sdel, "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon}

                # CASE: velocity check only
                if percent_diff != False and percent_diff <= 1 and overlap != False:
                    # print("Clean echo detected at", date, time)
                    # add to parent dictionary here; will do later
                    parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Geocentric velocity" : vel_geo, "Percent difference": percent_diff,
                                          "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon}
                
                # CASE: Interferometry check only
                # if del_int != False:
                #     parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Interferometry Error": int_error,
                #                           "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon}
                
                # # CASE: Radiant Location Check only
                # if del_radiant != False:
                #     parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Solid Angle Error": solid_angle_error,
                #                           "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon}
                
                #  CASE: Station Measurement Check only
                # if del_stations != False:
                #     parent_dict[date+time] = {"date": date, "time": time, "Number of Stations" : num_stations, "R0" : R0, "Theta" : theta, "Phi" : phi, "Time of flight velocity": vel_TimeofFlight, "Pre-t0 velocity": vel_PTN0, "Station Measurement Error": sdel,
                #                           "Ecliptic longitude": corrected_lon, "Ecliptic latitude": ecl_lat, "Solar longitude": slon}
                
                else:
                    # print("Discarded echo at", date, time, "due to speed disagreement")
                    continue
        
        clean_data = len(parent_dict)

        # print(f'File {filename} has {clean_data} clean echoes')

    return parent_dict, clean_data # dictionary of clean echo data, number of clean files (with and without defined coordinates)


# Filtering methods below

def vel_check(vel1, vel2, dvel1, dvel2):
    '''
    This function will be used to check if the time of flight velocity and the pre-t0 velocity agree to within 5%
    take vel m uncertainty too (velm - d_velm)
    '''
    if vel1[0] == '.' or vel2[0] == '.' or dvel1[0] == '.' or dvel2[0] == '.':
        return False, False # skip rows with missing speed data; keep return type consistent

    # changing string format to float numbers
    vel1, vel2, dvel1, dvel2 = float(vel1), float(vel2), float(dvel1), float(dvel2)
   
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

def convex_hull(vel_geo, file):
    '''
    put meteor distribution into a voxel (cubic space)
    voxels defined by buffer (5 degrees solar longitude)
        subtraction (5/10 solar longitude?)
    convex hull created as a boundary around meteors left after subtraction
    '''

    # Should perform this for a single file of orbit data (echo data for a single solar longitude - a day)

    # Define 3d plot of ecliptic latitude, longitude and geocentric velocities

    # 

    pass 


def duplicate(parent, times, dists, angles, dr_count, da_count):
    '''
    This function goes through all stored clean events, and checks if any two events have similar times (within 0.01 seconds from eachother)
    This will have to be called following the Parse function, but before the clean_echoes call, since we want the clean data to be written to our files
    '''

    dr = 0
    da = 0

    # creating a copy of the dictionary and updating this with non-duplicate events. Using the old parent dictionary to iterate through events
    parent_new = parent.copy()

    # # runs if the number of clean events seen are odd
    # if len(times) % 2 == 1:
    #     for i, key in enumerate(parent): 
    #     # in range(0, len(times)-1, 2): # each iteration goes through two events stored in the parent dictionary
    #         # small issue: skips the last event, so will not know if this is a duplicate or not unless we look at the file

    #         time1 = times[i].strip()
    #         time2 = times[i+1].strip()

    #         hour1 = time1[0:2]
    #         hour2 = time2[0:2]

    #         minutes1 = time1[3:5]
    #         minutes2 = time2[3:5]

    #         seconds1 = float(time1[6:12])
    #         seconds2 = float(time2[6:12])

    #         # same hour
    #         if hour1 == hour2:
    #             # same minute
    #             if minutes1 == minutes2:

    #                 # we assume this is close enough for the two events to be a duplicate
    #                 if abs(seconds1 - seconds2) < 0.010:
    #                     del parent_new[key] # removes one of the events from our dictionary
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
    sub_folder = f'/Users/aubz/Desktop/radar/clean file data/{date[0:4]} clean events'
    os.makedirs(sub_folder, exist_ok=True)

    path = os.path.join(sub_folder, f"clean-{date}-29.txt") # 29 MHz is the frequency used by CMOR

    num_clean = len(parent) # number of clean echoes that satisfy the set velocity condition in Parse

    # print(f'File {filename} has {num_clean} clean echoes, {num_locs} of which have defined coordinates that can be plotted')
    # will add to this line later the more filtering I include, meaning I need to return more objects from my functions above

    with open(path, "w") as clean_data:
        if method == vel_check:
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr, ")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Number of Stations':>12} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'Percent difference':>10} {'Ecliptic Lambda':>10} {'Ecliptic Beta':>10} {'Solar Longitude':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Percent difference']:>10.2f} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10}\n")
       
        elif method == int_check:
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")

            clean_data.write(f"{'Date':>12} {'Time':>8} {'Number of Stations':>12} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'Interferometry Error':>10} {'Ecliptic Lambda':>10} {'Ecliptic Beta':>10} {'Solar Longitude':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Interferometry Error']:>10} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10}\n")

        elif method == solid_angle_check:
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Number of Stations':>12} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'Solid Angle Error':>10} {'Ecliptic Lambda':>10} {'Ecliptic Beta':>10} {'Solar Longitude':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Solid Angle Error']:>10} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10}\n")

        elif method == station_check:
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")
            
            clean_data.write(f"{'Date':>12} {'Time':>8} {'Number of Stations':>12} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'Station Measurement Error':>10} {'Ecliptic Lambda':>10} {'Ecliptic Beta':>10} {'Solar Longitude':>10}\n\n")
            
            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Station Measurement Error']:>10} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10}\n")

        elif method == 'all checks':
            clean_data.write(f"This file {filename} has {num_locs_init} detected events that satisfy the restrictions set for what a clean echo is. There are {num_clean} events these have a defined set of ecliptic coordinates.\n\n")
            clean_data.write(f"For events within 10 milliseconds of eachother, {num_dr} events have been removed from being within 6km of each other relative to the Zehr")
            clean_data.write(f"and {num_da} events have been removed for being within 5 degrees from their zenithal and azimuthal positions in the sky.\n\n")
            clean_data.write(f"There are {num_locs_init - num_clean} events that were removed for not having a defined set of ecliptic coordinates.\n\n")

            clean_data.write(f"{'Date':>12} {'Time':>8} {'Number of Stations':>12} {'R0':>8} {'Theta':>8} {'Phi':>8} {'vel_m':>10} {'vel_ptn0':>10} {'vel_geo':>10} {'Percent difference':>10} {'Interferometry Error':>10} {'Solid Angle Error':>10} {'Station Measurement Error':>10} {'Ecliptic Lambda':>10} {'Ecliptic Beta':>10} {'Solar Longitude':>10}\n\n")

            for key, value in parent.items():
                clean_data.write(f"{value['date']:>12} {value['time']:>8} {value['Number of Stations']:>12} {value['R0']:>8} {value['Theta']:>8} {value['Phi']:>8} {value['Time of flight velocity']:>10} {value['Pre-t0 velocity']:>10} {value['Geocentric velocity']:>10} {value['Percent difference']:>10.2f} {value['Interferometry Error']:>10} {value['Solid Angle Error']:>10} {value['Station Measurement Error']:>10} {value['Ecliptic longitude']:>10} {value['Ecliptic latitude']:>10} {value['Solar longitude']:>10}\n")


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

    # lists for the function duplicate; checking for overlapping events
    times = []
    dists = []
    angles = []

    loc_count = 0 # to keep track of files that contain defined coordinates
    keys_to_delete = [] # collect keys during iteration

    # v are nested dictionaries containing important info for each meteor
    for k, v in parent.items():

        time = v['time']

        dist = v['R0']

        theta = v['Theta']
        phi = v['Phi']

        times.append(time)
        dists.append(dist)
        angles.append([theta, phi])


        beta = v['Ecliptic latitude']
        lmda = v['Ecliptic longitude'] # some longitude coordinates are negative, we want them from 0-360 for plotting

        vel_g = v['Geocentric velocity']

        # put this section after the duplicate function, giving a number of filtered events with undefined long/lat
        if beta == '0.00' or lmda == '0.00':
                keys_to_delete.append(k)
                # print('deleted for having undefined coordinates')
                continue # skip rows with missing speed data; does not seem to change anything, as any file without these coordinates already lacks pre-to velocity

        loc_count += 1
        lmda = long_transform(float(lmda)) # transforming longitude to 0-360 scale for plotting

        # Using these three for 3d plots
        latitudes.append(float(beta))
        longitudes.append(float(lmda))
        geo_vels.append(float(vel_g))

    # Delete collected keys after iteration completes
    for k in keys_to_delete:
        del parent[k]

    # print(times)
    return latitudes, longitudes, geo_vels, keys_to_delete, loc_count, times, dists, angles # also returning the number of files with defined coordinates for more precise tracking purposes


# Supplimentary functions for plotting
def long_transform(lmda):

    '''
    This function will be used to transform the ecliptic longitude values to plot on a 0-360 degree scale
    '''

    if lmda < 0:
        return 360 + lmda
    else:
        return lmda
    

def overplot():
    '''

    This function will get the ecliptic coordinates across specified years and plot them on the same graph to compare the meteor sources across years

    Does not work as intended yet, trying something else for the time being

    '''


    input_years = input("Enter folder names from different years to compare (or drag folders here, separated by commas): ").strip("'\"").split(",")  # Strip quotes and split by commas

    plt.figure, ax = plt.subplots(figsize=(10, 5))

    # checks if the user input is empty
    if input_years:
        
        # going through each year
        for year in input_years:
            folder_name = year.strip()  # Remove any leading/trailing whitespace
            # print(year, folder_name)
            folder = os.listdir(os.path.join(folder_name))

            # going through each file in the folder for that year
            for file in folder:

                parent_dict, num_clean = Parse(folder_name, file)

                lat, lon, num_locs = grab_coords(parent_dict)

                ax.scatter(lon, lat, s=10, alpha=0.5, label=f'{year.strip()}')
        
        ax.set_xlabel('Ecliptic Longitude (Lambda)')
        ax.set_ylabel('Ecliptic Latitude (Beta)')
        ax.set_title('Clean Meteor Sources observed in Ecliptic Coordinates - measurements from multiple years')
        ax.set_ylim(-90, 90)
        ax.set_xlim(0, 360)
        plt.grid()
        plt.legend()()
        plt.show()
    else:
        print("No valid years entered. Please try again with valid folder names.")


# Essentially another Parse function that organizes clean echo data by month for a given year; requires clean echo data to exist first
    # should integrate this with Parse in a new file
def monthly_echoes(year, folder_name, file):
    '''
    This will be used to organize the files for a given year by month in new folders within that year's clean events folder
    The folder will be the one put in by the user, and it's year will point to the location of its associated clean folder 
    This function call will be following the execution of clean_echoes so the clean data will be filed as needed
    '''
    
    # pointing to location of the folder the user entered
    home= '/Users/aubz/Desktop/radar/'
    path = os.path.join(home, folder_name, file)
    # print(os.path.exists(path)) # True if the directory exists
    # print(path)
    # contains all txt files; I want to organize them by month, which is part of their first entry

    # creating new folder to store monthly organized data
    sub_folder = f'/Users/aubz/Desktop/radar/clean file data/{year} clean events by month'
    os.makedirs(sub_folder, exist_ok=True)

    # longitudes = []
    # latitudes = []

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
                    continue

                date = params[0]
                month = date[4:6] # extract month from date string

                lmda = params[13] # ecliptic longitude
                beta = params[14] # ecliptic latitude

                # longitudes.append(lmda)
                # latitudes.append(beta)

                # monthly folder nested in the yearly folder
                month_folder = f'/Users/aubz/Desktop/radar/clean file data/{year} clean events by month/{year} {month} clean echoes'

                os.makedirs(month_folder, exist_ok=True)

                month_path = os.path.join(month_folder, f'clean-{date}-29.txt')

                with open(month_path, 'w') as month_file: # using append mode to add to the file for each echo in that month

                    month_file.write(clean_data.read()) # write the clean echo data for that month to the new file; works but need to check that it's writing the correct data to the correct file; will do later


def monthly_plotter(year, month, folder, file):
    '''
    This will be used to create plots for each month of data for a specified year by the user
    '''

    home = '/Users/aubz/Desktop/radar/clean file data/'
    if month < 10:
        month = f'0{month}' # add leading zero to month for file path  

    try:
        
        clean_month_name = f'{home}/{year} clean events by month/{year} {month} clean echoes'
        # print(os.path.exists(clean_month_name)) # True if the directory exists

        month_files = os.listdir(clean_month_name)
        # print(os.path.exists(clean_month_name)) # True if the directory exists
        # print(month_files)

        longitudes = []
        latitudes = []

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
                            
                        
                        date = params[0]
                        year = date[0:4]

                        # since these are list indices, anytime I add to the txt files that get saved I will have to go back and see if I have to change their indices here 
                        lmda = params[13] # ecliptic longitude, corrected for solar longitude in Parse
                        beta = params[14] # ecliptic latitude

                        # print(lmda, beta)

                        longitudes.append(float(lmda))
                        latitudes.append(float(beta))

        # print(longitudes)
        scaled_longitudes = scale(longitudes) # scaling longitudes to be centered at 270 degrees
        
        figure, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(scaled_longitudes, latitudes, s=10, alpha=0.5)
        ax.set_xlabel('Ecliptic Longitude (Lambda)')
        ax.set_ylabel('Ecliptic Latitude (Beta)')
        ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements from {month}/{year}')
        ax.set_ylim(-100, 100)
        ax.set_xlim(180, -180)

        ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
        ax.invert_xaxis()
        plt.grid()

        plt.savefig(f'{home}/{year} clean events by month/{year}{month}_radiantDist.png') # save the plot to the same folder as the data for that month
        plt.show()

        
    
    except FileNotFoundError:
        print(f"No data found for {month}/{year}. Please check that the month and year are correct and that the data has been organized by month using the monthly_echoes function.")
        

def scale(x):
    '''
    This function scales the x axis to be centered at 270 degrees longitude
    '''
    x = np.asarray(x) % 360
    res = (x - 270) % 360

    return np.where(res > 180, res - 360, res)


def relabel(x, pos):
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


if len(folder) == 0:
    print('Your folder is empty! Please check the folder name and try again.')
else:
    # goes through each event file in the folder
    for file in folder:

        orb_date = file[4:12] # year-solar longitude

        # Parse function call
        parent_dict, num_clean = Parse(folder_name, file) # num clean being written as pre duplicate number of events
        # print(len(parent_dict))
        
        # coordinate collection call
        lat, lon, vel_g, deleted_events, num_locs, times, dists, angles = grab_coords(parent_dict) # will be used to grab the ecliptic coordinates of the clean echoes for plotting; will do later
        # num locs is the number of events before duplicate correction below

        ecliptics.append([lat, lon])

        # lat and lon are lists themselves, so I am putting those elements into a bigger list for plotting
        plot_lons.extend(lon)
        plot_lats.extend(lat)

        num_without_coords += len(deleted_events)

        # print('nums', num_locs, num_clean)
        # function call to save located clean echo data
        num_echoes += num_clean # adding on to counter to track number of clean echoes per year; might not need this
        
        # removing any meteors we deem duplicates (check duplicate function for criteria)
        parent_new, dr, da = duplicate(parent_dict, times, dists, angles, delranges, delangles)

        delranges += dr
        delangles += da

        if dr != 0 or da != 0:
            print(f'For events within 10 milliseconds of eachother, {delranges} events were deleted for being within 6km of each other, {delangles} events were deleted for having zenith/azimuth angles within 5 degrees')

        
        # print(len(parent_dict), len(parent_new)) # same as num locs, the second number is number of events after duplicate check

        # running the writing function to only include data for clean echoes with a defined set of coordinates
        num_echo_locs += clean_echoes(parent_new, num_clean, orb_date, folder_name, file, dr, da, method=vel_check)

print('\n\nFULL YEAR DATA:\n')

print(f'The total number of clean echoes across all files is {num_echoes}, with {num_echo_locs} events observed at distinct times and that have defined ecliptic coordinate and can be plotted.')

print(f'For all events within 10 milliseconds of eachother, {delranges} events were deleted for being within 6km of each other, {delangles} events were deleted for having zenith/azimuth angles within 5 degrees')
print(f'There were {num_without_coords} events deleted for not having a defined set of ecliptic coordinates.\n')
# this might be working: trying on other years

# Putting coordinate check here so the dictionary size does not change any more than the duplicate function can currently handle
# duplicate is being called in the for loop above, so if an event were to be deleted within the same for loop, then that influences how duplicate works since it runs based on the length of the events dictionary

# this is no longer needed
# num_with_coords = 0

# for sets in ecliptics:

#     beta = sets[0]
#     lmda = sets[1]

#     if beta != '0.00' and lmda != '0.00':
#         num_with_coords += 1
    
# print(f'After filtering, there are {num_with_coords} events with a defined set of ecliptic coordinates of which can be plotted.')



# creating a map of the clean meteor sources based on their coordinates of ecliptic latitude and longitude

year = orb_date[0:4] # this is only valid if the data set being ran is within the same calendar year; need something more universal eventually 

scaled_lons = scale(plot_lons) # scaling longitudes to be centered at 270 degrees

figure, ax = plt.subplots(figsize=(10, 5))
ax.scatter(scaled_lons, plot_lats, s=10, color='blue', alpha=0.5)
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
plt.show()


# 3d plot
figure = plt.figure()
ax = figure.add_subplot(projection='3d')

ax.scatter(scaled_lons, plot_lats, vel_g, marker='o')
ax.set_xlabel('Ecliptic Longitude')
ax.set_ylabel('Ecliptic Latitude')
ax.set_zlabel('Geocentric Velocity')

# limits to where the gemenids are (reproducing Kipreos et al (2022)), no data seen here for me though; might have an issue with defining my coordinates in my scale or relabel function
# ax.set_xlim(320, 340)
# ax.set_ylim(-30, 0)

ax.xaxis.set_major_locator(plt.FixedLocator([-90, 0, 90, 180, 270, 360])) # defines tick marks on the x axis
ax.xaxis.set_major_formatter(plt.FuncFormatter(relabel)) # re labels the x axis tick marks to show we are labeled at 270 degrees
ax.invert_xaxis()
plt.grid()
plt.show()


# attempting to plot using an ellipse shaped figure


# calling the overplot function here
# overplot()

clean_folder_name = input('Enter the folder name of clean echo data you wish to organize by month (or drag folder here): ').strip("'\"")  # Strip quotes that may be included when dragging from file explorer

clean_folder = os.listdir(os.path.join(clean_folder_name))

for clean_file in clean_folder:
    # print(clean_file)

    monthly_echoes(year, clean_folder_name, clean_file)

    # next step is to write a function that will go through the monthly organized files and plot the coordinates for each month to compare meteor sources across months; will do later

for m in range(1, 13):
    # plot the coordinates for each month
    
    monthly_plotter(year, m, clean_folder_name, clean_file)
    
