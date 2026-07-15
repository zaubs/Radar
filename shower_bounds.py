'''
This file contains a class with stored information of meteor shower radiant spaces for the years 2011-2025
Each year may or may not have different bounds on their radiant spaces, and might include some showers that other years do not have
Cel2Hel calculation between celestial and heliocentric coordinates will also be called here so such radiant spaces can be defined using the same script
Those radiant spaces can then be called in parse_radar and can be used to identify shower data
'''

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from cel2hel2cel import * # celestial to heliocentric script from MCB
# from parse_radar_MAC import shower_slon_peaks, shower_rads, shower_mean_velocities # loop to convert each shower's heliocentric coordintates

# IMPORTANT DICTIONARIES #

# list of important solar longitudes per strong shower from MCB with 5 day buffer both before and after the active shower days - Using this
shower_slon = {'ARI' : np.arange(52, 110), 'DSX': np.arange(164, 208), 'ETA' : np.arange(20, 77),
            'GEM' : np.arange(230, 284), 'ORI' : np.arange(188, 238), 'QUA' : np.arange(265, 302), 'SDA' : np.arange(104, 175)}
# It might be worth including some other showers in here to see if there are any stronger ones in the distribution not mentioned before

# list of shower peak solar longitudes from MCB
shower_slon_peaks = {'ARI' : 81, 'DSX': 186, 'ETA' : 45,
            'GEM' : 261, 'ORI' : 208, 'QUA' : 283, 'SDA' : 126}

# ordered lists of geocentric right ascension (alpha) and declination (delta) from Brown et al. (2010) and MCB (same numbers) - Using this
shower_rads = {'ARI' : [45.7, 25], 'DSX': [154.3, -1], 'ETA' : [337.9, -0.9],
            'GEM' : [112.5, 32.1], 'ORI' : [95.5, 15.2], 'QUA' : [231.5, 48.5], 'SDA' : [340.8, -16.3]}

shower_mean_velocities = {'ARI' : 35, 'DSX' : 35, 'ETA' : 50,
                        'GEM' : 35, 'ORI' : 50, 
                        'QUA' : 35, 'SDA' : 35}

### YEARLY RADIANT SPACE INFO ###

# radii of longitude and latitude that showers fit in; some shower's need larger enclosed regions (quadrandids might be one of them)

shower_boundaries = {
                "all"  : {'ARI' : [[8, 8], [5, 7], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[8, 5], [5, 7], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[8, 5], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 6], [6, 6], [10, 10]]},
                   
                "2011" : {'ARI' : [[8, 8], [5, 7], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[8, 5], [5, 7], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[8, 5], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 6], [6, 6], [10, 10]]},
                   
                "2012" : {'ARI' : [[8, 8], [5, 7], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[8, 5], [5, 7], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[8, 5], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 6], [6, 6], [10, 10]]},
                   
                "2013" : {'ARI' : [[8, 8], [5, 7], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[8, 5], [5, 7], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[8, 5], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 6], [6, 6], [10, 10]]},
                   
                "2014" : {'ARI' : [[8, 8], [5, 7], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[7, 5], [5, 5], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[8, 5], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 6], [6, 6], [10, 10]]},
                   
                "2015" : {'ARI' : [[10, 10], [7, 10], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[7, 7], [5, 5], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[8, 5], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 8], [6, 6], [10, 10]]},
                   
                "2016" : {'ARI' : [[10, 10], [7, 10], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[10, 10], [6, 8], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[10, 5], [6, 6], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 8], [6, 6], [10, 10]]},
                   
                "2017" : {'ARI' : [[10, 10], [7, 10], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[10, 10], [6, 8], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[10, 5], [6, 6], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 8], [6, 6], [10, 10]]},
                    
                "2018" : {'ARI' : [[10, 10], [7, 10], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[10, 10], [6, 8], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[7, 6], [4, 6], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[18, 18], [8, 8], [10, 10]], 'SDA' : [[8, 8], [6, 6], [10, 10]]},
                    
                "2019" : {'ARI' : [[10, 10], [7, 10], [10, 10]], 'DSX' : [[10, 10], [10, 10], [10, 10]], 'ETA' : [[10, 10], [6, 8], [10, 10]],
                        'GEM' : [[7, 7], [7, 7], [10, 10]], 'ORI' : [[7, 6], [4, 6], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[15, 12], [8, 8], [10, 10]], 'SDA' : [[8, 6], [4, 6], [10, 10]]},
                        
                "2020" : {'ARI' : [[10, 10], [7, 10], [10, 10]], 'DSX' : [[20, 10], [5, 10], [10, 10]], 'ETA' : [[10, 10], [6, 8], [10, 10]],
                        'GEM' : [[5, 5], [5, 5], [10, 10]], 'ORI' : [[7, 6], [4, 6], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[15, 12], [8, 8], [10, 10]], 'SDA' : [[8, 6], [4, 6], [10, 10]]},

                "2021" : {'ARI' : [[10, 10], [7, 10], [10, 10]], 'DSX' : [[20, 10], [5, 10], [10, 10]], 'ETA' : [[10, 10], [6, 8], [10, 10]],
                        'GEM' : [[5, 5], [5, 5], [10, 10]], 'ORI' : [[7, 6], [4, 6], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[15, 12], [8, 8], [10, 10]], 'SDA' : [[8, 6], [4, 6], [10, 10]]}, 

                "2022" : {'ARI' : [[10, 10], [7, 10], [10, 10]], 'DSX' : [[20, 10], [5, 10], [10, 10]], 'ETA' : [[10, 10], [6, 8], [10, 10]],
                        'GEM' : [[5, 5], [5, 5], [10, 10]], 'ORI' : [[7, 6], [4, 7], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[12, 12], [8, 8], [10, 10]], 'SDA' : [[8, 6], [4, 6], [10, 10]]},

                "2023" : {'ARI' : [[6, 6], [5, 5], [10, 10]], 'DSX' : [[20, 10], [5, 10], [10, 10]], 'ETA' : [[10, 10], [8, 6], [10, 10]],
                        'GEM' : [[5, 5], [5, 5], [10, 10]], 'ORI' : [[6, 6], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[12, 12], [6, 6], [10, 10]], 'SDA' : [[6, 5], [5, 5], [10, 10]]},

                "2024" : {'ARI' : [[6, 6], [5, 5], [10, 10]], 'DSX' : [[20, 10], [5, 10], [10, 10]], 'ETA' : [[10, 10], [8, 6], [10, 10]],
                        'GEM' : [[5, 5], [5, 5], [10, 10]], 'ORI' : [[6, 6], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[12, 12], [8, 8], [10, 10]], 'SDA' : [[6, 5], [5, 5], [10, 10]]},

                "2025" : {'ARI' : [[6, 6], [5, 5], [10, 10]], 'DSX' : [[20, 10], [5, 10], [10, 10]], 'ETA' : [[10, 10], [8, 6], [10, 10]],
                        'GEM' : [[5, 5], [5, 5], [10, 10]], 'ORI' : [[6, 6], [5, 5], [10, 10]], 'PER' : [[10, 10], [8, 8], [10, 10]],
                        'QUA' : [[12, 12], [8, 8], [10, 10]], 'SDA' : [[6, 5], [5, 5], [10, 10]]}
                }

# Some notes below
    # 2023, 2024 and 2025 have identical radiant spaces
    # 2022 ARI shower is weak, so radiant space increased here
    # 2022 ETA shower has mirrored boundaries for its latitude to center the shower better for this year
    # 2022 ORI have different boundaries
    # 2021 and 2022 QUA has a wider latitude space
    # 2021 radiant spaces are identical to 3 latest years except for the QUA
    # 2020 and 2021 have identical radiant spaces
    # 2019 saw change in DSX radiant space by 10 deg lon and 5 deg lat
    # 2019 saw change in GEM radiant space by 4 degrees lon and lat
    # 2018 QUA min lon changed by 3 deg and max lon changed by 6 deg
    # 2018 SDA max lon changed by 2 deg and its min lat chnaged by 2 deg
    # 2017 identical to 2018 except fot the ORI shower, as its min lon changed by 3 deg, max lon by 1 deg and min lat by  2 deg
    # 2016 and 2017 have identical radiant spaces
    # 2015 ETA changed 3 deg lon and 1 deg min lat, 3 deg max lat
    # 2015 ORI changed 2 deg min lon, and 1 deg both min and max lat
    # 2014 ARI, ETA and SDA each had their boundaries reduced 
### YEARLY RADIANT SPACE INFO ###

class Shower_Bounds:

    def __init__(self, name, year, lon_bounds, lat_bounds, vel_bounds):
       self.name = name                          # name of the shower
       self.year = year                          # year the shower was observed
       self.lower_lon_bound = lon_bounds[0]      # lower bound of the shower's longitude
       self.upper_lon_bound = lon_bounds[1]      # upper bound of the shower's longitude
       self.lower_lat_bound = lat_bounds[0]      # lower bound of the shower's latitude
       self.upper_lat_bound = lat_bounds[1]      # upper bound of the shower's latitude
       self.lower_vel_bound = vel_bounds[0]      # lower bound of the shower's latitude
       self.upper_vel_bound = vel_bounds[1]      # upper bound of the shower's latitude

    def shower_hels(self, shower_rads, shower_slon_peaks):
        '''
        Takes the celestial coordinates of a given shower and computes its heliocentric coordinates
        The converted coordinates are used to precisely isolate a shower's location and all meteors within that location in a generated plot using echo_plot, which uses heliocentric lmda/beta
        '''
        shower_hel_dict = {}
        
        for shower, rads in shower_rads.items():

            v_cel = getvec(rads[0], rads[1])

            hel = cel2hel(v_cel, shower_slon_peaks[shower]) # currently doing peaks only, should do each one and tie into the correction done in shower parser for different ra/decs from the peak

            lon, lat = getangle(hel) # store these

            # scaled_lon = scale(lon) # bring to a 0-360 scale

            shower_hel_dict[shower] = [float(lon), float(lat)]
        
        # print(shower_hel_dict)
        # shower_helios = shower_hel_dict[shower_name] # two entry list

        return shower_hel_dict # two entry list
    

    def shower_radius(self, shower_helios, shower_velocity):
        '''
        Takes a given shower, computed coordinates from shower_hels and set boundaries for said location and calcuate defined heliocentric
        boundaries for both lmda and beta
        Will be used to plot shaded regions within these computed boundaries to highlight the region that showers should appear in
        '''

        lower_lon, upper_lon = self.lower_lon_bound, self.upper_lon_bound
        lower_lat, upper_lat = self.lower_lat_bound, self.upper_lat_bound
        lower_vel, upper_vel = self.lower_vel_bound, self.upper_vel_bound

        # defining bounds for the specific shower here - put into shower parser and only include meteors within these bounds using a mask
        lmda_bounds = [shower_helios[0] - lower_lon, shower_helios[0] + upper_lon]
        beta_bounds = [shower_helios[1] - lower_lat, shower_helios[1] + upper_lat]
        vel_bounds = [shower_velocity - lower_vel, shower_velocity + upper_vel]

        return lmda_bounds, beta_bounds, vel_bounds

# C2H Loop function for each shower

name='ETA' # this is a user input in parse_radar
year='2025'

# Put the next three lines into parse_radar's shower iteration loop
dataset = Shower_Bounds(name, year, shower_boundaries[year][name][0], shower_boundaries[year][name][1], shower_boundaries[year][name][2])

coords = dataset.shower_hels(shower_rads, shower_slon_peaks)

lmda, beta, vel = dataset.shower_radius(coords[name], shower_mean_velocities[name])

# print(lmda, beta, vel)

