'''
Feed radar data labelled 'final' to this code for plotting
This data should contain none of the following strong shower sources: 
    ARI, DSX, ETA, GEM, ORI, PER, QUA, SDA
The only orbits in the data is filtered sporadics and those from minor showers, or showers with a broad profile
The following will hope to be plotted here:
    A heat map of the distirbution of orbits
    Histograms of 
        geocentric velocity
        semi major axis (negative corresponds to hyperbolic orbits)
        eccentricity (e > 1 corresponds to hyperbolic orbits)
        inclination (0 < i < 180)
'''
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from shower_bounds import shower_slon


def scale(x):
    '''
    This function scales the x axis to be centered at 270 degrees longitude
    '''
    x = np.asarray(x) % 360
    res = (x - 270) % 360

    return np.where(res > 180, res - 360, res)


longitudes = []
latitudes = []
velocities = []
shower_count = 0


clean_folder = '/home/zaubs/Desktop/radar/clean shower data/sporadics'
shower_dates_folder = '/home/zaubs/Desktop/radar/clean shower data/shower meteor dates'

clean_folder_path = os.listdir(clean_folder)
shower_dates_folder_path = os.listdir(shower_dates_folder)

# load all shower dates from the txt files into a set for fast lookup
shower_dates = set()

for date_file in shower_dates_folder_path:
    date_file_path = os.path.join(shower_dates_folder, date_file)
    with open(date_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                shower_dates.add(line)

for folder in clean_folder_path:
    # print(folder) # these are years of data

    year = folder

    subfolder = f'{folder}/{year} final clean events'

    folder_path = os.listdir(os.path.join(clean_folder, subfolder))

    for file in folder_path:

        if file == 'figures':
            continue # skips the figure directory in this year of data
            
        file_path = os.path.join(clean_folder, subfolder, file)

        # deifning a file's solar longitude, and an integer equivalent solar longitude
        file_slon = file[11:14]
        print(file, year, file_slon)

        if file_slon[0:2] == '00':
            meteor_slon = int(file_slon[2]) # taking the leading zero out from solar longitude
            

        elif file_slon[0] == '0':
            meteor_slon = int(file_slon[1:]) # taking the leading zero out from solar longitude
        
        else:
            meteor_slon = int(file_slon)

        with open(file_path, 'r') as sporadic_data:

            # saving non active days to the file regardless
            if meteor_slon not in shower_slon:

                # print('not here')

                for line in sporadic_data:

                    line = line.strip()
                    params = line.split()

                    if params == [] or params[0][0] != '2': 
                        continue

                    date = params[0]
                    time = params[1]

                    lmda = float(params[3])
                    beta = float(params[4])
                    velg = float(params[11])

                    longitudes.append(lmda)
                    latitudes.append(beta)
                    velocities.append(velg)

            else:
                # print('here')

                for line in sporadic_data:

                    line = line.strip()
                    params = line.split()

                    # print(params)

                    # skipping empty lines
                    if params == []:
                        continue # skip lines that don't start with a date; works

                    else:

                        # skip lines that don't start with a date; works
                        if params[0][0] != '2':
                            # header += line
                            # header += '\n'
                            # # print(header)
                            continue

                        date = params[0]
                        time = params[1]

                        lmda = float(params[3])
                        beta = float(params[4])
                        slon = float(params[5])

                        velg = float(params[11])

                        datetime = f'{date} {time}'
                        # print(datetime)

                        # not writing lines with shower date/time to final dataset
                        if datetime in shower_dates: # need this written to a file
                            shower_count += 1

                        else:

                            longitudes.append(lmda)
                            latitudes.append(beta)
                            velocities.append(velg)

print(f'Shower meteors excluded: {shower_count}')
print(f'Sporadics kept: {len(longitudes)}')

scaled_lons = scale(longitudes)

figure, ax = plt.subplots(figsize=(10,5))

h = ax.hist2d(scaled_lons, latitudes, bins=200, cmap='plasma')

figure.colorbar(h[3], ax=ax, label='Number of meteors per bin')


ax.set_facecolor("#0D0F81")

ax.set_xlabel(r'Ecliptic Longitude $(\lambda - \lambda_{\odot})$')
ax.set_ylabel(r'Ecliptic Latitude $(\beta)$')

ax.set_title(f'Clean Meteor Sources observed in Ecliptic Coordinates - measurements excluding all shower meteors from 2011-2025')

lmda_min, lmda_max = min(scaled_lons) - 5, max(scaled_lons) + 5
beta_min, beta_max = min(latitudes) - 2.5, max(latitudes) + 2.5

ax.set_xlim(lmda_max, lmda_min)
ax.set_ylim(beta_min, beta_max)

plt.show()