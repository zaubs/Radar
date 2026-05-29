'''
This code will be used to generate overplotted velocity histograms of echo data before and after each filter is applied, in successive order
'''


import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# current working directory

home = Path.home() / 'Desktop/radar'

folder_name1 = home / 'clean file data/0528/raw check events'
file_name1 = 'raw check-velocities-2022-29.txt'

folder_name2 = home / 'clean file data/0528/vel only, no 40 km s check'
file_name2 = 'vel check, no 40 km s.txt'

folder_name3 =  home / 'clean file data/0528/vel only, with 40 km s check'
file_name3 = 'vel check, 40 km s.txt'


folder_name_raw = home / 'clean file data/0528_2/raw check events'
file_name_raw = 'raw check-velocities-2023-29.txt'

folder_name_v = home / 'clean file data/0528_2/vel only, with 48 km s check'
file_name_v = 'vel check, with 48 km s.txt'

folder_name_vi = home / 'clean file data/0528_2/vel_check and int_check events'
file_name_vi = 'vel_check and int_check-velocities-2023-29.txt'

folder_name_via = home / 'clean file data/0528_2/vel_check and int_check and solid_angle_check events'
file_name_via = 'vel_check and int_check and solid_angle_check-velocities-2023-29.txt'

folder_name_all = home / 'clean file data/0528_2/all checks events'
file_name_all = 'all checks-velocities-2023-29.txt'

def histo_plotter(folder_name, file_name):

    folder = os.listdir(folder_name)

    file = os.path.join(folder_name, file_name)

    velocities = []
    counts = []

    with open(file, 'r') as shower_data:

        for r in range(2):
            next(shower_data, None) # skips first two lines

        for line in shower_data:

            line = line.strip(",").split()
            # print(line)

            vel = float(line[0])
            count = float(line[1])

            velocities.append(vel)
            counts.append(count)

    # print(velocities, counts)

    return velocities, counts


# function calls
vels1, counts1 = histo_plotter(folder_name1, file_name1)
vels2, counts2 = histo_plotter(folder_name2, file_name2)
vels3, counts3 = histo_plotter(folder_name3, file_name3)

vels_raw, counts_raw = histo_plotter(folder_name_raw, file_name_raw)
vels_v, counts_v = histo_plotter(folder_name_v, file_name_v)
vels_vi, counts_vi = histo_plotter(folder_name_vi, file_name_vi)
vels_via, counts_via = histo_plotter(folder_name_via, file_name_via)

vels_all, counts_all = histo_plotter(folder_name_all, file_name_all)

# Raw data only
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels_raw, bins=1000, weights=counts_raw, label='Raw meteor data')


plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()


# Velocity filter
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels_raw, bins=200, weights=counts_raw, label='Raw meteor data')
plt.hist(vels_v, bins=200, weights=counts_v, label='Vel (48km/s) Filter Results')

plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()


# Velocity filter + interferometry filter
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels_raw, bins=200, weights=counts_raw, label='Raw meteor data')
plt.hist(vels_v, bins=200, weights=counts_v, label='Vel (48km/s) Results')
plt.hist(vels_vi, bins=200, weights=counts_vi, label='Vel (48 km/s) + Int Results')

plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()


# Velocity filter + interferometry filter + solid angle filter
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels_raw, bins=200, weights=counts_raw, label='Raw meteor data')
plt.hist(vels_v, bins=200, weights=counts_v, label='Vel (48km/s) Results')
plt.hist(vels_vi, bins=200, weights=counts_vi, label='Vel (48 km/s) + Int Results')
plt.hist(vels_via, bins=200, weights=counts_via, label='Vel (48 km/s) + Int + Solid Angle Results')

plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()


# All filters applied
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels_raw, bins=200, weights=counts_raw, label='Raw meteor data')
plt.hist(vels_v, bins=200, weights=counts_v, label='Vel (48km/s) Results')
plt.hist(vels_vi, bins=200, weights=counts_vi, label='Vel (48 km/s) + Int Results')
plt.hist(vels_via, bins=200, weights=counts_via, label='Vel (48 km/s) + Int + Solid Angle Results')
plt.hist(vels_all, bins=200, weights=counts_all, label='Final Results (incl >4 station error)')

plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend(fontsize=10)

plt.show()



# plot with raw data and both velocity checks, before and after the 40km/s condition is applied
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels1, bins=50, weights=counts1, label='Raw meteor data')
plt.hist(vels3, bins=50, weights=counts3, label='Velocity filtered, with 40 km/s')
plt.hist(vels2, bins=50, weights=counts2, label='Velocity filtered, no 40km/s')


plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()

# a plot of what is removed before and after the velocity condition is applied on the raw data
# the plot includes the difference in counts

figure = plt.figure(figsize=(10,5))

counts13_diff = []

for i in range(len(counts1)):
    # counts13 = counts1[i] - counts3[i]
    counts13_diff.append(counts1[i] - counts3[i])

plt.hist(vels1, bins=50, weights=counts1, label='Raw meteor data')
plt.hist(vels1, bins=50, weights=counts13_diff, label='Removed data') 
# some data is negative; the bug here is that some of the bins are not aligned but that is because the filtering changes what the average velocity is for each bin
# maybe there is a way to set the average velocity per bin manually?
# plt.hist(vels3, bins=50, weights=counts3, label='Velocity filtered, with 40 km/s')

plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()

# plot without a 40 km/s check
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels1, bins=50, weights=counts1, label='Raw meteor data')
plt.hist(vels2, bins=50, weights=counts2, label='Velocity filtered, no 40km/s')


plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()


# Plot of raw vs a 40 km/s check; not much of a difference than above
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels1, bins=50, weights=counts1, label='Raw meteor data')
plt.hist(vels3, bins=50, weights=counts3, label='Velocity filtered, with 40 km/s')


plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()


# plot before and after the velocity condition is applied
figure = plt.figure(figsize=(10,5))

# plt.scatter(velocities, counts)
plt.hist(vels3, bins=50, weights=counts3, label='Velocity filtered, with 40 km/s')
plt.hist(vels2, bins=50, weights=counts2, label='Velocity filtered, no 40 km/s')


plt.title('Velocity bins before/after filtering')

plt.xlabel('Geocentrc Velocities (km/s)')
plt.ylabel('Counts per bin')

plt.legend()

plt.show()

