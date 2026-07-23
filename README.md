# Radar

This repository includes some python code and resulting files of meteor orbit data detected by the Canadian Meteor Orbit Radar (CMOR). The parse script is used to collect important components of the meteor's velocity, position in the sky and measurment errors that contribute to the signal coming from the detected echo. Restricitons are defined so that clean signals have a certain threshold of agreement based on their measured values of speed or position relative to other meteors seen by CMOR. 
The functionalities of the parse radar script are as follows:
1. Parses through a given folder of orbit file data (.orb files) and extracts the following values for each orbit:
   - Date and time of observation
   - Number of radar stations used to measure the orbit
   - Ecliptic Latitude and Longitude of the meteor radiant (Heliocentric Coordinates)
   - Solar Longitude (day relative to the VE) the orbit was observed to have
   - Right ascension and Declination of the meteor radiant (Celestial Coordinates)
   - Time of flight velocity, Geocentric Velocity, and Pre-t0 velocity along with uncertainties respectively
   - Azimuthal and Zenithal Angles of the meteor radiant
   - Range of the meteor radiant to the zehr
   - Interferometry Error, Radiant Position Error, Station Measurement Error
2. Apply the following filters to each orbit:
   - If time of flight and pre-t0 velocities agree to within 10 km/s AND if their uncertainties overlap, this filter is satisfied
     - The above step is only applied to slow moving meteors with a geocentric velocity less than 48 km/s, as faster moving meteors have poor pre-t0 measurements
   - Any Interferometry Measurement Error or Radiant location error greater than 2 degrees is discarded
   - Any orbit observed using 3 stations or less is discarded, and any orbit otherwise with a station measurement error of greater than 3 is discarded
     - This filter is a bit more strict and might be discarded in the future
   - Any two sequential events within 10 milliseconds of occurance is checked if their celestial coordinates come to within 5 degrees of eachother OR if their values of Zehr range (R0) are within 6 km
     - For orbits satisfying the above, one of the two sequential orbits are considered a duplicate event and are discarded
3. Plotting (heat, velocity, scatter)
4. Shower removal (raw data vs filtered data, shower date checl/save, voxel emptying and coord sigma check)
5. Source plotting and shape changes before/after filtering (work in progress)

For future
- Distribution weighting using limiting mass/energy
- Atmospheric bias correction (initial train radius, pulse repitition factor, finite velocity method, Faraday rotation)
