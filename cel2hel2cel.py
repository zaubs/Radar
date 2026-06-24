
import numpy as np
import math
import matplotlib.pyplot as plt

def getvec(ra, dec):
    '''
    Takes coords of ra and dec and gets the vector form of it
    '''

    v = np.zeros(3)

    dtr = np.pi/180

    v[1] = np.cos(dtr*dec)*np.sin(dtr*ra) # y
    v[0] = np.cos(dtr*dec)*np.cos(dtr*ra) # x # flipped x and y
    v[2] = np.sin(dtr*dec) # z

    return v

def getangle(vec):
    '''
    takes a vector and returns the angles relative to that vector
    '''

    rtd = 180/np.pi

    a = rtd*atan3(vec[0], vec[1]) # fillped the values ; making the second value negative flips the image about the vertical axis
    b = rtd*np.arcsin(vec[2])

    return a, b

    
def atan3(x, y):
    """
    Custom atan returning angle in [0, 2pi].
    
    Parameters:
        x : float
        y : float
    Returns:
        z : float, angle in radians
    """

    if x != 0.0:

        # quadrant I and IV
        if x > 0.0:
            z = np.atan(y/x) - np.pi/2 # put this factor of pi/2 here
        # quadrant II and III
        else:
            z = np.atan(y/x) + np.pi/2 # changed from pi to pi/2

    # on the + and - y axis respectively
    else:
        if y > 0.0:
            z = np.pi

        else:
            z = -np.pi

    # if z < 0:
    #     z += 2* np.pi # 3 so in range from [0,2pi]

    return z

def cel2hel(cel, sl, q=23.55):
    """
    Convert celestial coordinates to heliocentric coordinates.
    
    Parameters:
        cel : array-like, [x, y, z] celestial coordinate vector
        sl  : float, solar longitude in degrees
        q   : float, tilt angle of earth's axis relative to the solar system's poles
    
    Returns:
        hel : np.array, [x, y, z] heliocentric coordinate vector
    """
    dtr = np.pi / 180.0  # degrees to radians

    # intermediate rotation by obliquity q
    w = np.zeros(3)
    w[0] = cel[0]
    w[1] = cel[1] * np.cos(dtr * q) + cel[2] * np.sin(dtr * q)
    w[2] = -cel[1] * np.sin(dtr * q) + cel[2] * np.cos(dtr * q) 

    # rotation by solar longitude sl
    hel = np.zeros(3)
    hel[0] = -w[0] * np.cos(dtr * sl) - w[1] * np.sin(dtr * sl)
    hel[1] =  w[0] * np.sin(dtr * sl) - w[1] * np.cos(dtr * sl)
    hel[2] =  w[2]
    # below is the opposite signed and transposed matrix, did not give right values
    # hel[0] =  w[0] * np.cos(dtr * sl) - w[1] * np.sin(dtr * sl)
    # hel[1] =  w[0] * np.sin(dtr * sl) + w[1] * np.cos(dtr * sl)
    # hel[2] =  w[2]

    return hel


def hel2cel(hel, sl, q=23.55):
    """
    Convert heliocentric coordinates to celestial coordinates.
    
    Parameters:
        hel : array-like, [x, y, z] heliocentric coordinate vector
        sl  : float, solar longitude in degrees
        q   : float, tilt angle of earth's axis relative to the solar system's poles
    
    Returns:
        cel : np.array, [x, y, z] celestial coordinate vector
    """

    dtr = np.pi / 180

    # reversing the solar longitude rotation
    w = np.zeros(3)
    w[0] = -hel[0] * np.cos(dtr * sl) + hel[1] * np.sin(dtr * sl) 
    w[1] = -hel[0] * np.sin(dtr * sl) - hel[1] * np.cos(dtr * sl)
    w[2] =  hel[2]

    # revsersing the axis tilt
    cel = np.zeros(3)
    cel[0] =  w[0]
    cel[1] =  w[1] * np.cos(dtr * q) - w[2] * np.sin(dtr * q)
    cel[2] =  w[1] * np.sin(dtr * q) + w[2] * np.cos(dtr * q)
    return cel

def slon_corr(sl, c, d, q=23.55):

    # pass # not using this function currently

    '''
    This is an approximation on the solar longitude correction oon ra / dec
    It currently mirrors the distribution of radiants for any data set I use
    It also relies on the computed lon/lat using c2h
    '''

    dtr = np.pi/180

    dec = np.sin((sl - c) * dtr) * q + d * np.cos(q * np.sin(sl * dtr) * dtr)
    if dec > 90:
        dec = 90 - (dec - 90)

    ra = (sl - c) + d * np.sin(q * np.sin(-sl * dtr) * dtr)
    if ra > 360:
        ra = ra - 360
    if ra < 0:
        ra = ra + 360

    return ra, dec

def main():
    q = 23.55
    ra  = 45.7
    dec = 25
    sl  = 81

    dtr = np.pi/180

    print(f'{ra} {dec}')

    # convert ra/dec to celestial vector, then to heliocentric
    cel = getvec(ra, dec)
    hel = cel2hel(cel, sl, q)
    c, d = getangle(hel)
    print(f'lon: {c}  lat: {d}')

    
    # # solar longitude correction
    # ra, dec = slon_corr(sl, c, d)

    # print(f'Ra: {ra} Dec: {dec}')

    
    # cel = getvec(ra, dec)
    # hel = cel2hel(cel, sl, q)
    # c, d = getangle(hel)
    # print(f'lon: {c}  lat: {d}')

    
    # convert heliocentric angles back to celestial
    hel = getvec(c, d)
    cel = hel2cel(hel, sl, q)
    ra, dec = getangle(cel)
    print(f'{ra} {dec}')

# Main function call
if __name__ == '__main__':
    main()
