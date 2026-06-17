
import numpy as np
import matplotlib.pyplot as plt

def getvec(ra, dec):
    '''
    Takes coords of ra and dec and gets the vector form of it
    '''

    v = np.zeros(3)

    dtr = np.pi/180

    v[0] = -np.cos(dtr*dec)*np.sin(dtr*ra)
    v[1] = np.cos(dtr*dec)*np.cos(dtr*ra)
    v[2] = np.sin(dtr*dec)

    return v

def getangle(vec):
    '''
    takes a vector and returns the angles relative to that vector
    '''

    rtd = 180/np.pi

    a = rtd*atan(vec[1], -vec[0])
    b = rtd*np.arcsin(np.clip(vec[2], -1.0, 1.0))

    return a, b

    
def atan(x, y):
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
            z = np.arctan(y/x)
        # quadrant II and III
        else:
            z = np.arctan(y/x) + np.pi

    # on the + and - y axis
    else:
        if y > 0.0:
            z = np.pi

        else:
            z = -np.pi

    if z < 0:
        z += 2* np.pi # 3 so in range from [0,2pi]

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
    w[2] = cel[2] * np.cos(dtr * q) - cel[1] * np.sin(dtr * q)

    # rotation by solar longitude sl
    hel = np.zeros(3)
    hel[0] = -w[0] * np.cos(dtr * sl) - w[1] * np.sin(dtr * sl)
    hel[1] =  w[0] * np.sin(dtr * sl) - w[1] * np.cos(dtr * sl)
    hel[2] =  w[2]

    return hel
# Main function call

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

    w = np.zeros(3)
    w[0] =  hel[1] * np.sin(dtr * sl) - hel[0] * np.cos(dtr * sl)
    w[1] = -hel[0] * np.sin(dtr * sl) - hel[1] * np.cos(dtr * sl)
    w[2] =  hel[2]

    cel = np.zeros(3)
    cel[0] =  w[0]
    cel[1] =  w[1] * np.cos(dtr * q) - w[2] * np.sin(dtr * q)
    cel[2] =  w[2] * np.cos(dtr * q) + w[1] * np.sin(dtr * q)
    return cel

a_list, b_list = [0, 90, 180, 270], [0, 90,180,270]  # ras, decs
sl, q = 261.0, 23.5  # solar longitude, obliquity

plt.figure(figsize=(10,8))

for a, b in zip(a_list,b_list):
    v_cel      = getvec(a, b)
    v_hel      = cel2hel(v_cel, sl, q)
    l_comp, b_comp = getangle(v_hel)
    v_cel_back = hel2cel(v_hel, sl, q)
    a_back, b_back = getangle(v_cel_back)

    print(f'Original : a={a:.4f}  b={b:.4f}')
    print(f'Computed: lmda={l_comp:.4f}, beta={b_comp:.4f}')
    print(f'Recovered: a={a_back:.4f}  b={b_back:.4f}')

    plt.scatter(l_comp, b_comp, label=f'RA={a}, Dec={b}')

plt.scatter(270, 30, color='k')

plt.title('Heliocentric coordinates translated from celestial RA/Dec')

plt.xlabel('Heliocentric Longitude (Lambda)')
plt.ylabel('Heliocentric Latitude (Beta)')

plt.grid()
plt.legend()
plt.show()

# lat/lon list next
v_cel      = getvec(a, b)
v_hel      = cel2hel(v_cel, sl, q)
l_comp, b_comp = getangle(v_hel)
v_cel_back = hel2cel(v_hel, sl, q)
a_back, b_back = getangle(v_cel_back)