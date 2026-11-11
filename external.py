'''
This code will hold functions external to the mpd_an_shrm.py file
In the original code (writtin in c++) these were called as functions external to the code
    Unless these functions exist in colossid like the original code did (maybe within the same directory), I will need to write my own versions of them
'''

"""
Python port of external.cpp (functions used by mpd_an_shrm).

NOT in external.cpp, so they must be ported from wherever they are defined:
  wgspos_map, conv_alpdel_etarho, conv_etarho_alpdel, rangeE, TavGainNew
  (and the constants pi, rtd). Stubs for them are marked below.
"""
import math
import numpy as np


# ASSUMPTION: same values as mpd_an_shrm_new.cpp. Change if your headers define them differently.
pi = 3.14159265358
rtd = 180 / pi
dtr = pi / 180


# ================================================================ C-style math helpers

def _pow(a, b):
    try:
        return math.pow(a, b)
    except OverflowError:
        return math.inf
    except ValueError:
        return math.inf if a == 0 else math.nan

def _acos(x):
    return math.acos(x) if -1.0 <= x <= 1.0 else math.nan
 
 
def _asin(x):
    return math.asin(x) if -1.0 <= x <= 1.0 else math.nan
 

def _sqrt(x):
    return math.sqrt(x) if x >= 0 else math.nan


def _div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
 
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

        # quadrant I and IV ; changing this portion causes trouble for h2c, but gives the correct helio coords
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

def _mag(v):
    """C++: vec::mag(). ASSUMED to be the Euclidean length (vec class not uploaded)."""
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

# ================================================================ ported functions
def speed_geo_app(vel_g, alp_g, del_g, sid, lat, lon, elv):
    """
    Returns (app, alp_p, del_p). The C++ version returned app and filled *alp_p, *del_p.
    sid, alp, del, lat, lon are in radians, elv in km.
    If vel_g <= 0 the C++ returned 0.0 and left alp_p/del_p untouched; here they are 0.0.
    """
    a = 6378.137
    b = 6356.752314245
    ee = (a * a - b * b) / (a * a)
    G = 6.67259e-11 / 1.0e9
    M = 5.9742e24

    if vel_g <= 0.0:
        return 0.0, 0.0, 0.0

    p = wgspos_map(lat, lon, elv)

    rad = math.hypot(p[0], math.hypot(p[1], p[2]))
    cen = atan3((1.0 - ee) * math.tan(lat))
    rot = 2.0 * pi * rad * math.cos(cen) / 86164.09054
    esc = math.sqrt(2.0 * G * M / rad)
    app = math.sqrt(vel_g * vel_g + esc * esc)

    eta, rho = conv_alpdel_etarho(alp_g, del_g, cen, sid)

    dif = 0.0
    zen = eta
    while True:  # loop: ... while (fabs(dif) > 1e-5) goto loop;
        zen -= dif
        dZc = 2.0 * atan3((app - vel_g) * math.tan(zen / 2.0) / (app + vel_g))
        dif = zen + dZc - eta
        if not abs(dif) > 1.0e-5:
            break

    alp_g, del_g = conv_etarho_alpdel(zen, rho, cen, sid)

    v = [app * math.cos(del_g) * math.cos(alp_g) + rot * math.cos(sid + pi / 2),
         app * math.cos(del_g) * math.sin(alp_g) + rot * math.sin(sid + pi / 2),
         app * math.sin(del_g)]

    app = math.hypot(v[0], math.hypot(v[1], v[2]))

    alp_p = math.atan2(v[1], v[0])
    del_p = math.atan2(v[2], math.hypot(v[0], v[1]))
    return app, alp_p, del_p


def SidTime(t, year, month, day, lng):
    """Sidereal time (fraction of a day) from UT time (hours), date and longitude (degrees)."""
    # (the C++ had two identical branches, for t>12 and t<=12)
    jdl = julday(day, month, year) + (t - 12) / 24.0
    jdd = math.trunc(jdl + 0.5) - 0.5
    hh = (jdl - jdd) * 24
    tt = (jdd - 2451545) / 36525.0
    gmst0 = 100.46061837 + 36000.770053608 * tt + 0.000387933 * tt * tt - tt * tt * tt / 38710000.0
    while gmst0 < 0:
        gmst0 += 360
    while gmst0 > 360:
        gmst0 -= 360
    gmst = gmst0 + hh * 60 * 0.250684477338
    lmst = gmst + lng
    while lmst < 0:
        lmst += 360
    while lmst > 360:
        lmst -= 360
    lmst = lmst / 360.0
    return lmst


def julday(id, im, iy):
    """Julian day. Based on Jim Jones' code."""
    igreg = 588829

    if iy == 0:
        return 1.0
    if iy < 0:
        iy = iy + 1
    if im > 2:
        jy = float(iy)
        jm = float(im + 1)
    else:
        jy = float(iy - 1)
        jm = float(im + 13)
    jul = float(math.trunc(365.25 * jy)) + float(math.trunc(30.6001 * jm)) + id + 1720995
    if id + 31 * (im + 12 * iy) >= igreg:
        ja = float(math.trunc(0.01 * jy))
        jul = jul + 2 - ja + float(math.trunc(0.25 * ja))
    return jul


def solar_long(year, month, day, t):
    """Solar longitude (degrees). t in hours."""
    ao = [334166, 3489, 350, 342, 314, 268, 234, 132, 127, 120, 99, 90, 86, 78, 75,
          51, 49, 36, 32, 28, 27, 24, 21, 21, 20, 16, 13, 13]
    bo = [4.669257, 4.6261, 2.744, 2.829, 3.628, 4.418, 6.135, 0.742, 2.037, 1.110,
          5.233, 2.045, 3.508, 1.179, 2.533, 4.58, 4.21, 2.92, 5.85, 1.90, 0.31,
          0.34, 4.81, 1.87, 2.46, 0.83, 3.41, 1.08]
    co = [6283.075850, 12566.1517, 5753.385, 3.523, 77713.771, 7860.419, 3930.210,
          11506.77, 529.691, 1577.344, 5884.927, 26.298, 398.149, 5223.694, 5507.553,
          18849.23, 775.52, 0.07, 11790.63, 796.30, 10977.08, 5486.78, 2544.31,
          5573.14, 6069.78, 213.30, 2942.46, 20.78]
    a1 = [20606, 430, 43]
    b1 = [2.67823, 2.635, 1.59]
    c1 = [6283.07585, 12566.152, 3.52]
    a2 = [872, 29]
    b2 = [1.073, 0.44]
    c2 = [6283.07585, 12566.15]
    a3 = 29
    b3 = 5.84
    c3 = 6283.07585

    julian = julday(day, month, year)
    jd = julian + (t - 12) / 24.0
    T = (jd - 2451545) / 365250.0
    Lo = 4.8950627 + 6283.0758500 * T - 0.0000099 * T * T
    S0 = S1 = S2 = S3 = 0.0
    for i in range(28):
        te = bo[i] + co[i] * T
        S0 = S0 + ao[i] * math.cos(te)
    for i in range(3):
        te = b1[i] + c1[i] * T
        S1 = S1 + a1[i] * math.cos(te)
    for i in range(2):
        te = b2[i] + c2[i] * T
        S2 = S2 + a2[i] * math.cos(te)
    te = b3 + c3 * T
    S3 = a3 * math.cos(te)
    L = Lo + (S0 + S1 * T + S2 * T * T + S3 * T * T * T) * 1e-7
    if L < 2 * pi:  # two separate ifs, as in the original
        L = 2 * pi - (math.trunc(L / (2 * pi)) - (L / (2 * pi))) * 2 * pi
    if L > 2 * pi:
        L = ((L / (2 * pi) - math.trunc(L / (2 * pi)))) * 2 * pi
    L = L * rtd
    return L


def DayofYear(yr, mon, d):
    """Day number given year, month and day."""
    days = {1: 0, 2: 31, 3: 59, 4: 90, 5: 120, 6: 151,
            7: 181, 8: 212, 9: 243, 10: 273, 11: 304, 12: 334}
    if mon not in days:
        raise ValueError("month out of range (undefined in the C++)")
    Yday = days[mon]
    if yr % 4 == 0 and mon > 2:  # leap year
        Yday += 1
    return Yday + d


def sens(sp, s, tht, TxCo, RxCo):
    """Sensitivity of the radar."""
    r = rangeE(sp)
    gn = TavGainNew(sp, tht, TxCo) * TavGainNew(sp, tht, RxCo)
    sns = _sqrt(_div(gn, _pow(r * 1000, 3)))
    return _pow(sns, s - 1)


def wgspos_map(lat, lon, elv):
    """WGS84 position (km) from lat, lon (radians) and elevation (km). Returns [x, y, z].
    (C++ filled a double u[3] passed in.)"""
    a = 6378.137
    b = 6356.752314245
    ee = (a * a - b * b) / (a * a)
    N = a / math.sqrt(1.0 - ee * math.sin(lat) * math.sin(lat))
 
    u = [0.0, 0.0, 0.0]
    u[0] = (N + elv) * math.cos(lat) * math.cos(lon)
    u[1] = (N + elv) * math.cos(lat) * math.sin(lon)
    u[2] = (N * (1.0 - ee) + elv) * math.sin(lat)
    return u
 
 
def conv_alpdel_etarho(alp, dec, lat, sid):
    """RA, Dec -> (eta, rho). C++ filled *eta, *rho (and the declination argument was named del)."""
    eta = _acos(math.sin(dec) * math.sin(lat) + math.cos(sid - alp) * math.cos(dec) * math.cos(lat))
    rho = math.atan2(math.cos(lat) * math.sin(dec) - math.sin(lat) * math.cos(dec) * math.cos(sid - alp),
                     -math.sin(sid - alp) * math.cos(dec))
    return eta, rho
 
 
def conv_etarho_alpdel(eta, rho, lat, sid):
    """(eta, rho) -> (alp, dec). C++ filled *alp, *del."""
    dec = _asin(math.cos(eta) * math.sin(lat) + math.sin(rho) * math.sin(eta) * math.cos(lat))
    ha = _acos(_div(math.cos(eta) - math.sin(dec) * math.sin(lat), math.cos(dec) * math.cos(lat)))
 
    if math.cos(rho) > 0.0:
        ha = sid + ha
    else:
        ha = sid - ha
 
    while ha < 0.0:
        ha += 2.0 * pi
    while ha > 2.0 * pi:
        ha -= 2.0 * pi
 
    return ha, dec
 
 

def rangeEarth(Re, tht, ht):
    """Range (km) to a height ht (km) for a curved Earth; tht in degrees."""
    if Re is None:
        raise NotImplementedError("set Re (Earth radius in metres) near the top of external.py")
    Rekm = Re / 1000
 
    a = 1
    b = -2 * Rekm * math.cos(dtr * (180 - tht))
    c = Rekm * Rekm - _pow(Rekm + ht, 2)
 
    rng = (-b + _sqrt(b * b - 4 * a * c)) / (2 * a)  # in km
    return rng

def rangeE(v):
    """Range to the Earth-surface point at the zenith angle of v, for 100 km height.
    math.hypot is used to calculate """
    Re = 6.37e6
    tht = rtd * _acos(_div(v.z, _mag(v)))
    rng = rangeEarth(Re, tht, 100)
    return rng
 
 
def TavGainNew(v, th, Coefs):
    """Antenna gain for Tavistock. th is the interferometer angle (degrees)."""
    # Gbroad: elevation dependence perpendicular to the antenna elements (fit to linear)
    # Gphi: difference between 0/180 and 90 degrees (from the dipoles), with a phi dependence
    phi = atan3(v.x, v.y) / dtr + th + 90  # phi=0 is at the south(sse) end of the antenna
    tht = _asin(_div(v.z, _mag(v))) / dtr
    while phi > 360:
        phi -= 360
    while phi < 0:
        phi += 360
    if tht > 90.01:
        Gt = 0
    else:
        c = math.cos(tht * dtr)
        Gbroad = (Coefs.broad[0] + Coefs.broad[1] * _pow(c, 2) + Coefs.broad[2] * _pow(c, 4)
                  + Coefs.broad[3] * _pow(c, 6) + Coefs.broad[4] * _pow(c, 8))
        GbroaddB = 10 * math.log10(Gbroad)
        if phi < 90 or phi > 270:
            n = 0
        else:
            n = 1
        GenddB = _div(Coefs.end[n][0] + Coefs.end[n][1] * _pow(c, 2)
                      + Coefs.end[n][2] * _pow(c, 4) + Coefs.end[n][3] * _pow(c, 6)
                      + Coefs.end[n][4] * _pow(c, 8),
                      Coefs.phidep[n][0] + Coefs.phidep[n][1] + Coefs.phidep[n][2])
        if GenddB < 0:
            GenddB = 0
        GenddB = _pow(math.cos(phi * dtr), 2) * _pow(GenddB, Coefs.exp)
        GtdB = GbroaddB - (Coefs.phidep[n][0] * GenddB + Coefs.phidep[n][1] * _pow(GenddB, 2)
                           + Coefs.phidep[n][2] * _pow(GenddB, 3))
        # Fudge only significant about 25 degrees above horizon or less (disabled in the original)
        gainfunc = 1
        Gt = _div(_pow(10, GtdB / 10), gainfunc)
    return Gt
 