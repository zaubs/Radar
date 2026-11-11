"""
Python port of mpd_an_shrm_new.cpp

Changes mpd files to mc, sc or xc files
  mc: echoes below 20 degrees removed, best amplitude calculated for multiple echoes
  sc: as above, but selects echoes consistent with a given radiant
  xc: removes showers active on given day, leaving only sporadics

The C++ source called routines defined elsewhere. Supply them in external.py:
  SidTime(t, year, month, day, lng)                  -> sidereal time (fraction of day)
  speed_geo_app(vel_g, alp_g, del_g, sid, lat, lon, elv)
                                                     -> (v_app, alp_p, del_p)   [radians, elv km]
  sens(sp, s, tht, TxCo, RxCo)                       -> float   (sp is a Vec, Tx/Rx are GainCo)
"""
import os
import math
import re
from dataclasses import dataclass, field

from pathlib import Path

from external import SidTime, speed_geo_app, sens

pi = 3.14159265358
dtr = pi / 180

# zach's directories
mac_path = str(Path.home() / f'Desktop/radar/mpd input files/')
mac_path_out = str(Path.home() / f'Desktop/radar/mpd output files/')

# colossid irectory containing mpd files - too many to transport to a local directory
colossid_path = f'/srv/meteor/radar/spool/mpd-38/'

os.makedirs(mac_path, exist_ok=True)
os.makedirs(mac_path_out, exist_ok=True)
# print(mac_path)

# ---------------------------------------------------------------- small C helpers
def c_round(x):
    """
    A function that rounds numbers to integers half away from zero.
    """
    a = abs(x)
    fl = math.floor(a)
    return math.copysign(fl + 1 if a - fl >= 0.5 else fl, x)


def _pow(a, b):
    """
    Tries to raise a to the power b unless the resulting value diverges to infinity
    """
    try:
        return math.pow(a, b)
    except OverflowError:
        return math.inf
    except ValueError:
        return math.inf if a == 0 else math.nan


def _div(a, b):
    """
    Tries to divide a by b unless b=0
    """
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _log10(x):
    """
    Tries to take the log of x unless x<0
    """
    if x > 0:
        return math.log10(x)
    return -math.inf if x == 0 else math.nan


# ---------------------------------------------------------------- data classes
@dataclass
class Vec:
    """
    Defines vector elements in 3D
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class GainCo:
    """
    Container for the antenna gain coefficients
    """
    broad: list = field(default_factory=lambda: [0.0] * 5)
    end: list = field(default_factory=lambda: [[0.0] * 5 for _ in range(2)])
    phidep: list = field(default_factory=lambda: [[0.0] * 3 for _ in range(2)])
    exp: float = 0.0


# ---------------------------------------------------------------- C++ ifstream emulation - a replication of how c++ reads the mpd file
_WS = " \t\n\v\f\r" # the set of whitespace characters C++ skips before reading a number. It includes \f, which matters because the files end with a form feed.
_INT_RX = re.compile(r"[+-]?\d+") # matches an integer (optional sign, then digits).
_FLT_RX = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?") # matches a float (optional sign, digits with an optional decimal point, optional exponent like e5).


class MpdStream:
    """Mimics the parts of ifstream used: >> int/double, get(char&), getline, eof().
    Once a failbit is set every later operation is a no-op, as in C++."""

    def __init__(self, path, filename):
        file = os.path.join(path, filename) # defining the path that the file is in
        print('here', path)
        with open(file, "rb") as fh:
            raw = fh.read()
        self.s = raw.decode("latin-1").replace("\r\n", "\n")  # text-mode newline translation
        self.pos = 0
        self.eofbit = False
        self.failbit = False

    def eof(self):
        return self.eofbit

    def _good(self):
        return not (self.eofbit or self.failbit)

    def get(self):
        if not self._good():
            self.failbit = True
            return None
        if self.pos >= len(self.s):
            self.eofbit = self.failbit = True
            return None
        ch = self.s[self.pos]
        self.pos += 1
        return ch

    def _num(self, rx, conv, zero, old):
        if not self._good():
            self.failbit = True
            return old
        while self.pos < len(self.s) and self.s[self.pos] in _WS:
            self.pos += 1
        if self.pos >= len(self.s):
            self.eofbit = self.failbit = True
            return old
        m = rx.match(self.s, self.pos)
        if not m:
            self.failbit = True
            return zero
        self.pos = m.end()
        if self.pos >= len(self.s):
            self.eofbit = True
        return conv(m.group())

    def read_int(self, old):
        return self._num(_INT_RX, int, 0, old)

    def read_float(self, old):
        return self._num(_FLT_RX, float, 0.0, old)

    def getline(self, n):
        """istream::getline(buf, n). Returns the text, or None if buffer untouched."""
        if not self._good():
            self.failbit = True
            return None
        out = []
        while True:
            if self.pos >= len(self.s):
                self.eofbit = True
                if not out:
                    self.failbit = True
                return "".join(out)
            ch = self.s[self.pos]
            if ch == "\n":
                self.pos += 1
                return "".join(out)
            if len(out) >= n - 1:
                self.failbit = True
                return "".join(out)
            out.append(ch)
            self.pos += 1


# ---------------------------------------------------------------- meteor record: holds one echo record from the mpd file and knows how to read and write it.
class Meteor:
    def __init__(self):
        self.yr = self.mn = self.d = 0
        self.t = 0.0
        self.f = [" "] * 5
        self.rng = self.ht = self.vrad = self.dvr = 0.0
        self.tht = self.phi = 0.0
        self.amb = 0
        self.delph = 0.0
        self.ap = self.irex = 0
        self.amp = self.tau = self.vmet = self.snr = 0.0

    def readdata(self, st, old):
        self.yr = st.read_int(self.yr)
        self.yr = self.yr - int(self.yr / 100.0) * 100  # 2 digit year
        st.get()
        self.mn = st.read_int(self.mn)
        st.get()
        self.d = st.read_int(self.d)
        th = st.read_int(0)
        st.get()
        tm = st.read_int(0)
        st.get()
        ts = st.read_float(0.0)
        self.t = th + tm / 60.0 + ts / 3600.0
        st.get()
        for ir in range(5):
            ch = st.get()
            if ch is not None:
                self.f[ir] = ch
        self.rng = st.read_float(self.rng)
        self.ht = st.read_float(self.ht)
        self.vrad = st.read_float(self.vrad)
        self.dvr = st.read_float(self.dvr)
        self.tht = st.read_float(self.tht)
        self.phi = st.read_float(self.phi)
        self.amb = st.read_int(self.amb)
        self.delph = st.read_float(self.delph)
        self.ap = st.read_int(self.ap)
        self.irex = st.read_int(self.irex)
        self.amp = st.read_float(self.amp)
        self.tau = st.read_float(self.tau)
        # be careful of ***** in vmet
        st.get()
        vm = []
        for _ in range(5):
            ch = st.get()
            vm.append(ch if ch is not None else "0")
        if vm[0] == "*" or vm[0] == "-":
            self.vmet = -9.99
        else:
            self.vmet = ((ord(vm[0]) - 48) * 10 + (ord(vm[1]) - 48)
                         + (ord(vm[3]) - 48) / 10.0 + (ord(vm[4]) - 48) / 100.0)
        if self.vmet > 99:
            self.vmet = 99
        if old != 0 and old != 4:
            self.snr = st.read_float(self.snr)
        st.getline(200)

    def writedata(self, out, old):
        s = " " + f"{self.yr:2d}" + "/"
        s += ("0" + f"{self.mn:1d}") if self.mn < 10 else f"{self.mn:2d}"
        s += "/"
        s += ("0" + f"{self.d:1d}") if self.d < 10 else f"{self.d:2d}"
        s += "  " + writetm(self.t) + " " + "".join(self.f)
        s += f"{self.rng:6.1f}{self.ht:7.1f}{self.vrad:8.2f}{self.dvr:7.2f}"
        s += f"{self.tht:9.1f}{self.phi:9.1f}{self.amb:3d}{self.delph:8.1f}"
        s += f"{self.ap:7d}{self.irex:9d}{self.amp:7.0f}."
        s += f"{self.tau:6.3f}{self.vmet:6.2f}"
        if old != 0 and old != 4:
            s += f"{self.snr:7.1f}"
        out.write(s + "\n")


def prev(m, mp):
    """Copy m into mp. Like the original, yr/mn/d are NOT copied."""
    mp.t = m.t
    mp.f = list(m.f)
    mp.rng = m.rng
    mp.ht = m.ht
    mp.vrad = m.vrad
    mp.dvr = m.dvr
    mp.tht = m.tht
    mp.phi = m.phi
    mp.amb = m.amb
    mp.delph = m.delph
    mp.ap = m.ap
    mp.irex = m.irex
    mp.amp = m.amp
    mp.tau = m.tau
    mp.vmet = m.vmet
    mp.snr = m.snr


def writetm(t):
    th = int(math.floor(t))
    tm = int(math.floor((t - th) * 60))
    tm1 = int(tm / 10)
    tm2 = tm - tm1 * 10
    ts = (t - th - tm / 60.0) * 3600
    ts1 = int(math.floor(ts / 10))
    ts2 = ts - ts1 * 10
    ts2 = c_round(ts2 * 10) / 10
    if ts2 >= 10:
        ts2 = ts2 - 10
        ts1 = ts1 + 1
    return f"{th:2d}:{tm1:1d}{tm2:1d}:{ts1:1d}{ts2:3.1f}"


# ---------------------------------------------------------------- maths helpers
def DecayHeight(tau, lam):
    # coefficients for 2001/12/12 Tavistock - update for present day radar?
    if lam < 9:
        a, b = 0.048, 3.63
    elif lam < 15:
        a, b = 0.06, 4.7
    else:
        a, b = 0.044, 3.11
    return (_log10(_div(lam * lam, 16 * pi * pi * tau)) + b) / a


def Dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def RDectoenu(ras, dc, t, lt):
    """RA, Dec -> east, north, up (needs latitude and lst). Returns [e, n, u]."""
    td = t * 360  # time in degrees
    r0 = math.cos(dtr * dc) * math.cos(dtr * ras)
    r1 = math.cos(dtr * dc) * math.sin(dtr * ras)
    r2 = math.sin(dtr * dc)
    v0 = -r0 * math.sin(dtr * td) + r1 * math.cos(dtr * td)
    v1 = (-r0 * math.cos(dtr * td) * math.sin(dtr * lt)
          - r1 * math.sin(dtr * td) * math.sin(dtr * lt) + r2 * math.cos(dtr * lt))
    v2 = (r0 * math.cos(dtr * td) * math.cos(dtr * lt)
          + r1 * math.sin(dtr * td) * math.cos(dtr * lt) + r2 * math.sin(dtr * lt))
    return [v0, v1, v2]


def norm(v):
    # NB: in place and sequential, exactly as the original (later components
    # are divided by a magnitude that already includes the normalised earlier ones)
    for i in range(3):
        v[i] = v[i] / math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def rang(r, r0, rf):
    if r > rf:
        r = r - (rf - r0)
    elif r < r0:
        r = r + (rf - r0)
    return r


def eq(a1, a2):
    return 1 if (a1[0] == a2[0] and a1[1] == a2[1] and a1[2] == a2[2]) else 0


def toupper(c):
    return c.upper() if "a" <= c <= "z" else c


def chardate(year, mnth, day):
    """Returns (ldate, date): yyyymmdd and yymmdd."""
    yr = [""] * 4
    if year < 1000:  # two digit year
        if year > 90:
            yr[0], yr[1] = "1", "9"
        else:
            yr[0], yr[1] = "2", "0"
    else:
        i = year // 1000
        j = (year - 1000 * i) // 100
        yr[0] = chr(i + 48)
        yr[1] = chr(j + 48)
    i = year % 10
    j = (year % 100 - i) // 10
    yr[2] = chr(j + 48)
    yr[3] = chr(i + 48)
    m = chr(mnth // 10 + 48) + chr(mnth % 10 + 48)
    d = chr(day // 10 + 48) + chr(day % 10 + 48)
    ldate = "".join(yr) + m + d
    date = yr[2] + yr[3] + m + d
    return ldate, date


def Shower2(met, rd, shv, year, mnth, day, lng, lat, elv, radsize):
    """0 if meteor is not from the listed shower, 1 if it is."""
    sti = SidTime(met.t, year, mnth, day, lng)
    ech = [math.cos(met.phi * dtr) * math.sin(met.tht * dtr),
           math.sin(met.phi * dtr) * math.sin(met.tht * dtr),
           math.cos(met.tht * dtr)]
    norm(ech)

    sti_rad = sti * 2 * pi
    rd_rad = [rd[0] * pi / 180, rd[1] * pi / 180]
    lat_rad = lat * dtr
    lng_rad = lng * dtr

    v_app, alp_p, del_p = speed_geo_app(shv, rd_rad[0], rd_rad[1], sti_rad, lat_rad, lng_rad, elv)
    rd_app = [alp_p * 180 / pi, del_p * 180 / pi]

    Sh = RDectoenu(rd_app[0], rd_app[1], sti, lat)
    norm(Sh)
    if Sh[2] > 0:  # shower above horizon
        if abs(Dot(Sh, ech)) < math.sin(dtr * radsize): # on shower echo line?
            return 1
        return 0
    return 0


# ---------------------------------------------------------------- main routine - first function that is being run
def mpd_an_shrm(year, mnth, day, freq, idir, showspor, lat, lng, srd, shv, GT, shrm,
                rdbig, shvb, thtlim, senslim, s, radsize, tgaps):
    """
    Returns (ret, GT, noise, snr).  tgaps (length-2 list) is modified in place.
    ret is 0 on success, 1 if the mpd file was not found (GT is then returned as passed in).
    """
    tgap = 2.0
    elv = 100.0  # approximate elevation above sea level for meteors
    met, metp, met1 = Meteor(), Meteor(), Meteor()
    ngaps = 0
    q = 0
    ng = 0
    a1 = 0
    c = "\0"

    # tgaps: first and last times of major gaps (in hours?). If more than one, day currently discarded.
    tgaps[0] = 0
    tgaps[1] = 1

    # frequency digit for the mpd file name
    f = chr(int(math.trunc(freq / 10)) + 48)
    if year == 1999 and (mnth == 10 or mnth == 11):
        th = 319.6  # in Alert
    elif year == 2011 and mnth == 10 and 3 <= day <= 11 and f == "3":
        th = 254.0
    else:
        th = 344.0  # angle of interferometer relative to North (degrees)

    # Transmitting Antenna Bin
    TxCo = GainCo()
    TxCo.broad = [5.5781236, -2.8200065, -3.5209044, 10.400722, -9.5485295]
    TxCo.end = [[0.031795133, 4.8990288, 13.669295, -25.170221, 22.814163],
                [-0.021062092, 7.1497901, -3.3448948, 13.533221, -4.7793303]]
    TxCo.phidep = [[7.7827215, -8.1684971, 11.627713],
                   [6.4776762, -4.5794876, 8.1254215]]
    TxCo.exp = 0.6

    # Recieving Antenna Bin
    RxCo = GainCo()
    RxCo.broad = [4.1036476, 1.5400743, -3.1886833, 2.9979406, -5.3852557]
    RxCo.end = [[-0.006819758, 7.9024553, -1.538963, 12.2719, -4.0899217],
                [-0.005270753, 6.3391005, 3.3638038, 0.36978112, 2.3794633]]
    RxCo.phidep = [[8.2011129, -9.6857106, 13.001805],
                   [6.2636256, -3.9807794, 7.4747311]]
    RxCo.exp = 0.65

    noise = 0.0
    snr = 0.0
    nnoise = 0
    lam = 3e8 / (freq * 1e6)  # wavelength of radar

    # branches to determine which version of mpd is being worked with
    if year == 1999 and (mnth == 10 or mnth == 11):
        old = 4
    elif year < 2001 or (year == 2001 and mnth < 5) or (year == 2001 and mnth == 5 and day < 27):
        old = 0  # mpd version 2.0
    elif year == 2001 or (year == 2002 and mnth < 11) or (year == 2002 and mnth == 11 and day < 22):
        old = 1  # version 2.1
    elif year < 2009 or (year == 2009 and (mnth < 4 or (mnth == 4 and day < 3)
                                           or (mnth == 4 and day > 27) or (mnth == 5 and day < 26))):
        old = 2  # version 2.2
    # this branch might only apply to the 29 MHz - 38 MHz mpd files after this time are still tindal3
    elif year < 2025 or (year == 2025 and (mnth < 11 or (mnth == 11 and day < 26))):
        old = 3  # tindal2a
    else:
        old = 5  # tindal2b

    ldate, date = chardate(year, mnth, day)

    # name of mpd file
    mpd_name = "mp" + ldate
    print(mpd_name)
    if old == 0:
        mpd_name += ".Tavis"
    elif old == 4:
        mpd_name += ".Alert"
    else:
        mpd_name += ".tindal"
    mpd_name += f 
    if old == 3 and f == "2":
        mpd_name += "a"
    if old == 5 and f == "2":
        mpd_name += "b"
    mpd_name += ".mpd"

    # name of mc, sc or xc output file - gets saved to the same directory as idir
    mpnm = ("mc" if showspor == 0 else "sc" if showspor == 1 else "xc") + date + ".dat"

    # directory the .dat files for verified meteors get saved to
    output_path = os.path.join(mac_path_out, mpnm)

    # check for existence of mpd file, then create result file
    try:
        st = MpdStream(mac_path, mpd_name)
    except OSError:
        print("Could not find file: " + mpd_name)
        return 1, GT, noise, snr
    out = open(output_path, "w", encoding="latin-1")  # native newline translation, like C++ text mode

    # skipping header lines
    nlines = 21 if old == 0 else 27 if old == 1 else 29
    header = ""
    for _ in range(nlines):
        got = st.getline(300)
        if got is not None:
            header = got
    # the last one read in was the column headers
    out.write(header + "\n")

    j = 0
    GT = 0

    # ---- closures for logic the original repeats inline
    def getc():
        nonlocal c
        ch = st.get()
        if ch is not None:
            c = ch

    def check_time(tm):
        nonlocal j, prevt, GT, ngaps
        if tm > 24.1 or tm < 0:
            j = 100000
        else:
            if tm - prevt > tgap:
                GT = 1
                ngaps += 1
                tgaps[0] = prevt
                tgaps[1] = tm
            prevt = tm

    def emit(m):
        """Good echo: add to noise calculation, then filter by shower membership and write."""
        nonlocal noise, snr, nnoise
        noise += _div(m.amp * m.amp, _pow(10, m.snr / 10))
        nnoise += 1
        snr += m.snr
        if showspor == 0:  # no filters
            m.writedata(out, old)
        else:
            shmem = Shower2(m, srd, shv, year, mnth, day, lng, lat, elv, radsize)
            if showspor == 2 and shmem == 0: # only write sporadic data
                m.writedata(out, old)
            elif showspor == 1 and shmem == 1:  # member of shower we want to keep
                if shrm == 0:
                    m.writedata(out, old) # not shower member, so only write sporadic data?
                else:
                    # on the major shower echo line? then remove
                    bgshmem = Shower2(m, rdbig, shvb, year, mnth, day, lng, lat, elv, radsize)
                    if bgshmem == 0: # not on echo line
                        m.writedata(out, old) # write sporadic data?

    # starting to read lines in the current file here
    metp.readdata(st, old)
    prevt = metp.t
    if prevt > tgap:
        GT = 1
        tgaps[0] = 0
        tgaps[1] = prevt
        ngaps += 1
    getc()
    slim = _pow(senslim, s - 1)

    # ---- find first good echo
    bp = 1
    while j < 100000 and c != "\f" and not st.eof() and bp == 1:
        bp = 0  # assume the meteor is good until we see that it's bad
        check_time(metp.t)
        # decay times valid, elevation above limit, sensitivity sufficient
        if metp.tau < 0.004 or metp.tau > 9 or metp.tht > thtlim:
            bp = 1
            metp.readdata(st, old)
            j += 1
            continue
        # vector representation of the previous meteor's position in spherical coordinates
        sp = Vec(metp.rng * math.cos(metp.phi * dtr) * math.sin(metp.tht * dtr),
                 metp.rng * math.sin(metp.phi * dtr) * math.sin(metp.tht * dtr),
                 metp.rng * math.cos(metp.tht * dtr))
        sen = sens(sp, s, th, TxCo, RxCo)
        if sen < slim:
            bp = 1
            metp.readdata(st, old)
            j += 1
            continue

        # if the previous meteor event is ambiguous
        if metp.amb != 1:
            a1 = metp.amb
            ng = 0
            if metp.tau == 0:
                metp.tau = 0.001
            htc = DecayHeight(metp.tau, lam)
            if abs(htc - metp.ht) < 3:
                ng += 1
                q = 1
            else:
                q = 0
            for k in range(1, a1):
                met1.readdata(st, old)
                getc()
                if abs(metp.t - met1.t) > 5e-7:  # shouldn't happen: same time expected
                    bp = 1
                    prev(met1, metp)
                    check_time(met1.t)
                    j += 1
                    break
                if ng > 1:
                    bp = 1
                    continue
                if met1.tau == 0:
                    met1.tau = 0.001
                htc = DecayHeight(met1.tau, lam)
                if abs(htc - met1.ht) < 3: # decay height too low?
                    ng += 1
                if ng == 1:
                    if q == 1:  # metp is good, met1 not
                        continue
                    else:  # met1 good
                        prev(met1, metp)
                        q = 1
                else:  # either both good or both bad: don't use either
                    bp = 1
                    continue

    ql = 0
    # ---- main loop
    while j < 100000 and c != "\f" and not st.eof():
        b = 0
        met.readdata(st, old)
        check_time(met.t)
        j += 1
        getc()
        if met.tau < 0.004 or met.tau > 9 or met.tht > thtlim:
            b = 1
            met.readdata(st, old)
            j += 1
            continue
        sp = Vec(met.rng * math.cos(met.phi * dtr) * math.sin(met.tht * dtr),
                 met.rng * math.sin(met.phi * dtr) * math.sin(met.tht * dtr),
                 met.rng * math.cos(met.tht * dtr))
        sen = sens(sp, s, th, TxCo, RxCo)
        if sen < slim:
            b = 1
            met.readdata(st, old)
            j += 1
            continue
        if met.amb != 1:
            a1 = met.amb
            ng = 0
            if met.tau == 0:
                met.tau = 0.001
            htc = DecayHeight(met.tau, lam)
            if abs(htc - met.ht) < 3:
                ng += 1
                q = 1
            else:
                q = 0
            for k in range(1, a1):
                met1.readdata(st, old)
                getc()
                if abs(met.t - met1.t) > 5e-7:  # shouldn't happen
                    bp = 1
                    prev(met1, met)
                    check_time(met1.t)
                    j += 1
                    break
                if ng > 1:
                    b = 1
                    continue
                if met1.tau == 0:
                    met1.tau = 0.001
                htc = DecayHeight(met1.tau, lam)
                if abs(htc - met1.ht) < 3:
                    ng += 1
                if ng == 1:
                    if q == 1:  # met is good, met1 not
                        continue
                    else:  # met1 good
                        prev(met1, met)
                        q = 1
                else:  # both good or both bad: don't use either
                    b = 1
                    continue
        ql = 0
        if b == 0:
            ql = 1
            # check to see if metp, met occur at same time in same place
            if (abs(metp.t - met.t) < 0.0002778 and abs(metp.phi - met.phi) < 12
                    and abs(metp.tht - met.tht) < 4 and abs(metp.rng - met.rng) <= 12):
                if metp.amp < met.amp:
                    prev(met, metp)
                bp = 1
            if bp == 0:
                if 0 <= metp.t <= 24.1:
                    emit(metp)
                prev(met, metp)
            bp = 0

    if ql == 1:  # end of file after reading good met
        if 0 <= met.t <= 24.1:
            emit(met)

    if nnoise > 1:
        noise /= nnoise
        snr /= nnoise
    else:
        noise = 0.0
    if prevt < (24 - tgap):
        GT = 1
        ngaps += 1
        tgaps[0] = prevt
        tgaps[1] = 24
    if ngaps > 1:
        GT = 2

    out.close() # out is an output file that meteor data is written to
    return 0, GT, noise, snr


if __name__ == '__main__':

    tgaps = [0, 0]
    
            
    ret, GT, noise, snr = mpd_an_shrm(
        2016, 3, 25, 28, mac_path, 2,   # year, month, day, freq, idir, showspor
        43.0, -81.3,                      # lat, lng
        [120, 20], 40, 0,                 # srd, shv, GT (good time - 1 if there's one large time gap)
        0,                                # shrm
        [200, -10], 35,                   # rdbig, shvb
        70, 8e9, 1.5, 25, tgaps)          # thtlim, senslim, s, radsize, tgaps
    print(ret, GT, noise, snr, tgaps)


    # script to put into linux version of this code - this desktop has listed path containing the remaining data needed
    # this loop runs only to define the date of the file so each one in the colossid directory can be passed through the mpd function call
        # mpd does all the work reading the file data for each new day
    for file in colossid_path:

        # skips pngs or other files that is not the explicit data
        if file[-4:] != '.mpd':
            continue

        year = int(file[2:6])
        month = int(file[6:8])
        date = int(file[8:10])

        tgaps = [0, 0]

        file_path = os.path.join(colossid_path, file)

        with open(file_path, 'r') as mpd_data:
        
            ret, GT, noise, snr = mpd_an_shrm(
                year, month, date, 38, colossid_path, 2,   # year, month, day, freq, idir, showspor
                43.2, -80.7,                      # lat, lng
                [120, 20], 40, 0,                 # srd, shv, GT (good time - 1 if there's one large time gap)
                0,                                # shrm
                [200, -10], 35,                   # rdbig, shvb
                70, 8e9, 1.5, 25, tgaps)          # thtlim, senslim, s, radsize, tgaps
            print(ret, GT, noise, snr, tgaps)