import psrchive as psr
import numpy as np
import os
import re
import shutil
import sys
from os.path import split
from os.path import basename
from os.path import dirname
import os.path
import errno
import subprocess
from subprocess import check_output

import astropy.coordinates as coord
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

from astropy.utils import iers
#from astropy.utils.iers import conf
try:
    iers.conf.auto_max_age = None
    iers.conf.auto_download = False
    iers.IERS.iers_table = iers.IERS_A.open('/home/lbondonneau/lib/python/astropy/utils/iers/data/finals2000A.all')
except:
    print('WARNING: Can not use iers.conf probably due to the astropy version')

#from coast_guard import cleaners

def clean_workingdir(WORKDIR):
    """
    creat a new dir name WORKDIR to the path or clean the WORKDIR if not empty
    """
    try:
        fp = open(WORKDIR)
    except IOError as e:
        if e.errno == errno.EACCES:
            print('Permission denied to WORKDIR = %s' % WORKDIR)
        if e.errno == errno.ENOENT:
            print('No such file or directory = %s' % WORKDIR)

    if(os.path.isdir(WORKDIR+'/WORKDIR/')):
        for the_file in os.listdir(WORKDIR+'/WORKDIR_'+str(os.getpid())+'/'):
            file_path = os.path.join(WORKDIR+'/WORKDIR_'+str(os.getpid())+'/', the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    else:
        try:
            original_umask = os.umask(0)
            os.mkdir(WORKDIR+'/WORKDIR_'+str(os.getpid()), 0777)
        finally:
            os.umask(original_umask)
    return WORKDIR+'/WORKDIR_'+str(os.getpid())+'/'

def remove_workingdir(WORKDIR):
    """
    rm dir name WORKDIR to the path or clean the WORKDIR if not empty
    """
    try:
        shutil.rmtree(WORKDIR)
        return 'OK'
    except Exception as e:
        print(e)
        return '0'

def archive_TO_elev_start_end(archive):
    """
    Fonction to give elevation in the first and last subintegration
    EarthLocation is set for Nancay

    Input:
        archives : list of PSRCHIVE archive objects
    Output
        float(elev0),float(elevlast)
    """
    try:
        x, y, z = archive.get_ant_xyz()
    except:
        print('warning: archive.get_ant_xyz() faild will used nancay location')
        x=4324016.70769
        y=165545.525467
        z=4670271.363
    Site = EarthLocation.from_geocentric(x=float(x)*u.m, y=float(y)*u.m, z=float(z)*u.m)

    ra = archive.get_coordinates().ra().getDegrees()
    dec = archive.get_coordinates().dec().getDegrees()
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)

    MJD0 = archive.get_Integration(0).get_epoch().in_days()
    MJDlast = archive.get_Integration(int(archive.get_nsubint())-int(1)).get_epoch().in_days()

    MJD0 = Time(MJD0, format='mjd')
    MJDlast = Time(MJDlast, format='mjd')

    elev0 = c.transform_to(AltAz(obstime=MJD0, location=Site))
    elevlast = c.transform_to(AltAz(obstime=MJDlast, location=Site))

    return elev0.alt.degree, elevlast.alt.degree



def load_some_chan_in_archive_data(path, WORKDIR='NULL', initmetadata=False,
                                   minfreq=0, maxfreq=2**24,
                                   verbose=True, freqappend=False, singlepulses_patch=False,
                                   bscrunch=1, tscrunch=1, fscrunch=1, pscrunch=False, dm=0.0, rm=0.0, nodefaraday=False):
    """Function to load .ar files and convert to PSRCHIVE archive objects.

    Input:
        path    : full path to location of the .ar files.
        verbose : option to run in verbose mode (default=False)
    Options:
        minfreq : high-band filter
        maxfreq : low-band filter
    Output:
        archives : list of PSRCHIVE archive objects
    """
    archives = []
    files = []
    for file in path:
        if(dirname(file) == ''):
            path = './'
        else:
            path = dirname(file)+'/'
        file = basename(file)
        files.append(file)
    files.sort()
    print("path used: %s" %(path))
    print("file(s) used: %s" %(files))
    archives = [psr.Archive_load(path+files[0])]

    if not(WORKDIR == 'NULL'):
        orig_stdout = sys.stdout
        printfile = open(WORKDIR+'9_processed_files.txt', 'w')
        sys.stdout = printfile

    if(verbose):
        print '======================================================================================================'
        print '                                     File(s) to be processed:                                         '
        print '======================================================================================================'
    for i in range(len(files)):
        archives = psr.Archive_load(path + files[i])
        buffarchive = archives.clone()

        freqs = np.zeros(archives.get_nchan())
        for ichan in range(archives.get_nchan()):
            freqs[ichan] = archives.get_Profile(0, 0, ichan).get_centre_frequency()

        if(np.min(freqs) < minfreq):
            minchan = (np.argmax(freqs[np.where(freqs < minfreq)])+1)
        else:
            minchan = 0

        if(np.max(freqs) > maxfreq):
            maxchan = (np.argmax(freqs[np.where(freqs < maxfreq)]))
        else:
            maxchan = archives.get_nchan()-1

        if (initmetadata):
            initTFB = [archives.get_nsubint(), archives.get_nchan(), archives.get_nbin()]
        else:
            initTFB = [0, 0, 0]
        if (maxchan < buffarchive.get_nchan()-1):
            buffarchive.remove_chan(maxchan+1, buffarchive.get_nchan()-1)
        if (minchan > 0):
            buffarchive.remove_chan(0, minchan-1)
        if (pscrunch):
            buffarchive.pscrunch()
        if(buffarchive.get_npol() == 1):
            nodefaraday = True
        if not (dm==0.0):
            buffarchive.set_dispersion_measure(float(dm))
            if not (nodefaraday): buffarchive.dedisperse()
        else:
            buffarchive.dedisperse()
        if not (rm == 0.0):
            buffarchive.set_rotation_measure(float(rm))
            if not (nodefaraday): buffarchive.defaraday()
        else:
            if not (buffarchive.get_rotation_measure() == 0.0) and (buffarchive.get_npol() > 1):
                if not (nodefaraday): buffarchive.defaraday()
        if (tscrunch > 1):
            buffarchive.tscrunch(tscrunch)
        if (bscrunch > 1):
            if(buffarchive.get_nbin()/bscrunch < 8):
                bscrunch = buffarchive.get_nbin()/8
            buffarchive.bscrunch(bscrunch)
        if (fscrunch > 1):
            buffarchive.fscrunch(fscrunch)
        if (i == 0):
            newarchive = buffarchive.clone()
            string = ''
            if (minchan > 0):
                string = string + ("minfreq = %.8f " % freqs[minchan])
            if (maxchan < buffarchive.get_nchan()-1):
               string = string + ("maxfreq = %.8f " % freqs[maxchan])
            if (pscrunch):
                string = string + ("pscrunch = True ")
            if (bscrunch > 1):
                string = string + ("bscrunch = %d " % bscrunch)
            if (tscrunch > 1):
                string = string + ("tscrunch = %d " % tscrunch)
            if (fscrunch > 1):
                string = string + ("fscrunch = %d " % fscrunch)
            if not string == '':
                print(string)
        else:
            if (freqappend):
                freqappend = psr.FrequencyAppend()
                patch = psr.PatchTime()
                # This is needed for single-pulse data:
                if (singlepulses_patch): patch.set_contemporaneity_policy("phase")
                freqappend.init(newarchive)
                freqappend.ignore_phase = True
                polycos = newarchive.get_model()
                buffarchive.set_model(polycos)
                patch.operate(newarchive,buffarchive)
                freqappend.append(newarchive,buffarchive)
                newarchive.update_centre_frequency()
            else:
                polycos = newarchive.get_model()
                buffarchive.set_model(polycos)
                newarchive.append(buffarchive)

        if verbose:
            print path + files[i]

    if not(WORKDIR == 'NULL'):
        sys.stdout = orig_stdout
        printfile.close()
    return newarchive, files[0].split('.')[0], initTFB


def load_archive_data(path, verbose=False):
    """Function to load .ar files and convert to PSRCHIVE archive objects.

    Input:
        path    : full path to location of the .ar files.
        verbose : option to run in verbose mode (default=False)

    Output:
        archives : list of PSRCHIVE archive objects
    """
    files = []
    for file in os.listdir(path):
        if file.endswith('.ar'):
            files.append(file)
    files.sort()
    archives = []
    archives = [psr.Archive_load(path + file) for file in files]
    if verbose:
        print '======================================================================================================'
        print '                                     Files to be processed:                                           '
        print '======================================================================================================'
    for i in range(1, len(archives)):
        archives[0].append(archives[i])
        # add the .ar files (added file is archive[0])
        if verbose:
            print archives[i]
    return archives

def bandpass_filter(ar, minfreq=0, maxfreq=2**24):
    """Fonction to app a bandpass filter on an archive

    Input:
        archive: PSRCHIVE Archive object.
        minfreq: minimal frequency in MHz.
        maxfreq: maximum frequency in MHz.

    Output:
        archive: PSRCHIVE Archive object.

    """
    if (ar.get_telescope() == 'nenufar'):
        if ( minfreq == 0 ):
            minfreq = 20
        if ( maxfreq == 2**24 ):
            maxfreq = 87

    AR_minfreq = float(ar.get_Profile(0, 0, 0).get_centre_frequency())
    AR_maxfreq = float(ar.get_Profile(0, 0, ar.get_nchan()-1).get_centre_frequency())
    AR_freq = np.linspace(AR_minfreq, AR_maxfreq, ar.get_nchan())

    for isub in range(ar.get_nsubint()):
        for ipol in range(ar.get_npol()):
            for ichan in range(ar.get_nchan()):
                freq = AR_freq[ichan]
                prof = ar.get_Profile(isub, ipol, ichan)
                if( minfreq > freq ) or ( freq > maxfreq):
                    prof.set_weight(0.0)
    return ar

def print_metadata(archive, WORKDIR='NOWORKDIR', arpath='/tmp/', initTFB=[0, 0, 0], initmetadata=False):
    """Function to print archive file metadata in a single string.

    Input:
        archive: PSRCHIVE Archive object.

    Output:
        print metadata in a nice table.
    """
    arx = archive.clone()
    if not(WORKDIR == 'NOWORKDIR'):
        orig_stdout = sys.stdout
        headerfile = open(WORKDIR+'0_header.txt', 'w')
        sys.stdout = headerfile

    # get metadata from header
    OUTPUT = ''
    PRECISETIME = ''
    UTC_START = ''
    ANTENNAE = ''
    if os.path.isfile(arpath+'/obs.header'):
        headerfile = open(arpath+'/obs.header', "r")
        for line in headerfile:
            if re.search("PRECISETIME_", line):
                lenstring = 51
                string = line.strip().split(' ')[0]
                PRECISETIME = PRECISETIME+'\n'+string+(lenstring-len(string))*' '+re.sub('\s+', ' ', line).split(' ')[1]
            if re.search("UTC_START", line):
                UTC_START = line.strip()
                UTC_START = re.sub('\s+', ' ', UTC_START).split(' ')[1]
            if re.search("ANTENNAE", line):
                ANTENNAE = line.strip() #.split(' ')[1:].strip(' ')
                ANTENNAE = sorted(re.sub('\s+', ' ', ANTENNAE).split(' ')[1].split(','))
       #OUTPUT = ("""name         Source name                           %s
        OUTPUT = ("""start        UTC start time                        %s
antennae     liste of used antennae                %s
nb antenna   number of antennae                    %s""" % (UTC_START,
                                                            ANTENNAE,
                                                            len(ANTENNAE)))
    # get metadata from file
    if not (initmetadata): # default
        nbin = archive.get_nbin()
        nchan = archive.get_nchan()
        nsubint = archive.get_nsubint()
        
    else:
        nsubint = initTFB[0]
        nchan = initTFB[1]
        nbin = initTFB[2]
        if (archive.get_nsubint() > initTFB[0]):
            nsubint = archive.get_nsubint()
        if (archive.get_nchan() > initTFB[1]):
            nchan = archive.get_nchan()
    npol = archive.get_npol()
    obs_type = archive.get_type()
    telescope_name = archive.get_telescope()
    source_name = archive.get_source()
    ra = archive.get_coordinates().ra().getHMS()
    dec = archive.get_coordinates().dec().getDMS()
    centre_frequency = archive.get_centre_frequency()
    bandwidth = archive.get_bandwidth()
    dm = archive.get_dispersion_measure()
    rm = archive.get_rotation_measure()
    is_dedispersed = archive.get_dedispersed()
    is_faraday_rotated = archive.get_faraday_corrected()
    is_pol_calib = archive.get_poln_calibrated()
    data_units = archive.get_scale()
    data_state = archive.get_state()
    obs_duration = archive.integration_length()
    receiver_name = archive.get_receiver_name()
    receptor_basis = archive.get_basis()
    backend_name = archive.get_backend_name()
    low_freq = centre_frequency - bandwidth / 2.0
    high_freq = centre_frequency + bandwidth / 2.0
    freq_vec = np.linspace(low_freq+(bandwidth/(2*nchan)), high_freq-(bandwidth/(2*nchan)), nchan)
    subint_duration = []
    for isub in range(0,archive.get_nsubint()):
        subint_duration.append(archive.get_Integration(isub).get_duration())
    subint_duration = np.median(subint_duration)
    MJDstart = archive.get_Integration(0).get_epoch().in_days()
    elevStart, elevEnd = archive_TO_elev_start_end(archive)
    weights = arx.get_weights()
    weights = weights/np.max(weights)
    RFI = 100.*(1.-np.mean(weights))
    if(high_freq < 100):
        weights20_85 = weights[:, np.argmin(np.abs(freq_vec-20)):np.argmin(np.abs(freq_vec-85))]
        RFI20_85 = 100.*(1.-np.mean(weights20_85))
        RFI_string = ("RFI 20-80    RFI (/100) from 20 to 80 MHz          %.2f"% RFI20_85)
    elif(high_freq < 200):
        weights20_85 = weights[:, np.argmin(np.abs(freq_vec-115)):np.argmin(np.abs(freq_vec-185))]
        RFI20_85 = 100.*(1.-np.mean(weights20_85))
        RFI_string = ("RFI 115-185  RFI (/100) from 115 to 185 MHz        %.2f"% RFI20_85)
    else:
        RFI_string = ''
    arx.pscrunch()
    arx.fscrunch()
    arx.tscrunch()
    SNR = arx.get_Profile(0, 0, 0).snr()
    SNRvlad, off_left, off_right, rot_bins, bscr = auto_snr_onpulse(arx)
    source_folding_period = arx.get_Integration(0).get_folding_period()
    julianDAY = MJD_TO_JULIEN(MJDstart)
    # Print out metadata
    HEADER = ("""==============================================================
Attribute    Description                           Value
==============================================================""")
    OUTPUT = OUTPUT+'\n'+("""name         Source name                           %s
Start        %s
nbin         Number of pulse phase bins            %s
nchan        Number of frequency channels          %s
npol         Number of polarizations               %s
nsubint      Number of sub-integrations            %s
length       Observation duration (s)              %s
dm           Dispersion measure (pc/cm^3)          %s
rm           Rotation measure (rad/m^2)            %s
period topo  Folding_period (s)                    %s
type         Observation type                      %s
site         Telescope name                        %s
coord ra     Source coordinates (hms)              %s
coord dec    Source coordinates (dms)              %s
freq         Centre frequency (MHz)                %s
bw           Bandwidth (MHz)                       %s
dmc          Dispersion corrected                  %s
rmc          Faraday Rotation corrected            %s
polc         Polarization calibrated               %s
scale        Data units                            %s
stat         Data state                            %s
rcvr:name    Receiver name                         %s
rcvr:basis   Basis of receptors                    %s
be:name      Name of the backend instrument        %s
MJDstart     MJD of the first subintegration       %s
SNR(psrstat) Signal noise ratio                    %.1f
SNR(range)   Signal noise ratio with vlad script   %.1f
RFI          Radio Frequency Interferency (/100)   %.2f
%s
elevStart    Elevation of the first subintegration %.2f
elevEnd      Elevation of the last subintegration  %.2f""" % (source_name,
                                                            str(julianDAY),
                                                            nbin,
                                                            nchan,
                                                            npol,
                                                            nsubint,
                                                            obs_duration,
                                                            dm,
                                                            rm,
                                                            source_folding_period,
                                                            obs_type,
                                                            telescope_name,
                                                            str(ra),
                                                            str(dec),
                                                            centre_frequency,
                                                            bandwidth,
                                                            is_dedispersed,
                                                            is_faraday_rotated,
                                                            is_pol_calib,
                                                            data_units,
                                                            data_state,
                                                            receiver_name,
                                                            receptor_basis,
                                                            backend_name,
                                                            MJDstart,
                                                            SNR,
                                                            SNRvlad,
                                                            RFI,
                                                            RFI_string,
                                                            elevStart,
                                                            elevEnd))
    if not (UTC_START == ''):
        OUTPUT = HEADER+'\n'+OUTPUT+'\n'+PRECISETIME
    if not(WORKDIR == 'NOWORKDIR'):
        print(OUTPUT)
        sys.stdout = orig_stdout
        headerfile.close()
    return OUTPUT

def apply_mask(archive, path_to_mask):
    if os.path.isfile(path_to_mask):
        mask = np.genfromtxt(path_to_mask)
        for isub in range(archive.get_nsubint()):
            for ipol in range(archive.get_npol()):
                for ichan in range(archive.get_nchan()):
                    prof = archive.get_Profile(isub, ipol, ichan)
                    prof.set_weight(mask[isub, ichan])
    else:
        print("Inout mask is not a file %s" % args.mask )
        exit()

def correct_bp(ar):
    """
    function for band-pass  (IN CONSTRUCTION)
    """
    arx = ar.clone()
    arx.dedisperse()
    #arx.tscrunch()

    # rescale the baseline
    arx2 = arx.clone()
    arx2.pscrunch()
    arx2.tscrunch()

    # weights
    weights = arx.get_weights()
    weights = weights.squeeze()
    weights = weights/np.max(weights)

    subint = arx2.get_Integration(0)
    (bl_mean, bl_var) = subint.baseline_stats()
    bl_mean = bl_mean.squeeze()
    non_zeroes = np.where(bl_mean != 0.0)
    arx.remove_baseline()
    for isub in range(arx.get_nsubint()):
        for ichan in range(arx.get_nchan()):
            for ipol in range(arx.get_npol()):
                prof = arx.get_Profile(isub, ipol, ichan)
                if ichan in non_zeroes[0] and (prof.get_weight() != 0):
                    prof.scale(1/bl_mean[ichan])
                else:
                    prof.set_weight(0.0)
                    prof.scale(0)
    return arx


def clean_archive_surgical(archive_clone, chan_threshold=3, subint_threshold=3, chan_numpieces=1, subint_numpieces=1):
    """Function to clean the archive files using coast_guard cleaner surgical.
    
       Input:
           archive_clone:    a clone of the PSRCHIVE archive object.
           chan_threshold:   threshold sigma of a profile in a channel.
           subint_threshold: threshold sigma of a profile in a sub-intergration.
           chan_numpieces:   the number of equally sized peices in each channel (used for detranding in surgical)
           subint_numpieces: the number of equally sized peices in each sub-int (used for detranding in surgical)
    """
    cleaner = cleaners.load_cleaner('surgical')
    cleaner.parse_config_string('chan_numpieces=%s,subint_numpieces=%s,chanthresh=%s,subintthresh=%s'\
                               % (str(chan_numpieces), str(subint_numpieces), str(chan_threshold),\
                                  str(subint_threshold)))
    cleaner.run(archive_clone)
    
def clean_archive_rcvrstd(archive_clone, bad_channels='0:210;3896:4095', bad_frequencies=None, bad_subints=None, \
                          trim_bw=0, trim_frac=0, trim_num=0):
    """Function to clean the archive files using coast_guard cleaner rcvrstd.
    
       Input:
           archive_clone:    a clone of the PSRCHIVE archive object to clean.
           bad_channels:     bad channels to de-weight (default: band edges (0:210, 3896:4095).
           bad_frequencies:  bad frequencies to de-weight (default: None).
           bad_subints:      bad sub-ints to de-weight (default: None).
           trim_bw:          bandwidth of each band-edge (in MHz) to de-weight (default: None).
           trim_frac:        fraction of each band-edge to de-weight, float between 0 - 0.5 (default: None).
           trim_num:         number of channels to de-weight at each edge of the band (default: None).
    """
    cleaner2 = cleaners.load_cleaner('rcvrstd')
    
    cleaner2.parse_config_string('badchans=%s,badfreqs=%s,badsubints=%s,trimbw=%s,trimfrac=%s,trimnum=%s' \
                                 %(str(bad_channels), str(bad_frequencies), str(bad_subints), \
                                   str(trim_bw), str(trim_frac), str(trim_num)))
    cleaner2.run(archive_clone)
    
def clean_archive_bandwagon(archive_clone, bad_chan_tol=0.9, bad_sub_tol=1.0):
    """Function to clean the archive files using coast_guard cleaner bandwagon.
       Input:
            archive_clone:    a clone of the PSRCHIVE archive object to clean.
            bad_chan_tol:     fraction of bad channels to be tolarated before mask (float between 0 - 1).
            bad_sub_tol:      fraction of bad sub-intergrations to be tolerated before mask (float between 0 - 1)
    """
    cleaner3 = cleaners.load_cleaner('bandwagon')
    cleaner3.parse_config_string('badchantol=0.99,badsubtol=1.0')
    cleaner3.run(archive_clone)

def bestprof_rotate (x, rot_bins):
    if rot_bins == 0: return x
    if abs(rot_bins) < 1.0: # it means that rot_bins is in pulse phase
        rot_bins = (rot_bins/abs(rot_bins))*int(abs(rot_bins)*len(x)+0.5)
    if abs(rot_bins) >= len(x):
        rot_bins = (rot_bins/abs(rot_bins))*(int(abs(rot_bins)+0.5)%len(x))
    if rot_bins > 0:
        out=np.append(x[int(rot_bins):],x[:int(rot_bins)])
    else:
        out=np.append(x[int(len(x)-abs(rot_bins)):],x[:int(len(x)-abs(rot_bins))])
    return out

def MJD_TO_JULIEN(MJD):
    t = Time(MJD, format='mjd', scale='utc')
    return t.iso

# exclude single bins representating 1-bin outliers
def trim_bins(x):
    x_diffs=[x[ii]-x[ii-1] for ii in xrange(1, len(x))]
    # trim left side
    cut_bin = 0
    for ii in xrange(0, len(x_diffs)/2):
        if x_diffs[ii] == 1: 
            if cut_bin != 0 : x=x[cut_bin:]
            break
        else: cut_bin += 1
    # trim right side
    cut_bin = 0
    for ii in xrange(len(x_diffs)-1, len(x_diffs)/2, -1):
        if x_diffs[ii] == 1: 
            if cut_bin != 0: x=x[:-cut_bin]
            break
        else: cut_bin += 1
    # trim in the middle
    x_diffs=[x[ii]-x[ii-1] for ii in xrange(1, len(x))]
    ii_to_trim=[]
    prev = 1
    for ii in xrange(0, len(x_diffs)):
        if x_diffs[ii] != 1 and prev == 1:
            prev = x_diffs[ii]
        elif x_diffs[ii] != 1 and prev != 1:
            ii_to_trim.append(ii)
            prev = x_diffs[ii]
        else: prev = 1
    x=np.delete(x, ii_to_trim, axis=0)
    x_diffs=[x[ii]-x[ii-1] for ii in xrange(1, len(x))]
    return x

# automatic search for the off-pulse window
# input profile will be rotated as necessary
# return tuple (data, rotphase, off-left, off-right)
def auto_find_off_window(data, rot_bins, nbins, adjust):
    # find first the bin with maximum value
    maxbin = np.argmax(data)
    # exclude the area of 60% of all bins around the maxbin
    # make the 60%-area the even number
    exclsize=int(nbins*0.6)+int(nbins*0.6)%2
    le=maxbin-exclsize/2
    re=maxbin+exclsize/2
    # extra rotation by "le" bins, so left edge will be at 0
    data = bestprof_rotate(data, le)
    # total rotation in phase
    if abs(rot_bins) < 1:
        rot_bins += float(le)/nbins
    else:
        rot_bins = float(rot_bins + le)/nbins
    amean = np.mean(data[re-le:nbins])
    arms = np.std(data[re-le:nbins])
    aprof = (data - amean)/arms
    abins=np.arange(0,nbins)[(aprof>2.5)]
    abins=trim_bins(abins) # trimming bins
    # updating pulse window
    try:
        exclsize=abins[-1]-abins[0]
    except:
        return (data, rot_bins, re-le, nbins)
    # to be extra-cautious, increase it by 15% of the pulse window on both sides
    le=abins[0]-int(0.15*exclsize)
    re=abins[-1]+1+int(0.15*exclsize)
    # doing manual adjustment of the off-pulse window
    if adjust[0] == 'l': # only adjusting the left edge
        le+=int(adjust[1:])
    elif adjust[0] == 'r': # only adjusting the right edge
        re+=int(adjust[1:])
    else: # adjusting both sides
        le-=int(adjust)
        re+=int(adjust)
    # extra rotation by "le" bins again, so left edge will be at 0
    data = bestprof_rotate(data, le)
    # total rotation in phase
    rot_bins += float(le)/nbins
    return (data, rot_bins, re-le, nbins)

def auto_snr_onpulse(raw):
    is_auto_off = True
    auto_off_adjust = "0"
    bscr = 1
    rot_bins = 0
    off_left = 0
    off_right = 1
    osm_min = None
    osm_max = 0.95
    if not(raw.get_dedispersed()):
        raw.dedisperse()
    raw.pscrunch()
    nchan = raw.get_nchan()
    nsubint = raw.get_nsubint()
    target = raw.get_source()
    if nchan > 1: raw.fscrunch()
    if nsubint > 1: raw.tscrunch()
    if bscr > 1: raw.bscrunch(bscr)
    nbins = raw.get_nbin()
    r = raw.get_data()
    #time stokes f phase
    data = r[0,0,0,:]
    weights = raw.get_weights()
    data[np.where(weights[0]==0)[0]] = 0.0
    # auto-find of the OFF-pulse window
    if is_auto_off:
        (data, rot_bins, off_left, off_right) = auto_find_off_window(data, rot_bins, nbins, auto_off_adjust)
        # and do total rotation for the input file as well (for psrstat)
        raw.rotate_phase(rot_bins)
    else:
        if rot_bins != 0:
            if abs(rot_bins) < 1:
                raw.rotate_phase(rot_bins)
            else:
                raw.rotate_phase(rot_bins/nbins)
            data = bestprof_rotate(data, rot_bins)
    if off_right == -1:
        off_right = int(nbins*0.1)
    if off_right-off_left<=1:
        off_right = nbins
    # Range
    range_mean = np.mean(data[off_left:off_right])
    range_rms = np.std(data[off_left:off_right])
    range_prof = (data - range_mean)/range_rms
    range_snrpeak = np.max(range_prof)
    range_weq = np.sum(range_prof)/range_snrpeak
    range_profsign = np.sum(range_prof)/np.sqrt(range_weq)
    return (range_profsign, off_left, off_right, rot_bins, bscr)

def auto_find_on_window_from_ar(ar, safe_fraction = 1/8.):
    # find first the bin with maximum value
    arx = ar.clone()
    arx.pscrunch()  # pscrunching again is not necessary if already pscrunched but prevents a bug
    arx.remove_baseline()
    arx.dedisperse()
    arx.fscrunch()
    arx.tscrunch()
    data = arx.get_Profile(0, 0, 0).get_amps() * 10000
    maxbin = np.argmax(data)
    nbins = len(data)
    # exclude the area of 60% of all bins around the maxbin
    # make the 60%-area the even number
    exclsize=int(nbins*0.6)+int(nbins*0.6)%2
    le=maxbin-exclsize/2
    re=maxbin+exclsize/2

    p_start = le%nbins
    p_end = re%nbins
    offbins = np.ones(nbins, dtype='bool')
    if(p_start<p_end):
        offbins[ p_start : p_end ] = False
    else:
        offbins[ p_start : ] = False
        offbins[ : p_end ] = False
    # extra rotation by "le" bins, so left edge will be at 0
    #data = bestprof_rotate(data, le)
    # total rotation in phase
    amean = np.mean(data[offbins])
    arms = np.std(data[offbins])
    aprof = (data - amean)/arms
    abins = np.arange(0,nbins)[(aprof>2.5)]
    abins = trim_bins(abins) # trimming bins
    # updating pulse window
    try:
        dabins = (abins - np.roll(abins, 1))%nbins
        le = abins[np.argmax(dabins)]%nbins
        re = abins[np.argmax(dabins)-1]%nbins
    except:
        le = maxbin-1
        re = maxbin+1
    # to be extra-cautious, ONpulse have to be largeur than 15% of the pulse window
    # to be extra-cautious, OFFpulse have to be largeur than 15% of the pulse window

    # 4 bin it is not enouth to calculate statistic
    if(nbins*safe_fraction < 5):
        safe_fraction = 1/4.
        if(nbins*safe_fraction < 5):
            safe_fraction = 1/2.

    if(le < re):
        onpulse = (re - le)/nbins
        offpulse = 1 - onpulse
        if(onpulse < safe_fraction):
            extrabin = ((safe_fraction - onpulse)/2.)*nbins
            re = re + int(extrabin)
            le = le - int(extrabin)
        if(offpulse < safe_fraction):
            extrabin = ((safe_fraction - offpulse)/2.)*nbins
            re = re - int(extrabin)
            le = le + int(extrabin)
    else: #(le > re)
        onpulse = (nbins-(le - re))/nbins
        offpulse = 1 - onpulse
        if(onpulse < safe_fraction):
            extrabin = ((safe_fraction - onpulse)/2.)*nbins
            re = re - int(extrabin)
            le = le + int(extrabin)
        if(offpulse < safe_fraction):
            extrabin = ((safe_fraction - offpulse)/2.)*nbins
            re = re + int(extrabin)
            le = le - int(extrabin)
    le = le%nbins
    re = re%nbins
    return le, re

def PNG_to_html_img_tag(png_file):
    """Function to convert a PNG file in html code.
       Input:
            png_file: a .png file
    """
    data_uri = open(png_file, 'rb').read().encode('base64').replace('\n', '')
    img_tag = '<div ><img src="data:image/png;base64,{0}" width="640" alt="Computer Hope"></div>'.format(data_uri)
    img_tag = '<h1>'+basename(png_file).split('.')[0]+'</h1>'+img_tag
    return img_tag

def ASCII_to_html(ascii_file):
    """Function to convert a txt file in html code.
       Input:
            ascii_file: a .txt file
    """
    data_uri = open(ascii_file, 'rb').read().replace('\n',
                                                     '</pre><pre class="tab">')
    html = ('<h1>%s</h1><body><pre class="tab">%s</pre></body>' %
            (basename(ascii_file).split('.')[0], data_uri))
    return html

def make_html_code(html_body):
    """Function to finalize the html code.
       Input:
            html_body: string containing the html body of the code
    """
    html_code = ('<!DOCTYPE html>%s<html></html>' % (html_body))
    return html_code


def reduce_pdf(pdf_path, pdf_name, dpi=400):
    """Function to reduce the size of a pdf using Ghostscript
    """
    commande = 'gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -r'+str(int(dpi))+' -dNOPAUSE -dQUIET -dBATCH -sOutputFile='+pdf_path+pdf_name+'TMPfile.tmp '+pdf_path+pdf_name
    output = check_output(commande, shell=True)
    commande = 'mv '+pdf_path+pdf_name+'TMPfile.tmp '+pdf_path+pdf_name
    output = check_output(commande, shell=True)



def all_to_html(WORKDIR):
    """Function to convert and stack all txt an PNG file in an html code

       Input:
            WORKDIR: path to the working directory

    """
    html_body = ''
    if(os.path.isdir(WORKDIR)):
        for the_file in sorted(os.listdir(WORKDIR)):
            file_path = os.path.join(WORKDIR, the_file)
            try:
                if os.path.isfile(file_path):
                    print(file_path)
                    if (file_path.split('.')[1] == 'png'):
                        html_body = html_body + PNG_to_html_img_tag(file_path)
                    if (file_path.split('.')[1] == 'txt'):
                        html_body = html_body + ASCII_to_html(file_path)
            except Exception as e:
                print(e)
    html_code = make_html_code(html_body)
    return html_code
