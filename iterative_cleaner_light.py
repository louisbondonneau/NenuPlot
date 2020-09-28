#!/usr/bin/env python

# Tool to remove RFI from pulsar archives.
# Originally written by Patrick Lazarus. Modified by Lars Kuenkel.

import numpy as np
import datetime
import scipy.optimize
from scipy import stats
#from statsmodels import robust
import psrchive

#import matplotlib
#import matplotlib.pyplot as plt

def flattenBP(ar):
    ar.pscrunch()
    ar2 = ar.clone()
    ar2.tscrunch()
    subint = ar2.get_Integration(0)
    (bl_mean, bl_var) = subint.baseline_stats()
    bl_mean = bl_mean.squeeze()
    ar.remove_baseline()
    if (ar.get_nchan()>1):
        for isub in range(ar.get_nsubint()):
            for ichan in range(ar.get_nchan()):
                for ipol in range(ar.get_npol()):
                    if( bl_mean[ichan] != 0 ):
                        prof = ar.get_Profile(isub, ipol, ichan)
                        #prof.offset(-bl_mean[ichan])
                        if(ichan == 0):
                            prof.scale(1/(0.8*bl_mean[ichan]+0.2*bl_mean[ichan+1]))
                        elif(ichan == ar.get_nchan()-1):
                            prof.scale(1/(0.2*bl_mean[ichan-1]+0.8*bl_mean[ichan]+0.1))
                        else:
                            prof.scale(1/(0.1*bl_mean[ichan-1]+0.8*bl_mean[ichan]+0.1*bl_mean[ichan+1]))
    return ar


def clean(ar, zapfirstsubint = False, fast = False, flat_cleaner=False, chanthresh=6.0, subintthresh=3.0, bad_subint=0.9, bad_chan=0.50):
    unload_res = False
    no_log = True
    pscrunch = False
    unload_res = False
    max_iterations = 12
    if (fast):
        max_iterations = 3
    orig_weights = ar.get_weights()

    if (pscrunch):
        ar.pscrunch()

    # Mask the first subintegration if required
    if (zapfirstsubint) and (ar.get_nsubint() > 1):
        for ichan in range(ar.get_nchan()):
            prof = ar.get_Profile(0, 0, ichan)
            prof.set_weight(0.0)

    patient = ar.clone()
    ar_name = ar.get_filename().split()[-1]
    x = 0
    pulse_region = [0,0,1] #metavar=('pulse_start', 'pulse_end', 'scaling_factor'), help="Defines the range of the pulse and a suppression factor.")

    # Create list that is used to end the iteration
    test_weights = []
    test_weights.append(patient.get_weights())
    profile_number = orig_weights.size
    while (x < max_iterations):
        x += 1
        print("Loop: %s" % x)
        # Prepare the data for template creation
        archive_template = patient.clone()
        archive_template.pscrunch()  # pscrunching again is not necessary if already pscrunched but prevents a bug
        archive_template.remove_baseline()
        archive_template.dedisperse()
        archive_template.fscrunch()
        archive_template.tscrunch()
        template = archive_template.get_Profile(0, 0, 0).get_amps() * 10000
        #import matplotlib
        #import matplotlib.pyplot as plt
        pulse_region[1], pulse_region[2] = auto_find_on_window( template )
        print("ON pulse window is %d-%d" % (pulse_region[1], pulse_region[2]))
        #plt.plot(template)
        #plt.show()

        # Reset patient
        patient = ar.clone()
        if (flat_cleaner):
            if ( x > 2 ):apply_weights_archive(patient, new_weights)
            patient = flattenBP(patient) #pscrunch & remove_baseline inside
        else:
            patient.pscrunch()
            patient.remove_baseline()
        patient.dedisperse()
        remove_profile_inplace(patient, template, pulse_region)

        # re-set DM to 0
        patient.dededisperse()

        if (unload_res):
            residual = patient.clone()

        # Get data (select first polarization - recall we already P-scrunched)
        data = patient.get_data()[:, 0, :, :]
        if ( x <= 2 ): # template is refined
            data = apply_weights(data, orig_weights)
            # Mask profiles where weight is 0
            mask_2d = np.bitwise_not(np.expand_dims(orig_weights, 2).astype(bool))
        else: # mask is refined using the new mask
            #set_weights_archive(patient, new_weights)
            data = apply_weights(data, new_weights)
            # Mask profiles where weight is 0
            mask_2d = np.bitwise_not(np.expand_dims(new_weights, 2).astype(bool))
            #print(new_weights[41,:]) #result array of integration 42

        mask_3d = mask_2d.repeat(ar.get_nbin(), axis=2)
        data = np.ma.masked_array(data, mask=mask_3d)

        # RFI-ectomy must be recommended by average of tests
        avg_test_results = comprehensive_stats(data, x, axis=2, fast = fast, default_chanthresh=float(chanthresh), default_subintthresh=float(subintthresh))
        #print(avg_test_results[41,:]) #result array of integration 42
        # Reset patient and set weights in patient
        del patient
        patient = ar.clone()
        set_weights_archive(patient, avg_test_results)

        # find bad part
        patient = find_bad_parts(patient, bad_subint=bad_subint, bad_chan=bad_chan)

        # Test whether weigths were already used in a previous iteration
        new_weights = patient.get_weights()
        if ( x > 2 ):
            for (isub, ichan) in np.argwhere(test_weights[-1] == 0):
                new_weights[isub, ichan] = 0.0
            #print('previous weights')
            #print(test_weights[-1][41,:])
        diff_weigths = np.sum(new_weights != test_weights[-1])
        rfi_frac = 1-(np.mean(new_weights)/np.max(new_weights)) #(new_weights.size - np.count_nonzero(new_weights)) / float(new_weights.size)

        print("RFI fraction is %.2f percent" % (rfi_frac*100.))
        # Print the changes to the previous loop to help in choosing a suitable max_iter
        if np.all(new_weights == test_weights[-1]):
            loops = x
            x = 1000000
        test_weights.append(new_weights)

    if (x == max_iterations):
        print("Cleaning was interrupted after %s loops" % max_iterations)
        loops = max_iterations

    # Set weights in archive.
    apply_weights_archive(ar, new_weights)



    # Unload residual if needed
    if (unload_res):
        residual.unload("%s_residual_%s.ar" % (ar_name, loops))

    # Create log that contains the used parameters
    if not (no_log):
        with open("clean.log", "a") as myfile:
            myfile.write("\n %s: Cleaned %s, required loops=%s"
             % (datetime.datetime.now(), ar_name, loops))
    return ar


def comprehensive_stats(data, x, axis, fast = False, default_chanthresh=6.0, default_subintthresh=3.0):
    """The comprehensive scaled stats that are used for
        the "Surgical Scrub" cleaning strategy.

        Inputs:
            data: A 3-D numpy array.
            axis: The axis that should be used for computing stats.
            args: argparse namepsace object that need to contain the
                following two parameters:
            chanthresh: The threshold (in number of sigmas) a
                profile needs to stand out compared to others in the
                same channel for it to be removed.
                (Default: use value defined in config files)
            subintthresh: The threshold (in number of sigmas) a profile
                needs to stand out compared to others in the same
                sub-int for it to be removed.
                (Default: use value defined in config files)

        Output:
            stats: A 2-D numpy array of stats.
    """

    if (x==1):
        chanthresh = 8.0
        subintthresh = 8.0
    else:
        chanthresh = default_chanthresh
        subintthresh = default_subintthresh
    
    print('chanthresh = %.1f  subintthresh = %.1f' % (chanthresh, subintthresh))

    nsubs, nchans, nbins = data.shape


    diagnostic_functions = [
        np.ma.std,
        np.ma.mean,
        np.ma.ptp,
        lambda data, axis: np.max(np.abs(np.fft.rfft(
            data - np.expand_dims(data.mean(axis=axis), axis=axis),
            axis=axis)), axis=axis)
    ]
    # Compute diagnostics
    diagnostics = []
    for func in diagnostic_functions:
        diagnostics.append(func(data, axis=2))

    # Now step through data and identify bad profiles
    scaled_diagnostics = []
    for diag in diagnostics:
        chan_scaled = np.abs(channel_scaler(diag)) / chanthresh
        subint_scaled = np.abs(subint_scaler(diag)) / subintthresh
        chan_scaled[np.where(np.isnan(chan_scaled))] = 2.0
        subint_scaled[np.where(np.isnan(subint_scaled))] = 2.0
        scaled_diagnostics.append(np.max((chan_scaled, subint_scaled), axis=0))
    test_results = np.median(scaled_diagnostics, axis=0)
    return test_results


def channel_scaler(array2d):
    """For each channel scale it.
    """
    scaled = np.empty_like(array2d)
    nchans = array2d.shape[1]
    for ichan in np.arange(nchans):
        with np.errstate(invalid='ignore', divide='ignore'):
            channel = array2d[:, ichan]
            median = np.ma.median(channel)
            channel_rescaled = channel - median
            mad = np.ma.median(np.abs(channel_rescaled))
            scaled[:, ichan] = (channel_rescaled) / mad
    return scaled


def subint_scaler(array2d):
    """For each sub-int scale it.
    """
    scaled = np.empty_like(array2d)
    nsubs = array2d.shape[0]
    for isub in np.arange(nsubs):
        with np.errstate(invalid='ignore', divide='ignore'):
            subint = array2d[isub, :]
            median = np.ma.median(subint)
            subint_rescaled = subint - median
            mad = np.ma.median(np.abs(subint_rescaled))
            scaled[isub, :] = (subint_rescaled) / mad
    return scaled


def remove_profile_inplace(ar, template, pulse_region):
    """Remove the temnplate pulse from the individual profiles.
    """
    data = ar.get_data()[:, 0, :, :]  # Select first polarization channel
                                # archive is P-scrunched, so this is
                                # total intensity, the only polarization
                                # channel
    for isub, ichan in np.ndindex(ar.get_nsubint(), ar.get_nchan()):
        amps = remove_profile1d(data[isub, ichan], isub, ichan, template, pulse_region)[1]
        prof = ar.get_Profile(isub, 0, ichan)
        if amps is None:
            prof.set_weight(0)
        else:
            prof.get_amps()[:] = amps


def remove_profile1d(prof, isub, ichan, template, pulse_region):

    err = lambda amp: amp * template - prof
    params, status = scipy.optimize.leastsq(err, [1.0])
    err2 = np.asarray(err(params))
    if pulse_region != [0, 0, 1]:   # ('pulse_start', 'pulse_end', 'scaling_factor')
        p_start = int(pulse_region[1])
        p_end = int(pulse_region[2])
        nbins = len(template)

        offbins = np.ones(nbins, dtype='bool')
        if(p_start<p_end):
            offbins[ p_start : p_end ] = False
            sizeon=p_end-p_start
        else:
            offbins[ p_start : ] = False
            offbins[ : p_end ] = False
            sizeon=nbins-(p_start-p_end)
        #onbins = np.zeros(len(template), dtype='bool')
        #onbins[ p_start : p_end ] = True

        #err2[p_start:p_end] = err2[p_start:p_end] * pulse_region[0]
        mad = np.std(err2[offbins])
        #mad = np.median(np.abs(err2[offbins]-np.median(err2[offbins])))
        #err2[~offbins] = mad*np.random.standard_normal(sizeon)
        err2[~offbins] = mad*np.random.rand(sizeon)

    if status not in (1, 2, 3, 4):
        print "Bad status for least squares fit when removing profile."
        return (isub, ichan), np.zeros_like(prof)
    else:
        return (isub, ichan), err2


def apply_weights(data, weights):
    """Apply the weigths to an array.
    """
    nsubs, nchans, nbins = data.shape
    for isub in range(nsubs):
        data[isub] = data[isub] * weights[isub, ..., np.newaxis]
    return data

def apply_weights_archive(archive, weights):
    """Apply the weigths to an array.
    """
    for (isub, ichan) in np.argwhere(weights == 0):
        integ = archive.get_Integration(int(isub))
        integ.set_weight(int(ichan), 0.0)


def set_weights_archive(archive, test_results):
    """Apply the weigths to an archive according to the test results.
    """
    for (isub, ichan) in np.argwhere(np.isnan(test_results)):
        integ = archive.get_Integration(int(isub))
        integ.set_weight(int(ichan), 0.0)
        test_results[isub, ichan] = 2
    for (isub, ichan) in np.argwhere(test_results >= 1):
        integ = archive.get_Integration(int(isub))
        integ.set_weight(int(ichan), 0.0)

def auto_find_on_window(data, safe_fraction = 1/8.):
    # find first the bin with maximum value
    maxbin = np.argmax(data)
    nbins = len( data )
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
    # updating pulse windo
    # to be extra-cautious, ONpulse have to be largeur than 15% of the pulse window
    # to be extra-cautious, OFFpulse have to be largeur than 15% of the pulse window
    try:
        dabins = (abins - np.roll(abins, 1))%nbins
        le = abins[np.argmax(dabins)]%nbins
        re = abins[np.argmax(dabins)-1]%nbins
    except:
        le = maxbin-1
        re = maxbin+1

    
    if(nbins*safe_fraction < 5) and (safe_fraction <= 1/4):
        safe_fraction = 1/4.
        if(nbins*safe_fraction < 5):
            safe_fraction = 1/2.

    if(le < re):
        onpulse = (re - le)/float(nbins)
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
        onpulse = (nbins-(le - re))/float(nbins)
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

def find_bad_parts(archive, bad_subint=1, bad_chan=1, quiet=True):
    """Checks whether whole channels or subints should be removed
    """
    weights = archive.get_weights()
    n_subints = archive.get_nsubint()
    n_channels = archive.get_nchan()
    n_bad_channels = 0
    n_bad_subints = 0

    for i in range(n_subints):
        bad_frac = 1 - np.count_nonzero(weights[i, :]) / float(n_channels)
        if bad_frac > bad_subint:
            for j in range(n_channels):
                integ = archive.get_Integration(int(i))
                integ.set_weight(int(j), 0.0)
            n_bad_subints += 1

    for j in range(n_channels):
        bad_frac = 1 - np.count_nonzero(weights[:, j]) / float(n_subints)
        if bad_frac > bad_chan:
            for i in range(n_subints):
                integ = archive.get_Integration(int(i))
                integ.set_weight(int(j), 0.0)
            n_bad_channels += 1

    if not quiet and n_bad_channels + n_bad_subints != 0:
        print("Removed %s bad subintegrations and %s bad channels." % (n_bad_subints, n_bad_channels))
    return archive

def bandcut (ar, freqList ,quiet=True ) :
        if not quiet :
                print 'Frequency band cutting.'
        count=0
        for fmin , fmax in freqList :
                for isub in range(ar.get_nsubint()):
                        for ipol in range(ar.get_npol()):
                                for ichan in range(ar.get_nchan()):
                                        prof = ar.get_Profile(isub, ipol, ichan)
                                        freq = float(ar.get_Profile(isub, ipol, ichan).get_centre_frequency())
                                        if( freq >= float(fmin) ) and ( freq <= float(fmax) ):
                                                prof.set_weight(0.0)
                                                count=count+1
        if not quiet :
                mask=100*float(count)/float(ar.get_nsubint()*ar.get_npol()*ar.get_nchan())
                print "coupbande mask = "+str(mask)+" %"


def clean_hotbins(ar, thresh=5.0, onpulse=[]):
    """Replace hot bits with white noise.

    Inputs:
        ar: The archive to be cleaned
        thresh: The threshold (in number of sigmas) for a
            bin to be removed.
        onpulse: On-pulse regions to be ignored when computing
            profile statistics. A list of 2-tuples is expected.

    Outputs:
        None - The archive is modified in place
    """
    nbins = ar.get_nbin()
    indices = np.arange(nbins)
    offbins = np.ones(nbins, dtype='bool')
    offbins[ onpulse[0] : onpulse[1] ] = False
    offbin_indices = indices[offbins]

    for isub in np.arange(ar.get_nsubint()):
            for ichan in np.arange(ar.get_nchan()):
                    for ipol in np.arange(ar.get_npol()):
                            prof = ar.get_Profile(int(isub), int(ipol), int(ichan))
                            data = prof.get_amps() #inplace
                            offdata = data[offbins]
                            med = np.median(offdata)
                            mad = np.median(np.abs(offdata-med))
                            std = mad*1.4826 # This is the approximate relation between the
                                             # standard deviation and the median absolute
                                             # deviation (assuming normally distributed data).
                            ioffbad = np.abs(offdata-med) > std*thresh
                            ibad = offbin_indices[ioffbad]
                            igood = offbin_indices[~ioffbad]
                            nbad = np.sum(ioffbad)
                            gooddata = data[igood]
                            avg = gooddata.mean()
                            std = gooddata.std()
                            if std > 0:
                                    noise = np.random.normal(avg, std, size=nbad).astype('float32')
                                    data[ibad] = noise

def hotbins ( ar, quiet=True) :
    if not ar.get_dedispersed() :
            if not quiet :
                    print 'Dedispersion of the original archive.\n'
            ar.dedisperse()
    ar.centre_max_bin()

    data = ar.get_data()
    weights = ar.get_weights()
    data = data.mean( 1 )           # Pol scr
    data *= weights                 # Weights applying
    prof = data.mean( 0 )           # Subint scr
    prof = prof.mean( 0 )           # Freq scr

    if not quiet :
            print 'Off area computing.\n'
    offL , offR = auto_find_on_window( prof )

    if not quiet :
            print 'Clean hotbins.\n'
    ar.dededisperse()
    clean_hotbins( ar , thresh=5.0 , onpulse=[ offL , offR ] )