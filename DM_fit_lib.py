#!/usr/bin/env python

# To get help type 'man python' at the terminal

# Written by L. Bondonneau, 2020

import psrchive as psr
import numpy as np
import os
import sys
from multiprocessing import Pool, TimeoutError
import time
#Nancay_astroplan = Observer(latitude=47.37583*u.deg, longitude=2.1925*u.deg, elevation=139*u.m)


def smooth_Gaussian(list,degree=5):  
    window=degree*2-1  
    weight=np.array([1.0]*window)  
    weightGauss=[]  
    for i in range(window):  
        i=i-degree+1  
        frac=i/float(window)  
        gauss=1/(np.exp((4*(frac))**2))  
        weightGauss.append(gauss)  
    weight=np.array(weightGauss)*weight  
    smoothed=[0.0]*(len(list))  
    for i in range(len(smoothed)-window):  
        smoothed[i+int(window/2)]=sum(np.array(list[i:i+window])*weight)/sum(weight)  
    return smoothed  

def flatten_from_mad(p_start, p_end):
    global AR
    AR.pscrunch()
    ar2 = AR.clone()
    ar2.tscrunch()
    data = ar2.get_data()
    nbins = ar2.get_nbin()
    #print(np.shape(data))

    offbins = np.ones(nbins, dtype='bool')
    if(p_start<p_end):
        offbins[ p_start : p_end ] = False
    else:
        offbins[ p_start : ] = False
        offbins[ : p_end ] = False

    mad_off = mad(data[0,0,:, offbins], axis=0)
    bl_mean = 0
    AR.remove_baseline()
    if (AR.get_nchan()>1):
        for isub in range(AR.get_nsubint()):
            for ichan in range(AR.get_nchan()):
                for ipol in range(AR.get_npol()):
                    if( mad_off[ichan] != 0 ):
                        prof = AR.get_Profile(isub, ipol, ichan)
                        #prof.offset(-bl_mean[ichan])
                        prof.scale(1/mad_off[ichan])
    return AR

def freq_vector(ar, coh_rm):
    min_freq = ar.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = ar.get_Profile(0, 0, ar.get_nchan()-1).get_centre_frequency()
    doppler = np.sqrt(ar.get_Integration(0).get_doppler_factor()) # doppler comparatively to the frequency Freq * doppler = dop_freq
    #  double dmfac = s->dm*2.0*M_PI/(2.41e-10*(1.0+s->earth_z4/1.0e4));
    # psrchive_doppler => (1.0+s->earth_z4/1.0e4)
    #print("%.18f" % doppler)
    #exit(0)
    #doppler = 1/doppler
    bw_chan = ((max_freq-min_freq)/(ar.get_nchan()-1))*doppler
    #freqs = np.linspace(min_freq, max_freq, ar.get_nchan())*doppler
    #print(np.linspace(min_freq, max_freq, ar.get_nchan()))
    #print(max_freq,min_freq)
    #print(ar.get_nchan())
    #print(bw_chan)
    global rechan_factor
    if (coh_rm):
        rechan_factor = 256
    else:
        rechan_factor = 1
        #print('WARNING: no coherent dedispertion give with -coh_rm. Default value is the exact RM value (no depolarisation).')

    freqs = np.zeros(ar.get_nchan())
    freqs_extended = np.zeros(rechan_factor*ar.get_nchan())
    fraction_bw = np.linspace(-0.5+0.5/rechan_factor, 0.5-0.5/rechan_factor, rechan_factor)

    for ichan in range(ar.get_nchan()):
        freqs[ichan] = ar.get_Profile(0, 0, ichan).get_centre_frequency()*doppler
        #print(" ichan %d  --> %.8f MHz " % (ichan, freqs[ichan]))
        if (rechan_factor > 1 ):
            for ichan_extended in range(rechan_factor):
                freqs_extended[ichan*rechan_factor + ichan_extended] = freqs[ichan] + fraction_bw[ichan_extended]*bw_chan
                #print(" %.8f MHz " % freqs_extended[ichan*rechan_factor + ichan_extended])
        else:
            freqs_extended[ichan] = freqs[ichan]
    #exit(0)


    nchan_extended = ar.get_nchan()*rechan_factor
    bw_chan_extended = bw_chan/rechan_factor
    #freqs_extended = np.linspace(min_freq-bw_chan/2.+bw_chan_extended/2., max_freq+bw_chan/2.-bw_chan_extended/2., nchan_extended)*doppler
    centerfreq = ar.get_centre_frequency()*doppler
    freqs_extended_test = np.mean(np.reshape(freqs_extended,(len(freqs), rechan_factor)), axis=1)
    return freqs, centerfreq, freqs_extended, max_freq

def auto_find_on_window(ar, safe_fraction = 1/8., archive=True):
    # find first the bin with maximum value
    if (archive):
        ar2 = ar.clone()
        ar2.tscrunch()
        ar2.fscrunch()
        ar2.pscrunch()
        data = 1000.*ar2.get_Profile(0, 0, 0).get_amps()
        nbins = ar2.get_nbin()
    else:
        data = ar
        nbins = np.size(data)
    maxbin = np.argmax(data)
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

    
    if(nbins*safe_fraction < 5):
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
            re = re + int(extrabin)
            le = le - int(extrabin)
        if(offpulse < safe_fraction):
            extrabin = ((safe_fraction - offpulse)/2.)*nbins
            re = re - int(extrabin)
            le = le + int(extrabin)
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


def mad(data, axis=0):
    return np.nanmedian( np.abs(data - np.nanmedian(data, axis=axis)) , axis=axis)

def auto_snr(data, off_left, off_right):
    off_data = data[off_left:off_right]
    #plt.plot(data)
    #plt.show()
    range_mean = np.nanmedian(off_data)
    range_rms = 1.5*mad(off_data)
    if(range_rms == 0):
        print('ERROR401 This is the Murphy law')
        print(off_left,off_right)
        print(off_data)
        print(np.nanmedian(off_data))
        print(np.abs(off_data - np.nanmedian(off_data)))
        exit(0)
    if(np.isnan(range_rms)):
        print('ERROR402 This is the Murphy law')
        print(off_left, off_right)
        print(off_data)
        print(np.nanmedian(off_data))
        print(np.abs(off_data - np.nanmedian(off_data)))
        exit(0)
    range_prof = (data - range_mean)/range_rms
    range_snrpeak = np.nanmax(range_prof)
    range_weq = np.nansum(range_prof)/range_snrpeak
    #print(np.nansum(range_prof), range_weq)
    if(range_weq < 0):
        return 0
    range_profsign = np.nansum(range_prof)/np.sqrt(range_weq)
    return (range_profsign)

def auto_snr_peak(data, off_left, off_right):
    off_data = data[off_left:off_right]
    on_data = np.append(data[:off_left], data[off_right:])
    range_mean = np.median(off_data)
    range_rms = 1.48*mad(off_data)
    #data[off_left:off_right] = range_mean
    range_prof = (on_data - range_mean)/range_rms
    range_snrpeak = np.max(range_prof)
    return (range_snrpeak)

def sharp_lvl(dm, plot=False, pulse_region_sharp=[0, 0]):
    global AR
    ar2 = AR.clone()
    ar2.tscrunch()
    ar2.pscrunch()
    ar2.set_dispersion_measure(dm)
    ar2.dedisperse()
    ar2.fscrunch()
    nbin = ar2.get_nbin()
    #print('0', pulse_region_sharp)
    if (pulse_region_sharp[0]==0) and (pulse_region_sharp[1]==0) :
        pulse_region_sharp[0], pulse_region_sharp[1] = auto_find_on_window(ar2)
    AR = flatten_from_mad(pulse_region_sharp[0], pulse_region_sharp[1])
    template = ar2.get_data().squeeze()
    template = np.roll(template, -pulse_region_sharp[0])
    #if(pulse_region[0] < pulse_region[1]):
    on_right = (pulse_region_sharp[1]-pulse_region_sharp[0])%nbin
    on_left = 0
    #else:

    sharp_template = np.abs(template - np.roll(template, 1))
    #pond_sharp_template = sharp_template
    #pond_sharp_template[(sharp_template > 0)] *= 8
    #pond_sharp_template = np.abs(pond_sharp_template)
    #sharp_template = np.abs(sharp_template)   I do not see any difference on the fit
    snr_peak = auto_snr_peak(template, on_right+1, nbin)
    #print('1', pulse_region_sharp)
    real_snr = auto_snr(template, on_right+1, nbin)
    #sharpness = auto_snr(sharp_template, pulse_region[1]+1, nbin)
    #print(on_left, on_right, nbin, pulse_region)
    flux_peak = np.max(template[on_left: on_right]) - np.median(template[on_right+1:])
    ## ponderate sharp_template positive derivative is twice the negative
    #print(0, on_right, len(sharp_template))
    sharpness = np.nanmean(sharp_template[0: on_right])
    #sharpness = np.nanmean(sharp_template)
    return real_snr, flux_peak, snr_peak, sharpness, pulse_region_sharp


def search_pulse_region(dm):
    global AR
    ar2 = AR.clone()
    ar2.tscrunch()
    ar2.pscrunch()
    ar2.set_dispersion_measure(dm)
    ar2.dedisperse()
    ar2.fscrunch()
    nbin = ar2.get_nbin()
    pulse_region_search = [0, 0]
    pulse_region_search[0], pulse_region_search[1] = auto_find_on_window(ar2)
    #print('search_pulse_region', nbin, pulse_region)
    return pulse_region_search



def dm_trials(dm, diff, plot=False, mode='sharpness', Force_pulse_region=False, ncore=8, lim_min_dm=0.5):
    dm_min = dm-float(diff)/2. #-16.6
    dm_max = dm+float(diff)/2. #-16.1
    if(dm_min < lim_min_dm): dm_min = lim_min_dm
    delt = (dm_max - dm_min) / float(200)
    
    dm_vec = np.linspace(float(dm_min), float(dm_max), 1+int(((float(dm_max)-float(dm_min))/float(delt))))
    real_snr = np.zeros(len(dm_vec))
    flux_peak = np.zeros(len(dm_vec))
    sharpness = np.zeros(len(dm_vec))
    snr_peak = np.zeros(len(dm_vec))


    if (Force_pulse_region):
        pulse_region_trial = search_pulse_region(dm) # pulse_region for the best dm

    #sharp_lvl(ar, 12.439)
    pool = Pool(processes=int(ncore))        # Creation de l iterable a envoyer comme argument a la fon
    #result = pool.apply_async(sharp_lvl, (ar, dm_vec[idm], ))
    #multiple_results = [pool.apply_async(sharp_lvl, (ar, dm_vec[idm],)) for idm in range(len(dm_vec))]
    #multiple_results = pool.apply_async(sharp_lvl, iter( [12.3, 12.4] ))
    if (Force_pulse_region):
        multiple_results = [pool.apply_async(sharp_lvl,
                                            (dm_vec[n], False, pulse_region_trial))
                                            for n in range(len(dm_vec))]
    else:
        multiple_results = [pool.apply_async(sharp_lvl,
                                            (dm_vec[n], False, [0, 0]))
                                            for n in range(len(dm_vec))]
    for idm in range(len(dm_vec)):
        real_snr[idm], flux_peak[idm], snr_peak[idm], sharpness[idm], pulse_region_trial = multiple_results[idm].get()

    snr_peak = smooth_Gaussian(snr_peak)
    sharpness = smooth_Gaussian(sharpness)
    real_snr = smooth_Gaussian(real_snr)
    flux_peak = smooth_Gaussian(flux_peak)

    if (mode == 'sharpness'):
        return dm_vec[np.nanargmax(sharpness)]
    elif (mode == 'snr_peak'):
        return dm_vec[np.nanargmax(snr_peak)]
    elif (mode == 'real_snr'):
        return dm_vec[np.nanargmax(real_snr)]
    elif (mode == 'flux_peak'):
        return dm_vec[np.nanargmax(flux_peak)]

def estimation_dm_error(pulse_region_est, real_snr, dm):
    global AR
    if (real_snr< 100):
        nchan_new = 10
    else:
        nchan_new = int(real_snr/10.)

    nchan = AR.get_nchan()

    rechan_factor = int(np.ceil(float(nchan)/float(nchan_new)))
    #print(float(nchan)/float(nchan_new), rechan_factor)
    if(rechan_factor > 1): AR.fscrunch(rechan_factor)
    AR.tscrunch()
    AR.pscrunch()
    AR.set_dispersion_measure(dm)
    AR.dedisperse()

    freqs, centerfreq, freqs_extended, max_freq = freq_vector(AR, coh_rm=0)
    nchan = AR.get_nchan()
    nchan_bw = (np.max(freqs)-np.min(freqs))/(nchan-1)
    weights = AR.get_weights()
    weights = weights.squeeze()
    weights = weights/np.max(weights)
    #print(np.shape(weights))
    #ar2.fscrunch()
    nbin = AR.get_nbin()
    #if (pulse_region[0]==0) and (pulse_region[1]==0) :
    #    pulse_region[0], pulse_region[1] = auto_find_on_window(ar2)
    #AR = flatten_from_mad(pulse_region[0], pulse_region[1])
    data = AR.get_data().squeeze()
    delta_dm_vec = []
    period = AR.get_Integration(0).get_folding_period() # in sec
    first = True
    top_valid_chan = False
    for ichan in range(AR.get_nchan()-1,0, -1):
        if(weights[ichan] == 0):
            real_snr = 0
        else:
            template = data[ichan, :]
            #print(pulse_region)
            template = np.roll(template, -pulse_region_est[0])
            on_right = (pulse_region_est[1]-pulse_region_est[0])%nbin
            on_left = 0
            #print(np.shape(template),  on_right+1, nbin)
            real_snr = auto_snr(template, on_right+1, nbin)
            #print(freqs[ichan], real_snr)
            on_in_sec = period*(float(on_right)/float(nbin))
            top_freq = freqs[ichan] + nchan_bw/2
            bot_freq = freqs[ichan] - nchan_bw/2
            if(real_snr == 0): real_snr = np.nan
            if(real_snr > 8) and (first==True):
                top_valid_chan = top_freq
                first = False
            if (first == False):
                dt = on_in_sec/(real_snr/5.)
                delta_dm = dt/(4150.*((bot_freq)**(-2)-(top_valid_chan)**(-2)))
                #print(freqs[ichan], real_snr, dt, delta_dm )
                delta_dm_vec.append(delta_dm)
    try:
        result = np.nanmin(delta_dm_vec)
    except:
        dt = on_in_sec/(8/5.)
        if not top_valid_chan:
            print("WARING: SNR too small can not compute a DM error")
            result = np.nan
        else:
            result = dt/(4150.*((top_valid_chan- nchan_bw)**(-2)-(top_valid_chan)**(-2)))
    return result

def smoothGaussian(list):
    degree = int(np.ceil(0.07 * np.size(list) /4.))
    window = degree*2-1  
    weight = np.array([1.0]*window)
    weightGauss = []  
    for i in range(window):  
        i = i-degree+1  
        frac=i/float(window)  
        gauss = 1/(np.exp((4*(frac))**2))  
        weightGauss.append(gauss)  
    weight = np.array(weightGauss)*weight  
    smoothed = [0.0]*(len(list))  
    for i in range(len(smoothed)-window):  
        smoothed[i+int(window/2)] = sum(np.array(list[i:i+window])*weight)/sum(weight)  
    return np.asarray(smoothed)

def auto_rebin(ar, ichan='NONE'):
    arx = ar.clone()
    arx.dedisperse()
    arx.pscrunch()
    arx.tscrunch()
    if (ichan == 'NONE'):
        arx.fscrunch()
        ichan = 0
    arx.remove_baseline()
    nbin = arx.get_nbin()

    prof = arx.get_Profile(0, 0, ichan).get_amps() * 10000
    prof = np.asarray(prof)
    prof_smooth = smoothGaussian( prof )
    p_start, p_end = auto_find_on_window( prof_smooth, safe_fraction = 3/100. ,archive=False)
    offbins = np.ones(nbin, dtype='bool')
    if( p_start < p_end ):
        offbins[ p_start : p_end ] = False
        sizeon = p_end-p_start
    else:
        offbins[ p_start : ] = False
        offbins[ : p_end ] = False
    std = np.std(prof[offbins])
    delta = np.max(np.abs(prof_smooth[~offbins] - np.roll(prof_smooth[~offbins], 1)))

    bscrunch_factor = 1
    #print(nbin, delta, 2.5*std)
    while (nbin >32) and (delta < 2.9*std):
        arx.bscrunch(2)
        nbin = arx.get_nbin()
        prof = arx.get_Profile(0, 0, ichan).get_amps() * 10000
        prof = np.asarray(prof)
        prof_smooth = smoothGaussian( prof )
        p_start, p_end = auto_find_on_window( prof_smooth, safe_fraction = 7/100. ,archive=False)
        offbins = np.ones(nbin, dtype='bool')
        if( p_start < p_end ):
            offbins[ p_start : p_end ] = False
            sizeon = p_end-p_start
        else:
            offbins[ p_start : ] = False
            offbins[ : p_end ] = False
        std = np.std(prof[offbins])
        delta = np.max(np.abs(prof_smooth[~offbins] - np.roll(prof_smooth[~offbins], 1)))
        bscrunch_factor *= 2
        print(nbin, delta, 2.9*std)
    if (bscrunch_factor > 1):
        bscrunch_factor /= 2
        if (bscrunch_factor > 1):
            ar.bscrunch(bscrunch_factor)
            print('automatic rebin by factor: '+str(bscrunch_factor))
        else:
            print('automatic rebin by factor: '+str(bscrunch_factor))
    return ar, bscrunch_factor


def DM_fit(AR0, verbose=False, ncore=8, autorebin=True, lim_min_dm=0.5):
    global AR
    AR = AR0.clone()
    dm_archive = AR.get_dispersion_measure()
    if(verbose):print("Coherent dm = %.4f pc cm-3" % dm_archive)
    AR.dedisperse()
    AR.tscrunch()
    AR.pscrunch()
    
    #diff = 0.1
    dm = AR.get_dispersion_measure()
    period = AR.get_Integration(0).get_folding_period() # in sec
    nbin = AR.get_nbin()
    bandwidth = AR.get_bandwidth()
    centre_frequency = AR.get_centre_frequency()
    chan_bw = bandwidth/AR.get_nchan()
    low_freq = centre_frequency - bandwidth / 2.0
    high_freq = centre_frequency + bandwidth / 2.0
    
    lastbroadfreq = low_freq + (high_freq-low_freq)/2.
    
    dt = period/nbin
    dm_window = 8*dt/(4.15e3*((lastbroadfreq-(chan_bw/2))**(-2)-(lastbroadfreq+(chan_bw/2))**(-2)))
    dm_minstep = dt/(4.15e3*((low_freq-(chan_bw/2))**(-2)-(high_freq+(chan_bw/2))**(-2))) 
    if (dm_minstep < 1e-5): dm_minstep = 1e-5
    if (dm_window < 0.01): dm_window = 0.01
    #print(lastbroadfreq, dr, ds)
    
    #exit(0)
    
    real_snr, flux_peak, snr_peak, sharpness, pulse_region = sharp_lvl(dm, pulse_region_sharp=[0, 0])
    
    snr_limit = 250
    if(real_snr > snr_limit):
        mode = 'sharpness'
        print("The S/N is %.1f > %d -> will use the %s" %(real_snr, snr_limit, mode))
    else:
        mode = 'flux_peak'
        print("The S/N is %.1f < %d -> will use the %s" %(real_snr, snr_limit, mode))
    
    first_dm = dm
    first_dm_window = dm_window
    first_snr = flux_peak #real_snr
    Force_pulse_region = False
    First = True
    while (dm_window > dm_minstep):
        #print(dm_window/first_dm_window)
        if (dm_window/first_dm_window < 0.02) and (First):
            First = False
            #print('ICCI4', pulse_region)
            real_snr, flux_peak, snr_peak, sharpness, pulse_region = sharp_lvl(dm, pulse_region_sharp=[0, 0])
            #print('snr_peak = ', snr_peak)
            Force_pulse_region = True
            snr_limit = 80
            if(verbose):print("SNR peak = %.1f" % snr_peak)
            if(0.9*first_snr > flux_peak):
                if(verbose):print("the flux_peak is smaler than at 0.9*start_flux_peak %.1f < %.1f -> will be bscrunch by 2" %(0.9*flux_peak, first_snr))
                AR.bscrunch(2)
                if (AR.get_nbin() < 32):
                    dm = first_dm
                    if(verbose):print("nbin is now < 32 and the SNR is lower thant the sart SNR -> dm_fit stop with the initial dm")
                    break
                dm = first_dm
                dt = period/float(AR.get_nbin())
                dm_window = 8*dt/(4.15e3*((lastbroadfreq-(chan_bw/2))**(-2)-(lastbroadfreq+(chan_bw/2))**(-2)))
                dm_minstep = dt/(4.15e3*((low_freq-(chan_bw/2))**(-2)-(high_freq+(chan_bw/2))**(-2)))
                if (dm_minstep < 1e-5): dm_minstep = 1e-5
                if (dm_window < 0.01): dm_window = 0.01
                pulse_region = search_pulse_region(dm)
                real_snr, flux_peak, snr_peak, sharpness, pulse_region = sharp_lvl(dm, pulse_region_sharp=pulse_region)
                first_snr = flux_peak
                Force_pulse_region = False
                First = True
                continue
            if( snr_peak < 5 ) and (AR.get_nbin() > 32):
                if(verbose):print("The snr_peak %.1f is smaler than 5  -> will be bscrunch by 2" %(snr_peak))
                AR.bscrunch(2)
                dm = first_dm
                dt = period/float(AR.get_nbin())
                dm_window = 8*dt/(4.15e3*((lastbroadfreq-(chan_bw/2))**(-2)-(lastbroadfreq+(chan_bw/2))**(-2)))
                dm_minstep = dt/(4.15e3*((low_freq-(chan_bw/2))**(-2)-(high_freq+(chan_bw/2))**(-2)))
                if (dm_minstep < 1e-5): dm_minstep = 1e-5
                if (dm_window < 0.01): dm_window = 0.01
                pulse_region = search_pulse_region(dm)
                real_snr, flux_peak, snr_peak, sharpness, pulse_region = sharp_lvl(dm, pulse_region_sharp=pulse_region)
                first_snr = flux_peak
                Force_pulse_region = False
                First = True
                continue
            if(real_snr > snr_limit):
                mode = 'sharpness'
                if(verbose):print("The SNR is %.1f > %d -> will use the %s" %(real_snr, snr_limit, mode))
            else:
                mode = 'flux_peak'
                if(verbose):print("The SNR is %.1f < %d -> will use the %s" %(real_snr, snr_limit, mode))
        dm = dm_trials(dm, dm_window, mode=mode, Force_pulse_region=Force_pulse_region, ncore=ncore, lim_min_dm=lim_min_dm)
        dm_window /= 2
        print(dm, dm_window)
    
    AR.set_dispersion_measure(dm)
    AR.dedisperse()
    
    real_snr, flux_peak, snr_peak, sharpness, pulse_region = sharp_lvl(dm, pulse_region_sharp=[0, 0])
    
    if (snr_peak >= 5) and (real_snr >= 20):
        dm_err = estimation_dm_error(pulse_region, real_snr, dm)
        if(verbose):print("Best dm is %.5f +- %.5f" % (dm, dm_err))
        AR0.set_dispersion_measure(dm)
        #AR0.dedisperse()  dedisperse + dededisperse add error in TOAs
        if (autorebin):
            if (AR.get_nbin() >= 8):
                AR0, rebin = auto_rebin(AR0)
    else:
        if (snr_peak < 5) and verbose: print("The snr_peak %.1f is smaler than 5" %(snr_peak))
        if (real_snr < 20) and verbose: print("and real_snr %.1f is smaler than 10  -> back to initial DM" %(real_snr))
        dm = first_dm
        dm_err = 0.0
        AR.set_dispersion_measure(dm)
        AR.dedisperse()

    return (AR0, dm, dm_err)









