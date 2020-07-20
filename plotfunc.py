import numpy as np
import psrchive as psr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import ScalarFormatter
# import seaborn as sns
# sns.set()
import os

"""
this code is a ploting library for pulsar archive data
    profil
    phase_freq
    phase_freq_no_ON
    phase_time
    bandpass
    zaped_bandpass
    diff_baseline
    dynaspect_bandpass
    dynaspect_onpulse
    snr_vs_frequency
    snr_vs_subintegration
    snr_vs_incremental_subintegration
"""


def profil(ar, AX):
    """
    plot the profil I L V in the requested area AX
    """
    arx = ar.clone()
    arx.dedisperse()
    arx.tscrunch()
    arx.fscrunch()
    arx.remove_baseline()
    phase = np.linspace(0, 1, arx.get_nbin())
    if arx.get_npol() > 1:
        arx.convert_state('Stokes')
        data = arx.get_data()
        data = data.squeeze()
        AX.plot(phase, data[0, :],
                'k', alpha=0.75, label='Total Intensity')
        AX.plot(phase, np.sqrt((data[1, :])**2 + (data[2, :])**2),
                'r', alpha=0.6, label='Linear polarization')
        AX.plot(phase, data[3, :],
                'b', alpha=0.6, label='Circular polarization')
        AX.legend(loc='upper right')
    else:
        data = arx.get_data()
        data = data.squeeze()
        AX.plot(phase, data, 'k')
    AX.grid(True, which="both", ls="-", alpha=0.65)
    AX.set_xlabel('Pulse Phase')
    AX.set_ylabel('Amplitude (AU)')

def ticks_format_func(value, tick_number):
    if (value == 0):
        return ''
    else:
        value = str("%.6f" % (value))
        while(value[-1] == '0'):
            value = value[0:-1]
        if(value[-1] == '.'):
            value = value[0:-1]
        return value


def phase_freq(ar, AX, pol=0, leftaxis=False, flatband=True, stokes=True, threshold=False):
    """
    Phase vs Freq plot with polarization selection in AX
    baseline is removed with a remove_baseline
    and the signal is adjusted by 1/baseline
    """
    arx = ar.clone()
    arx.dedisperse()
    arx.tscrunch()
    if arx.get_npol() > 1:
        if(stokes): arx.convert_state('Stokes')

    # min and max freq for the extent in imshow
    min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = arx.get_Profile(0, 0, ar.get_nchan()-1).get_centre_frequency()

    # rescale the baseline
    arx2 = arx.clone()
    arx2.pscrunch()

    # weights
    weights = arx.get_weights()
    weights = weights.squeeze()
    weights = weights/np.max(weights)

    subint = arx2.get_Integration(0)
    (bl_mean, bl_var) = subint.baseline_stats()
    bl_mean = bl_mean.squeeze()
    non_zeroes = np.where(bl_mean != 0.0)
    arx.remove_baseline()
    if(flatband):
        for ichan in range(arx.get_nchan()):
            for ipol in range(arx.get_npol()):
                prof = arx.get_Profile(0, ipol, ichan)
                if ichan in non_zeroes[0] and (prof.get_weight() != 0):
                    prof.scale(1/bl_mean[ichan])
                else:
                    prof.set_weight(0.0)
                    prof.scale(0)
    data = arx.get_data()
    data = data[:, pol, :, :].squeeze()
    if threshold:
        std_data = np.nanstd(data)
        ind = np.where(data > threshold*std_data)
        data[ind] = threshold*std_data
        ind = np.where(data < -threshold*std_data)
        data[ind] = -threshold*std_data

    for abin in range(arx.get_nbin()):
        data[:,abin] = data[:,abin]*weights

    data = np.flipud(data)
    if(stokes):
        if (pol > 0):
            cmap = 'bwr'
        else:
            cmap = 'afmhot'
    else:
        if (pol > 1):
            cmap = 'bwr'
        else:
            cmap = 'afmhot'
    fig = AX.imshow(data, interpolation='none', cmap=cmap,
              extent=[0, 1, min_freq, max_freq], aspect='auto')
    if (pol > 0):
        lim = np.max(np.abs(data))
        fig.set_clim(-lim, lim)
    if(stokes):
        string = ['I', 'Q', 'U', 'V']
    else:
        string = ['XX', 'YY', 'XY', 'YX']
    AX.text(0.95, min_freq+0.90*(max_freq-min_freq), string[pol],
            horizontalalignment='right',
            verticalalignment='top',
            fontdict={'family': 'DejaVu Sans Mono'},
            size=14,
            color='k',
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    AX.set_xlabel('Pulse Phase')
    AX.set_ylabel('Frequency (MHz)')
    AX.xaxis.set_major_locator(plt.MaxNLocator(2))
    AX.xaxis.set_major_formatter(plt.FuncFormatter(ticks_format_func))
    if (leftaxis):
        AXchan = AX.twinx()
        AXchan.yaxis.set_ticks_position('right')
        AXchan.set_ylim(top=ar.get_nchan())
        AXchan.set_ylabel('Channels')


def phase_freq_no_ON(ar, AX, pol=0, leftaxis=False, flatband=True, normesubint=False):
    """
    Phase vs Freq plot with polarization selection in AX
    All signal > 1.5 and  < -1.5 x the standar deviation is masked
    baseline is removed with a remove_baseline
    and the signal is adjusted by 1/baseline
    pol=4 is for the total intensity
    """
    arx = ar.clone()
    arx.dedisperse()
    if pol == 4:
        arx.pscrunch()
    arx.tscrunch()

    # min and max freq for the extent in imshow
    min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = arx.get_Profile(0, 0, ar.get_nchan()-1).get_centre_frequency()

    # rescale the baseline
    arx2 = arx.clone()
    arx2.pscrunch()
    subint = arx2.get_Integration(0)
    (bl_mean, bl_var) = subint.baseline_stats()
    bl_mean = bl_mean.squeeze()
    non_zeroes = np.where(bl_mean != 0.0)

    arx.remove_baseline()
    if(flatband):
        for ichan in range(arx.get_nchan()):
            for ipol in range(arx.get_npol()):
                prof = arx.get_Profile(0, ipol, ichan)
                if ichan in non_zeroes[0] and (prof.get_weight() != 0):
                    prof.scale(1/bl_mean[ichan])
                else:
                    prof.set_weight(0.0)
                    prof.scale(0)
    arx.bscrunch_to_nbin(128)
    if arx.get_nchan() > 64:
        arx.fscrunch(arx.get_nchan()/64)
    data = arx.get_data()
    if pol == 4:
        data = data[:, 0, :, :].squeeze()
    else:
        data = data[:, pol, :, :].squeeze()
    data = np.flipud(data)
    for i in range(2):
        if np.nanmax(data) > 1.5*np.nanstd(data):
            data[np.where(data > 1.5*np.nanstd(data))] = np.nanmedian(data)
        if np.nanmin(data) < -1.5*np.nanstd(data):
            data[np.where(data < -1.5*np.nanstd(data))] = np.nanmedian(data)
    if (normesubint):
        for subint in range(arx.get_nsubint()):
            data[subint, :] = data[subint, :] / np.nanmax(data[subint, :])
    AX.imshow(data, interpolation='none', cmap='afmhot',
              extent=[0, 1, min_freq, max_freq], aspect='auto')
    string = ['xx', 'yy', 'xy', 'yx', 'I']
    AX.text(0.95, min_freq+0.90*(max_freq-min_freq), string[pol],
            horizontalalignment='right',
            verticalalignment='top',
            fontdict={'family': 'DejaVu Sans Mono'},
            size=14,
            color='k',
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    AX.set_xlabel('Pulse Phase')
    AX.set_ylabel('Frequency (MHz)')
    if leftaxis:
        AXchan = AX.twinx()
        AXchan.yaxis.set_ticks_position('right')
        AXchan.set_ylim(top=ar.get_nchan())
        AXchan.set_ylabel('Channels')


def phase_time(ar, AX, pol=0, leftaxis=False, stokes=True, timenorme=False, threshold=False):
    """
    plot phase time prof in area AX
    the polarization can be selected
    """
    arx = ar.clone()
    arx.dedisperse()
    arx.remove_baseline()
    arx.fscrunch()
    if arx.get_npol() > 1:
        if(stokes): arx.convert_state('Stokes')
    # arx.pscrunch()
    tsubint = arx.integration_length() / arx.get_nsubint()

    # weights
    weights = arx.get_weights()
    weights = weights.squeeze()
    weights = weights/np.max(weights)

    data = arx.get_data()
    data = data[:, pol, 0, :]

    for abin in range(arx.get_nbin()):
        data[:,abin] = data[:,abin]*weights

    if (threshold):
        std_data = np.nanstd(data)
        ind = np.where(data > threshold*std_data)
        data[ind] = threshold*std_data
        ind = np.where(data < -threshold*std_data)
        data[ind] = -threshold*std_data

    if (arx.get_nsubint() == 1):
        data = np.repeat(data, 2, axis=0)
    else:
        data = np.flipud(data)
    if(stokes):
        if (pol > 0):
            cmap = 'bwr'
        else:
            cmap = 'afmhot'
    else:
        if (pol > 1):
            cmap = 'bwr'
        else:
            cmap = 'afmhot'
    if (timenorme):
        for subint in range(arx.get_nsubint()):
            if(np.nanmax(np.abs(data[subint,:])) != 0):
                data[subint,:] = data[subint,:] / np.nanmax(np.abs(data[subint,:]))

    fig = AX.imshow(data, interpolation='none', cmap=cmap,
              aspect='auto', extent=[0, 1, 0, arx.get_nsubint()*tsubint/60.])
    if (pol > 0):
        lim = np.max(np.abs(data))
        fig.set_clim(-lim, lim)
    if(stokes):
        string = ['I', 'Q', 'U', 'V']
    else:
        string = ['XX', 'YY', 'XY', 'YX']
    AX.text(0.95, 0.90*(arx.get_nsubint()*tsubint/60.), string[pol],
         horizontalalignment='right',
         verticalalignment='top',
         fontdict={'family': 'DejaVu Sans Mono'},
         size=10, color='k',
         bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    AX.set_xlabel('Pulse phase')
    AX.set_ylabel('Time (minutes)')
    if (leftaxis):
        AXsubint = AX.twinx()
        AXsubint.yaxis.set_ticks(np.arange(0, ar.get_nsubint()-1,
                                           ar.get_nsubint()/5))
        AXsubint.set_ylabel('Subint index')


def bandpass(ar, AX, botaxis=False, mask=False):
    """
    plot bandpass xx and yy in AX
    """
    arx = ar.clone()
    arx.tscrunch()
    if (arx.get_nbin() > 16):
        arx.bscrunch_to_nbin(16)
    if(mask):
        weights = arx.get_weights()
        weights = weights.squeeze()
        weights = weights/np.max(weights)

    min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = arx.get_Profile(0, 0, arx.get_nchan()-1).get_centre_frequency()
    freqs = np.linspace(min_freq, max_freq, arx.get_nchan())

    subint = arx.get_Integration(0)
    (bl_mean, bl_var) = subint.baseline_stats()
    bl_mean = bl_mean.squeeze()
    POL = ['xx', 'yy', 'xy', 'yx']
    color = ['b', 'r', 'g', 'm']
    if (arx.get_npol() > 1):
        for Xpol in range(arx.get_npol()):
            if(Xpol < 2):
                if(mask): bl_mean[Xpol,np.where(weights[:] <= 0.5)] = np.nan
                AX.semilogy(freqs, bl_mean[Xpol, :],
                            color[Xpol], alpha=0.5,
                            label='Polarization '+POL[Xpol])
    else:
        if(mask): bl_mean[np.where(weights[:] <= 0.5)] = np.nan
        AX.semilogy(freqs, bl_mean[:], color[0], alpha=0.5, label='Total intensity')

    if(mask):
        trans = mtransforms.blended_transform_factory(AX.transData,
                                                      AX.transAxes)
        if(mask):
            AX.fill_between(freqs, 0, 1,
                            where=weights[:] <= 0.5, facecolor='k',
                            alpha=0.6, transform=trans, label='masked channels')
    AX.legend(loc='upper right')
    AX.grid(True, which="both", ls="-", alpha=0.75)
    AX.set_xlabel('Frequency (MHz)')
    AX.set_ylabel('Amplitude (AU)')
    if (botaxis):
        AX_secondary = AX.twiny()
        AX_secondary.set_frame_on(True)
        AX_secondary.patch.set_visible(False)
        AX_secondary.xaxis.set_ticks_position('bottom')
        AX_secondary.set_xlabel('Channels')
        AX_secondary.xaxis.set_label_position('bottom')
        AX_secondary.spines['bottom'].set_position(('outward', 50))
        AX_secondary.set_xlim(1, ar.get_nchan())
        AX.set_xlim(min_freq, max_freq)


def zaped_bandpass(ar, AX, botaxis=False):
    """
    plot a no clean bandpass xx and yy in AX
    """
    arx = ar.clone()
    arx.tscrunch()

    weights = arx.get_weights()
    weights = weights.squeeze()
    weights = weights/np.max(weights)

    min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = arx.get_Profile(0, 0, arx.get_nchan()-1).get_centre_frequency()
    freqs = np.linspace(min_freq, max_freq, arx.get_nchan())
    zaped_bandpass = np.copy(weights)
    zaped_bandpass[np.where(weights[:] <= 0.01)] = np.nan
    zaped_bandpass = 1 - zaped_bandpass
    AX.plot(freqs, 100.*zaped_bandpass, 'b',
                alpha=0.5, label='Zaped bandpass')
    #AX.yaxis.set_major_locator(plt.MaxNLocator(2))
    AX.yaxis.set_major_formatter(plt.FuncFormatter(ticks_format_func))
    AX.fill_between(freqs, -1, 1000, where=weights <= 0.5,
                    facecolor='k', alpha=0.6, label='masked channels')
    AX.legend(loc='upper right')
    AX.grid(True, which="both", ls="-", alpha=0.75)
    AX.set_xlabel('Frequency (MHz)')
    AX.set_ylabel('masked channels (%)')
    AX.set_ylim(0, 100)
    #AX.ticklabel_format(axis="y", style="plain")
    if (botaxis):
        AX_secondary = AX.twiny()
        AX_secondary.set_frame_on(True)
        AX_secondary.patch.set_visible(False)
        AX_secondary.xaxis.set_ticks_position('bottom')
        AX_secondary.set_xlabel('Channels')
        AX_secondary.xaxis.set_label_position('bottom')
        AX_secondary.spines['bottom'].set_position(('outward', 50))
        AX_secondary.set_xlim(1, ar.get_nchan())
        AX.set_xlim(min_freq, max_freq)


def dynaspect_bandpass(ar, AX, left_onpulse=0, righ_onpulse=0, leftaxis=False, botaxis=False, flatband=True, threshold=False):
    """
    plot dynamic_spect plot of the baseline in AX
    with a correction of the bandpass
    """
    arx = ar.clone()
    arx.pscrunch()
    arx.dedisperse()

    if(left_onpulse == 0) and (righ_onpulse == 0):
        if(arx.get_nbin() > 16):
            arx.bscrunch_to_nbin(16)
    arx2 = arx.clone()
    arx2.tscrunch()

    tsubint = arx.integration_length() / arx.get_nsubint()
    integ = arx2.get_Integration(0)
    (bl_mean, bl_var) = integ.baseline_stats()
    bl_mean = bl_mean.squeeze()
    bl_var = bl_var.squeeze()
    non_zeroes = np.where(bl_mean != 0.0)

    weights = arx.get_weights()
    weights = weights.squeeze()
    weights = weights/np.max(weights)

    bl_mean_avg = np.average(bl_mean[non_zeroes])
    if (flatband):
        for isub in range(arx.get_nsubint()):
            for ipol in range(arx.get_npol()):
                for ichan in range(arx.get_nchan()):
                    prof = arx.get_Profile(isub, ipol, ichan)
                    if ichan in non_zeroes[0]:
                        prof.offset(-bl_mean[ichan])
                        prof.scale(bl_mean_avg / bl_mean[ichan])
                    else:
                        prof.set_weight(0.0)
    if(left_onpulse == 0) and (righ_onpulse == 0):
        data = np.min(arx.get_data(), axis=3).squeeze()
    else:
        nbins = arx.get_nbin()
        offbins = np.ones(nbins, dtype='bool')
        if(left_onpulse < righ_onpulse):
            offbins[ left_onpulse : righ_onpulse ] = False
        else:
            offbins[ left_onpulse : ] = False
            offbins[ : righ_onpulse ] = False
        #data = np.median(arx.get_data()[:, :, :, offbins], axis=3).squeeze()
        data = arx.get_data()[:, 0, :, :]
        data = np.median(data[:, :, offbins], axis=2)
        #print(left_onpulse, righ_onpulse)
        #plt.plot(np.mean(np.mean(np.mean(arx.get_data()*weights, axis=0), axis=0), axis=0))
        #plt.show()

    if (arx.get_nsubint() == 1):
        data = np.repeat(data*weights, 2, axis=0)
    data = np.rot90(data*weights)
    data = data-np.nanmean(data)

    DATAstd = np.nanstd(data)

    if threshold:
        sigma = threshold
    else:
        sigma = 3.

    if(np.nanmax(data) > DATAstd*sigma):
        isub, ichan = np.where(data > DATAstd*sigma)
        data[isub, ichan] = DATAstd*sigma
    if(np.nanmin(data) < -DATAstd*sigma):
        isub, ichan = np.where(data < -DATAstd*sigma)
        data[isub, ichan] = -DATAstd*sigma

    data = data-np.nanmean(data)

    #percent1 = np.nanpercentile(data, 2)
    #percent50 = np.nanpercentile(data, 50)
    #percent99 = np.nanpercentile(data, 98)
    #test_data = np.copy(data)
    #test_data[np.isnan(test_data)] = percent50
    #test_data[np.isinf(test_data)] = percent50
    #condition1 = (test_data < percent1)
    #condition2 = (test_data > percent99)
    #if np.any(condition1): data[np.where( condition1 )] = percent1
    #if np.any(condition2): data[np.where( condition2 )] = percent99
    
    min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = arx.get_Profile(0, 0, arx.get_nchan()-1).get_centre_frequency()
    AX.imshow(data, interpolation='nearest', cmap='afmhot',
              aspect='auto',
              extent=[0, arx.get_nsubint()*tsubint/60.,
              min_freq, max_freq])
    string = 'Bandpass'
    AX.text(0.95*arx.get_nsubint()*tsubint/60.,
            min_freq+0.90*(max_freq-min_freq), string,
            horizontalalignment='right',
            verticalalignment='top',
            fontdict={'family': 'DejaVu Sans Mono'},
            size=10,
            color='k',
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    AX.set_ylabel('Frequency (MHz)')
    AX.set_xlabel('Time (minutes)')
    if (leftaxis):
        AXchan = AX.twinx()
        AXchan.yaxis.set_ticks_position('right')
        AXchan.set_ylim(top=ar.get_nchan())
        AXchan.set_ylabel('Channels')
    if (botaxis):
        AX_secondary = AX.twiny()
        AX_secondary.set_frame_on(True)
        AX_secondary.patch.set_visible(False)
        AX_secondary.xaxis.set_ticks_position('bottom')
        AX_secondary.set_xlabel('Subintegration')
        AX_secondary.xaxis.set_label_position('bottom')
        AX_secondary.spines['bottom'].set_position(('outward', 50))
        AX_secondary.set_xlim(1, ar.get_nsubint())
        AX.set_xlim(0, ar.integration_length()/60.)


def dynaspect_onpulse(ar, AX, left_onpulse=0, righ_onpulse=0, leftaxis=False, flatband=True, botaxis=False, threshold=False):
    """
    plot dynamic_spect plot of the onpulse
    note: it's not the real onpulse, but the highest bin on 64bin
    """
    arx = ar.clone()
    arx.pscrunch()
    arx.dedisperse()
    #arx.remove_baseline()
    if(left_onpulse == 0) and (righ_onpulse == 0):
        if(arx.get_nbin() > 32):
            arx.bscrunch_to_nbin(32)
    arx2 = arx.clone()
    arx2.tscrunch()

    weights = ar.get_weights()
    weights = weights.squeeze()
    weights = weights/np.max(weights)

    tsubint = arx.integration_length() / arx.get_nsubint()
    integ = arx2.get_Integration(0)
    (bl_mean, bl_var) = integ.baseline_stats()
    bl_mean = bl_mean.squeeze()
    bl_var = bl_var.squeeze()
    non_zeroes = np.where(bl_mean != 0.0)

    bl_mean_avg = np.average(bl_mean[non_zeroes])
    if (flatband):
        for isub in range(arx.get_nsubint()):
            integ = arx.get_Integration(isub)
            (bl_mean, bl_var) = integ.baseline_stats()
            bl_mean = bl_mean.squeeze()
            bl_var = bl_var.squeeze()
            non_zeroes = np.where(bl_mean != 0.0)
            for ipol in range(arx.get_npol()):
                for ichan in range(arx.get_nchan()):
                    prof = arx.get_Profile(isub, ipol, ichan)
                    if ichan in non_zeroes[0]:
                        prof.offset(-bl_mean[ichan])
                        prof.scale(bl_mean_avg / bl_mean[ichan])
                    else:
                        prof.set_weight(0.0)

    min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = arx.get_Profile(0, 0, arx.get_nchan()-1).get_centre_frequency()

    if(left_onpulse == 0) and (righ_onpulse == 0):
        data = np.max(arx.get_data(), axis=3).squeeze() - np.min(arx.get_data(), axis=3).squeeze()
    else:
        nbins = arx.get_nbin()
        onbins = np.zeros(nbins, dtype='bool')
        offbins = np.ones(nbins, dtype='bool')
        if(left_onpulse < righ_onpulse):
            onbins[ left_onpulse : righ_onpulse ] = True
            offbins[ left_onpulse : righ_onpulse ] = False
        else:
            onbins[ left_onpulse : ] = True
            onbins[ : righ_onpulse ] = True
            offbins[ left_onpulse : ] = False
            offbins[ : righ_onpulse ] = False
        data = arx.get_data()
        dataON = data[:, 0, :, :]
        dataOFF = data[:, 0, :, :]
        dataON = dataON[:, :, onbins]
        dataOFF = dataOFF[:, :, offbins]
        data = np.mean(dataON, axis=2) - np.median(dataOFF, axis=2)
        #print(left_onpulse, righ_onpulse)
        #data = arx.get_data()
        #print(np.shape(data))
        #nsubint, npol, nchan, nbin = np.shape(data)
        #for ibin in range(nbin):
        #    data[:, 0, :, ibin] = data[:, 0, :, ibin]*weights
        #plt.plot(np.mean(np.mean(data[:, 0, :, :], axis=0), axis=0))
        #plt.show()
        #exit(0)
    if (arx.get_nsubint() == 1):
        data = np.repeat(data*weights, 2, axis=0)
    data = np.rot90(data*weights)

    data = data-np.nanmean(data)

    DATAstd = np.nanstd(data)

    if threshold:
        sigma = threshold
    else:
        sigma = 3.

    if(np.nanmax(data) > DATAstd*sigma):
        isub, ichan = np.where(data > DATAstd*sigma)
        data[isub, ichan] = DATAstd*sigma
    if(np.nanmin(data) < -DATAstd*sigma):
        isub, ichan = np.where(data < -DATAstd*sigma)
        data[isub, ichan] = -DATAstd*sigma

    data = data-np.nanmean(data)


    #percent1 = np.nanpercentile(data, 2)
    #percent50 = np.nanpercentile(data, 50)
    #percent99 = np.nanpercentile(data, 98)
    #test_data = np.copy(data)
    #test_data[np.isnan(test_data)] = percent50
    #test_data[np.isinf(test_data)] = percent50
    #condition1 = (test_data < percent1)
    #condition2 = (test_data > percent99)
    #if (np.any(condition1)): data[np.where( condition1 )] = percent1
    #if (np.any(condition2)): data[np.where( condition2 )] = percent99

    AX.imshow(data, interpolation='none', cmap='afmhot',
              aspect='auto',
              extent=[0, arx.get_nsubint()*tsubint/60., min_freq, max_freq])
    string = 'ON pulse'
    AX.text(0.95*arx.get_nsubint()*tsubint/60.,
            min_freq+0.90*(max_freq-min_freq), string,
            horizontalalignment='right',
            verticalalignment='top',
            fontdict={'family': 'DejaVu Sans Mono'},
            size=10,
            color='k',
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    AX.set_ylabel('Frequency (MHz)')
    AX.set_xlabel('Time (minutes)')
    if (leftaxis):
        AXchan = AX.twinx()
        AXchan.yaxis.set_ticks_position('right')
        AXchan.set_ylim(top=ar.get_nchan())
        AXchan.set_ylabel('Channels')
    if (botaxis):
        AX_secondary = AX.twiny()
        AX_secondary.set_frame_on(True)
        AX_secondary.patch.set_visible(False)
        AX_secondary.xaxis.set_ticks_position('bottom')
        AX_secondary.set_xlabel('Subintegration')
        AX_secondary.xaxis.set_label_position('bottom')
        AX_secondary.spines['bottom'].set_position(('outward', 50))
        AX_secondary.set_xlim(1, ar.get_nsubint())
        AX.set_xlim(0, ar.integration_length()/60.)


def snr_vs_frequency(ar, AX, botaxis=False):
    """
    plot signal noise ratio versus frequency
    """
    arx = ar.clone()
    arx.pscrunch()
    arx.tscrunch()
    min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = arx.get_Profile(0, 0, arx.get_nchan()-1).get_centre_frequency()

    SNR = np.zeros(arx.get_nchan())
    for chan in range(arx.get_nchan()):
        prof = arx.get_Profile(0, 0, chan)
        SNR[chan] = prof.snr()

    channels = np.linspace(1, ar.get_nchan(), ar.get_nchan())
    AX.plot(channels, SNR)
    AX.set_ylabel('Signal noise ratio')
    AX.set_xlabel('Channels')
    if (botaxis):
        AX_secondary = AX.twiny()
        AX_secondary.set_frame_on(True)
        AX_secondary.patch.set_visible(False)
        AX_secondary.xaxis.set_ticks_position('bottom')
        AX_secondary.set_xlabel('Frequency (MHz)')
        AX_secondary.xaxis.set_label_position('bottom')
        AX_secondary.spines['bottom'].set_position(('outward', 50))
        AX_secondary.set_xlim(min_freq, max_freq)
        AX.set_xlim(1, ar.get_nchan())


def snr_vs_subintegration(ar, AX, botaxis=False):
    """
    plot signal noise ratio in each subintegration
    """
    arx = ar.clone()
    arx.pscrunch()
    arx.fscrunch()

    SNR = np.zeros(arx.get_nsubint())
    for isub in range(arx.get_nsubint()):
        prof = arx.get_Profile(isub, 0, 0)
        SNR[isub] = prof.snr()

    subintegration = np.linspace(1, ar.get_nsubint(), ar.get_nsubint())
    AX.plot(subintegration, SNR)
    AX.set_ylabel('Signal noise ratio')
    AX.set_xlabel('Subintegration')
    string = 'SNR per suintegration'
    AX.text(0.95*arx.get_nsubint(),
            np.min(SNR)+0.90*(np.max(SNR)-np.min(SNR)), string,
            horizontalalignment='right',
            verticalalignment='top',
            fontdict={'family': 'DejaVu Sans Mono'},
            size=10,
            color='k',
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    if (botaxis):
        AX_secondary = AX.twiny()
        AX_secondary.set_frame_on(True)
        AX_secondary.patch.set_visible(False)
        AX_secondary.xaxis.set_ticks_position('bottom')
        AX_secondary.set_xlabel('Time (minutes)')
        AX_secondary.xaxis.set_label_position('bottom')
        AX_secondary.spines['bottom'].set_position(('outward', 50))
        AX_secondary.set_xlim(0, ar.integration_length()/60.)
        AX.set_xlim(1, ar.get_nsubint())


def snr_vs_incremental_subintegration(ar, AX, botaxis=False):
    """
    plot signal noise ratio for integrated subintegration
    """
    arx = ar.clone()
    arx.pscrunch()
    arx.fscrunch()
    new_integration = arx.get_Integration(0).total()
    SNR = np.zeros(arx.get_nsubint())
    SNR[0] = new_integration.get_Profile(0, 0).snr()
    for isub in range(1, arx.get_nsubint()):
        next_integration = arx.get_Integration(isub).total()
        new_integration.combine(next_integration)
        prof = new_integration.get_Profile(0, 0)
        SNR[isub] = prof.snr()
    subintegration = np.linspace(1, ar.get_nsubint(), ar.get_nsubint())
    AX.plot(subintegration, SNR)
    AX.set_ylabel('Signal noise ratio')
    AX.set_xlabel('Subintegration')
    string = 'integrated SNR'
    AX.text(0.95*arx.get_nsubint(),
            np.min(SNR)+0.90*(np.max(SNR)-np.min(SNR)), string,
            horizontalalignment='right',
            verticalalignment='top',
            fontdict={'family': 'DejaVu Sans Mono'},
            size=10,
            color='k',
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    if (botaxis):
        AX_secondary = AX.twiny()
        AX_secondary.set_frame_on(True)
        AX_secondary.patch.set_visible(False)
        AX_secondary.xaxis.set_ticks_position('bottom')
        AX_secondary.set_xlabel('Time (minutes)')
        AX_secondary.xaxis.set_label_position('bottom')
        AX_secondary.spines['bottom'].set_position(('outward', 50))
        AX_secondary.set_xlim(0, ar.integration_length()/60.)
        AX.set_xlim(1, ar.get_nsubint())
