#!/usr/bin/env python

# To get help type 'man python' at the terminal

# Written by L. Bondonneau, 2022

import os
import numpy as np
from multiprocessing import Pool
from methode_class import smoothGaussian
from mypsrchive import psrchive_class

CONFIG_FILE = os.path.dirname(os.path.realpath(__file__)) + '/' + 'NenuPlot.conf'


class DM_fit_class(psrchive_class):
    def __init__(self, *args, **kwargs):
        super(DM_fit_class, self).__init__(*args, **kwargs)

    def SNR_methode(self, dm, plot=False, pulse_region_sharp=(0, 0), indice_mp=None):
        global scrunch_Tp
        ar2 = scrunch_Tp.myclone()
        ar2.set_dispersion_measure(dm)
        ar2.dedisperse()
        # ar2.flatten_from_mad()
        ar2.fscrunch()
        rebuild_local_scrunch = True

        if (pulse_region_sharp[0] == 0) and (pulse_region_sharp[1] == 0):
            ar2.get_on_window(safe_fraction=1 / 8., rebuild_local_scrunch=True)
            pulse_region_sharp = ar2.get_onpulse()
            rebuild_local_scrunch = False
        real_snr, real_snr_err = ar2.snr_range(on_left=pulse_region_sharp[0], on_right=pulse_region_sharp[1], rebuild_local_scrunch=rebuild_local_scrunch)
        flux_peak = ar2.flux_peak(on_left=pulse_region_sharp[0], on_right=pulse_region_sharp[1], rebuild_local_scrunch=False)
        snr_peak = ar2.snr_peak(on_left=pulse_region_sharp[0], on_right=pulse_region_sharp[1], rebuild_local_scrunch=False)
        sharpness = ar2.sharpness(on_left=pulse_region_sharp[0], on_right=pulse_region_sharp[1], rebuild_local_scrunch=False)

        if indice_mp is not None:
            return real_snr, flux_peak, snr_peak, sharpness, pulse_region_sharp, indice_mp
        else:
            return real_snr, flux_peak, snr_peak, sharpness, pulse_region_sharp

    def resultCollector(self, result):
        self.__result.append(result)

    def search_pulse_region(self, dm):
        global scrunch_Tp
        ar2 = scrunch_Tp.myclone()
        ar2.set_dispersion_measure(dm)
        pulse_region_search = ar2.get_on_window(safe_fraction=1 / 8., rebuild_local_scrunch=True)
        return pulse_region_search

    def dm_trials(self, dm, diff, plot=False, mode='sharpness', Force_pulse_region=False, ncore=8, lim_min_dm=0.5):
        dm_min = dm - float(diff) / 2.  # -16.6
        dm_max = dm + float(diff) / 2.  # -16.1
        if(dm_min < lim_min_dm):
            dm_min = lim_min_dm
        delt = (dm_max - dm_min) / float(200)

        dm_vec = np.linspace(float(dm_min), float(dm_max), 1 + int(((float(dm_max) - float(dm_min)) / float(delt))))

        if (Force_pulse_region):
            pulse_region_trial = self.search_pulse_region(dm)  # pulse_region for the best dm
        else:
            pulse_region_trial = [0, 0]

        # removing the doppler from the DM
        dm_vec_dedoppler = dm_vec * (self.get_doppler()**2)

        real_snr, flux_peak, snr_peak, sharpness, pulse_region_trial = multiprocessing_SNR_methode(ncore, dm_vec_dedoppler,
                                                                                                   Force_pulse_region, pulse_region_trial)

        dm_vec = dm_vec

        snr_peak = smoothGaussian(snr_peak, size=0.01)
        sharpness = smoothGaussian(sharpness, size=0.01)
        real_snr = smoothGaussian(real_snr, size=0.01)
        flux_peak = smoothGaussian(flux_peak, size=0.01)

        if (plot):
            import matplotlib.pyplot as plt
            AX = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
            AX.plot(dm_vec, snr_peak / np.max(snr_peak), label='snr_peak')
            AX.plot(dm_vec, sharpness / np.max(sharpness), label='sharpness')
            AX.plot(dm_vec, real_snr / np.max(real_snr), label='real_snr')
            AX.plot(dm_vec, flux_peak / np.max(flux_peak), label='flux_peak')
            AX.legend(loc='upper right')
            plt.show()

        if (mode == 'sharpness'):
            return dm_vec[np.nanargmax(sharpness)]
        elif (mode == 'snr_peak'):
            return dm_vec[np.nanargmax(snr_peak)]
        elif (mode == 'real_snr'):
            return dm_vec[np.nanargmax(real_snr)]
        elif (mode == 'flux_peak'):
            return dm_vec[np.nanargmax(flux_peak)]

    # def estimation_dm_error(pulse_region_est, real_snr, dm):
    #     global AR
    #     if (real_snr < 100):
    #         nchan_new = 10
    #     else:
    #         nchan_new = int(real_snr / 10.)
    #
    #     nchan = AR.get_nchan()
    #
    #     rechan_factor = int(np.ceil(float(nchan) / float(nchan_new)))
    #     # print(float(nchan)/float(nchan_new), rechan_factor)
    #     if(rechan_factor > 1):
    #         AR.fscrunch(rechan_factor)
    #     AR.tscrunch()
    #     AR.pscrunch()
    #     AR.set_dispersion_measure(dm)
    #     AR.dedisperse()
    #
    #     freqs = AR.get_freqs(doppler=True)
    #     nchan = AR.get_nchan()
    #     nchan_bw = (np.max(freqs) - np.min(freqs)) / (nchan - 1)
    #     weights = AR.get_weights()
    #     weights = weights.squeeze()
    #     weights = weights / np.max(weights)
    #     # print(np.shape(weights))
    #     # ar2.fscrunch()
    #     nbin = AR.get_nbin()
    #     # if (pulse_region[0]==0) and (pulse_region[1]==0) :
    #     #    pulse_region[0], pulse_region[1] = auto_find_on_window(ar2)
    #     # AR = flatten_from_mad(pulse_region[0], pulse_region[1])
    #     data = AR.get_data().squeeze()
    #     delta_dm_vec = []
    #     period = AR.get_Integration(0).get_folding_period()  # in sec
    #     first = True
    #     top_valid_chan = False
    #     for ichan in range(AR.get_nchan() - 1, 0, -1):
    #         if(weights[ichan] == 0):
    #             real_snr = 0
    #         else:
    #             template = data[ichan, :]
    #             # print(pulse_region)
    #             template = np.roll(template, -pulse_region_est[0])
    #             on_right = (pulse_region_est[1] - pulse_region_est[0]) % nbin
    #             # print(np.shape(template),  on_right+1, nbin)
    #             real_snr = auto_snr(template, on_right + 1, nbin)
    #             # print(freqs[ichan], real_snr)
    #             on_in_sec = period * (float(on_right) / float(nbin))
    #             top_freq = freqs[ichan] + nchan_bw / 2
    #             bot_freq = freqs[ichan] - nchan_bw / 2
    #             if (real_snr == 0):
    #                 real_snr = np.nan
    #             if (real_snr > 8) and (first is True):
    #                 top_valid_chan = top_freq
    #                 first = False
    #             if (first is False):
    #                 dt = on_in_sec / (real_snr / 5.)
    #                 delta_dm = dt / (4150. * ((bot_freq)**(-2) - (top_valid_chan)**(-2)))
    #                 # print(freqs[ichan], real_snr, dt, delta_dm )
    #                 delta_dm_vec.append(delta_dm)
    #     try:
    #         result = np.nanmin(delta_dm_vec)
    #     except:
    #         dt = on_in_sec / (8 / 5.)
    #         if not top_valid_chan:
    #             print("WARING: SNR too small can not compute a DM error")
    #             result = np.nan
    #         else:
    #             result = dt / (4150. * ((top_valid_chan - nchan_bw)**(-2) - (top_valid_chan)**(-2)))
    #     return result

    # def auto_rebin(ar, ichan=None):
    #     arx = ar.myclone()
    #     arx.dedisperse()
    #     arx.pscrunch()
    #     arx.tscrunch()
    #     if (ichan is None):
    #         arx.fscrunch()
    #         ichan = 0
    #     arx.remove_baseline()
    #     arx.get_on_window(safe_fraction=3 / 100., rebuild_local_scrunch=True)
    #
    #     prof = np.asarray(arx.get_Profile(0, 0, ichan).get_amps() * 10000)
    #     nbin = arx.get_nbin()
    #     prof_smooth = smoothGaussian(prof)
    #
    #     std = np.std(prof[arx.offbins])
    #     delta = np.max(np.abs(prof_smooth[arx.onbins] - np.roll(prof_smooth[arx.onbins], 1)))
    #
    #     bscrunch_factor = 1
    #     # print(nbin, delta, 2.5*std)
    #     while (nbin > 32) and (delta < 2.9 * std):
    #         arx.bscrunch(2)
    #         nbin = arx.get_nbin()
    #         arx.get_on_window(safe_fraction=7 / 100., rebuild_local_scrunch=True)
    #         prof = np.asarray(arx.get_Profile(0, 0, ichan).get_amps() * 10000)
    #         prof_smooth = smoothGaussian(prof)
    #         std = np.std(prof[arx.offbins])
    #         delta = np.max(np.abs(prof_smooth[arx.onbins] - np.roll(prof_smooth[arx.onbins], 1)))
    #         bscrunch_factor *= 2
    #         print(nbin, delta, 2.9 * std)
    #     if (bscrunch_factor > 1):
    #         bscrunch_factor /= 2
    #         if (bscrunch_factor > 1):
    #             ar.bscrunch(bscrunch_factor)
    #             print('automatic rebin by factor: ' + str(bscrunch_factor))
    #         else:
    #             print('automatic rebin by factor: ' + str(bscrunch_factor))
    #     return ar, bscrunch_factor

    def DM_fit(self, verbose=False, ncore=8, autorebin=True, lim_min_dm=0.5, plot=False):
        global scrunch_Tp
        scrunch_Tp = self.myclone()
        dm_archive = self.get_dispersion_measure()
        if(verbose):
            self.log.log("Coherent dm = %.4f pc cm-3" % dm_archive, objet='DM_fit')
        scrunch_Tp.tscrunch()
        scrunch_Tp.pscrunch()
        self.scrunch_Tpd = scrunch_Tp.myclone()
        self.scrunch_Tpd.dedisperse()

        # diff = 0.1
        dm = self.scrunch_Tpd.get_dispersion_measure()
        period = self.scrunch_Tpd.get_Integration(0).get_folding_period()  # in sec
        nbin = self.scrunch_Tpd.get_nbin()
        # bandwidth = self.scrunch_Tpd.get_bandwidth()
        # centre_frequency = self.scrunch_Tpd.get_centre_frequency()
        chan_bw = np.mean(self.scrunch_Tpd.get_chan_bw())
        # low_freq = centre_frequency - bandwidth / 2.0
        # high_freq = centre_frequency + bandwidth / 2.0
        low_freq = self.scrunch_Tpd.freqs[0] - chan_bw / 2.
        high_freq = self.scrunch_Tpd.freqs[-1] + chan_bw / 2.

        lastbroadfreq = low_freq + (high_freq - low_freq) / 2.

        dt = period / nbin
        dm_window = 8 * dt / (4.15e3 * ((lastbroadfreq - (chan_bw / 2))**(-2) - (lastbroadfreq + (chan_bw / 2))**(-2)))
        dm_minstep = dt / (4.15e3 * ((low_freq - (chan_bw / 2))**(-2) - (high_freq + (chan_bw / 2))**(-2)))
        if (dm_minstep < 1e-5):
            dm_minstep = 1e-5
        if (dm_window < 0.01):
            dm_window = 0.01
        # print(lastbroadfreq, dr, ds)

        # exit(0)

        real_snr, flux_peak, snr_peak, sharpness, pulse_region = self.SNR_methode(dm, pulse_region_sharp=[0, 0])

        snr_limit = 250
        if(real_snr > snr_limit):
            mode = 'sharpness'
            self.log.log("The S/N is %.1f > %d -> will use the %s" % (real_snr, snr_limit, mode), objet='DM_fit')
        else:
            mode = 'flux_peak'
            self.log.log("The S/N is %.1f < %d -> will use the %s" % (real_snr, snr_limit, mode), objet='DM_fit')

        first_dm = dm
        first_dm_window = dm_window
        first_snr = flux_peak  # real_snr
        Force_pulse_region = False
        First = True
        while (dm_window > dm_minstep):
            # print(dm_window/first_dm_window)
            if (dm_window / first_dm_window < 0.02) and (First):
                First = False
                # print('ICCI4', pulse_region)
                real_snr, flux_peak, snr_peak, sharpness, pulse_region = self.SNR_methode(dm, pulse_region_sharp=[0, 0])
                # print('snr_peak = ', snr_peak)
                Force_pulse_region = True
                snr_limit = 80
                if(verbose):
                    self.log.log("SNR peak = %.1f" % snr_peak, objet='DM_fit')
                if(0.9 * first_snr > flux_peak):
                    if(verbose):
                        self.log.log("the flux_peak is smaler than at 0.9*start_flux_peak %.1f < %.1f -> will be bscrunch by 2" %
                                     (0.9 * flux_peak, first_snr), objet='DM_fit')
                    self.scrunch_Tpd.bscrunch(2)
                    if (self.scrunch_Tpd.get_nbin() < 32):
                        dm = first_dm
                        if(verbose):
                            self.log.log("nbin is now < 32 and the SNR is lower thant the sart SNR -> dm_fit stop with the initial dm", objet='DM_fit')
                        break
                    dm = first_dm
                    dt = period / float(self.scrunch_Tpd.get_nbin())
                    dm_window = 8 * dt / (4.15e3 * ((lastbroadfreq - (chan_bw / 2))**(-2) - (lastbroadfreq + (chan_bw / 2))**(-2)))
                    dm_minstep = dt / (4.15e3 * ((low_freq - (chan_bw / 2))**(-2) - (high_freq + (chan_bw / 2))**(-2)))
                    if (dm_minstep < 1e-5):
                        dm_minstep = 1e-5
                    if (dm_window < 0.01):
                        dm_window = 0.01
                    pulse_region = self.search_pulse_region(dm)
                    real_snr, flux_peak, snr_peak, sharpness, pulse_region = self.SNR_methode(dm, pulse_region_sharp=pulse_region)
                    first_snr = flux_peak
                    Force_pulse_region = False
                    First = True
                    continue
                if(snr_peak < 5) and (self.scrunch_Tpd.get_nbin() > 32):
                    if(verbose):
                        self.log.log("The snr_peak %.1f is smaler than 5  -> will be bscrunch by 2" % (snr_peak), objet='DM_fit')
                    self.scrunch_Tpd.bscrunch(2)
                    dm = first_dm
                    dt = period / float(self.scrunch_Tpd.get_nbin())
                    dm_window = 8 * dt / (4.15e3 * ((lastbroadfreq - (chan_bw / 2))**(-2) - (lastbroadfreq + (chan_bw / 2))**(-2)))
                    dm_minstep = dt / (4.15e3 * ((low_freq - (chan_bw / 2))**(-2) - (high_freq + (chan_bw / 2))**(-2)))
                    if (dm_minstep < 1e-5):
                        dm_minstep = 1e-5
                    if (dm_window < 0.01):
                        dm_window = 0.01
                    pulse_region = self.search_pulse_region(dm)
                    real_snr, flux_peak, snr_peak, sharpness, pulse_region = self.SNR_methode(dm, pulse_region_sharp=pulse_region)
                    first_snr = flux_peak
                    Force_pulse_region = False
                    First = True
                    continue
                if(real_snr > snr_limit):
                    mode = 'sharpness'
                    if(verbose):
                        self.log.log("The SNR is %.1f > %d -> will use the %s" % (real_snr, snr_limit, mode), objet='DM_fit')
                else:
                    mode = 'flux_peak'
                    if(verbose):
                        self.log.log("The SNR is %.1f < %d -> will use the %s" % (real_snr, snr_limit, mode), objet='DM_fit')
            dm = self.dm_trials(dm, dm_window, mode=mode, Force_pulse_region=Force_pulse_region, ncore=ncore, lim_min_dm=lim_min_dm, plot=plot)
            dm_window /= 2
            # print(dm, dm_window)

        scrunch_Tp.set_dispersion_measure(dm)
        scrunch_Tp.dedisperse()

        real_snr, flux_peak, snr_peak, sharpness, pulse_region = self.SNR_methode(dm, pulse_region_sharp=[0, 0])

        # if (snr_peak >= 5) and (real_snr >= 20):
        #    dm_err = estimation_dm_error(pulse_region, real_snr, dm)
        #    if(verbose):
        #        print("Best dm is %.5f +- %.5f" % (dm, dm_err))
        #    AR0.set_dispersion_measure(dm)
        #    # AR0.dedisperse()  dedisperse + dededisperse add error in TOAs
        #    if (autorebin):
        #        if (AR.get_nbin() >= 8):
        #            AR0, rebin = auto_rebin(AR0)
        # else:
        #    if (snr_peak < 5) and verbose:
        #        print("The snr_peak %.1f is smaler than 5" % (snr_peak))
        #    if (real_snr < 20) and verbose:
        #        print("and real_snr %.1f is smaler than 10  -> back to initial DM" % (real_snr))
        #    dm = first_dm
        #    dm_err = 0.0
        #    AR.set_dispersion_measure(dm)
        #    AR.dedisperse()
        self.log.log("Best dm is %.5f pc.cm-3" % (dm), objet='DM_fit')
        # return (AR0, dm, 0)
        self.set_dispersion_measure(dm)
        self.dedisperse()


def SNR_methode_wrapper(args):
    return DM_fit_class.SNR_methode(DM_fit_class(), *args)


def multiprocessing_SNR_methode(ncore, dm_vec, Force_pulse_region, pulse_region_trial, MP=True):
    real_snr = np.zeros(len(dm_vec))
    flux_peak = np.zeros(len(dm_vec))
    sharpness = np.zeros(len(dm_vec))
    snr_peak = np.zeros(len(dm_vec))
    # with Pool(processes=int(ncore)) as pool:

    if (MP is True):
        pool = Pool(processes=int(ncore))
        if (Force_pulse_region):
            multiple_results = pool.imap_unordered(SNR_methode_wrapper,
                                                   [(dm_vec[n], False, pulse_region_trial, n)
                                                    for n in range(len(dm_vec))])
        else:
            multiple_results = pool.imap_unordered(SNR_methode_wrapper,
                                                   [(dm_vec[n], False, [0, 0], n)
                                                    for n in range(len(dm_vec))])
        for snr1, snr2, snr3, snr4, pulse_region, idm in multiple_results:
            real_snr[idm], flux_peak[idm], snr_peak[idm], sharpness[idm], pulse_region_trial = snr1, snr2, snr3, snr4, pulse_region
        pool.close()
    else:
        for n in range(len(dm_vec)):
            snr1, snr2, snr3, snr4, pulse_region, idm = SNR_methode_wrapper((dm_vec[n], False, [0, 0], n))
            real_snr[idm], flux_peak[idm], snr_peak[idm], sharpness[idm], pulse_region_trial = snr1, snr2, snr3, snr4, pulse_region

    return (real_snr, flux_peak, snr_peak, sharpness, pulse_region_trial)


if __name__ == "__main__":
    ar = DM_fit_class(ar_name=['/databf2/nenufar-pulsar/DATA/B1919+21/PSR/B1919+21_D20220303T1301_59641_500675_0057_BEAM2.fits'],
                      bscrunch=1, tscrunch=2, pscrunch=True, verbose=True)
    # ar.MyArchive_load(['/databf2/nenufar-pulsar/DATA/B1508+55/PSR/B1508+55_D20220707T1701_59767_002409_0057_BEAM0.fits'],
    #                 bscrunch=4, tscrunch=8, pscrunch=True)

    ar.clean(fast=True)
    print(ar.snr_range())

    ar.DM_fit(ncore=16, plot=False)
