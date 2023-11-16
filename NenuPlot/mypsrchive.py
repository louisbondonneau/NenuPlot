#!/usr/bin/env python

# Tool to remove RFI from pulsar archives.
# Originally written by Patrick Lazarus. Modified by Lars Kuenkel.

import os
import numpy as np
import scipy.optimize
# from scipy import stats
# from statsmodels import robust
import psrchive as psr
import warnings

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, FK5
from astropy.time import Time
from datetime import datetime, timedelta

from log_class import Log_class
from configuration_class import Config_Reader
from methode_class import smoothGaussian

ARCHIVE_METHODE_OBJET = 'Archive_methode'

ARCHIVE_LOAD_OBJET = 'Archive_load'

PULSE_OBJET = 'Archive_on_pulse'

CLEANER_OBJET = 'Archive_cleaner'

ARCHIVE_GETTER_OBJET = 'Archive_getter'

# class psrchive_class(psr.Archive_load, psr.Archive)
CONFIG_FILE = os.path.dirname(os.path.realpath(__file__)) + '/' + 'NenuPlot.conf'


class psrchive_class(psr.Archive):
    def __init__(self, ar=None, ar_name=None, logname='cleaner', config_file=CONFIG_FILE, log_obj=None, verbose=True,
                 minfreq=None, maxfreq=None, mintime=None, maxtime=None, singlepulses_patch=False,
                 bscrunch=1, tscrunch=1, fscrunch=1, pscrunch=False, dm=None, rm=None, defaraday=True):
        self.verbos = verbose
        self.config_file = config_file
        if log_obj is None:
            self.log = Log_class(logname=logname, verbose=self.verbos)
        else:
            self.log = log_obj
        self.__init_clean_configuration()

        if(ar_name is not None):
            if isinstance(ar_name, list):
                self.name = self.MyArchive_load(ar_name, minfreq=minfreq, maxfreq=maxfreq, mintime=mintime, maxtime=maxtime,
                                                verbose=self.verbos, singlepulses_patch=singlepulses_patch,
                                                bscrunch=bscrunch, tscrunch=tscrunch, fscrunch=fscrunch, pscrunch=pscrunch, dm=dm, rm=rm, defaraday=defaraday)
            else:
                self.this = psr.Archive_load(ar_name)
                self.name = os.path.basename(ar_name).split('.')[0]
                self.set_freqs()
                self.set_times()
                self.set_onpulse(on_left=None, on_right=None, rebuild_local_scrunch=True)
        if(ar is not None):
            self.set_Archive(ar)

    def set_Archive(self, ar):
        self.this = ar
        self.name = ar.get_filename()
        self.set_freqs()
        self.set_times()
        self.set_onpulse(on_left=None, on_right=None, rebuild_local_scrunch=True)

    def __init_clean_configuration(self):
        nenuplot_config = Config_Reader(config_file=self.config_file, log_obj=self.log, verbose=False)
        self.cleaner_zapfirstsubint = nenuplot_config.get_config('CLEANER', 'zapfirstsubint')
        self.cleaner_fast = nenuplot_config.get_config('CLEANER', 'fast')
        self.cleaner_flat_cleaner = nenuplot_config.get_config('CLEANER', 'flat_cleaner')
        self.cleaner_chanthresh = nenuplot_config.get_config('CLEANER', 'chanthresh')
        self.cleaner_subintthresh = nenuplot_config.get_config('CLEANER', 'subintthresh')
        self.cleaner_first_chanthresh = nenuplot_config.get_config('CLEANER', 'first_chanthresh')
        self.cleaner_first_subintthresh = nenuplot_config.get_config('CLEANER', 'first_subintthresh')
        self.cleaner_bad_subint = nenuplot_config.get_config('CLEANER', 'bad_subint')
        self.cleaner_bad_chan = nenuplot_config.get_config('CLEANER', 'bad_chan')
        self.cleaner_max_iter = nenuplot_config.get_config('CLEANER', 'max_iter')

    def myclone(self):
        archive = psrchive_class(log_obj=self.log)
        archive.this = self.clone()
        archive.set_freqs()
        archive.set_times()
        return archive

    def mytscrunch(self, tscrunch_factor):
        self.tscrunch(np.round(tscrunch_factor))
        self.set_times()

    def myfscrunch(self, fscrunch_factor):
        self.fscrunch(np.round(fscrunch_factor))
        self.set_freqs()

    def mybscrunch(self, bscrunch_factor):
        self.bscrunch(np.round(bscrunch_factor))
        self.set_onpulse(on_left=None, on_right=None, rebuild_local_scrunch=True)

    def myfscrunch_to_nchan(self, nchan):
        factor = np.ceil(float(self.get_nchan()) / nchan)
        self.myfscrunch(int(factor))

    def mytscrunch_to_nsub(self, nsub):
        factor = np.ceil(float(self.get_nsubint()) / nsub)
        self.mytscrunch(int(factor))

    def mybscrunch_to_nbin(self, nbin):
        factor = np.ceil(float(self.get_nbin()) / nbin)
        self.mybscrunch(int(factor))

    def flattenBP(self):
        self.pscrunch()
        ar2 = self.myclone()
        # ar2.uniform_weight(new_weight=1.0)
        ar2.tscrunch()
        subint = ar2.get_Integration(0)
        (bl_mean, bl_var) = subint.baseline_stats()
        bl_mean = bl_mean.squeeze()
        bl_mean[np.where(bl_mean == 0)[0]] = np.nan
        # import matplotlib
        # import matplotlib.pyplot as plt
        # plt.plot(bl_mean); plt.show()
        self.remove_baseline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if (self.get_nchan() > 1):
                for isub in range(self.get_nsubint()):
                    for ichan in range(self.get_nchan()):
                        for ipol in range(self.get_npol()):
                            if(bl_mean[ichan] != 0):
                                prof = self.get_Profile(isub, ipol, ichan)
                                # prof.offset(-bl_mean[ichan])
                                if(ichan == 0):
                                    bl = np.nanmean([bl_mean[ichan], bl_mean[ichan + 1]])
                                elif(ichan == self.get_nchan() - 1):
                                    bl = np.nanmean([bl_mean[ichan - 1], bl_mean[ichan]])
                                else:
                                    bl = np.nanmedian([bl_mean[ichan - 1], bl_mean[ichan], bl_mean[ichan + 1]])
                                if (np.isnan(bl)):
                                    bl = np.nanmean(bl_mean)
                                prof.scale(1 / bl)

    def flatten_from_mad(self, rebuild_local_scrunch=True):
        if (rebuild_local_scrunch):
            self.local_scrunch()
            self.local_onpulse(rebuild_local_scrunch=False)
        # self.pscrunch()
        ar2 = self.myclone()
        ar2.tscrunch()
        data = ar2.get_data()
        mad_off = np.nanmedian(np.abs(data[0, 0, :, self.offbins] - np.nanmedian(data[0, 0, :, self.offbins], axis=0)), axis=0)
        self.remove_baseline()
        if (self.get_nchan() > 1):
            for isub in range(self.get_nsubint()):
                for ichan in range(self.get_nchan()):
                    for ipol in range(self.get_npol()):
                        if(mad_off[ichan] != 0):
                            prof = self.get_Profile(isub, ipol, ichan)
                            # prof.offset(-bl_mean[ichan])
                            prof.scale(1 / mad_off[ichan])

    def freq_cutter(self, minfreq=None, maxfreq=None):
        if (minfreq is not None):
            if(np.max(self.freqs) >= minfreq) and (np.min(self.freqs) <= minfreq):
                minchan = (np.argmax(self.freqs[np.where(self.freqs <= minfreq)]))
                if (minchan == 0):
                    minchan = None
            else:
                minchan = None
                if(minfreq > np.max(self.freqs)):
                    self.log.warning("minfreq %f MHz in freq_cutter is outside the frequency range %f-%f MHz" %
                                     (minfreq, np.min(self.freqs), np.max(self.freqs)), objet=ARCHIVE_METHODE_OBJET)
        else:
            minchan = None

        if (maxfreq is not None):
            if(np.max(self.freqs) >= maxfreq) and (np.min(self.freqs) <= maxfreq):
                maxchan = (np.argmax(self.freqs[np.where(self.freqs <= maxfreq)]))
            else:
                maxchan = None
                if(maxfreq < np.min(self.freqs)):
                    self.log.warning("maxfreq %f MHz in freq_cutter is outside the frequency range %f-%f MHz" %
                                     (maxfreq, np.min(self.freqs), np.max(self.freqs)), objet=ARCHIVE_METHODE_OBJET)
        else:
            maxchan = None

        if (maxchan is not None):
            if (maxchan < self.get_nchan() - 1):
                self.remove_chan(int(maxchan + 1), int(self.get_nchan() - 1))
        if (minchan is not None):
            if (minchan > 0):
                self.remove_chan(int(0), int(minchan - 1))
        self.set_freqs()
        return (minchan, maxchan)

    def time_cutter(self, mintime=None, maxtime=None):
        if (mintime is not None):
            if(isinstance(mintime, datetime)):
                mintime = Time(mintime)
            elif(isinstance(mintime, timedelta)):
                mintime = Time(self.times[0].to_datetime() + mintime)
            elif(isinstance(mintime, int)):
                mintime = Time(self.times[0].to_datetime() + timedelta(seconds=mintime * self.get_subint_duration()))
            if(isinstance(mintime, Time)):
                # mintime is a subintegration number
                if(mintime >= np.min(self.times)) and (mintime <= np.max(self.times)):
                    minsubint = np.argmax(self.times[np.where(self.times <= mintime)])
                else:
                    minsubint = None
                    if(mintime > np.max(self.times)):
                        self.log.warning("mintime %s in time_cutter is outside the time range %s - %s" %
                                         (mintime.isot, np.min(self.times).isot, np.max(self.times).isot), objet=ARCHIVE_METHODE_OBJET)
            else:
                self.log.warning("mintime in time_cutter is not a type astropy.time/datetime/timedela", objet=ARCHIVE_METHODE_OBJET)
                minsubint = None
        else:
            minsubint = None

        if (maxtime is not None):
            if(isinstance(maxtime, datetime)):
                maxtime = Time(maxtime)
            elif(isinstance(maxtime, timedelta)):
                maxtime = Time(self.times[0].to_datetime() + maxtime)
            elif(isinstance(maxtime, int)):
                maxtime = Time(self.times[0].to_datetime() + timedelta(seconds=maxtime * self.get_subint_duration()))

            if(isinstance(maxtime, Time)):
                # mintime is a subintegration number
                if(maxtime >= np.min(self.times)) and (maxtime <= np.max(self.times)):
                    maxsubint = np.argmax(self.times[np.where(self.times <= maxtime)])
                else:
                    maxsubint = None
                    if(maxtime < np.min(self.times)):
                        self.log.warning("maxtime %s in time_cutter is outside the time range %s - %s" %
                                         (maxtime.isot, np.min(self.times).isot, np.max(self.times).isot), objet=ARCHIVE_METHODE_OBJET)
            else:
                self.log.warning("maxtime in time_cutter is not a type astropy.time/datetime/timedela", objet=ARCHIVE_METHODE_OBJET)
                maxsubint = None
        else:
            maxsubint = None
        if(maxsubint is not None):
            for isub in range(self.get_nsubint() - 1, maxsubint - 1, -1):
                self.erase(isub)
        if(minsubint is not None):
            for isub in range(minsubint - 1, - 1, -1):
                self.erase(isub)
        self.set_times()
        return (minsubint, maxsubint)

    def remove_profile_inplace(self, pulse_region):
        """Remove the temnplate pulse from the individual profiles.
        """
        data = self.get_data()[:, 0, :, :]  # Select first polarization channel
        # archive is P-scrunched, so this is
        # total intensity, the only polarization
        # channel
        for isub, ichan in np.ndindex(self.get_nsubint(), self.get_nchan()):
            amps = self.__remove_profile1d(data[isub, ichan], pulse_region)
            prof = self.get_Profile(isub, 0, ichan)
            if amps is None:
                prof.set_weight(0)
            else:
                prof.get_amps()[:] = amps

    def __remove_profile1d(self, prof, pulse_region):
        if pulse_region != [0, 0, 1]:   # ('pulse_start', 'pulse_end', 'scaling_factor')
            p_start = int(pulse_region[1])
            p_end = int(pulse_region[2])
            nbins = len(prof)
            offbins = np.ones(nbins, dtype='bool')
            if(p_start < p_end):
                offbins[p_start: p_end] = False
                sizeon = p_end - p_start
            else:
                offbins[p_start:] = False
                offbins[: p_end] = False
                sizeon = nbins - (p_start - p_end)
            mad = np.std(prof[offbins])
            mean = np.mean(prof[offbins])
            prof[~offbins] = mad * np.random.standard_normal(sizeon) + mean
        return prof

    def remove_profile2d(self):
        ar_ref = self.myclone()
        ar_ref.tscrunch()
        ar_ref.remove_baseline()
        ref_amps = ar_ref.get_data()  # [0, ipol, ichan, :]
        amps = self.get_data()  # [isub, ipol, ichan, :]
        for ipol in range(self.get_npol()):
            for ichan in range(self.get_nchan()):
                ref_amps_tscunch = np.mean(ref_amps[:, ipol, ichan, :], axis=0)
                for isub in range(self.get_nsubint()):
                    prof = self.get_Profile(isub, ipol, ichan)

                    def res_func(amp):
                        res = amps[isub, ipol, ichan, :] - amp * ref_amps_tscunch
                        res[res < 0] = res[res < 0] * 4
                        return res
                    params, status = scipy.optimize.leastsq(res_func, [1.0])
                    prof.get_amps()[:] = np.asarray(amps[isub, ipol, ichan, :] - params * ref_amps_tscunch)

    def apply_weights(self, weights):
        """Apply the weigths to an array.
        """
        for (isub, ichan) in np.argwhere(weights == 0):
            integ = self.get_Integration(int(isub))
            integ.set_weight(int(ichan), 0.0)

    def __apply_weights_data(self, data, weights):
        """Apply the weigths to an array.
        """
        nsubs, nchans, nbins = data.shape
        for ibin in range(nbins):
            data[:, :, ibin] *= weights
            Xsub, Xchan = np.where(weights == 0)
            if(len(Xsub) != 0):
                data[Xsub, Xchan, ibin] = np.nan
            else:
                continue
        return data

    def apply_test_results(self, test_results):
        """Apply the weigths to an archive according to the test results.
        """
        nsub = self.get_nsubint()
        nchan = self.get_nchan()

        if (nsub > 1) and (nchan > 1):
            for (isub, ichan) in np.argwhere(np.isnan(test_results)):
                integ = self.get_Integration(int(isub))
                integ.set_weight(int(ichan), 0.0)
                test_results[isub, ichan] = 2
            for (isub, ichan) in np.argwhere(test_results >= 1):
                integ = self.get_Integration(int(isub))
                integ.set_weight(int(ichan), 0.0)
        elif (nsub > 1):
            for (isub) in np.argwhere(np.isnan(test_results)):
                integ = self.get_Integration(int(isub))
                integ.set_weight(int(0), 0.0)
                test_results[isub, ichan] = 2
            for (isub) in np.argwhere(test_results >= 1):
                integ = self.get_Integration(int(isub))
                integ.set_weight(int(0), 0.0)
        else:
            for (ichan) in np.argwhere(np.isnan(test_results)):
                integ = self.get_Integration(0)
                integ.set_weight(int(ichan), 0.0)
                test_results[isub, ichan] = 2
            for (ichan) in np.argwhere(test_results >= 1):
                integ = self.get_Integration(0)
                integ.set_weight(int(ichan), 0.0)

    def local_scrunch(self):
        self.scrunch = self.myclone()
        self.scrunch.pscrunch()
        self.scrunch.remove_baseline()
        self.scrunch.dedisperse()
        self.scrunch.tscrunch()
        self.scrunch.fscrunch()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -------------------------------- ON pulse -----------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def local_onpulse(self, rebuild_local_scrunch=False):
        self.on_left, self.on_right = self.get_on_window(rebuild_local_scrunch=rebuild_local_scrunch)
        self.local_offbins(on_left=self.on_left, on_right=self.on_right)

    def set_onpulse(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        if (on_left is None) or (on_right is None):
            self.local_onpulse(rebuild_local_scrunch=rebuild_local_scrunch)
        else:
            self.on_left, self.on_right = on_left, on_right
        self.local_offbins(on_left=self.on_left, on_right=self.on_right)

    def get_onpulse(self):
        return (self.on_left, self.on_right)

    def local_offbins(self, on_left=None, on_right=None):
        self.offbins = np.ones(self.get_nbin(), dtype='bool')
        if(on_left < on_right):
            self.offbins[on_left: on_right] = False
        else:
            self.offbins[on_left:] = False
            self.offbins[: on_right] = False
        self.onbins = ~self.offbins

    def center(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        pass

    def get_on_window(self, safe_fraction=1 / 8., rebuild_local_scrunch=True):
        if (rebuild_local_scrunch):
            self.local_scrunch()
        try:
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        except AttributeError:
            self.local_scrunch()
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        except NameError:
            self.local_scrunch()
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000

        # find first the bin with maximum value
        maxbin = np.argmax(prof)
        nbins = len(prof)
        # exclude the area of 60% of all bins around the maxbin
        # make the 60%-area the even number
        exclsize = int(nbins * 0.6) - (int(nbins * 0.6) % 2)
        le = maxbin - exclsize / 2
        re = maxbin + exclsize / 2
        # print('0', le, re)

        p_start = le % nbins
        p_end = re % nbins
        offbins = np.ones(nbins, dtype='bool')
        if(p_start < p_end):
            offbins[int(p_start): int(p_end)] = False
        else:
            offbins[int(p_start):] = False
            offbins[: int(p_end)] = False
        # extra rotation by "le" bins, so left edge will be at 0
        # prof = bestprof_rotate(prof, le)
        # total rotation in phase
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            amean = np.mean(prof[offbins])
            arms = np.std(prof[offbins])
            aprof = (prof - amean) / arms
            abins = np.arange(0, nbins)[(aprof > 2.5)]
            abins = self.__trim_bins(abins)  # trimming bins

        # updating pulse windo
        # print("len(prof):", len(prof))
        # print("maxbin:", maxbin)
        # print("exclsize:", exclsize)
        # print("ON LE RE:", le, re)
        # print("ON PULSE ZERO:", p_start, p_end)
        # import matplotlib.pyplot as plt
        # print("amean:", amean)
        # print("arms:", arms)
        # plt.plot((prof - amean) / arms)
        # plt.plot(abins)

        # updating pulse windo
        # to be extra-cautious, ONpulse have to be largeur than 15% of the pulse window
        # to be extra-cautious, OFFpulse have to be largeur than 15% of the pulse window
        if(np.size(abins) > 2):
            dabins = (abins - np.roll(abins, 1)) % nbins
            le = abins[np.argmax(dabins)] % nbins
            re = abins[np.argmax(dabins) - 1] % nbins
        else:
            le = maxbin - 1
            re = maxbin + 1

        if(nbins * safe_fraction < 5) and (safe_fraction <= 1 / 4):
            safe_fraction = 5 / nbins

        # print('8', le, re)
        if(le < re):
            onpulse = (re - le) / float(nbins)
            offpulse = 1 - onpulse
            if(onpulse < safe_fraction):
                extrabin = ((safe_fraction - onpulse) / 2.) * nbins
                # print("on0 too small extrabin=", extrabin)
                re = re + int(extrabin)
                le = le - int(extrabin)
                # print(le, re)
            if(offpulse < safe_fraction):
                extrabin = ((safe_fraction - offpulse) / 2.) * nbins
                # print("off0 too small extrabin=", extrabin)
                re = re - int(extrabin)
                le = le + int(extrabin)
                # print(le, re)
        else:  # (le > re)
            onpulse = (nbins - (le - re)) / float(nbins)
            offpulse = 1 - onpulse
            if(onpulse < safe_fraction):
                extrabin = ((safe_fraction - onpulse) / 2.) * nbins
                # print("on1 too small extrabin=", extrabin)
                re = re + int(extrabin)
                le = le - int(extrabin)
                # print(le, re)
            if(offpulse < safe_fraction):
                extrabin = ((safe_fraction - offpulse) / 2.) * nbins
                # print("off1 too small extrabin=", extrabin)
                re = re - int(extrabin)
                le = le + int(extrabin)
                # print(le, re)
        # print('9', le, re)
        le = le % nbins
        re = re % nbins
        self.set_onpulse(on_left=le, on_right=re, rebuild_local_scrunch=False)
        # print('10', le, re)
        # plt.show()
        return le, re

    # exclude single bins representating 1-bin outliers
    def __trim_bins(self, x):
        x_diffs = [x[ii] - x[ii - 1] for ii in range(1, len(x))]
        # trim left side
        cut_bin = 0
        for ii in range(0, int(len(x_diffs) / 2)):
            if x_diffs[ii] == 1:
                if cut_bin != 0:
                    x = x[cut_bin:]
                break
            else:
                cut_bin += 1
        # trim right side
        cut_bin = 0
        for ii in range(len(x_diffs) - 1, int(len(x_diffs) / 2), -1):
            if x_diffs[ii] == 1:
                if cut_bin != 0:
                    x = x[:-cut_bin]
                break
            else:
                cut_bin += 1
        # trim in the middle
        x_diffs = [x[ii] - x[ii - 1] for ii in range(1, len(x))]
        ii_to_trim = []
        prev = 1
        for ii in range(0, len(x_diffs)):
            if x_diffs[ii] != 1 and prev == 1:
                prev = x_diffs[ii]
            elif x_diffs[ii] != 1 and prev != 1:
                ii_to_trim.append(ii)
                prev = x_diffs[ii]
            else:
                prev = 1
        x = np.delete(x, ii_to_trim, axis=0)
        x_diffs = [x[ii] - x[ii - 1] for ii in range(1, len(x))]
        return x

    def auto_rebin(self, ichan=None, rebuild_local_scrunch=True):
        if (rebuild_local_scrunch):
            self.local_scrunch()
        if (ichan is None):
            arx = self.scrunch.myclone()
            ichan = 0
        else:
            arx = self.myclone()
            arx.dedisperse()
            arx.pscrunch()
            arx.tscrunch()
        arx.remove_baseline()
        arx.get_on_window(safe_fraction=3 / 100., rebuild_local_scrunch=False)

        prof = np.asarray(arx.get_Profile(0, 0, ichan).get_amps() * 10000)
        nbin = arx.get_nbin()
        prof_smooth = smoothGaussian(prof)

        std = np.std(prof[arx.offbins])
        prof_deriv = np.abs(prof_smooth[arx.onbins] - np.roll(prof_smooth[arx.onbins], 1))
        prof_deriv[0] = 0
        delta = np.max(prof_deriv)
        # self.log.log('delta = %d' % (delta), objet=PULSE_OBJET)
        # self.log.log('2.9 * std = %d' % (3.5 * std), objet=PULSE_OBJET)
        # import matplotlib.pyplot as plt
        # ax0 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1, frameon=True)
        # ax0.plot(prof_smooth)
        # ax0.plot(prof)
        # ax0.plot(prof_smooth[arx.onbins])
        # ax0.plot(prof[arx.onbins])
        # ax0.plot(prof_deriv)
        # plt.show()

        bscrunch_factor = 1
        # print(nbin, delta, 2.5*std)
        while (nbin > 32) and (delta < 2.9 * std):
            arx.bscrunch(2)
            nbin = arx.get_nbin()
            arx.get_on_window(safe_fraction=7 / 100., rebuild_local_scrunch=True)
            prof = np.asarray(arx.get_Profile(0, 0, ichan).get_amps() * 10000)
            prof_smooth = smoothGaussian(prof)
            std = np.std(prof[arx.offbins])
            prof_deriv = np.abs(prof_smooth[arx.onbins] - np.roll(prof_smooth[arx.onbins], 1))
            prof_deriv[0] = 0
            delta = np.max(prof_deriv)
            bscrunch_factor *= 2
            # print(nbin, delta, 2.9 * std)
            # self.log.log('delta = %d' % (delta), objet=PULSE_OBJET)
            # self.log.log('2.9 * std = %d' % (3.5 * std), objet=PULSE_OBJET)
            # ax0.plot(prof_smooth[arx.onbins])
            # ax0.plot(prof_deriv)
            # plt.show()
        bscrunch_factor /= 2
        if (bscrunch_factor > 1):
            self.mybscrunch(bscrunch_factor)
            self.log.log('automatic rebin by factor: ' + str(bscrunch_factor), objet=PULSE_OBJET)
        else:
            self.log.log('No need of rebin', objet=PULSE_OBJET)
        return bscrunch_factor

    def snr_range(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        if (rebuild_local_scrunch):
            self.local_scrunch()
        try:
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        except NameError:
            self.local_scrunch()
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000

        if on_left is None and on_right is None:
            if (rebuild_local_scrunch):
                self.local_onpulse()
        else:
            self.set_onpulse(on_left=on_left, on_right=on_right)

        off_prof = prof[self.offbins]
        range_mean = np.nanmean(off_prof)
        range_rms = np.nanstd(off_prof)
        if(range_rms == 0):
            self.log.error('ERROR401 This is the Murphy law (if this append there is probably a problem with the profile or the onpulse  calculation)', objet=PULSE_OBJET)
            self.log.error('ERROR401 off_prof = %s' % str(off_prof), objet=PULSE_OBJET)
            self.log.error('ERROR401 on_left = %d on_right = %d' % (on_left, on_right), objet=PULSE_OBJET)
            return (0, 0)
        if(np.isnan(range_rms)):
            self.log.error('ERROR402 This is the Murphy law (if this append there is probably a problem with the profile or the onpulse  calculation)', objet=PULSE_OBJET)
            self.log.error('ERROR402 off_prof = %s' % str(off_prof), objet=PULSE_OBJET)
            self.log.error('ERROR402 on_left = %d on_right = %d' % (on_left, on_right), objet=PULSE_OBJET)
            return (0, 0)
        range_prof = (prof - range_mean) / range_rms
        range_snrpeak = np.nanmax(range_prof)
        range_weq = np.abs(np.nansum(range_prof) / range_snrpeak)
        # print(np.nansum(range_prof), range_weq)
        if(range_weq == 0):
            return (0, 0)
        range_profsign = np.nansum(range_prof[self.onbins]) / np.sqrt(np.abs(range_weq))
        range_profsign_err = 5 / np.sqrt(np.abs(range_weq))
        return (range_profsign, range_profsign_err)

    def snr_norm(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        range_profsign, range_profsign_err = self.snr_range(on_left=on_left, on_right=on_right, rebuild_local_scrunch=rebuild_local_scrunch)
        elev, az = self.get_altaz()
        weights = self.get_weights()
        weights /= np.nanmax(weights)
        RFI = 100. * (1. - np.mean(weights))
        SNR_norm = range_profsign * np.sqrt(3600. / self.integration_length())
        SNR_norm *= (np.sin(90. * np.pi / 180.) / np.sin(np.mean(elev) * np.pi / 180.))**2 / ((100 - float(RFI)) / 100.)
        return SNR_norm

    def snr_psrstat(self, rebuild_local_scrunch=True):
        if (rebuild_local_scrunch):
            self.local_scrunch()
        try:
            snr_psrstat = self.scrunch.get_Profile(0, 0, 0).snr()
        except NameError:
            self.local_scrunch()
            snr_psrstat = self.scrunch.get_Profile(0, 0, 0).snr()
        return snr_psrstat

    def snr_peak(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        if (rebuild_local_scrunch):
            self.local_scrunch()

        if (on_left is None) or (on_right is None):
            self.local_onpulse()
        else:
            self.set_onpulse(on_left=on_left, on_right=on_right)

        try:
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        except NameError:
            self.local_scrunch()
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        off_data = prof[self.offbins]
        on_data = prof[self.onbins]
        range_mean = np.median(off_data)
        range_rms = 1.48 * np.nanmedian(np.abs(off_data - np.nanmedian(off_data, axis=0)), axis=0)
        range_prof = (on_data - range_mean) / range_rms
        range_snrpeak = np.max(range_prof)
        return range_snrpeak

    def flux_peak(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        if (rebuild_local_scrunch):
            self.local_scrunch()

        if (on_left is None) or (on_right is None):
            self.local_onpulse()
        else:
            self.set_onpulse(on_left=on_left, on_right=on_right)

        try:
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        except NameError:
            self.local_scrunch()
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        off_data = prof[self.offbins]
        range_mean = np.median(off_data)
        flux_peak = np.max(prof) - range_mean
        return flux_peak

    def sharpness(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        if (rebuild_local_scrunch):
            self.local_scrunch()

        if (on_left is None) or (on_right is None):
            self.local_onpulse()
        else:
            self.set_onpulse(on_left=on_left, on_right=on_right)

        try:
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        except NameError:
            self.local_scrunch()
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        prof_var = np.abs(prof - np.roll(prof, 1))
        sharpness = np.nanmean(prof_var[self.onbins])
        return sharpness

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -------------------------------- CLEANER ------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def clean(self, zapfirstsubint=None, fast=None, flat_cleaner=None,
              first_chanthresh=None, first_subintthresh=None,
              chanthresh=None, subintthresh=None,
              bad_subint=None, bad_chan=None,
              max_iter=None):
        if (zapfirstsubint is not None):
            self.cleaner_zapfirstsubint = zapfirstsubint

        if (fast is not None):
            self.cleaner_fast = fast

        if (flat_cleaner is not None):
            self.cleaner_flat_cleaner = flat_cleaner

        if (chanthresh is not None):
            self.cleaner_chanthresh = chanthresh

        if (subintthresh is not None):
            self.cleaner_subintthresh = subintthresh

        if (first_chanthresh is not None):
            self.cleaner_first_chanthresh = first_chanthresh

        if (first_subintthresh is not None):
            self.cleaner_first_subintthresh = first_subintthresh

        if (bad_subint is not None):
            self.cleaner_bad_subint = bad_subint

        if (bad_chan is not None):
            self.cleaner_bad_chan = bad_chan

        if (max_iter is not None):
            self.cleaner_max_iter = max_iter

        if (self.cleaner_fast):
            self.cleaner_max_iter = 3

        self.stop_flag = False
        orig_weights = self.get_weights()

        # Mask the first subintegration if required
        if (self.cleaner_zapfirstsubint) and (self.get_nsubint() > 2):
            for ichan in range(self.get_nchan()):
                prof = self.get_Profile(0, 0, ichan)
                prof.set_weight(0.0)

        patient_zero = self.myclone()
        patient_zero.pscrunch()
        # ar_name = self.get_filename().split()[-1]
        Niter = 0
        pulse_region = [0, 0, 1]  # metavar=('pulse_start', 'pulse_end', 'scaling_factor'), help="Defines the range of the pulse and a suppression factor.")
        pulse_region_old = [0, 0, 1]
        static_pulse_cont = 0

        # Create list that is used to end the iteration
        weights_liste = []
        weights_liste.append(patient_zero.get_weights())
        patient = patient_zero.myclone()
        while (Niter < self.cleaner_max_iter) and not self.stop_flag:
            Niter += 1
            # Prepare the data for template creation-
            pulse_region[1], pulse_region[2] = patient.get_on_window(rebuild_local_scrunch=True)
            SNR, SNRerr = patient.snr_range(pulse_region[1], pulse_region[2])
            self.log.log("ON pulse window is %d-%d for S = %.1f +- %.4f AU" % (pulse_region[1], pulse_region[2], SNR, SNRerr), objet=CLEANER_OBJET)
            self.log.log("Loop: %s" % Niter, objet=CLEANER_OBJET)

            if (Niter > 1):
                if (pulse_region[1] == pulse_region_old[1]) and (pulse_region[2] == pulse_region_old[2]):
                    static_pulse_cont += 1
                    if(static_pulse_cont > 2):
                        self.log.log("Cleaning was stop at loop %s after a constant on_pulse" % Niter, objet=CLEANER_OBJET)
                        self.stop_flag = True
                        continue
                else:
                    static_pulse_cont = 0
            pulse_region_old = np.copy(pulse_region)
            # plt.plot(template)
            # plt.show()

            # Reset patient
            patient = patient_zero.myclone()
            patient.pscrunch()  # should be useless because patient_zero is already pscrunch
            if (Niter > 2):
                patient.apply_weights(new_weights)
            if (self.cleaner_flat_cleaner):
                patient.flattenBP()  # pscrunch and remove_baseline inside
            else:
                patient.remove_baseline()
            patient.dedisperse()
            patient.remove_profile_inplace(pulse_region)

            # re-set DM to 0
            patient.dededisperse()

            # Get data (select first polarization - recall we already P-scrunched)
            data = patient.get_data()[:, 0, :, :]
            if (Niter <= 2):  # template is refined
                data = self.__apply_weights_data(data, orig_weights)
                curent_weights = orig_weights
            else:  # mask is refined using the new mask
                data = self.__apply_weights_data(data, new_weights)
                curent_weights = new_weights

            # RFI-ectomy must be recommended by average of tests
            avg_test_results = self.__comprehensive_stats(data, curent_weights, Niter, fast=self.cleaner_fast)

            # Reset patient and set weights in patient
            patient = patient_zero.myclone()
            patient.apply_test_results(avg_test_results)

            # find bad part
            patient.clean_bad_parts(bad_subint=self.cleaner_bad_subint, bad_chan=self.cleaner_bad_chan)

            # Test whether weigths were already used in a previous iteration
            new_weights = patient.get_weights()

            diff_frac = float(np.sum(new_weights != weights_liste[-1])) / np.size(new_weights)
            if(np.nanmax(new_weights) == 0):
                rfi_frac = 100.
            else:
                rfi_frac = 1 - (np.nanmean(new_weights) / np.nanmax(new_weights))
            self.log.log("RFI fraction is %.2f percent diff = %.2f" % (rfi_frac * 100, diff_frac * 100), objet=CLEANER_OBJET)
            # Print the changes to the previous loop to help in choosing a suitable max_iter
            if np.all(new_weights == weights_liste[-1]) and (Niter > 2):
                self.log.log("Cleaning was stop at loop %s after a constant interation" % Niter, objet=CLEANER_OBJET)
                self.stop_flag = True
            elif (rfi_frac * 100. > 85.0):
                self.log.warning("WARNING: Cleaning was force to stop at loop %s after %f percent of cleaning" %
                                 (Niter, 100. * rfi_frac), objet=CLEANER_OBJET)
                self.stop_flag = True
            elif (diff_frac * 100. <= 0.1) and (Niter > 2):
                self.stop_flag = True
                self.log.log("Cleaning was stop at loop %s after a diff of %.2f percent btw tow iteration" % (Niter, diff_frac * 100.), objet=CLEANER_OBJET)
            weights_liste.append(new_weights)

        self.log.log("Cleaning was stop after %s loops" % Niter, objet=CLEANER_OBJET)
        # Set weights in archive.
        self.apply_weights(new_weights)

    def clean_bad_parts(self, bad_subint=1, bad_chan=1, quiet=True):
        """Checks whether whole channels or subints should be removed
        """
        weights = self.get_weights()
        frac_max = np.nanmax(weights)
        frac_mean = np.nanmean(weights)
        frac_bad_tot = (frac_max - frac_mean) / frac_max

        n_subints = self.get_nsubint()
        n_channels = self.get_nchan()
        n_bad_channels = 0
        n_bad_subints = 0

        if (frac_bad_tot < bad_subint):
            for i in range(n_subints):
                bad_frac = 1 - np.count_nonzero(weights[i, :]) / float(n_channels)
                if bad_frac > bad_subint:
                    for j in range(n_channels):
                        integ = self.get_Integration(int(i))
                        integ.set_weight(int(j), 0.0)
                    n_bad_subints += 1

        if (frac_bad_tot < bad_chan):
            for j in range(n_channels):
                bad_frac = 1 - np.count_nonzero(weights[:, j]) / float(n_subints)
                if bad_frac > bad_chan:
                    for i in range(n_subints):
                        integ = self.get_Integration(int(i))
                        integ.set_weight(int(j), 0.0)
                    n_bad_channels += 1

        if not quiet and n_bad_channels + n_bad_subints != 0:
            self.log.log("Removed %s bad subintegrations and %s bad channels." % (n_bad_subints, n_bad_channels), objet=CLEANER_OBJET)

    def __comprehensive_stats(self, data, weights, x, fast=False):
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

        if (x == 1):
            chanthresh = self.cleaner_first_chanthresh
            subintthresh = self.cleaner_first_subintthresh
        else:
            chanthresh = self.cleaner_chanthresh
            subintthresh = self.cleaner_subintthresh

        self.log.log("chanthresh = %.1f  subintthresh = %.1f" % (chanthresh, subintthresh), objet=CLEANER_OBJET)

        nsubs, nchans, nbins = data.shape

        # remaining nchan and nsubs after applying weight.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data_2d = np.nanmean(data, axis=2)
            bad_perchan = np.nanmean(data_2d, axis=0)
            bad_persubs = np.nanmean(data_2d, axis=1)
            nsubs_remain = np.sum(~(bad_persubs == 0))
            nchans_remai = np.sum(~(bad_perchan == 0))

        def nanptp(data, axis=0):
            return (np.nanmax(data, axis=axis) - np.nanmin(data, axis=axis))

        diagnostic_functions = [
            np.nanstd,
            np.nanmean,
            nanptp
            # lambda data, axis: np.max(np.abs(np.fft.rfft(
            #    data - np.expand_dims(data.mean(axis=2), axis=2),
            #    axis=2)), axis=2)
        ]
        # Compute diagnostics
        diagnostics = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for func in diagnostic_functions:
                diagnostics.append(func(data, axis=2))

            # Now step through data and identify bad profiles
            scaled_diagnostics = []
            for diag in diagnostics:
                if(nsubs_remain > 1) and (nchans_remai > 1):
                    chan_scaled = np.abs(self.__channel_scaler(diag)) / chanthresh
                    subint_scaled = np.abs(self.__subint_scaler(diag)) / subintthresh
                    chan_scaled[np.where(np.isnan(chan_scaled))] = 2.0
                    subint_scaled[np.where(np.isnan(subint_scaled))] = 2.0
                    scaled_diagnostics.append(np.max((chan_scaled, subint_scaled), axis=0))
                elif(nsubs_remain > 1):
                    chan_scaled = np.abs(self.__channel_scaler(diag)) / chanthresh
                    chan_scaled[np.where(np.isnan(chan_scaled))] = 2.0
                    scaled_diagnostics.append(np.max(chan_scaled, axis=0))
                else:
                    subint_scaled = np.abs(self.__subint_scaler(diag)) / subintthresh
                    subint_scaled[np.where(np.isnan(subint_scaled))] = 2.0
                    scaled_diagnostics.append(np.max(subint_scaled, axis=0))

            test_results = np.mean(scaled_diagnostics, axis=0)
        return test_results

    def __channel_scaler(self, array2d):
        """For each channel scale it.
        """
        scaled = np.empty_like(array2d)
        nchans = array2d.shape[1]
        for ichan in np.arange(nchans):
            with np.errstate(invalid='ignore', divide='ignore'):
                channel = array2d[:, ichan]
                median = np.nanmedian(channel)
                channel_rescaled = channel - median
                mad = np.nanmedian(np.abs(channel_rescaled))
                scaled[:, ichan] = (channel_rescaled) / (mad)
        return scaled

    def __subint_scaler(self, array2d):
        """For each sub-int scale it.
        """
        scaled = np.empty_like(array2d)
        nsubs = array2d.shape[0]
        for isub in np.arange(nsubs):
            with np.errstate(invalid='ignore', divide='ignore'):
                subint = array2d[isub, :]
                median = np.nanmedian(subint)
                subint_rescaled = subint - median
                mad = np.nanmedian(np.abs(subint_rescaled))
                scaled[isub, :] = (subint_rescaled) / (mad)
        return scaled

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # ------------------------------ Archive_load ---------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def MyArchive_load(self, path,
                       minfreq=None, maxfreq=None, mintime=None, maxtime=None,
                       verbose=True, singlepulses_patch=False,
                       bscrunch=1, tscrunch=1, fscrunch=1, pscrunch=False, dm=None, rm=None, defaraday=False):
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
        directory_list = []
        for file in path:
            if(os.path.dirname(file) == ''):
                directory_list.append('./')
            else:
                directory_list.append(os.path.dirname(file) + '/')
            file = os.path.basename(file)
            files.append(file)
        files.sort()
        self.log.log("file(s) used: %s" % (files), objet=ARCHIVE_LOAD_OBJET)
        # archives = [self.Archive_load(path + files[0])]

        if(verbose):
            self.log.log('======================================================================================================', objet=ARCHIVE_LOAD_OBJET)
            self.log.log('                                     File(s) to be processed:                                         ', objet=ARCHIVE_LOAD_OBJET)
            self.log.log('======================================================================================================', objet=ARCHIVE_LOAD_OBJET)

        first = True
        for i in range(len(files)):
            # archives = psr.Archive_load(directory_list[i] + files[i])
            archives = psrchive_class(ar_name=directory_list[i] + files[i], log_obj=self.log)
            buffarchive = archives.myclone()

            # ignore file if it is outside the requesed freq range
            if minfreq is not None:
                if (np.max(buffarchive.freqs) < minfreq):
                    continue
            if maxfreq is not None:
                if (np.min(buffarchive.freqs) > maxfreq):
                    continue

            minchan, maxchan = buffarchive.freq_cutter(minfreq=minfreq, maxfreq=maxfreq)

            if (pscrunch):
                buffarchive.pscrunch()
            if(buffarchive.get_npol() == 1):
                defaraday = False
            if (dm is not None):
                buffarchive.set_dispersion_measure(float(dm))
                if (defaraday):
                    buffarchive.dedisperse()
            # else:
            #    buffarchive.dedisperse()
            if (rm is not None):
                buffarchive.set_rotation_measure(float(rm))
                if (defaraday):
                    buffarchive.defaraday()
            else:
                # if not (buffarchive.get_rotation_measure() == 0.0) and (buffarchive.get_npol() > 1):
                if (buffarchive.get_npol() > 1):
                    if (defaraday):
                        print("defaraday ICCCIIII")
                        buffarchive.defaraday()
            if (tscrunch > 1):
                buffarchive.tscrunch(tscrunch)
            if (bscrunch > 1):
                if(buffarchive.get_nbin() / bscrunch < 8):
                    bscrunch = buffarchive.get_nbin() / 8
                if (bscrunch > 1):
                    buffarchive.bscrunch(bscrunch)
            if (fscrunch > 1):
                buffarchive.fscrunch(fscrunch)
            string = ''
            if (minchan is not None):
                string = string + ("minfreq = %.8f " % buffarchive.freqs[0])
            if (maxchan is not None):
                string = string + ("maxfreq = %.8f " % buffarchive.freqs[-1])
            if (pscrunch):
                string = string + ("pscrunch = True ")
            if (dm is not None):
                string = string + ("dm = %6f " % float(dm))
            if (rm is not None):
                string = string + ("rm = %6f " % float(rm))
            if (bscrunch > 1):
                string = string + ("bscrunch = %d " % bscrunch)
            if (tscrunch > 1):
                string = string + ("tscrunch = %d " % tscrunch)
            if (fscrunch > 1):
                string = string + ("fscrunch = %d " % fscrunch)
            if (first):
                self.this = buffarchive.clone()
                self.set_freqs()
                self.set_times()
                # self.append(buffarchive)
                first = False
            else:
                newarchive_freq = self.get_Profile(0, 0, 0).get_centre_frequency()
                buffarchive_freq = buffarchive.get_Profile(0, 0, 0).get_centre_frequency()
                if (np.abs(newarchive_freq - buffarchive_freq) > 1):  # diff center of freq
                    # if (freqappend):
                    freqappend = psr.FrequencyAppend()
                    patch = psr.PatchTime()
                    # This is needed for single-pulse data:
                    if (singlepulses_patch):
                        patch.set_contemporaneity_policy("phase")
                    freqappend.init(self)
                    freqappend.ignore_phase = True
                    polycos = self.get_model()
                    buffarchive.set_model(polycos)
                    patch.operate(self, buffarchive)
                    freqappend.append(self, buffarchive)
                    self.update_centre_frequency()
                    self.set_model(polycos)
                    self.set_freqs()
                else:
                    # polycos = self.get_model()
                    # buffarchive.set_model(polycos)
                    # self.append(buffarchive)
                    timeappend = psr.TimeAppend()
                    timeappend.ignore_phase = True
                    timeappend.append(self, buffarchive)
                    self.set_times()
            if not string == '':
                self.log.log(string, objet=ARCHIVE_LOAD_OBJET)

            if verbose:
                self.log.log(directory_list[i] + files[i], objet=ARCHIVE_LOAD_OBJET)
        minsub, maxsub = self.time_cutter(mintime=mintime, maxtime=maxtime)
        if(minsub is not None):
            self.log.log("minsub   = %d" % minsub, objet=ARCHIVE_LOAD_OBJET)
        if(maxsub is not None):
            self.log.log("maxsub   = %d" % maxsub, objet=ARCHIVE_LOAD_OBJET)

        return files[0].split('.')[0]

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # ----------------------------- Mask_managment --------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def get_mask(self):
        return self.get_weights()

    def apply_mask(self, mask):
        if(isinstance(mask, str)):
            # TODO verif exist
            mask = np.genfromtxt(mask)
        if (isinstance(mask, type(np.array(1)))):
            # TODO verif les dim
            for isub in range(self.get_nsubint()):
                for ipol in range(self.get_npol()):
                    for ichan in range(self.get_nchan()):
                        prof = self.get_Profile(isub, ipol, ichan)
                        prof.set_weight(mask[isub, ichan])

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # ---------------------------------- getter -----------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def get_EarthLocation(self):
        try:
            x, y, z = self.get_ant_xyz()
        except AttributeError:
            self.log.warning('WARNING: archive.get_ant_xyz() faild will used nancay location', objet=ARCHIVE_GETTER_OBJET)
            x = 4324016.70769
            y = 165545.525467
            z = 4670271.363
        return EarthLocation.from_geocentric(x=float(x) * u.m, y=float(y) * u.m, z=float(z) * u.m)

    def get_SkyCoord(self, equinox='J2000'):
        try:
            if self.get_ephemeris().get_value("RAJ") != '':
                psr_raj = psr.Angle()
                psr_raj.setHMS(self.get_ephemeris().get_value("RAJ"))
                psr_decj = psr.Angle()
                psr_decj.setDMS(self.get_ephemeris().get_value("DECJ"))
                psr_coor = SkyCoord(ra=psr_raj.getDegrees() * u.degree, dec=psr_decj.getDegrees() * u.degree, frame=FK5(equinox='J2000'))
            elif(self.get_ephemeris().get_value("ELONG") != ''):
                elong = float(self.get_ephemeris().get_value("ELONG").replace("D", "e"))
                elat = float(self.get_ephemeris().get_value("ELAT").replace("D", "e"))
                psr_coor = SkyCoord(lon=elong * u.degree, lat=elat * u.degree, frame='geocentrictrueecliptic')
                psr_coor = psr_coor.transform_to(FK5(equinox='J2000'))
            else:
                print("ERROR can not retrive RAJ/DECJ or ELONG/ELAT")
                exit(0)

        except RuntimeError:
            from astropy.io import fits
            data = fits.getdata(self.get_filename(), do_not_scale_image_data=True, scale_back=True)
            header = fits.getheader(self.get_filename(), do_not_scale_image_data=True, scale_back=True)
            if (data.columns[4].name == 'RAJ'):
                psr_raj = psr.Angle()
                psr_raj.setHMS(header["RA"])
                psr_decj = psr.Angle()
                psr_decj.setDMS(header["DEC"])
                psr_coor = SkyCoord(ra=psr_raj.getDegrees() * u.degree, dec=psr_decj.getDegrees() * u.degree, frame=FK5(equinox='J2000'))
            elif (data.columns[4].name == 'ELAT'):
                elat = float(data.field(4)[0])
                elong = float(data.field(5)[0])
                psr_coor = SkyCoord(lon=elong * u.degree, lat=elat * u.degree, frame='geocentrictrueecliptic')
                psr_coor = psr_coor.transform_to(FK5(equinox='J2000'))

        return psr_coor

    def get_ra_dec(self, deg=False, equinox='J2000'):
        psr_coor = self.get_SkyCoord(equinox=equinox)
        # print(PSR.to_string('hmsdms'))
        if (deg is True):
            return psr_coor.ra.degree, psr_coor.dec.degree
        else:
            return str(psr_coor.ra.to_string(u.hour, sep=':')), str(psr_coor.dec.to_string(u.degree, sep=':'))

    def get_mjd(self):
        mjd = []
        for isub in range(self.get_nsubint()):
            int_mjd = self.get_Integration(isub).get_epoch().intday()
            int_sec = self.get_Integration(isub).get_epoch().get_secs()
            int_fracsec = self.get_Integration(isub).get_epoch().get_fracsec()
            mjd.append(np.float64(int_mjd) + (np.float64(int_sec) + np.float64(int_fracsec)) / 86400.)
            # if(isub > 0):
            #     print((mjd[isub] - mjd[isub - 1]) * 24. * 3600., "%.8f" % self.get_Integration(isub).get_duration())
            #  This should be exactely 10.73741824 sec .... but it's not
        return Time(np.array(mjd), format='mjd')

    def get_times(self):
        return self.get_mjd()

    def get_altaz(self):
        """
        Fonction to give elevation in the first and last subintegration
        EarthLocation is set for Nancay

        Input:
            archives : list of PSRCHIVE archive objects
        Output
            float(elev0),float(elevlast)
        """
        Site = self.get_EarthLocation()

        c = self.get_SkyCoord()

        c_altaz = c.transform_to(AltAz(obstime=self.times, location=Site))

        elevation = c_altaz.alt.degree
        azimut = c_altaz.az.degree
        return (elevation, azimut)

    def get_subint_duration(self, subint=None):
        if(subint is None):
            subint_duration = []
            for isub in range(0, self.get_nsubint()):
                subint_duration.append(self.get_Integration(isub).get_duration())
            subint_duration = np.median(subint_duration)
        else:
            subint_duration = self.get_Integration(subint).get_duration()
        return subint_duration

    def get_doppler(self, subint=None):
        if(subint is None):
            doppler = []
            for isub in range(0, self.get_nsubint()):
                doppler.append(np.sqrt(self.get_Integration(isub).get_doppler_factor()))  # doppler comparatively to the frequency Freq * doppler = dop_freq
            doppler = np.median(doppler)
        else:
            doppler = np.sqrt(self.get_Integration(subint).get_doppler_factor())
        return doppler

    def get_freqs(self, doppler=False):
        freq = []
        for ichan in range(self.get_nchan()):
            freq.append(self.get_Profile(0, 0, ichan).get_centre_frequency())
        if (doppler):
            freq = np.array(freq) * self.get_doppler()
        return np.array(freq)

    def get_chan_bw(self):
        mean_chan_bw = float(self.get_bandwidth()) / self.get_nchan()
        chan_bw = []
        for ichan in range(self.get_nchan()):
            chan_bw.append(mean_chan_bw)
        return chan_bw

    def get_period(self):
        return self.get_Integration(0).get_folding_period()

    def get_snr_range(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        return self.snr_range(on_left=on_left, on_right=on_right, rebuild_local_scrunch=rebuild_local_scrunch)

    def get_snr_norm(self, on_left=None, on_right=None, rebuild_local_scrunch=True):
        return self.snr_norm(on_left=on_left, on_right=on_right, rebuild_local_scrunch=rebuild_local_scrunch)

    def get_snr_psrstat(self, rebuild_local_scrunch=True):
        return self.snr_psrstat(rebuild_local_scrunch=rebuild_local_scrunch)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # ---------------------------------- setter -----------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def set_freqs(self):
        self.freqs = self.get_freqs(doppler=False)
        self.freqs_doppler = self.get_freqs(doppler=True)
        self.chan_bw = self.get_chan_bw()

        self.centre_frequency = self.get_centre_frequency()
        self.bandwidth = self.get_bandwidth()

    def set_times(self):
        self.times = self.get_times()
        self.duration = self.integration_length()
        self.times_subint = []
        for isub in range(self.get_nsubint()):
            self.times_subint.append(self.get_Integration(isub).get_duration())
        self.times_subint = np.array(self.times_subint)

    # ---------------------------------- setter - cleaner -------------------------

    def set_zapfirstsubint(self, zapfirstsubint):
        self.cleaner_zapfirstsubint = zapfirstsubint

    def set_fast(self, fast):
        self.cleaner_fast = fast

    def set_flat_cleaner(self, flat_cleaner):
        self.cleaner_flat_cleaner = flat_cleaner

    def set_chanthresh(self, chanthresh):
        self.cleaner_chanthresh = float(chanthresh)

    def set_subintthresh(self, subintthresh):
        self.cleaner_subintthresh = float(subintthresh)

    def set_first_chanthresh(self, first_chanthresh):
        self.cleaner_first_chanthresh = first_chanthresh

    def set_first_subintthresh(self, first_subintthresh):
        self.cleaner_first_subintthresh = first_subintthresh

    def set_bad_subint(self, bad_subint):
        self.cleaner_bad_subint = float(bad_subint)

    def set_bad_chan(self, bad_chan):
        self.cleaner_bad_chan = float(bad_chan)

    def set_max_iter(self, max_iter):
        self.cleaner_max_iter = int(max_iter)


if __name__ == "__main__":
    ar = psrchive_class(verbos=True)
    # ar.MyArchive_load(['/databf2/nenufar-pulsar/DATA/B1508+55/PSR/B1508+55_D20220707T1701_59767_002409_0057_BEAM0.fits'], bscrunch=1, tscrunch=4, pscrunch=True)
    # ar.MyArchive_load(['/databf2/nenufar-pulsar/DATA/B1919+21/PSR/B1919+21_D20220303T1301_59641_500675_0057_BEAM2.fits'], bscrunch=1, tscrunch=2, pscrunch=True)
    # ar.MyArchive_load(['/databf/nenufar-pulsar/DATA/J1022+1001/PSR/J1022+1001_D20220815T1201_59806_252326_0067_BEAM1.fits', '/databf/nenufar-pulsar/DATA/J1022+1001/PSR/J1022+1001_D20220815T1201_59806_002534_0030_BEAM0.fits'], bscrunch=4, tscrunch=2, pscrunch=True)
    # ar.MyArchive_load(['B0950+08_D20220715T0801_59775_252248_0055_BEAM1_0001_dspsr.auxRM'], bscrunch=1, tscrunch=1, pscrunch=False)

    ar = psrchive_class(ar_name=['B0950+08_D20220715T0801_59775_252248_0055_BEAM1_0001_dspsr.auxRM'],
                        bscrunch=1, tscrunch=1, pscrunch=False, verbos=True)
    time_vec = ar.get_mjd()

    print(ar.get_nsubint())
    for i in range(len(time_vec)):
        print("%s, %f" % (time_vec[i].isot, time_vec[i].mjd))

    # exit(0)
    # ar.clean(fast=False)
    print(ar.snr_range())

    from plot_class import PlotArchive
    import matplotlib.pyplot as plt
    plot_ar = PlotArchive(ar)
    ax0 = plt.subplot2grid((5, 5), (0, 0), colspan=4, rowspan=4)
    plot_ar.snr_vs_subintegration(ax0, botaxis=True)
    plt.show()
    exit(0)

    plot_ar.profil(ax0, color=['b'])

    from DM_fit_lib import DM_fit
    print(ar.get_dispersion_measure())
    arx, dm, dm_err = DM_fit(ar, verbose=True, ncore=16, log_obj=ar.log)

    ar.set_dispersion_measure(dm)
    ar.dedisperse()
    ar.auto_rebin()

    print(ar.get_dispersion_measure())
    print(ar.snr_range())
    plot_ar.set_archive(ar)
    plot_ar.profil(ax0, color=['r'])
    plt.show()
