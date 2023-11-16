#!/usr/bin/env python

# To get help type 'man python' at the terminal

# Written by L. Bondonneau, 2022

import os
import numpy as np
import warnings
import csv
from multiprocessing import Pool
from methode_class import smoothGaussian
from methode_class import mad
from mypsrchive import psrchive_class
from DM_fit_class import DM_fit_class

RMFIT_OBJET = "RM_fit"
CONFIG_FILE = os.path.dirname(os.path.realpath(__file__)) + '/' + 'NenuPlot.conf'


class RM_fit_class(DM_fit_class):
    def __init__(self, *args, **kwargs):
        super(RM_fit_class, self).__init__(*args, **kwargs)
        rmdelt_60mhz = 0.002
        self.rmdelt = (rmdelt_60mhz / 60**2) * (self.centre_frequency)**2
        self.ncore = 40
        self.RM_sigma_limit = 8
        self.bin_sigma = 4.5

    def init_RM_fit(self):
        global fit_RM_archive
        fit_RM_archive = self.myclone()
        fit_RM_archive.dedisperse()
        fit_RM_archive.remove_baseline()
        fit_RM_archive.convert_state('Stokes')
        self.coh_rotation_measure = self.get_rotation_measure()
        self.max_duration = self.get_caracteristic_depolarisation_time()
        self.min_duration = 30  # self.max_duration / 16.
        self.set_freqs_extended(rechan_factor=256)
        # print(self.freqs_extended)
        # print(self.freqs)
        self.freqs_extended = self.freqs
        self.set_RM_vec(rm=self.coh_rotation_measure)
        self.coherent_derotaion = True
        if(self.get_faraday_corrected()):
            fit_RM_archive.set_rotation_measure(float(0.0))
            fit_RM_archive.defaraday()

        # self.data = np.memmap(self.name.split('.')[0] + '.memmap', dtype=np.float32,
        #                       mode='w+', shape=np.shape(self.get_data()))
        self.data = fit_RM_archive.get_data()
        self.weights = fit_RM_archive.get_weights()
        self.weights /= np.max(self.weights)
        for isub2 in range(self.get_nsubint()):
            for ichan in range(self.get_nchan()):
                if(self.weights[isub2, ichan] == 0):
                    self.data[isub2, :, ichan, :] = np.nan

        subint_mjd = np.asarray([i.mjd for i in self.times])
        self.subint_mjd = np.copy(subint_mjd)  # Time array
        self.subint_dur = np.copy(self.times_subint)  # Numpy array in sec
        self.subint_ind = np.linspace(0, self.get_nsubint() - 1, self.get_nsubint() - 1)
        self.RM_flag = False
        self.absolute_phase_flag = False

    def compute_subint(self, isub, rebuild_local_scrunch=True, rm_perbin=False, only_bestbin=False, sum_stokes_bin=False):
        ibin_vec = self.get_intence_bins(isub=isub, rebuild_local_scrunch=rebuild_local_scrunch, sigma=self.bin_sigma, only_bestbin=only_bestbin)
        if (len(ibin_vec) == 0):
            if(rm_perbin):
                return(np.nan, 0, [])
            else:
                return (np.nan, 0)
        # print("ibin_vec = ", ibin_vec)
        RM, RM_sigma, spectra = self.get_RMspectrum(ibin_vec, isub, rm_perbin=rm_perbin, sum_stokes_bin=sum_stokes_bin)
        if(rm_perbin):
            return (RM, RM_sigma, ibin_vec)
        else:
            return (RM, RM_sigma)

    # def defaraday(self, Q_tmp, U_tmp, RM, centre_frequency=None, freqs=None, n=None):
    #    if centre_frequency is None:
    #        centre_frequency = self.centre_frequency
    #    if freqs is None:
    #        freqs = self.freqs
    #    rmfac = RM * 89875.51787368176  # /(1.0+ds.earth_z4/1.0e4);
    #    rot = (rmfac * ((centre_frequency**-2) - (freqs**-2)))
    #    Q_new = Q_tmp * np.cos(2 * rot) - U_tmp * np.sin(2 * rot)
    #    U_new = Q_tmp * np.sin(2 * rot) + U_tmp * np.cos(2 * rot)
    #    if n is None:
    #        return (Q_new, U_new)
    #    else:
    #        return (Q_new, U_new, n)

    def get_IQUW(self, isub_vec, ibin_vec, sum_stokes_bin=True):
        # self.weights [isub, ichan]
        # self.data    [isub, ipol, ichan, ibin]
        if (isinstance(isub_vec, list)) and (len(isub_vec) == 1):
            isub_data = self.data[isub_vec[0], :, :, :]
            isub_weights = self.weights[isub_vec[0], :]
        else:
            isub_data = self.data[isub_vec, :, :, :]
            isub_weights = self.weights[isub_vec, :]
        if (len(isub_vec) > 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # isub_std = np.nanstd(isub_data, axis=0)
                isub_data = np.nanmean(isub_data, axis=0)
                isub_weights = np.nanmean(isub_weights, axis=0)
        # isub_weights [ichan]
        # isub_data    [ipol, ichan, ibin]
        bad_ibin_vec = np.array(np.ones(self.get_nbin()), dtype=bool)
        bad_ibin_vec[ibin_vec] = False
        # bad_ibin_vec = np.np.linspace(0, self.get_nbin() - 1, self.get_nbin())[bad_ibin_vec]
        isub_std = isub_data[:, :, list(bad_ibin_vec)]
        # isub_std    [ipol, ichan, ibin]
        isub_std = np.nanstd(isub_std, axis=2)
        isub_std = np.expand_dims(isub_std, axis=2)

        if (isinstance(ibin_vec, list)) and (len(ibin_vec) == 1):
            isub_data = isub_data[:, :, ibin_vec[0]]
        else:
            isub_data = isub_data[:, :, ibin_vec]

        if (sum_stokes_bin):
            isub_data = np.nanmean(isub_data, axis=2)
            isub_data = np.expand_dims(isub_data, axis=2)
            ibin_vec = [0]
        # isub_std[ipol, ichan, ibin]
        # isub_data[ipol, ichan, ibin]
        I = isub_data[0, :, :]
        Q = isub_data[1, :, :]
        U = isub_data[2, :, :]
        I_std = isub_std[0, :, :]
        Q_std = isub_std[1, :, :]
        U_std = isub_std[2, :, :]
        # norm = np.sqrt((Q**2 + U**2) / 2.0)
        # Q /= norm
        # U /= norm
        # Q_std /= (norm * np.sqrt(len(ibin_vec)))
        # U_std /= (norm * np.sqrt(len(ibin_vec)))

        return (I, Q, U, I_std, Q_std, U_std, isub_weights)

    def get_isub_data(self, isub_vec):
        isub_data = self.data[isub_vec, :, :, :]
        if (np.size(isub_vec) > 1):
            isub_data = np.nanmean(isub_data, axis=0)
        return isub_data

    def get_intence_bins(self, isub=None, rebuild_local_scrunch=True, sigma=3, only_bestbin=False):
        self.set_onpulse(rebuild_local_scrunch=rebuild_local_scrunch)
        if(isub is None):
            prof = self.scrunch.get_Profile(0, 0, 0).get_amps() * 10000
        if (len(isub) <= 1):
            if(isinstance(isub, list)):
                prof = self.data[int(isub[0]), 0, :, :]
            else:
                prof = self.data[int(isub), 0, :, :]
            prof = np.nanmean(prof, axis=0)
        else:
            prof = self.data[isub, 0, :, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                prof = np.nanmean(np.nanmean(prof, axis=0), axis=0)
        median = np.median(prof[self.offbins])
        rms = np.std(prof[self.offbins])
        prof_norm = (prof - median) / rms
        if (only_bestbin):
            return np.array([np.argmax(prof_norm)])
        else:
            return np.where(prof_norm > sigma)[0]

    def get_caracteristic_depolarisation_time(self, freq=None):
        if (freq is None):
            freq = self.centre_frequency
        dRM = 1.0 / 3600.  # 1 rad.cm-3 in 60 min
        depol_rot = np.pi / 2.5
        return depol_rot / (dRM * 89875.51787368176 * (freq**-2))

    def set_subint_depolarisation_time(self):
        tscrunch = int(np.ceil(self.max_duration / self.get_subint_duration()))
        if(tscrunch > self.get_nsubint()):
            self.log.warning("The caracteristic depolarisation time (%.0f sec) is largeur than the total duration (%.0f)" %
                             (self.get_caracteristic_depolarisation_time(), self.duration), objet=RMFIT_OBJET)
            tscrunch = self.get_nsubint()

        nscrunched_subint = int(np.ceil(float(self.get_nsubint()) / tscrunch))
        self.scrunch_subint_start_ind = []
        self.scrunch_subint_stop_ind = []
        self.scrunch_subint_mjd = []
        self.scrunch_subint_dur = []
        for isub_scrunched in range(nscrunched_subint):
            start_ind = (isub_scrunched) * tscrunch
            stop_ind = (isub_scrunched + 1) * tscrunch
            if (stop_ind > self.get_nsubint()):
                stop_ind = self.get_nsubint()
            self.scrunch_subint_start_ind.append(start_ind)
            self.scrunch_subint_stop_ind.append(stop_ind)
            self.scrunch_subint_mjd.append(self.subint_mjd[start_ind])
            self.scrunch_subint_dur.append(np.sum(self.subint_dur[start_ind: stop_ind]))

        self.scrunch_subint_start_ind = np.array(self.scrunch_subint_start_ind)
        self.scrunch_subint_stop_ind = np.array(self.scrunch_subint_stop_ind)
        self.scrunch_subint_mjd = np.array(self.scrunch_subint_mjd)
        self.scrunch_subint_dur = np.array(self.scrunch_subint_dur)

    def set_freqs_extended(self, rechan_factor=256):
        self.freqs_extended = np.zeros(rechan_factor * self.get_nchan())
        fraction_bw = np.linspace(-0.5 + 0.5 / rechan_factor, 0.5 - 0.5 / rechan_factor, rechan_factor)
        for ichan in range(self.get_nchan()):
            chan_bw = self.chan_bw[ichan]
            if (rechan_factor > 1):
                for ichan_extended in range(rechan_factor):
                    self.freqs_extended[ichan * rechan_factor + ichan_extended] = self.freqs[ichan] + fraction_bw[ichan_extended] * chan_bw
            else:
                self.freqs_extended[ichan] = self.freqs[ichan]

    class Integration_limit(Exception):
        pass

    def expand_integration(self, isub_scrunched):
        nsubint = self.scrunch_subint_stop_ind[isub_scrunched] - self.scrunch_subint_start_ind[isub_scrunched]

        if (nsubint <= 1):
            raise self.Integration_limit
            return

        if (self.scrunch_subint_dur[isub_scrunched] / 2. < self.min_duration):
            raise self.Integration_limit
            return

        start_ind = self.scrunch_subint_start_ind[isub_scrunched] + int(np.round(nsubint / 2.))
        stop_ind = self.scrunch_subint_stop_ind[isub_scrunched]

        self.scrunch_subint_start_ind = np.insert(self.scrunch_subint_start_ind, isub_scrunched + 1, start_ind)
        self.scrunch_subint_stop_ind = np.insert(self.scrunch_subint_stop_ind, isub_scrunched + 1, stop_ind)
        # self.scrunch_subint_mjd = np.insert(self.scrunch_subint_mjd, isub_scrunched + 1, self.subint_mjd[start_ind])
        self.scrunch_subint_mjd = np.insert(self.scrunch_subint_mjd, isub_scrunched + 1, np.mean(self.subint_mjd[start_ind: stop_ind]))
        self.scrunch_subint_dur = np.insert(self.scrunch_subint_dur, isub_scrunched + 1, np.sum(self.subint_dur[start_ind: stop_ind]))

        self.scrunch_subint_mjd[isub_scrunched] = np.mean(self.subint_mjd[self.scrunch_subint_start_ind[isub_scrunched]: start_ind])
        self.scrunch_subint_dur[isub_scrunched] = np.sum(self.subint_dur[self.scrunch_subint_start_ind[isub_scrunched]: start_ind])
        self.scrunch_subint_stop_ind[isub_scrunched] = start_ind

        self.scrunch_subint_RM = np.insert(self.scrunch_subint_RM, isub_scrunched + 1, 0)
        self.scrunch_subint_sigma = np.insert(self.scrunch_subint_sigma, isub_scrunched + 1, 0)

    def unexpand_integration(self, isub_scrunched):
        if (isub_scrunched + 1 >= len(self.scrunch_subint_dur)):
            raise self.Integration_limit
            return
        self.scrunch_subint_stop_ind[isub_scrunched] = self.scrunch_subint_stop_ind[isub_scrunched + 1]
        self.scrunch_subint_dur[isub_scrunched] = np.sum(
            self.subint_dur[self.scrunch_subint_start_ind[isub_scrunched]: self.scrunch_subint_stop_ind[isub_scrunched + 1]])
        self.scrunch_subint_mjd[isub_scrunched] = np.mean(
            self.subint_mjd[self.scrunch_subint_start_ind[isub_scrunched]: self.scrunch_subint_stop_ind[isub_scrunched + 1]])

        self.scrunch_subint_start_ind = np.delete(self.scrunch_subint_start_ind, isub_scrunched + 1)
        self.scrunch_subint_stop_ind = np.delete(self.scrunch_subint_stop_ind, isub_scrunched + 1)
        self.scrunch_subint_mjd = np.delete(self.scrunch_subint_mjd, isub_scrunched + 1)
        self.scrunch_subint_dur = np.delete(self.scrunch_subint_dur, isub_scrunched + 1)

        self.scrunch_subint_RM = np.delete(self.scrunch_subint_RM, isub_scrunched + 1)
        self.scrunch_subint_sigma = np.delete(self.scrunch_subint_sigma, isub_scrunched + 1)

    def resize_integration(self, isub_scrunched, resize_fraction, only_bestbin=True, sum_stokes_bin=False):
        # resize_fraction = RM_sigma1 / RM_sigma2
        nsub1 = self.scrunch_subint_stop_ind[isub_scrunched] - self.scrunch_subint_start_ind[isub_scrunched]
        nsub2 = self.scrunch_subint_stop_ind[isub_scrunched + 1] - self.scrunch_subint_start_ind[isub_scrunched + 1]
        nb_subint = self.scrunch_subint_stop_ind[isub_scrunched + 1] - self.scrunch_subint_start_ind[isub_scrunched]

        if (resize_fraction > 1):  # need to feed isub2 with isub1
            nsub2 += round(nsub1 * 0.5 * (1 - 1 / resize_fraction))
            nsub1 = nb_subint - nsub2
        elif (resize_fraction <= 1):
            nsub1 += round(nsub2 * 0.5 * (1 - resize_fraction))
            nsub2 = nb_subint - nsub1

        if(nsub1 < 1) or (nsub2 < 1):
            raise self.Integration_limit
            return
        if (nsub1 == self.scrunch_subint_stop_ind[isub_scrunched] - self.scrunch_subint_start_ind[isub_scrunched]):
            raise self.Integration_limit
            return

        self.scrunch_subint_start_ind[isub_scrunched] = self.scrunch_subint_start_ind[isub_scrunched]
        self.scrunch_subint_stop_ind[isub_scrunched] = self.scrunch_subint_start_ind[isub_scrunched] + nsub1
        self.scrunch_subint_start_ind[isub_scrunched + 1] = self.scrunch_subint_stop_ind[isub_scrunched]
        self.scrunch_subint_stop_ind[isub_scrunched + 1] = self.scrunch_subint_stop_ind[isub_scrunched] + nsub2
        # self.scrunch_subint_mjd[isub_scrunched] = self.subint_mjd[self.scrunch_subint_start_ind[isub_scrunched]]
        # self.scrunch_subint_mjd[isub_scrunched + 1] = self.subint_mjd[self.scrunch_subint_start_ind[isub_scrunched + 1]]
        self.scrunch_subint_mjd[isub_scrunched] = np.mean(self.subint_mjd[self.scrunch_subint_start_ind[isub_scrunched]: self.scrunch_subint_stop_ind[isub_scrunched]])
        self.scrunch_subint_mjd[isub_scrunched + 1] = np.mean(self.subint_mjd[self.scrunch_subint_start_ind[isub_scrunched + 1]: self.scrunch_subint_stop_ind[isub_scrunched + 1]])
        self.scrunch_subint_dur[isub_scrunched] = np.sum(self.subint_dur[self.scrunch_subint_start_ind[isub_scrunched]: self.scrunch_subint_stop_ind[isub_scrunched]])
        self.scrunch_subint_dur[isub_scrunched + 1] = np.sum(self.subint_dur[self.scrunch_subint_start_ind[isub_scrunched + 1]: self.scrunch_subint_stop_ind[isub_scrunched + 1]])

        isub1 = range(self.scrunch_subint_start_ind[isub_scrunched], self.scrunch_subint_stop_ind[isub_scrunched])
        isub2 = range(self.scrunch_subint_start_ind[isub_scrunched + 1], self.scrunch_subint_stop_ind[isub_scrunched + 1])

        RM1, RM_sigma1 = self.compute_subint(isub1, rm_perbin=False, only_bestbin=only_bestbin, sum_stokes_bin=sum_stokes_bin)
        RM2, RM_sigma2 = self.compute_subint(isub2, rm_perbin=False, only_bestbin=only_bestbin, sum_stokes_bin=sum_stokes_bin)
        self.log.log("    RM1 = %.6f sigma = %.1f" % (RM1, RM_sigma1), objet=RMFIT_OBJET)
        self.log.log("    RM2 = %.6f sigma = %.1f" % (RM2, RM_sigma2), objet=RMFIT_OBJET)
        self.scrunch_subint_RM[isub_scrunched] = RM1
        self.scrunch_subint_RM[isub_scrunched + 1] = RM2
        self.scrunch_subint_sigma[isub_scrunched] = RM_sigma1
        self.scrunch_subint_sigma[isub_scrunched + 1] = RM_sigma2

    def RM_reduction(self, only_bestbin=False, sum_stokes_bin=False, QU_fit=True):
        self.set_subint_depolarisation_time()
        self.scrunch_subint_RM = []
        self.scrunch_subint_sigma = []
        for isub_scrunched in range(len(self.scrunch_subint_mjd)):
            self.log.log("progress : %d/%d" % (isub_scrunched + 1, len(self.scrunch_subint_mjd)), objet=RMFIT_OBJET)
            isub = range(self.scrunch_subint_start_ind[isub_scrunched], self.scrunch_subint_stop_ind[isub_scrunched])
            RM, RM_sigma = self.compute_subint(isub, rm_perbin=False, only_bestbin=only_bestbin, sum_stokes_bin=sum_stokes_bin)
            self.log.log("RM = %.6f sigma = %.1f duration = %.3f sec" % (RM, RM_sigma,
                                                                         self.scrunch_subint_dur[isub_scrunched]), objet=RMFIT_OBJET)
            self.scrunch_subint_RM.append(RM)
            self.scrunch_subint_sigma.append(RM_sigma)
        self.scrunch_subint_RM = np.array(self.scrunch_subint_RM)
        self.scrunch_subint_sigma = np.array(self.scrunch_subint_sigma)

        isub_scrunched = 0
        while (isub_scrunched < len(self.scrunch_subint_mjd)):
            # print(self.scrunch_subint_start_ind)
            # print(self.scrunch_subint_stop_ind)
            # print(self.scrunch_subint_dur)
            percent = float(100. * np.sum(self.scrunch_subint_dur[0:isub_scrunched]) / self.duration)
            log_msg = "progress %04d/%d -> %02.2f percent" % (self.scrunch_subint_start_ind[isub_scrunched], self.get_nsubint(), percent)
            self.log.log(log_msg, objet=RMFIT_OBJET)
            self.log.log("RM = %.6f sigma = %.1f duration = %.3f sec" % (self.scrunch_subint_RM[isub_scrunched], self.scrunch_subint_sigma[isub_scrunched],
                                                                         self.scrunch_subint_dur[isub_scrunched]), objet=RMFIT_OBJET)
            if (self.scrunch_subint_sigma[isub_scrunched] > self.RM_sigma_limit):
                try:
                    self.expand_integration(isub_scrunched)
                except self.Integration_limit:
                    isub_scrunched += 1
                    continue
                isub1 = range(self.scrunch_subint_start_ind[isub_scrunched], self.scrunch_subint_stop_ind[isub_scrunched])
                isub2 = range(self.scrunch_subint_start_ind[isub_scrunched + 1], self.scrunch_subint_stop_ind[isub_scrunched + 1])
                RM1, RM_sigma1 = self.compute_subint(isub1, rm_perbin=False, only_bestbin=only_bestbin, sum_stokes_bin=sum_stokes_bin)
                RM2, RM_sigma2 = self.compute_subint(isub2, rm_perbin=False, only_bestbin=only_bestbin, sum_stokes_bin=sum_stokes_bin)
                print("EXPAND!!! ")
                self.log.log("    RM1 = %.6f sigma = %.1f" % (RM1, RM_sigma1), objet=RMFIT_OBJET)
                self.log.log("    RM2 = %.6f sigma = %.1f" % (RM2, RM_sigma2), objet=RMFIT_OBJET)
                if (RM_sigma1 > self.RM_sigma_limit) and (RM_sigma2 > self.RM_sigma_limit):
                    self.scrunch_subint_RM[isub_scrunched] = RM1
                    self.scrunch_subint_RM[isub_scrunched + 1] = RM2
                    self.scrunch_subint_sigma[isub_scrunched] = RM_sigma1
                    self.scrunch_subint_sigma[isub_scrunched + 1] = RM_sigma2
                    print("REPROCESS!!! ")
                    continue
                elif (RM_sigma1 > self.RM_sigma_limit) and (RM_sigma2 < self.RM_sigma_limit) and (isub_scrunched < len(self.scrunch_subint_mjd) - 2):
                    print("UNEXPAND PROGRADE!!!")
                    self.scrunch_subint_RM[isub_scrunched] = RM1
                    self.scrunch_subint_sigma[isub_scrunched] = RM_sigma1
                    self.unexpand_integration(isub_scrunched + 1)
                    isub = range(self.scrunch_subint_start_ind[isub_scrunched + 1], self.scrunch_subint_stop_ind[isub_scrunched + 1])
                    RM, RM_sigma = self.compute_subint(isub, rm_perbin=False, only_bestbin=only_bestbin, sum_stokes_bin=sum_stokes_bin)
                    self.scrunch_subint_RM[isub_scrunched + 1] = RM
                    self.scrunch_subint_sigma[isub_scrunched + 1] = RM_sigma
                    if(RM_sigma1 > self.RM_sigma_limit):
                        print("REPROCESS!!! ")
                        continue
                elif (RM_sigma1 < self.RM_sigma_limit) and (RM_sigma2 > self.RM_sigma_limit):
                    print("RESIZE!!!")
                    try:
                        self.resize_integration(isub_scrunched, RM_sigma1 / RM_sigma2, only_bestbin=only_bestbin, sum_stokes_bin=sum_stokes_bin)
                    except self.Integration_limit:
                        self.unexpand_integration(isub_scrunched)
                else:
                    print("UNEXPAND!!!")
                    self.unexpand_integration(isub_scrunched)
            else:
                try:
                    print("UNEXPAND!!!")
                    RM1, RM_sigma1 = self.scrunch_subint_RM[isub_scrunched], self.scrunch_subint_sigma[isub_scrunched]
                    RM2, RM_sigma2 = self.scrunch_subint_RM[isub_scrunched + 1], self.scrunch_subint_sigma[isub_scrunched + 1]
                    self.unexpand_integration(isub_scrunched)
                    isub = range(self.scrunch_subint_start_ind[isub_scrunched], self.scrunch_subint_stop_ind[isub_scrunched])
                    RM, RM_sigma = self.compute_subint(isub, rm_perbin=False, only_bestbin=only_bestbin, sum_stokes_bin=sum_stokes_bin)
                    self.log.log("RM = %.6f sigma = %.1f duration = %.3f sec" % (RM, RM_sigma,
                                                                                 self.scrunch_subint_dur[isub_scrunched]), objet=RMFIT_OBJET)
                    if(RM_sigma < self.RM_sigma_limit) and (RM_sigma2 > self.RM_sigma_limit):
                        self.expand_integration(isub_scrunched)
                        self.scrunch_subint_RM[isub_scrunched] = RM1
                        self.scrunch_subint_RM[isub_scrunched + 1] = RM2
                        self.scrunch_subint_sigma[isub_scrunched] = RM_sigma1
                        self.scrunch_subint_sigma[isub_scrunched + 1] = RM_sigma2
                    else:
                        self.scrunch_subint_RM[isub_scrunched] = RM
                        self.scrunch_subint_sigma[isub_scrunched] = RM_sigma
                        continue
                except self.Integration_limit:
                    pass
                except IndexError:  # if last integration
                    pass
            isub_scrunched += 1

        popid = np.where(np.isnan(self.scrunch_subint_sigma) == True)[0]
        for i in np.flip(popid, axis=0):
            self.scrunch_subint_start_ind = np.delete(self.scrunch_subint_start_ind, i)
            self.scrunch_subint_stop_ind = np.delete(self.scrunch_subint_stop_ind, i)
            self.scrunch_subint_mjd = np.delete(self.scrunch_subint_mjd, i)
            self.scrunch_subint_dur = np.delete(self.scrunch_subint_dur, i)
            self.scrunch_subint_RM = np.delete(self.scrunch_subint_RM, i)
            self.scrunch_subint_sigma = np.delete(self.scrunch_subint_sigma, i)

        popid = np.where(self.scrunch_subint_sigma < self.RM_sigma_limit)[0]
        for i in np.flip(popid, axis=0):
            self.scrunch_subint_start_ind = np.delete(self.scrunch_subint_start_ind, i)
            self.scrunch_subint_stop_ind = np.delete(self.scrunch_subint_stop_ind, i)
            self.scrunch_subint_mjd = np.delete(self.scrunch_subint_mjd, i)
            self.scrunch_subint_dur = np.delete(self.scrunch_subint_dur, i)
            self.scrunch_subint_RM = np.delete(self.scrunch_subint_RM, i)
            self.scrunch_subint_sigma = np.delete(self.scrunch_subint_sigma, i)

        # if (QU_fit):
        #     from lmfit import Parameters, minimize
        #     from lmfit.printfuncs import report_fit
        #     out = minimize(Simulated_IQUV, fit_params, args=(freqs,),
        #                    kws={'ratio_LI': ratio_LI, 'ratio_VI': ratio_VI,
        #                         'I_data': I, 'Q_data': Q, 'U_data': U, 'V_data': V,
        #                         'ind_nan': ind_nan, 'ind_nan_extended': ind_nan_extended,
        #                         'errQ': errsigma * errQ_norm, 'errU': errsigma * errU_norm, 'coh_rm': coh_rm})  # , method='emcee'
    def RM_refining(self, sum_stokes_bin=True):
        # after RM reduction because need a good starting point
        import sys
        if (sys.version_info.major < 3):
            sys.path.insert(0, "/home/lbondonneau/.local/lib/python2.7_alt/site-packages")
        from lmfit import Parameters, minimize
        from lmfit.printfuncs import report_fit

        self.scrunch_subint_RM_refining = np.zeros(len(self.scrunch_subint_RM))
        self.scrunch_subint_RM_refining_err = np.zeros(len(self.scrunch_subint_RM))
        self.scrunch_subint_phase_refining = np.zeros(len(self.scrunch_subint_RM))
        self.scrunch_subint_phase_refining_err = np.zeros(len(self.scrunch_subint_RM))

        def QU_residual(param, freqs, rm=None, max_freq=None, freqs_extended=None, isub_vec=None, ibin_vec=None, coh_rm=None, plot=None):
            Q_fit, U_fit = self.Simulated_IQUV(param, max_freq, freqs, freqs_extended, rm=rm, coh_rm=coh_rm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                I_data, Q_data, U_data, I_std, Q_std, U_std, weights = self.get_IQUW(isub_vec, ibin_vec, sum_stokes_bin=True)
            I_data = np.squeeze(I_data)
            Q_data = np.squeeze(Q_data)
            U_data = np.squeeze(U_data)
            I_std = np.squeeze(I_std)
            Q_std = np.squeeze(Q_std)
            U_std = np.squeeze(U_std)

            def QU_to_Lsum(Q, U):
                Lsum = np.sqrt(np.nansum(Q)**2 + np.nansum(U)**2)
                Ldata = np.sqrt((Q)**2 + (U)**2)
                Ldata = Ldata / np.nanmean(Ldata)
                return (Lsum, Ldata)

            # normalisation of Q_data, U_data and std
            norm = np.sqrt((Q_data**2 + U_data**2) / 2.0)
            Q_data_norm = Q_data / norm
            U_data_norm = U_data / norm
            Q_std_norm = Q_std / (norm * np.sqrt(len(ibin_vec)))
            U_std_norm = U_std / (norm * np.sqrt(len(ibin_vec)))

            # calcul of the linear signal L
            if (rm is None):
                Q_data_tmp, U_data_tmp = self.apply_defaraday(Q_data_norm, U_data_norm, param['RM'].value)
                # print(np.nansum(L_data), "RM = ", param['RM'].value, "Phase = ", param['rotation'].value)
            else:
                Q_data_tmp, U_data_tmp = self.apply_defaraday(Q_data_norm, U_data_norm, rm)
            Lsum_data, L_data = QU_to_Lsum(Q_data_tmp / Q_std_norm, U_data_tmp / U_std_norm)
            L_data = np.squeeze(L_data)
            Q_residual = (np.squeeze(Q_fit) - np.squeeze(Q_data_norm)) / np.squeeze(Q_std_norm)
            U_residual = (np.squeeze(U_fit) - np.squeeze(U_data_norm)) / np.squeeze(U_std_norm)

            if (plot is not None):
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(12, 3))
                plt.subplots_adjust(top=0.92, bottom=0.145,
                                    left=0.045, right=0.995,
                                    hspace=0.095, wspace=0.2)
                ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=1)
                ax1 = plt.subplot2grid((2, 3), (1, 0), colspan=2, rowspan=1, sharex=ax0, sharey=ax0)
                ax0.axes.get_xaxis().set_visible(False)

                ax0.plot(freqs, Q_fit, 'b--', label='Fit Q')
                ax0.plot(freqs, U_fit, '--', color='orange', label='Fit U')

                ax0.errorbar(freqs, Q_data_tmp, yerr=Q_std_norm, fmt='r+', label='Data Q_tmp')
                ax0.errorbar(freqs, U_data_tmp, yerr=U_std_norm, fmt='m+', label='Data U_tmp')
                ax0.errorbar(freqs, Q_data_norm, yerr=Q_std_norm, fmt='b+', label='Data Q')
                ax0.errorbar(freqs, U_data_norm, yerr=U_std_norm, fmt='+', color='orange', label='Data U')
                ax1.errorbar(freqs, np.squeeze(Q_data_norm) - np.squeeze(Q_fit), yerr=Q_std_norm, fmt='.b', label='Residual of Q')
                ax1.errorbar(freqs, np.squeeze(U_data_norm) - np.squeeze(U_fit), yerr=U_std_norm, fmt='.', color='orange', label='Residual of U')

                ax0.plot(freqs, np.zeros(len(freqs)), 'k--', alpha=0.5)
                ax0.legend(loc='upper right')
                ax0.set_ylabel('Amplitude (AU)')
                ax0.set_ylim([-2, 2])
                ax1.plot(freqs, np.zeros(len(freqs)), 'k--', alpha=0.5)
                ax1.legend(loc='upper right')
                ax1.set_xlabel('Frequency (MHz)')
                ax1.set_ylabel('Residual (AU)')

                ax2 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=2)  # , polar=True
                PA_fit = 0.5 * np.arctan2(np.squeeze(U_fit), np.squeeze(Q_fit))
                nPA = 8.
                ind_PA = np.array((PA_fit - (PA_fit % (np.pi / nPA))) / (np.pi / nPA), dtype=int)
                rad = np.linspace(-np.pi, np.pi, int(nPA))

                L_data_tmp = np.sqrt((Q_data)**2 + (U_data)**2)
                Imean = np.zeros(int(nPA))
                Lmean = np.zeros(int(nPA))
                Qmean = np.zeros(int(nPA))
                Umean = np.zeros(int(nPA))

                Icont = np.zeros(int(nPA))
                Lcont = np.zeros(int(nPA))
                Qcont = np.zeros(int(nPA))
                Ucont = np.zeros(int(nPA))

                Q_res = np.squeeze(Q_data_norm) - np.squeeze(Q_fit)
                U_res = np.squeeze(U_data_norm) - np.squeeze(U_fit)

                for i in range(len(PA_fit)):
                    ind = int(ind_PA[i] + nPA / 2)
                    if not np.isnan(I_data[i]):
                        Imean[ind] += I_data[i]
                        Icont[ind] += 1.0
                    if not np.isnan(Q_data[i]):
                        Qmean[ind] += Q_data[i] / I_data[i]
                        Qcont[ind] += 1.0
                    if not np.isnan(U_data[i]):
                        Umean[ind] += U_data[i] / I_data[i]
                        Ucont[ind] += 1.0
                    if not np.isnan(L_data_tmp[i]):
                        Lmean[ind] += np.sqrt(Q_data[i]**2 + U_data[i]**2) / I_data[i]
                        Lcont[ind] += 1.0
                Imean = Imean / Icont
                Qmean = Qmean / Qcont
                Umean = Umean / Ucont
                Lmean = Lmean / Lcont

                # ax2.plot(rad, Imean, '+k')
                ax2.plot(rad, Lmean, '+r')
                ax2.plot(rad, Qmean, '+g--')
                ax2.plot(rad, Umean, '+m--')
                # ax2.set_rmax(2)
                # ax2.set_rmin(0)
                plt.show()

            Q_residual[np.isnan(Q_residual)] = 0
            U_residual[np.isnan(U_residual)] = 0
            L_data[np.isnan(L_data)] = 0
            Q_residual[np.isinf(Q_residual)] = 0
            U_residual[np.isinf(U_residual)] = 0
            L_data[np.isinf(L_data)] = 0
            print(Lsum_data, np.nansum(np.sqrt((Q_residual)**2 + (U_residual)**2) * L_data / np.nansum(Lsum_data)), param['RM'].value, param['rotation'].value)

            return np.sqrt((Q_residual)**2 + (U_residual)**2) * L_data / np.nansum(Lsum_data)

        for isub_scrunched in range(len(self.scrunch_subint_RM)):
            isub_vec = range(self.scrunch_subint_start_ind[isub_scrunched], self.scrunch_subint_stop_ind[isub_scrunched])
            ibin_vec = self.get_intence_bins(isub=isub_vec, rebuild_local_scrunch=True, sigma=self.bin_sigma, only_bestbin=False)

            # Q [ichan, ibin] with nan
            # U [ichan, ibin] with nan
            # weights [ichan]

            max_freq = self.freqs[-1]
            coh_rm = self.coh_rotation_measure
            RM = self.scrunch_subint_RM[isub_scrunched]

            # fit_params = Parameters()
            # fit_params.add('rotation', value=0.00, min=-2 * np.pi, max=2 * np.pi)
            # out = minimize(QU_residual, fit_params, args=(self.freqs,),
            #                kws={'rm': RM, 'max_freq': max_freq, 'freqs_extended': self.freqs_extended,
            #                     'isub_vec': isub_vec, 'ibin_vec': ibin_vec, 'coh_rm': coh_rm})  # , method='emcee'
            # report_fit(out, show_correl=True, modelpars=fit_params)

            p_true = Parameters()
            p_true.add('RM', value=RM)
            p_true.add('rotation', value=0)
            # p_true.add('gain', value=0.0)

            fit_params = Parameters()
            RM_window60 = 0.2
            RM_window = (RM_window60 / 60**2) * (self.centre_frequency)**2
            if (RM > 0):
                offset = RM_window / 3.  # force searching
            else:
                offset = -RM_window / 3.  # force searching
            offset = 0
            fit_params.add('RM', value=RM + offset, max=RM + RM_window, min=RM - RM_window)
            if (isub_scrunched > 0):
                rotation_ini = self.scrunch_subint_phase_refining[isub_scrunched - 1]
            else:
                rotation_ini = 0.66
            fit_params.add('rotation', value=rotation_ini, min=-2 * np.pi, max=2 * np.pi)
            # fit_params.add('gain', value=0.0, min=-10, max=+10)

            # QU_residual(param, freqs, max_freq=None, freqs_extended=None, isub_vec=None, ibin_vec=None, coh_rm=None):
            out = minimize(QU_residual, fit_params, args=(self.freqs,),
                           kws={'max_freq': max_freq, 'freqs_extended': self.freqs_extended,
                                'isub_vec': isub_vec, 'ibin_vec': ibin_vec, 'coh_rm': coh_rm, 'plot': None})  # , method='leastsq' 'emcee' 'brute' 'nelder'
            report_fit(out, show_correl=True, modelpars=p_true)

            subint_doppler = int(np.floor(np.mean(isub_vec)))
            self.scrunch_subint_RM_refining[isub_scrunched] = out.params['RM'].value / (self.get_doppler(subint=subint_doppler)**2)
            self.scrunch_subint_RM_refining_err[isub_scrunched] = out.params['RM'].stderr
            self.scrunch_subint_phase_refining[isub_scrunched] = out.params['rotation'].value
            self.scrunch_subint_phase_refining_err[isub_scrunched] = out.params['rotation'].stderr
            # QU_residual(out.params, self.freqs, max_freq=max_freq, freqs_extended=self.freqs_extended,
            #             isub_vec=isub_vec, ibin_vec=ibin_vec, coh_rm=coh_rm, plot=True)
        self.absolute_phase_flag = True

    def apply_rotation(self, Q, U, rot):
        Q_tmp = Q * np.cos(-2 * rot) - U * np.sin(-2 * rot)
        U_tmp = Q * np.sin(-2 * rot) + U * np.cos(-2 * rot)
        return Q_tmp, U_tmp

    def apply_defaraday(self, Q, U, rm):
        rot = rm * 89875.51787368176 * ((self.freqs[-1]**-2) - (self.freqs**-2))
        Q_tmp = Q * np.cos(2 * rot) - U * np.sin(2 * rot)
        U_tmp = Q * np.sin(2 * rot) + U * np.cos(2 * rot)
        return Q_tmp, U_tmp

    def Simulated_IQUV(self, param, max_freq, freqs, freqs_extended, rm=None, coh_rm=None):
        if (rm is None):
            rm = param['RM']

        try:
            rotation = param['rotation']
        except KeyError:
            rotation = None
        try:
            gain = param['gain']
        except KeyError:
            gain = None

        if coh_rm is None:
            coh_rm = 0

        # print('rm = %.4f rotation =  %.2f' % (rm, rotation))
        # print('rm = %.4f rotation =  %.2f gain = %.1f phase = %.1f' %(rm, rotation, gain, phase))

        # rmfac to the requesed RM value
        rmfac = rm * 89875.51787368176  # /(1.0+ds.earth_z4/1.0e4);
        # rmfac to the requesed RM used for coherent dedispersion
        rmfac_coh_rm = coh_rm * 89875.51787368176  # /(1.0+ds.earth_z4/1.0e4);
        # rotating angle to RM for the vector freqs_extended
        rot = (rmfac * ((max_freq**-2) - (freqs_extended**-2)))
        # rotating angle to coherent RM for the vector freqs_extended
        rot_coh_rm = (rmfac_coh_rm * ((max_freq**-2) - (freqs_extended**-2)))
        # delta angle btw coheerent RM and requested RM for vector freqs_extended
        delta_rot = rot - rot_coh_rm
        # rotating angle to coherent RM for the vector freqs
        rot_coh_rm_rechan = (rmfac_coh_rm * ((max_freq**-2) - (freqs**-2)))

        Q = np.ones(len(freqs_extended))
        U = np.copy(Q)

        # appling the true RM (sum of delta_rot and rot_coh_rm) (PSR is rotate by the true RM)
        Q, U = self.apply_rotation(Q, U, delta_rot + rot_coh_rm)

        if (gain is not None):
            Q += gain / 2
        # U, V = apply_phase(phase, U, V)

        # removing of dRM between 0 and the value of the coherent derotation (PSR is rotate by the dRM)
        if (coh_rm != 0):
            Q, U = self.apply_rotation(Q, U, -rot_coh_rm)

        # depolarisation due to the dDM between coherent derotation and the true RM
        rechan_factor = int(len(freqs_extended) / len(freqs))
        Q = np.mean(np.reshape(Q, (len(freqs), rechan_factor)), axis=1)
        U = np.mean(np.reshape(U, (len(freqs), rechan_factor)), axis=1)

        # appling of dRM between 0 and the value of the coherent derotation (PSR is rotate by the true RM)
        if (coh_rm != 0):
            Q, U = self.apply_rotation(Q, U, rot_coh_rm_rechan)
        Q, U = self.apply_rotation(Q, U, rotation)
        norm = np.sqrt((Q**2 + U**2) / 2)
        Q /= norm
        U /= norm
        return Q, U

    def get_RMspectrum(self, ibin_vec, isub_vec, rm_perbin=False, sum_stokes_bin=False):

        # isub_data = self.get_isub_data(isub_vec)
        if (isinstance(isub_vec, list)) and (len(isub_vec) == 1):
            isub_data = self.data[isub_vec[0], :, :, :]
        else:
            isub_data = self.data[isub_vec, :, :, :]
        if (len(isub_vec) > 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                isub_data = np.nanmean(isub_data, axis=0)

        if(len(ibin_vec) == 0):
            self.log.error("This should never append (len(ibin_vec) == 0 )...", objet=RMFIT_OBJET)
            return (0, 0, 0 * self.RM_vec)

        if (isinstance(ibin_vec, list)) and (len(ibin_vec) == 1):
            isub_data = isub_data[:, :, ibin_vec[0]]
        else:
            isub_data = isub_data[:, :, ibin_vec]

        if (sum_stokes_bin):
            isub_data = np.sum(isub_data, axis=2)
            isub_data = np.expand_dims(isub_data, axis=2)
            ibin_vec = [0]

        subint_doppler = int(np.floor(np.mean(isub_vec)))

        if(rm_perbin):
            best_RM, RM_sigma, spectra = multiprocessing_RM_specrum_perbin(self.ncore, isub_data, ibin_vec, self.centre_frequency,
                                                                           self.freqs, self.RM_vec, self.get_doppler(subint=subint_doppler), MP=True)
        else:
            best_RM, RM_sigma, spectra = multiprocessing_RM_specrum(self.ncore, isub_data, ibin_vec, self.centre_frequency,
                                                                    self.freqs, self.RM_vec, self.get_doppler(subint=subint_doppler), MP=True)
        return (best_RM, RM_sigma, spectra)

    def set_RM_vec(self, rm):
        self.rmmax = rm + 4000 * self.rmdelt
        self.rmmin = rm - 4000 * self.rmdelt
        self.RM_vec = np.arange(self.rmmin, self.rmmax + self.rmdelt, self.rmdelt)
        RM_zeros_idx = np.where(np.abs(self.RM_vec) < 0.25)[0]
        if(len(RM_zeros_idx) > 0):
            for idx in np.flip(RM_zeros_idx):
                self.RM_vec = np.delete(self.RM_vec, idx)

    def save_RM(self, name='example.csv'):
        with open(name, 'w') as file:
            writer = csv.writer(file)
            # 3. Write data to the file
            writer.writerow(['MJD', 'RM', 'RM_err', 'phase', 'phase_err'])
            for i in range(len(self.scrunch_subint_RM_refining)):
                MJD = self.scrunch_subint_mjd[i]
                RM = self.scrunch_subint_RM_refining[i]
                RM_err = self.scrunch_subint_RM_refining_err[i]
                Phase = self.scrunch_subint_phase_refining[i]
                Phase_err = self.scrunch_subint_phase_refining_err[i]
                writer.writerow([MJD, RM, RM_err, Phase, Phase_err])

    def open_RM(self, csvname):
        open_subint_mjd = []
        open_subint_RM_refining = []
        open_subint_RM_refining_err = []
        open_subint_phase_refining = []
        open_subint_phase_refining_err = []
        with open(csvname) as file:
            reader = csv.reader(file)
            for row in reader:
                ncolomn = len(row)
                try:
                    float(row[0])
                except ValueError:
                    continue  # this is a header
                for i in range(len(row)):
                    if (i == 0):
                        open_subint_mjd.append(float(row[i]))
                    elif (i == 1):
                        open_subint_RM_refining.append(float(row[i]))
                        self.RM_flag = True
                    elif (i == 2):
                        open_subint_RM_refining_err.append(float(row[i]))
                    elif (i == 3):
                        self.absolute_phase_flag = True
                        open_subint_phase_refining.append(float(row[i]))
                    elif (i == 4):
                        open_subint_phase_refining_err.append(float(row[i]))
                if (ncolomn == 2):
                    open_subint_RM_refining_err = 0 * open_subint_RM_refining
                if (ncolomn == 3):
                    open_subint_phase_refining_err = 0 * open_subint_phase_refining_err
        open_subint_mjd = np.array(open_subint_mjd)
        self.mjd_file = open_subint_mjd

        # RM
        self.RM_file = np.array(open_subint_RM_refining)
        self.RM_file_interp = self.interpolate_model(open_subint_mjd, self.RM_file)
        # RM err
        if (ncolomn > 2):
            self.RM_err_file = np.array(open_subint_RM_refining_err)
            self.RM_err_file_interp = self.interpolate_model(open_subint_mjd, self.RM_err_file)
        if (ncolomn > 3):
            # Phase
            self.meanPA_file = np.array(open_subint_phase_refining)
            self.meanPA_file_interp = self.interpolate_model(open_subint_mjd, self.meanPA_file)
            # Phase err
            self.meanPA_err_file = np.array(open_subint_phase_refining_err)
            self.meanPA_err_file_interp = self.interpolate_model(open_subint_mjd, self.meanPA_err_file)

    def interpolate_model(self, mjd, array):
        try:
            interpolate
        except NameError:
            from scipy import interpolate
        try:
            self.subint_mjd
        except AttributeError:
            subint_mjd = np.asarray([i.mjd for i in self.times])
            self.subint_mjd = subint_mjd
        if (len(array) < 2):
            self.log.warning("There is not enough RM value (min 2) to do an interpolation.", objet=RMFIT_OBJET)
            return array
        else:
            model = interpolate.interp1d(mjd, array, fill_value="extrapolate")
            return model(self.subint_mjd)

    def RM_interpolate_result(self):
        # RM
        self.interp_RM_refining = self.interpolate_model(self.scrunch_subint_mjd, self.scrunch_subint_RM_refining)
        # RM err
        self.interp_RM_refining_err = self.interpolate_model(self.scrunch_subint_mjd, self.scrunch_subint_RM_refining_err)
        # Phase
        self.interp_phase_refining = self.interpolate_model(self.scrunch_subint_mjd, self.scrunch_subint_phase_refining)
        # Phase err
        self.interp_phase_refining_err = self.interpolate_model(self.scrunch_subint_mjd, self.scrunch_subint_phase_refining_err)


def RMspectrum(Q_tmp, U_tmp, RM_vec, centre_frequency, freqs, n=None):
    spectrum = np.zeros((np.size(RM_vec)))
    norm = np.sqrt((Q_tmp**2 + U_tmp**2) / 2)
    Q_tmp /= norm
    U_tmp /= norm
    for i in range(len(RM_vec)):
        rmfac = RM_vec[i] * 89875.51787368176
        rot = (rmfac * ((centre_frequency**-2) - (freqs**-2)))
        Q_new = Q_tmp * np.cos(2 * rot) - U_tmp * np.sin(2 * rot)
        U_new = Q_tmp * np.sin(2 * rot) + U_tmp * np.cos(2 * rot)
        spectrum[i] = np.sqrt(np.nansum(Q_new)**2 + np.nansum(U_new)**2)
    # best_RM = RM_vec[np.nanargmax(spectrum)]
    sigma = np.nanmax(spectrum) / mad(spectrum)
    if (np.isnan(sigma)):
        print("MAD=", mad(spectrum))
        print("MAX=", np.nanmax(spectrum))
        print("spectrum=", spectrum)
    del Q_new, U_new, Q_tmp, U_tmp, norm, rot, centre_frequency, freqs
    if n is None:
        return (spectrum, sigma)
    else:
        return (spectrum, sigma, n)


def RMspectum_wrapper(args):
    return RMspectrum(*args)


def multiprocessing_RM_specrum_perbin(ncore, isub_data, ibin_vec, centre_frequency, freqs, RM_vec, doppler, MP=True):
    ibin_spectra = np.zeros((np.size(ibin_vec), np.size(RM_vec)))
    ibin_sigma = np.zeros(np.size(ibin_vec))
    # import matplotlib.pyplot as plt

    RM_vec_dedoppler = RM_vec * (doppler**2)
    if (MP is True):
        pool = Pool(processes=int(ncore))
        multiple_results = pool.imap_unordered(RMspectum_wrapper,
                                               [(isub_data[1, :, n], isub_data[2, :, n], RM_vec_dedoppler, centre_frequency, freqs, n)
                                                for n in range(len(ibin_vec))])
        pool.close()
        for spectrum, sigma, n in multiple_results:
            ibin_spectra[n, :] = spectrum * sigma
            # plt.plot(RM_vec, ibin_spectra[n, :])
        del multiple_results
    else:
        for n in range(len(ibin_vec)):
            ibin_spectra[n, :], ibin_sigma = RMspectum_wrapper((isub_data[1, :, n], isub_data[2, :, n], RM_vec_dedoppler, centre_frequency, freqs))
            ibin_spectra[n, :] *= ibin_sigma

            # plt.plot(RM_vec, ibin_spectra[n, :])
    del isub_data
    best_RM = RM_vec[np.nanargmax(ibin_spectra, axis=1)]
    RMsigma = (np.nanmax(ibin_spectra, axis=1) - np.nanmedian(ibin_spectra, axis=1)) / mad(ibin_spectra, axis=1)
    # plt.show()

    return (best_RM, RMsigma, ibin_spectra)


def multiprocessing_RM_specrum(ncore, isub_data, ibin_vec, centre_frequency, freqs, RM_vec, doppler, MP=True):
    best_RM, RMsigma, ibin_spectra = multiprocessing_RM_specrum_perbin(ncore=ncore, isub_data=isub_data, ibin_vec=ibin_vec,
                                                                       centre_frequency=centre_frequency, freqs=freqs, RM_vec=RM_vec, doppler=doppler, MP=MP)
    spectra = np.nansum(ibin_spectra, axis=0)
    best_RM = RM_vec[np.nanargmax(spectra)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        RMsigma = (np.nanmax(spectra) - np.nanmedian(spectra)) / mad(spectra)

    if (np.isnan(RMsigma)):
        print("MAD=", mad(spectra))
        print("MAX=", np.nanmax(spectra))
        print("MED=", np.nanmedian(spectra))
        print('np.shape(data)=', np.shape(isub_data))
        print('np.nanmean(data)=', np.nanmean(isub_data))
        print('np.nanmax(data)=', np.nanmax(isub_data))
    return best_RM, RMsigma, spectra


if __name__ == "__main__":
    ar = RM_fit_class(ar_name=['/data/lbondonneau/rmme/B0950+08_D20220827T1151_59818_750520_0063_BEAM3.ar.clear'],
                      tscrunch=1, bscrunch=1, minfreq=60, maxfreq=71, defaraday=False, maxtime=20)
    # ar = RM_fit_class(ar_name=['/databf/nenufar-pulsar/DATA/B1919+21/PSR/B1919+21_D20220909T1948_59831_252376_0068_BEAM1.fits'], tscrunch=2, bscrunch=2)
    # ar = RM_fit_class(ar_name=['/home/lbondonneau/NenuPlot_dev/B1919+21_D20220909T1948_59831_252376_0068_BEAM1.ar.clear'], maxtime=20, defaraday=False)
    # ar = RM_fit_class(ar_name=['/databf/nenufar-pulsar/DATA/B0329+54/PSR/B0329+54_D20220910T0402_59832_002597_0035_BEAM0.fits',
    #                            '/databf/nenufar-pulsar/DATA/B0329+54/PSR/B0329+54_D20220910T0402_59832_252388_0072_BEAM1.fits'],
    #                   tscrunch=4, bscrunch=2, maxfreq=85, minfreq=40)
    # ar = RM_fit_class(ar_name=['/databf/nenufar-pulsar/DATA/B0919+06/PSR/B0919+06_D20220914T0802_59836_002609_0048_BEAM0.fits'],
    #                    tscrunch=1, bscrunch=2, maxfreq=85, minfreq=40)

    # ar = RM_fit_class(ar_name=[#'/databf/nenufar-pulsar/DATA/B1133+16/PSR/B1133+16_D20221006T1111_59858_002683_0028_BEAM0.fits',
    #                           '/databf/nenufar-pulsar/DATA/B1133+16/PSR/B1133+16_D20221006T1111_59858_252471_0065_BEAM1.fits'],
    #                           tscrunch=2, bscrunch=2, maxfreq=85, minfreq=30, defaraday=False, maxtime=30)

    # ar = RM_fit_class(ar_name=['/databf/nenufar-pulsar/DATA/B1839+56/PSR/B1839+56_D20220909T1905_59831_002583_0035_BEAM0.fits',
    #                            '/databf/nenufar-pulsar/DATA/B1839+56/PSR/B1839+56_D20220909T1905_59831_252374_0072_BEAM1.fits'],
    #                            tscrunch=2, bscrunch=2, maxfreq=85, minfreq=25, defaraday=False)
    # ar = RM_fit_class(ar_name=['/databf/nenufar-pulsar/DATA/J2145-0750/PSR/J2145-0750_D20221005T2001_59857_002674_0029_BEAM0.fits',
    #                           '/databf/nenufar-pulsar/DATA/J2145-0750/PSR/J2145-0750_D20221005T2001_59857_252462_0067_BEAM1.fits'],
    #                           tscrunch=4, bscrunch=8, maxfreq=70, minfreq=40, defaraday=False)

    # ar = RM_fit_class(ar_name=['/databf2/artemis/FR606-2014B-03-Griessmeier/J2145-0750/raw/J2145-0750_D20211130T180301_122_133.ar',
    #                            '/databf2/artemis/FR606-2014B-03-Griessmeier/J2145-0750/raw/J2145-0750_D20211130T180301_122_255.ar',
    #                            '/databf2/artemis/FR606-2014B-03-Griessmeier/J2145-0750/raw/J2145-0750_D20211130T180301_122_377.ar',
    #                            '/databf2/artemis/FR606-2014B-03-Griessmeier/J2145-0750/raw/J2145-0750_D20211130T180301_122_499.ar'],
    #                           tscrunch=4, bscrunch=4, maxfreq=200, minfreq=110, defaraday=False)

    # ar = RM_fit_class(ar_name=["/home/lbondonneau/data/rmme/2021-11-29-16:28.ar"],
    #                            tscrunch=1, bscrunch=1, maxfreq=141, minfreq=131, defaraday=False)

    # ar = RM_fit_class(ar_name=['/databf2/nenufar-pulsar/DATA/B0950+08/PSR/B0950+08_D20220715T1001_59775_002457_0055_BEAM0.fits'],
    #                  tscrunch=4, bscrunch=4, maxfreq=85, minfreq=35, maxtime=42, defaraday=False)

    # ar = RM_fit_class(ar_name=['/databf/nenufar-pulsar/DATA/B1237+25/PSR/B1237+25_D20221006T1211_59858_002684_0029_BEAM0.fits',
    #                            '/databf/nenufar-pulsar/DATA/B1237+25/PSR/B1237+25_D20221006T1211_59858_252472_0067_BEAM1.fits'],
    #                   tscrunch=2, bscrunch=4, maxfreq=85, minfreq=40, defaraday=False)

    # ar = RM_fit_class(ar_name=['/databf/nenufar-pulsar/DATA/B0943+10/PSR/B0943+10_D20220927T1111_59849_002671_0032_BEAM0.fits',
    #                           '/databf/nenufar-pulsar/DATA/B0943+10/PSR/B0943+10_D20220927T1111_59849_252459_0069_BEAM1.fits'],
    #                            defaraday=False, tscrunch=2, bscrunch=4, maxfreq=85, minfreq=40)
    ar.clean(fast=False, bad_subint=1.0, bad_chan=1.0)
    ar.DM_fit(ncore=32, plot=False)
    ar.auto_rebin()

    # ar = RM_fit_class(ar_name=['/home/lbondonneau/data/rmme/tmp.ar.clear'], defaraday=False, maxtime=20)
    # ar.unload('/home/lbondonneau/data/rmme/tmp.ar.clear')

    ar.set_times()

    # from plot_class import PlotArchive
    # import matplotlib.pyplot as plt
    # plot_ar = PlotArchive(ar)
    # ax0 = plt.subplot2grid((5, 5), (0, 0), colspan=4, rowspan=4)
    # plot_ar.phase_freq(ax0, leftaxis=True)
    # plt.show()
    ar.init_RM_fit()

    ar.RM_reduction(only_bestbin=False, sum_stokes_bin=False, QU_fit=True)
    ar.RM_refining(sum_stokes_bin=True)
    ar.RM_interpolate_result()
    ar.save_RM()
    ar.open_RM()

    import matplotlib.pyplot as plt
    mjd = ar.scrunch_subint_mjd
    rm = ar.scrunch_subint_RM
    rm_refine = ar.scrunch_subint_RM_refining
    rm_refine_err = ar.scrunch_subint_RM_refining_err
    phase_refine = ar.scrunch_subint_phase_refining
    phase_refine_err = ar.scrunch_subint_phase_refining_err
    phase_refine2 = ar.scrunch_subint_phase_refining + rm_refine * 89875.51787368176 * (ar.freqs[-1]**-2)
    phase_refine_err2 = ar.scrunch_subint_phase_refining_err + rm_refine_err * 89875.51787368176 * (ar.freqs[-1]**-2)

    fig = plt.figure(figsize=(12, 6))
    ax0 = plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=1)
    ax1 = plt.subplot2grid((2, 1), (1, 0), colspan=1, rowspan=1, sharex=ax0)
    ax0.plot(mjd, rm, 'g--', label='RM [rad.m-2] (RM spectrum)')
    ax0.errorbar(mjd, rm_refine, yerr=rm_refine_err, fmt='b+', label='RM [rad.m-2] (QU fit)')
    ax0.errorbar(ar.subint_mjd_center, ar.interp_RM_refining, yerr=ar.interp_RM_refining_err, fmt='r+', label='RM [rad.m-2] (interpolation)', alpha=0.3)
    ax0.legend(loc='upper right')
    ax0.set_xlabel('Time (MJD)')
    ax0.set_ylabel('RM (rad.m-2)')

    def rad_to_deg(rad):
        return ((((rad * 180. / np.pi) + 180) / 2.) % 180) - 90

    ax1.errorbar(mjd, rad_to_deg(phase_refine), yerr=rad_to_deg(phase_refine_err), fmt='r+', label='Phase [rad]')
    ax1.errorbar(mjd, rad_to_deg(phase_refine2), yerr=rad_to_deg(phase_refine_err2), fmt='b+', label='abs Phase [rad]')
    ax1.errorbar(ar.subint_mjd_center,
                 rad_to_deg(ar.interp_phase_refining + ar.interp_RM_refining * 89875.51787368176 * (ar.freqs[-1]**-2)),
                 yerr=rad_to_deg(ar.interp_phase_refining_err + ar.interp_RM_refining_err * 89875.51787368176 * (ar.freqs[-1]**-2)),
                 fmt='r+', label='abs Phase [rad] (interp)', alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Time (MJD)')
    ax1.set_ylabel('Phase (Degree)')
    ax1.set_ylim([-100, 100])

    # for i in range(len(ar.scrunch_subint_mjd)):
    #    mjd = [ar.scrunch_subint_mjd[i], ar.scrunch_subint_mjd[i] + ar.scrunch_subint_dur[i] / 86400.]
    #    rm = [ar.scrunch_subint_RM[i], ar.scrunch_subint_RM[i]]
    #    sigma = [ar.scrunch_subint_sigma[i], ar.scrunch_subint_sigma[i]]
    #    plt.plot(mjd, rm, '-')
    #    plt.plot(mjd, sigma, '-o')
    #    plt.xlabel("Time (MJD)")
    #    plt.ylabel("RM (rad.m-2)")
    plt.show()

    exit(0)

    # rm_vec = np.zeros(ar.get_nsubint())
    # rm_sigma_vec = np.zeros(ar.get_nsubint())
    # for isub in range(len(rm_vec)):
    #     print(float(isub) / len(rm_vec))
    #     rm_vec[isub], rm_sigma_vec[isub] = ar.compute_subint(isub, rm_perbin=False, only_bestbin=True)
    # import matplotlib.pyplot as plt
    # ar.set_times()
    # plt.plot(ar.times.mjd, rm_vec)
    # plt.show()
    print(ar.compute_subint(range(tscrunch), rm_perbin=True))
    print(ar.compute_subint(range(tscrunch + 1, ar.get_nsubint()), rm_perbin=True))
    exit(0)

    import matplotlib.pyplot as plt
    ar.set_times()
    for isub in range(ar.get_nsubint()):
        print(float(isub) / ar.get_nsubint())
        rm_vec, rm_sigma_vec, ibin_vec = ar.compute_subint(isub, rm_perbin=True)
        plt.plot([ar.times.mjd[isub]], np.mean(rm_sigma_vec), '-o')
        plt.plot([ar.times.mjd[isub]] * len(rm_vec), rm_vec, '-')
    plt.show()
