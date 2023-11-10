import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
# from matplotlib.ticker import ScalarFormatter
# import seaborn as sns
# sns.set()
# from mypsrchive import psrchive_class
# from DM_fit_class import DM_fit_class
from RM_fit_class import RM_fit_class
from astropy.coordinates import Angle
import astropy.units as u

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
PLOT_OBJET = 'plot'


# class PlotArchive(psrchive_class):
class PlotArchive(RM_fit_class):
    def __init__(self, *args, **kwargs):
        super(PlotArchive, self).__init__(*args, **kwargs)

    def profil(self, AX, stokes=True, color=None):
        """
        plot the profil I L V in the requested area AX
        """
        if (self.verbose):
            self.log.log("plotting 1D profil", objet=PLOT_OBJET)
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        if (color is None):
            color = ['k', 'r', 'b', 'm', 'g']
        if not (arx.get_dedispersed()):
            arx.dedisperse()
        arx.tscrunch()
        # arx.flattenBP()
        arx.fscrunch()
        arx.remove_baseline()
        phase = np.linspace(0, 1, arx.get_nbin())
        if arx.get_npol() > 1:
            if (stokes):
                arx.convert_state('Stokes')
                data = arx.get_data()
                data = data.squeeze()
                AX.plot(phase, data[0, :],
                        color[0], alpha=0.75, label='Total Intensity')
                AX.plot(phase, np.sqrt((data[1, :])**2 + (data[2, :])**2),
                        color[1], alpha=0.6, label='Linear polarization')
                AX.plot(phase, data[3, :],
                        color[2], alpha=0.6, label='Circular polarization')
                AX.legend(loc='upper right')
            else:
                arx.convert_state('Coherence')
                data = arx.get_data()
                data = data.squeeze()
                AX.plot(phase, data[0, :],
                        color[0], alpha=0.75, label='XX')
                AX.plot(phase, data[1, :],
                        color[1], alpha=0.6, label='YY')
                AX.plot(phase, data[2, :],
                        color[2], alpha=0.6, label='Re(X*Y)')
                AX.plot(phase, data[3, :],
                        color[3], alpha=0.6, label='Im(X*Y)')
                AX.legend(loc='upper right')
        else:
            data = arx.get_data()
            data = data.squeeze()
            AX.plot(phase, data, color[0])
        AX.grid(True, which="both", ls="-", alpha=0.65)
        AX.set_xlabel('Pulse Phase')
        AX.set_ylabel('Amplitude (AU)')

    def __ticks_format_func(self, value, tick_number):
        if (value == 0):
            return ''
        else:
            value = str("%.6f" % (value))
            while(value[-1] == '0'):
                value = value[0:-1]
            if(value[-1] == '.'):
                value = value[0:-1]
            return value

    def phase_freq(self, AX, pol=0, rightaxis=False, flatband=True, stokes=True, threshold=False, nchan=None, nbin=None):
        """
        Phase vs Freq plot with polarization selection in AX
        baseline is removed with a remove_baseline
        and the signal is adjusted by 1/baseline
        """
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()

        if not (arx.get_dedispersed()):
            arx.dedisperse()
        arx.tscrunch()

        if not (nchan is None):
            arx.myfscrunch_to_nchan(int(nchan))
        if not (nbin is None):
            arx.mybscrunch_to_nbin(int(nbin))

        if arx.get_npol() > 1:
            if(stokes):
                if (pol == 0) and (self.verbose):
                    self.log.log("plotting phase/freq I profil", objet=PLOT_OBJET)
                elif (pol == 1) and (self.verbose):
                    self.log.log("plotting phase/freq Q profil", objet=PLOT_OBJET)
                elif (pol == 2) and (self.verbose):
                    self.log.log("plotting phase/freq U profil", objet=PLOT_OBJET)
                elif (pol == 3) and (self.verbose):
                    self.log.log("plotting phase/freq V profil", objet=PLOT_OBJET)
                arx.convert_state('Stokes')
            else:
                if (pol == 0) and (self.verbose):
                    self.log.log("plotting phase/freq XX profil", objet=PLOT_OBJET)
                elif (pol == 1) and (self.verbose):
                    self.log.log("plotting phase/freq YY profil", objet=PLOT_OBJET)
                elif (pol == 2) and (self.verbose):
                    self.log.log("plotting phase/freq XY profil", objet=PLOT_OBJET)
                elif (pol == 3) and (self.verbose):
                    self.log.log("plotting phase/freq YX profil", objet=PLOT_OBJET)
        else:
            if (self.verbose):
                self.log.log("plotting phase/freq I profil", objet=PLOT_OBJET)

        # min and max freq for the extent in imshow
        min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
        max_freq = arx.get_Profile(0, 0, arx.get_nchan() - 1).get_centre_frequency()

        # rescale the baseline
        try:
            arx2 = arx.myclone()
        except AttributeError:
            arx2 = arx.clone()
        arx2.pscrunch()

        # weights
        weights = arx.get_weights()
        weights = weights.squeeze()
        weights = weights / np.max(weights)

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
                        prof.scale(1 / bl_mean[ichan])
                    else:
                        prof.set_weight(0.0)
                        prof.scale(0)
        data = arx.get_data()
        data = data[:, pol, :, :].squeeze()
        if threshold:
            threshold = float(threshold)
            if (threshold > 0):
                std_data = np.nanstd(data)
                ind = np.where(data > threshold * std_data)
                data[ind] = threshold * std_data
                ind = np.where(data < -threshold * std_data)
                data[ind] = -threshold * std_data

        for abin in range(arx.get_nbin()):
            data[:, abin] = data[:, abin] * weights

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
        AX.text(0.95, min_freq + 0.90 * (max_freq - min_freq), string[pol],
                horizontalalignment='right',
                verticalalignment='top',
                fontdict={'family': 'DejaVu Sans Mono'},
                size=14,
                color='k',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
        AX.set_xlabel('Pulse Phase')
        AX.set_ylabel('Frequency (MHz)')
        AX.xaxis.set_major_locator(plt.MaxNLocator(2))
        AX.xaxis.set_major_formatter(plt.FuncFormatter(self.__ticks_format_func))
        if (rightaxis):
            AX.yaxis.tick_right()
            AX.yaxis.set_ticks_position('both')
            AX.yaxis.set_label_position("right")
            # AX.yaxis.set_ticks(np.arange(0, self.get_nsubint() - 1,
            #                                    self.get_nsubint() / 5))

    def phase_freq_no_ON(self, AX, pol=0, leftaxis=False, flatband=True, normesubint=False, nchan=None, nbin=None):
        """
        Phase vs Freq plot with polarization selection in AX
        All signal > 1.5 and  < -1.5 x the standar deviation is masked
        baseline is removed with a remove_baseline
        and the signal is adjusted by 1/baseline
        pol=4 is for the total intensity
        """
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        if not (arx.get_dedispersed()):
            arx.dedisperse()
        if pol == 4:
            arx.pscrunch()
        arx.tscrunch()

        if not (nchan is None):
            arx.myfscrunch_to_nchan(int(nchan))
        if not (nbin is None):
            arx.mybscrunch_to_nbin(int(nbin))

        # min and max freq for the extent in imshow
        min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
        max_freq = arx.get_Profile(0, 0, self.get_nchan() - 1).get_centre_frequency()

        # rescale the baselin
        try:
            arx2 = arx.myclone()
        except AttributeError:
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
                        prof.scale(1 / bl_mean[ichan])
                    else:
                        prof.set_weight(0.0)
                        prof.scale(0)
        arx.bscrunch_to_nbin(128)
        if arx.get_nchan() > 64:
            arx.fscrunch(arx.get_nchan() / 64)
        data = arx.get_data()
        if pol == 4:
            data = data[:, 0, :, :].squeeze()
        else:
            data = data[:, pol, :, :].squeeze()
        data = np.flipud(data)
        for i in range(2):
            if np.nanmax(data) > 1.5 * np.nanstd(data):
                data[np.where(data > 1.5 * np.nanstd(data))] = np.nanmedian(data)
            if np.nanmin(data) < -1.5 * np.nanstd(data):
                data[np.where(data < -1.5 * np.nanstd(data))] = np.nanmedian(data)
        if (normesubint):
            for subint in range(arx.get_nsubint()):
                data[subint, :] = data[subint, :] / np.nanmax(data[subint, :])
        AX.imshow(data, interpolation='none', cmap='afmhot',
                  extent=[0, 1, min_freq, max_freq], aspect='auto')
        string = ['xx', 'yy', 'xy', 'yx', 'I']
        AX.text(0.95, min_freq + 0.90 * (max_freq - min_freq), string[pol],
                horizontalalignment='right',
                verticalalignment='top',
                fontdict={'family': 'DejaVu Sans Mono'},
                size=14,
                color='k',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
        AX.set_xlabel('Pulse Phase')
        AX.set_ylabel('Frequency (MHz)')
        if leftaxis:
            AX.yaxis.tick_right()
            AX.yaxis.set_ticks_position('both')
            AX.set_ylim(top=self.get_nchan())
            AX.set_ylabel('Channels')
            AX.yaxis.set_label_position("right")

    def phase_time(self, AX, pol=0, rightaxis=False, stokes=True, timenorme=False, threshold=False, nsub=None, nbin=None, rmprof2d=None):
        """
        plot phase time prof in area AX
        the polarization can be selected
        """
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        if not (arx.get_dedispersed()):
            arx.dedisperse()
        arx.remove_baseline()
        if not (rmprof2d is None):
            arx.remove_profile2d()
        arx.fscrunch()
        if not (nsub is None):
            arx.mytscrunch_to_nsub(int(nsub))
        if not (nbin is None):
            arx.mybscrunch_to_nbin(int(nbin))

        # print('icci')
        if(arx.get_npol() > 1):
            # print('icci')
            if(stokes):
                # print(arx.get_state())
                arx.convert_state('Stokes')

        if arx.get_npol() > 1:
            if(stokes):
                if (pol == 0) and (self.verbose):
                    self.log.log("plotting phase/time I profil", objet=PLOT_OBJET)
                elif (pol == 1) and (self.verbose):
                    self.log.log("plotting phase/time Q profil", objet=PLOT_OBJET)
                elif (pol == 2) and (self.verbose):
                    self.log.log("plotting phase/time U profil", objet=PLOT_OBJET)
                elif (pol == 3) and (self.verbose):
                    self.log.log("plotting phase/time V profil", objet=PLOT_OBJET)
                arx.convert_state('Stokes')
            else:
                if (pol == 0) and (self.verbose):
                    self.log.log("plotting phase/time XX profil", objet=PLOT_OBJET)
                elif (pol == 1) and (self.verbose):
                    self.log.log("plotting phase/time YY profil", objet=PLOT_OBJET)
                elif (pol == 2) and (self.verbose):
                    self.log.log("plotting phase/time XY profil", objet=PLOT_OBJET)
                elif (pol == 3) and (self.verbose):
                    self.log.log("plotting phase/time YX profil", objet=PLOT_OBJET)
        else:
            if (self.verbose):
                self.log.log("plotting phase/time I profil", objet=PLOT_OBJET)
        tsubint = arx.integration_length() / arx.get_nsubint()

        # weights
        weights = arx.get_weights()
        weights = weights.squeeze()
        weights = weights / np.max(weights)

        data = arx.get_data()
        data = data[:, pol, 0, :]

        for abin in range(arx.get_nbin()):
            data[:, abin] = data[:, abin] * weights

        if (threshold):
            threshold = float(threshold)
            if (threshold > 0):
                std_data = np.nanstd(data)
                ind = np.where(data > threshold * std_data)
                data[ind] = threshold * std_data
                ind = np.where(data < -threshold * std_data)
                data[ind] = -threshold * std_data

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
                if(np.nanmax(np.abs(data[subint, :])) != 0):
                    data[subint, :] = data[subint, :] / np.nanmax(np.abs(data[subint, :]))

        fig = AX.imshow(data, interpolation='none', cmap=cmap,
                        aspect='auto', extent=[0, 1, 0, arx.get_nsubint() * tsubint / 60.])
        if (pol > 0):
            lim = np.max(np.abs(data))
            fig.set_clim(-lim, lim)
        if(stokes):
            string = ['I', 'Q', 'U', 'V']
        else:
            string = ['XX', 'YY', 'XY', 'YX']
        AX.text(0.95, 0.90 * (arx.get_nsubint() * tsubint / 60.), string[pol],
                horizontalalignment='right',
                verticalalignment='top',
                fontdict={'family': 'DejaVu Sans Mono'},
                size=10, color='k',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
        AX.set_xlabel('Pulse phase')
        # AX.yaxis.set_ticks(np.arange(0, self.get_nsubint() - 1,
        #                                    self.get_nsubint() / 5))
        if (rightaxis):
            AX.yaxis.tick_right()
            AX.yaxis.set_ticks_position('both')
            AX.yaxis.set_label_position("right")
            AX.set_ylabel('Time (min)')
        else:
            AX.set_ylabel('Time (minutes)')

    def bandpass(self, AX, botaxis=False, mask=False, rightaxis=False):
        """
        plot bandpass xx and yy in AX
        """
        if (self.verbose):
            self.log.log("plotting bandpass", objet=PLOT_OBJET)
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        arx.tscrunch()
        if (arx.get_nbin() > 16):
            arx.bscrunch_to_nbin(16)
        if(mask):
            weights = arx.get_weights()
            weights = weights.squeeze()
            weights = weights / np.max(weights)

        min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
        max_freq = arx.get_Profile(0, 0, arx.get_nchan() - 1).get_centre_frequency()
        freqs = np.linspace(min_freq, max_freq, arx.get_nchan())

        subint = arx.get_Integration(0)
        (bl_mean, bl_var) = subint.baseline_stats()
        bl_mean = bl_mean.squeeze()
        POL = ['xx', 'yy', 'xy', 'yx']
        color = ['b', 'r', 'g', 'm']
        if (arx.get_npol() > 1):
            for Xpol in range(arx.get_npol()):
                if(Xpol < 2):
                    if(mask):
                        bl_mean[Xpol, np.where(weights[:] <= 0.5)] = np.nan
                    AX.semilogy(freqs, bl_mean[Xpol, :],
                                color[Xpol], alpha=0.5,
                                label='Polarization ' + POL[Xpol])
        else:
            if(mask):
                bl_mean[np.where(weights[:] <= 0.5)] = np.nan
            AX.semilogy(freqs, bl_mean[:], color[0], alpha=0.5, label='Total intensity')

        if(mask):
            trans = mtransforms.blended_transform_factory(AX.transData,
                                                          AX.transAxes)
            if(mask):
                AX.fill_between(freqs, 0, 1,
                                where=weights[:] <= 0.5, facecolor='k',
                                alpha=0.6, transform=trans, label='masked channels')
        AX.legend(loc='upper right', fontsize=5)
        AX.grid(True, which="both", ls="-", alpha=0.75)
        AX.set_ylabel('Amplitude (AU)')
        if (botaxis):
            AX_secondary = AX.twiny()
            AX_secondary.set_frame_on(True)
            AX_secondary.patch.set_visible(False)
            AX_secondary.xaxis.set_ticks_position('bottom')
            AX_secondary.set_xlabel('Channels')
            AX_secondary.xaxis.set_label_position('bottom')
            AX_secondary.spines['bottom'].set_position(('outward', 50))
            AX_secondary.set_xlim(1, self.get_nchan())
            AX.set_xlim(min_freq, max_freq)
        if (rightaxis):
            AX.yaxis.tick_right()
            AX.yaxis.set_ticks_position('both')
            AX.yaxis.set_label_position("right")
            AX.set_xlabel('Freq (MHz)')
        else:
            AX.set_xlabel('Frequency (MHz)')

    def zaped_bandpass(self, AX, botaxis=False, rightaxis=False):
        """
        plot a no clean bandpass xx and yy in AX
        """
        if (self.verbose):
            self.log.log("plotting RFI mitigated bandpass", objet=PLOT_OBJET)
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        arx.tscrunch()

        weights = arx.get_weights()
        weights = weights.squeeze()
        weights = weights / np.max(weights)

        min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
        max_freq = arx.get_Profile(0, 0, arx.get_nchan() - 1).get_centre_frequency()
        freqs = np.linspace(min_freq, max_freq, arx.get_nchan())
        zaped_bandpass = np.copy(weights)
        zaped_bandpass[np.where(weights[:] <= 0.01)] = np.nan
        zaped_bandpass = 1 - zaped_bandpass
        AX.plot(freqs, 100. * zaped_bandpass, 'b',
                alpha=0.5, label='Zaped bandpass')
        # AX.yaxis.set_major_locator(plt.MaxNLocator(2))
        AX.yaxis.set_major_formatter(plt.FuncFormatter(self.__ticks_format_func))
        AX.fill_between(freqs, -1, 1000, where=weights <= 0.5,
                        facecolor='k', alpha=0.6, label='masked channels')
        AX.legend(loc='upper right')
        AX.grid(True, which="both", ls="-", alpha=0.75)
        AX.set_xlabel('Frequency (MHz)')
        AX.set_ylabel('masked channels (%)')
        AX.set_ylim(0, 100)
        # AX.ticklabel_format(axis="y", style="plain")
        if (botaxis):
            AX_secondary = AX.twiny()
            AX_secondary.set_frame_on(True)
            AX_secondary.patch.set_visible(False)
            AX_secondary.xaxis.set_ticks_position('bottom')
            AX_secondary.set_xlabel('Channels')
            AX_secondary.xaxis.set_label_position('bottom')
            AX_secondary.spines['bottom'].set_position(('outward', 50))
            AX_secondary.set_xlim(1, self.get_nchan())
            AX.set_xlim(min_freq, max_freq)
        if (rightaxis):
            AX.yaxis.tick_right()
            AX.yaxis.set_ticks_position('both')
            AX.yaxis.set_label_position("right")

    def dynaspect_bandpass(self, AX, left_onpulse=None, righ_onpulse=None, leftaxis=False, botaxis=False, flatband=True, threshold=False):
        """
        plot dynamic_spect plot of the baseline in AX
        with a correction of the bandpass
        """
        if (self.verbose):
            self.log.log("plotting dynamic spectrum bandpass", objet=PLOT_OBJET)
        if (left_onpulse is None) and (righ_onpulse is None):
            try:
                left_onpulse, righ_onpulse = self.get_on_window(safe_fraction=0.05, rebuild_local_scrunch=False)
            except AttributeError:
                left_onpulse, righ_onpulse = 0, 0
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        arx.pscrunch()
        if not (arx.get_dedispersed()):
            arx.dedisperse()

        if(left_onpulse == 0) and (righ_onpulse == 0):
            if(arx.get_nbin() > 16):
                arx.bscrunch_to_nbin(16)

        try:
            arx2 = arx.myclone()
        except AttributeError:
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
        weights = weights / np.max(weights)

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
                offbins[left_onpulse: righ_onpulse] = False
            else:
                offbins[left_onpulse:] = False
                offbins[: righ_onpulse] = False
            # data = np.median(arx.get_data()[:, :, :, offbins], axis=3).squeeze()
            data = arx.get_data()[:, 0, :, :]
            data = np.median(data[:, :, offbins], axis=2)
            # print(left_onpulse, righ_onpulse)
            # plt.plot(np.mean(np.mean(np.mean(arx.get_data()*weights, axis=0), axis=0), axis=0))
            # plt.show()

        if (arx.get_nsubint() == 1):
            data = np.repeat(data * weights, 2, axis=0)
        data = np.rot90(data * weights)
        data = data - np.nanmean(data)

        DATAstd = np.nanstd(data)

        if threshold:
            sigma = float(threshold)
        else:
            sigma = 3.

        if(np.nanmax(data) > DATAstd * sigma):
            isub, ichan = np.where(data > DATAstd * sigma)
            data[isub, ichan] = DATAstd * sigma
        if(np.nanmin(data) < -DATAstd * sigma):
            isub, ichan = np.where(data < -DATAstd * sigma)
            data[isub, ichan] = -DATAstd * sigma

        data = data - np.nanmean(data)

        # percent1 = np.nanpercentile(data, 2)
        # percent50 = np.nanpercentile(data, 50)
        # percent99 = np.nanpercentile(data, 98)
        # test_data = np.copy(data)
        # test_data[np.isnan(test_data)] = percent50
        # test_data[np.isinf(test_data)] = percent50
        # condition1 = (test_data < percent1)
        # condition2 = (test_data > percent99)
        # if np.any(condition1): data[np.where( condition1 )] = percent1
        # if np.any(condition2): data[np.where( condition2 )] = percent99

        min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
        max_freq = arx.get_Profile(0, 0, arx.get_nchan() - 1).get_centre_frequency()
        AX.imshow(data, interpolation='nearest', cmap='afmhot',
                  aspect='auto',
                  extent=[0, arx.get_nsubint() * tsubint / 60.,
                          min_freq, max_freq])
        string = 'Bandpass'
        AX.text(0.95 * arx.get_nsubint() * tsubint / 60.,
                min_freq + 0.90 * (max_freq - min_freq), string,
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
            AXchan.set_ylim(top=self.get_nchan())
            AXchan.set_ylabel('Channels')
        if (botaxis):
            AX_secondary = AX.twiny()
            AX_secondary.set_frame_on(True)
            AX_secondary.patch.set_visible(False)
            AX_secondary.xaxis.set_ticks_position('bottom')
            AX_secondary.set_xlabel('Subintegration')
            AX_secondary.xaxis.set_label_position('bottom')
            AX_secondary.spines['bottom'].set_position(('outward', 50))
            AX_secondary.set_xlim(1, self.get_nsubint())
            AX.set_xlim(0, self.integration_length() / 60.)

    def dynaspect_onpulse(self, AX, left_onpulse=None, righ_onpulse=None, leftaxis=False, flatband=True, botaxis=False, threshold=3):
        """
        plot dynamic_spect plot of the onpulse
        note: it's not the real onpulse, but the highest bin on 64bin
        """
        if (self.verbose):
            self.log.log("plotting on-pulse dynamic spectrum", objet=PLOT_OBJET)
        if (left_onpulse is None) and (righ_onpulse is None):
            try:
                left_onpulse, righ_onpulse = self.get_on_window(safe_fraction=0.05, rebuild_local_scrunch=False)
            except AttributeError:
                left_onpulse, righ_onpulse = 0, 0

        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        if not (arx.get_dedispersed()):
            arx.dedisperse()
        arx.pscrunch()
        # arx.remove_baseline()
        if(left_onpulse == 0) and (righ_onpulse == 0):
            if(arx.get_nbin() > 32):
                arx.bscrunch_to_nbin(32)
        arx2 = arx.clone()
        try:
            arx2 = arx.myclone()
        except AttributeError:
            arx2 = arx.clone()
        arx2.tscrunch()

        weights = self.get_weights()
        weights = weights.squeeze()
        weights = weights / np.max(weights)

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
        max_freq = arx.get_Profile(0, 0, arx.get_nchan() - 1).get_centre_frequency()

        if(left_onpulse is 0) and (righ_onpulse is 0):
            data = np.max(arx.get_data(), axis=3).squeeze() - np.min(arx.get_data(), axis=3).squeeze()
        else:
            nbins = arx.get_nbin()
            onbins = np.zeros(nbins, dtype='bool')
            offbins = np.ones(nbins, dtype='bool')
            if(left_onpulse < righ_onpulse):
                onbins[left_onpulse: righ_onpulse] = True
                offbins[left_onpulse: righ_onpulse] = False
            else:
                onbins[left_onpulse:] = True
                onbins[: righ_onpulse] = True
                offbins[left_onpulse:] = False
                offbins[: righ_onpulse] = False
            data = arx.get_data()
            dataON = data[:, 0, :, :]
            dataOFF = data[:, 0, :, :]
            dataON = dataON[:, :, onbins]
            dataOFF = dataOFF[:, :, offbins]
            data = np.mean(dataON, axis=2) - np.median(dataOFF, axis=2)
            # print(left_onpulse, righ_onpulse)
            # data = arx.get_data()
            # print(np.shape(data))
            # nsubint, npol, nchan, nbin = np.shape(data)
            # for ibin in range(nbin):
            #    data[:, 0, :, ibin] = data[:, 0, :, ibin]*weights
            # plt.plot(np.mean(np.mean(data[:, 0, :, :], axis=0), axis=0))
            # plt.show()
            # exit(0)
        if (arx.get_nsubint() == 1):
            data = np.repeat(data * weights, 2, axis=0)
        data = np.rot90(data * weights)

        data = data - np.nanmean(data)

        DATAstd = np.nanstd(data)

        if threshold:
            sigma = float(threshold)
        else:
            sigma = 3.

        if(np.nanmax(data) > DATAstd * sigma):
            isub, ichan = np.where(data > DATAstd * sigma)
            data[isub, ichan] = DATAstd * sigma
        if(np.nanmin(data) < -DATAstd * sigma):
            isub, ichan = np.where(data < -DATAstd * sigma)
            data[isub, ichan] = -DATAstd * sigma

        data = data - np.nanmean(data)

        AX.imshow(data, interpolation='none', cmap='afmhot',
                  aspect='auto',
                  extent=[0, arx.get_nsubint() * tsubint / 60., min_freq, max_freq])
        string = 'ON pulse'
        AX.text(0.95 * arx.get_nsubint() * tsubint / 60.,
                min_freq + 0.90 * (max_freq - min_freq), string,
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
            AXchan.set_ylim(top=self.get_nchan())
            AXchan.set_ylabel('Channels')
        if (botaxis):
            AX_secondary = AX.twiny()
            AX_secondary.set_frame_on(True)
            AX_secondary.patch.set_visible(False)
            AX_secondary.xaxis.set_ticks_position('bottom')
            AX_secondary.set_xlabel('Subintegration')
            AX_secondary.xaxis.set_label_position('bottom')
            AX_secondary.spines['bottom'].set_position(('outward', 50))
            AX_secondary.set_xlim(1, self.get_nsubint())
            AX.set_xlim(0, self.integration_length() / 60.)

    def snr_vs_frequency(self, AX, botaxis=False):
        """
        plot signal noise ratio versus frequency
        """
        if (self.verbose):
            self.log.log("plotting snr vs frequency channels", objet=PLOT_OBJET)
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        arx.pscrunch()
        arx.tscrunch()
        min_freq = arx.get_Profile(0, 0, 0).get_centre_frequency()
        max_freq = arx.get_Profile(0, 0, arx.get_nchan() - 1).get_centre_frequency()

        SNR = np.zeros(arx.get_nchan())
        for chan in range(arx.get_nchan()):
            prof = arx.get_Profile(0, 0, chan)
            SNR[chan] = prof.snr()

        channels = np.linspace(1, self.get_nchan(), self.get_nchan())
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
            AX.set_xlim(1, self.get_nchan())

    def snr_vs_subintegration(self, AX, botaxis=False):
        """
        plot signal noise ratio in each subintegration
        """
        if (self.verbose):
            self.log.log("plotting snr vs time subintegrations", objet=PLOT_OBJET)
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        arx.pscrunch()
        arx.fscrunch()

        SNR = np.zeros(arx.get_nsubint())
        for isub in range(arx.get_nsubint()):
            prof = arx.get_Profile(isub, 0, 0)
            SNR[isub] = prof.snr()

        subintegration = np.linspace(1, self.get_nsubint(), self.get_nsubint())
        AX.plot(subintegration, SNR)
        AX.set_ylabel('Signal noise ratio')
        AX.set_xlabel('Subintegration')
        string = 'SNR per suintegration'
        AX.text(0.95 * arx.get_nsubint(),
                np.min(SNR) + 0.90 * (np.max(SNR) - np.min(SNR)), string,
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
            AX_secondary.set_xlim(0, self.integration_length() / 60.)
            AX.set_xlim(1, self.get_nsubint())

    def RM_vs_time(self, AX, rightaxis=False):
        """
        plot RM vs time
        """
        if (self.verbose):
            self.log.log("plotting RM vs time", objet=PLOT_OBJET)
        time_ref = 0
        try:
            time = self.this.scrunch_subint_mjd
            time = (time - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
            rm = self.this.scrunch_subint_RM
            AX.plot(time, rm, 'g--', label='RM [rad.m-2] (RM spectrum)')
        except AttributeError:
            pass
        try:
            time = self.this.scrunch_subint_mjd
            time = (time - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
            rm_refine = self.this.scrunch_subint_RM_refining
            rm_refine_err = self.this.scrunch_subint_RM_refining_err
            AX.errorbar(time, rm_refine, yerr=rm_refine_err, fmt='r+', label='RM [rad.m-2] (fit)')
        except AttributeError:
            pass
        try:
            time_interp = self.this.subint_mjd
            time_interp = (time_interp - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
            rm_interp = self.this.interp_RM_refining
            rm_interp_err = self.this.interp_RM_refining_err
            AX.errorbar(time_interp, rm_interp, yerr=rm_interp_err, fmt='r+', label='RM [rad.m-2] (interp fit)', alpha=0.2)
        except AttributeError:
            pass
        try:
            time_file = self.this.mjd_file
            time_file = (time_file - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
            rm_file = self.this.RM_file
            rm_err_file = self.this.RM_err_file
            AX.errorbar(time_file, rm_file, yerr=rm_err_file, fmt='b+', label='RM [rad.m-2] (file)')
        except AttributeError:
            pass
        try:
            time_file_interp = self.this.subint_mjd
            time_file_interp = (time_file_interp - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
            rm_file_interp = self.this.RM_file_interp
            rm_err_file_interp = self.this.RM_err_file_interp
            AX.errorbar(time_file_interp, rm_file_interp, yerr=rm_err_file_interp, fmt='b+', label='RM [rad.m-2] (interp file)', alpha=0.2)
        except AttributeError:
            pass
        AX.legend(loc='upper right', fontsize=6)
        AX.set_xlabel('Time (minutes)')
        AX.set_ylabel('RM (rad.m-2)')
        if (rightaxis):
            # AXchan = AX.twinx()
            # AXchan.yaxis.set_ticks_position('right')
            # AXchan.set_ylabel("RM\n(rad.m-2)")
            AX.set_ylabel('RM\n(rad.m-2)')
            AX.yaxis.tick_right()
            AX.yaxis.set_ticks_position('both')
            AX.yaxis.set_label_position("right")

    def PA_vs_time(self, AX, rightaxis=False):
        if (self.verbose):
            self.log.log("plotting PA_vs_time", objet=PLOT_OBJET)
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
        if not (arx.get_dedispersed()):
            arx.dedisperse()
        arx.mytscrunch_to_nsub(16)

        # arx.flattenBP()
        arx.fscrunch()
        arx.remove_baseline()
        arx.convert_state('Stokes')

        data = arx.get_data().squeeze()  # (subint, pol, bins)

        profil_Q = data[:, 1, :]  # (subint, bins)
        profil_U = data[:, 2, :]  # (subint, bins)
        print(self.offbins)
        profil_Q_std = []
        profil_U_std = []
        for isub in range(arx.get_nsubint()):
            profil_Q_std.append(np.std(profil_Q[isub, self.offbins]))
            profil_U_std.append(np.std(profil_U[isub, self.offbins]))
        profil_Q_std = np.array(profil_Q_std)
        profil_U_std = np.array(profil_U_std)

        profil_linear = np.sqrt(profil_Q**2 + profil_U**2)  # (subint, bins)
        mean_profil_linear = np.nanmean(profil_linear, axis=1)  # (bins)
        best_bin = np.argmax(mean_profil_linear)

        # Calcul des derivees partielles
        dPPA_dQ = -0.5 * profil_U / (profil_Q**2 + profil_U**2)  # (subint, bins)
        dPPA_dU = 0.5 * profil_Q / (profil_Q**2 + profil_U**2)  # (subint, bins)

        # Calcul de l'erreur sur PPA en radians
        print(np.shape(dPPA_dQ))
        print(np.shape(dPPA_dU))
        print(np.shape(profil_Q_std))
        print(np.shape(profil_U_std))
        profil_PPA_err_rad = []
        for isub in range(arx.get_nsubint()):
            profil_PPA_err_rad.append(0.5 * np.sqrt((dPPA_dQ[isub, :]**2) * (profil_Q_std[isub]**2) +
                                                    (dPPA_dU[isub, :]**2) * (profil_U_std[isub]**2)))  # (subint, bins)
        profil_PPA_err_rad = np.array(profil_PPA_err_rad)

        # Conversion de l'erreur en degres
        profil_PPA = np.degrees(0.5 * np.arctan2(profil_U, profil_Q))  # (subint, bins)
        profil_PPA_err = np.degrees(profil_PPA_err_rad)  # (subint, bins)

        print(best_bin)
        print(np.shape(profil_PPA))
        print(np.shape(profil_PPA))
        print(np.shape(profil_PPA_err))
        print(self.times)
        print(profil_PPA[:, best_bin])
        print(profil_PPA_err[:, best_bin])
        AX.errorbar(self.times, profil_PPA[:, best_bin], yerr=profil_PPA_err[:, best_bin], fmt='r+', label='PPA (best bin)')

    def PA_vs_time_old(self, AX, rightaxis=False):
        def phase_to_PPA(angle_rad):
            # return ((((angle_rad * 180. / np.pi) + 180) / 2.) % 180) - 90
            angle = Angle(angle_rad, unit=u.rad)
            angle_wrapped = angle.wrap_at(180 * u.degree)
            return angle_wrapped.degree / 2.
        try:
            time = self.this.scrunch_subint_mjd
            time = (time - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
            rm_refine = self.this.scrunch_subint_RM_refining
            meanPA_refine = self.this.scrunch_subint_phase_refining
            meanPA_refine_abs = self.this.scrunch_subint_phase_refining + rm_refine * 89875.51787368176 * (self.freqs[-1]**-2)
            try:
                rm_refine_err = self.this.scrunch_subint_RM_refining_err
                meanPA_refine_err = self.this.scrunch_subint_phase_refining_err
                meanPA_refine_abs_err = self.this.scrunch_subint_phase_refining_err + rm_refine_err * 89875.51787368176 * (self.freqs[-1]**-2)
                AX.errorbar(time, phase_to_PPA(meanPA_refine), yerr=phase_to_PPA(meanPA_refine_err), fmt='r+', label='meanPA [rad] (fit)')
                AX.errorbar(time, phase_to_PPA(meanPA_refine_abs), yerr=phase_to_PPA(meanPA_refine_abs_err), fmt='b+', label='abs meanPA [rad] (fit)')
            except AttributeError:
                AX.plot(time, phase_to_PPA(meanPA_refine), 'r+', label='meanPA [rad] (fit)')
                AX.plot(time, phase_to_PPA(meanPA_refine_abs), 'b+', label='abs meanPA [rad] (fit)')
        except AttributeError:
            pass

        if (False):  # do not plot interp abs PA
            try:
                time_interp = self.this.subint_mjd
                time_interp = (time_interp - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
                interp_RM_refining = self.this.interp_RM_refining
                interp_phase_refining = self.this.interp_phase_refining
                try:
                    interp_RM_refining_err = self.this.interp_RM_refining_err
                    interp_phase_refining_err = self.this.interp_phase_refining_err
                    AX.errorbar(time_interp,
                                phase_to_PPA(interp_phase_refining + interp_RM_refining * 89875.51787368176 * (self.freqs[-1]**-2)),
                                yerr=phase_to_PPA(interp_phase_refining_err + interp_RM_refining_err * 89875.51787368176 * (self.freqs[-1]**-2)),
                                fmt='b+', label='abs meanPA [rad] (interp fit)', alpha=0.2)
                except AttributeError:
                    AX.plot(time_interp, phase_to_PPA(interp_phase_refining + interp_RM_refining * 89875.51787368176 *
                                                      (self.freqs[-1]**-2)), 'b+', label='abs meanPA [rad] (interp fit)', alpha=0.2)
            except AttributeError:
                pass

        try:
            time_file_interp = self.this.subint_mjd
            time_file_interp = (time_file_interp - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
            time_file = self.this.mjd_file
            time_file = (time_file - self.times[0].mjd) * 1440 + self.times_subint[0] / 120.
            meanPA_file = self.this.meanPA_file
            # meanPA_file_interp = self.this.meanPA_file_interp
            meanPA_file_interp_abs = self.this.meanPA_file_interp + self.this.RM_file_interp * 89875.51787368176 * (self.freqs[-1]**-2)
            try:
                meanPA_err_file = self.this.meanPA_err_file
                # meanPA_err_file_interp = self.this.meanPA_err_file_interp
                meanPA_err_file_interp_abs = self.this.meanPA_err_file_interp + self.this.RM_err_file_interp * 89875.51787368176 * (self.freqs[-1]**-2)
                AX.errorbar(time_file, phase_to_PPA(meanPA_file), yerr=phase_to_PPA(meanPA_err_file), fmt='g+', label='meanPA [rad] (file)')
                AX.errorbar(time_file_interp, phase_to_PPA(meanPA_file_interp_abs), yerr=phase_to_PPA(
                    meanPA_err_file_interp_abs), fmt='g+', label='abs meanPA [rad] (interp file)', alpha=0.2)
            except AttributeError:
                AX.plot(time_file, phase_to_PPA(meanPA_file), 'g+', label='meanPA [rad] (file)')
                AX.plot(time_file_interp, phase_to_PPA(meanPA_file_interp_abs), 'g+', label='abs meanPA [rad] (interp file)', alpha=0.2)
        except AttributeError:
            pass
        AX.legend(loc='upper right', fontsize=6)
        AX.set_xlabel('Time (minutes)')
        if (rightaxis):
            # AXchan = AX.twinx()
            # AXchan.set_ylabel('abs PA\n(deg)')
            AX.set_ylim([-95, 95])
            AX.set_ylabel('abs PA\n(deg)')
            AX.yaxis.tick_right()
            AX.yaxis.set_ticks_position('both')
            AX.yaxis.set_label_position("right")
        else:
            AX.set_ylim([-95, 95])
            AX.set_ylabel('abs PA\n(deg)')

    def snr_vs_incremental_subintegration(self, AX, botaxis=False):
        """
        plot signal noise ratio for integrated subintegration
        """
        if (self.verbose):
            self.log.log("plotting incremental snr vs subintegrations", objet=PLOT_OBJET)
        try:
            arx = self.myclone()
        except AttributeError:
            arx = self.clone()
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
        subintegration = np.linspace(1, self.get_nsubint(), self.get_nsubint())
        AX.plot(subintegration, SNR)
        AX.set_ylabel('Signal noise ratio')
        AX.set_xlabel('Subintegration')
        string = 'integrated SNR'
        AX.text(0.95 * arx.get_nsubint(),
                np.min(SNR) + 0.90 * (np.max(SNR) - np.min(SNR)), string,
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
            AX_secondary.set_xlim(0, self.integration_length() / 60.)
            AX.set_xlim(1, self.get_nsubint())

    def color_text(self, AX, x, y, string_list, colors, orientation='horizontal', **kwargs):
        """
        Take a list of *strings* and *colors* and place them next to each
        other, with text strings[i] being shown in colors[i].
        Parameters
        ----------
        x, y : float
            Text position in data coordinates.
        strings : list of str
            The strings to draw.
        colors : list of color
            The colors to use.
        orientation : {'horizontal', 'vertical'}
        ax : Axes,
            The Axes to draw into. If None, the current axes will be used.
        **kwargs
            All other keyword arguments are passed to plt.text(), so you can
            set the font size, family, etc.
        """
        text_obj = AX.transData
        fig = AX.figure
        canvas = fig.canvas
        assert orientation in ['horizontal', 'vertical']
        if orientation == 'vertical':
            kwargs.update(rotation=90, verticalalignment='bottom')
        for i in range(len(string_list)):
            text = AX.text(x, y, string_list[i].rstrip("\n"), color=colors[i % len(colors)], transform=text_obj, **kwargs)
            # Need to draw to update the text position.
            text.draw(canvas.get_renderer())
            ex = text.get_window_extent()
            # Convert window extent from pixels to inches
            # to avoid issues displaying at different dpi
            ex = fig.dpi_scale_trans.inverted().transform_bbox(ex)
            if orientation == 'horizontal':
                text_obj = text.get_transform() + \
                    mtransforms.offset_copy(mtransforms.Affine2D(), fig=fig, x=0, y=-1.1 * ex.height)
            else:
                text_obj = text.get_transform() + \
                    mtransforms.offset_copy(mtransforms.Affine2D(), fig=fig, x=ex.width, y=0)


if __name__ == "__main__":
    ar = PlotArchive(ar_name=['/databf2/nenufar-pulsar/DATA/B1919+21/PSR/B1919+21_D20220303T1301_59641_500675_0057_BEAM2.fits'],
                     bscrunch=1, tscrunch=2, pscrunch=False, verbose=True)

    ar.clean(fast=True)
    print(ar.snr_range())

    # ar.DM_fit(ncore=16, plot=True)
    AX = []
    AX.append(plt.subplot2grid((5, 5), (0, 0), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (1, 0), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (2, 0), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (3, 0), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (0, 1), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (1, 1), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (2, 1), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (3, 1), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (0, 2), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (1, 2), colspan=1, rowspan=1))
    AX.append(plt.subplot2grid((5, 5), (2, 2), colspan=1, rowspan=1))
    ar.profil(AX[0])
    ar.phase_freq(AX[1])
    ar.phase_freq_no_ON(AX[2])
    ar.phase_time(AX[3])
    ar.bandpass(AX[4])
    ar.zaped_bandpass(AX[5])
    ar.dynaspect_bandpass(AX[6])
    ar.dynaspect_onpulse(AX[7])
    ar.snr_vs_frequency(AX[8])
    ar.snr_vs_subintegration(AX[9], botaxis=True)
    ar.snr_vs_incremental_subintegration(AX[10])
    plt.show()
