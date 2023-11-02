import numpy as np

# from NenuPlot.log_class import Log_class
from log_class import Log_class
from tinypsrdb import Psrdb
from tinypsrdb import UpdateException
from configuration_class import Config_Reader
from methode_class import string_error_formating


ARCHIVE_GETTER_OBJET = 'metadata'
# try:
#    iers.conf.auto_max_age = None
#    iers.conf.auto_download = False
#    iers.IERS.iers_table = iers.IERS_A.open('/home/lbondonneau/lib/python/astropy/utils/iers/data/finals2000A.all')
# except:
#    print('WARNING: Can not use iers.conf probably due to the astropy version')


class Metadata():
    def __init__(self, log_obj=None, verbose=False):
        if log_obj is None:
            self.log = Log_class()
        else:
            self.log = log_obj
        self.psrdb = Psrdb(verbose=verbose, log_obj=self.log)
        self.psrdb.log.set_print(True)
        self.psrdb.set_db_name('test_db.json')

    def select_archive(self, archive):
        self.archive = archive

    def ext_version(self, archive=None):
        nenuplot_config = Config_Reader(config_file=self.psrdb.config_file, log_obj=self.log)
        self.version = nenuplot_config.get_config('NENUPLOT', 'version')

    def ext_source(self, archive=None):
        if archive is not None:
            self.source = archive.get_source()
        else:
            self.source = self.archive.get_source()

    def ext_telescope(self, archive=None):
        if archive is not None:
            self.telescope = archive.get_telescope()
        else:
            self.telescope = self.archive.get_telescope()

    def ext_receiver_name(self, archive=None):
        if archive is not None:
            self.receiver_name = archive.get_receiver_name()
        else:
            self.receiver_name = self.archive.get_receiver_name()

    def ext_basis(self, archive=None):
        if archive is not None:
            self.basis = archive.get_basis()
        else:
            self.basis = self.archive.get_basis()

    def ext_mjd_start(self, archive=None):
        if archive is not None:
            self.mjd_start = archive.times[0].mjd
        else:
            self.mjd_start = self.archive.times[0].mjd

    def ext_isot_start(self, archive=None):
        if archive is not None:
            self.isot_start = archive.times[0].isot
        else:
            self.isot_start = self.archive.times[0].isot

    def ext_mjd(self, archive=None):
        if archive is not None:
            self.mjd = archive.times.mjd
        else:
            self.mjd = self.archive.times.mjd

    def ext_backend_name(self, archive=None):
        if archive is not None:
            self.backend_name = archive.get_backend_name()
        else:
            self.backend_name = self.archive.get_backend_name()

    def ext_nbin(self, archive=None):
        if archive is not None:
            self.nbin = archive.get_nbin()
        else:
            self.nbin = self.archive.get_nbin()

    def ext_nchan(self, archive=None):
        if archive is not None:
            self.nchan = archive.get_nchan()
        else:
            self.nchan = self.archive.get_nchan()

    def ext_nsubint(self, archive=None):
        if archive is not None:
            self.nsubint = archive.get_nsubint()
        else:
            self.nsubint = self.archive.get_nsubint()

    def ext_obs_duration(self, archive=None):
        if archive is not None:
            self.obs_duration = archive.integration_length()
        else:
            self.obs_duration = self.archive.integration_length()

    def ext_npol(self, archive=None):
        if archive is not None:
            self.npol = archive.get_npol()
        else:
            self.npol = self.archive.get_npol()

    def ext_radec(self, archive=None):
        if archive is not None:
            coord = archive.get_SkyCoord()
            self.ra = coord.ra.degree
            self.dec = coord.dec.degree
        else:
            coord = self.archive.get_SkyCoord()
            self.ra = coord.ra.degree
            self.dec = coord.dec.degree

    def ext_centre_frequency(self, archive=None):
        if archive is not None:
            self.centre_frequency = archive.get_centre_frequency()
            self.bottom_frequency = np.min(archive.freqs)
        else:
            self.centre_frequency = self.archive.get_centre_frequency()
            self.bottom_frequency = np.min(self.archive.freqs)

    def ext_bandwidth(self, archive=None):
        if archive is not None:
            self.bandwidth = archive.get_bandwidth()
        else:
            self.bandwidth = self.archive.get_bandwidth()

    def ext_chan_bw(self, archive=None):
        if archive is not None:
            self.chan_bw = np.mean(archive.get_chan_bw())
        else:
            self.chan_bw = np.mean(self.archive.get_chan_bw())

    def ext_dispersion_measure(self, archive=None):
        if archive is not None:
            self.dispersion_measure = archive.get_dispersion_measure()
        else:
            self.dispersion_measure = self.archive.get_dispersion_measure()
        if (self.dispersion_measure != self.ini_dispersion_measure):
            self.dispersion_measure_string = ("%.6f" % self.dispersion_measure)
        else:
            self.dispersion_measure_string = "None"

    def ext_rotation_measure(self, archive=None):
        if archive is not None:
            self.RM_mjd = list(archive.scrunch_subint_mjd)
            self.RM = list(archive.scrunch_subint_RM_refining)
            self.RM_err = list(archive.scrunch_subint_RM_refining_err)
            self.RM_phase_refining = list(archive.scrunch_subint_phase_refining)
            self.RM_phase_refining_err = list(archive.scrunch_subint_phase_refining_err)
        else:
            self.RM_mjd = list(self.archive.scrunch_subint_mjd)
            self.RM = list(self.archive.scrunch_subint_RM_refining)
            self.RM_err = list(self.archive.scrunch_subint_RM_refining_err)
            self.RM_phase_refining = list(self.archive.scrunch_subint_phase_refining)
            self.RM_phase_refining_err = list(self.archive.scrunch_subint_phase_refining_err)
        if(np.sum(~np.isnan(self.RM)) > 1):
            error = (np.nanmax(self.RM) - np.nanmin(self.RM) + self.RM_err[np.nanargmax(self.RM)] + self.RM_err[np.nanargmin(self.RM)]) / 2.
        else:
            error = self.RM_err[0]
        self.rotation_measure_string = string_error_formating(np.nanmean(self.RM), error)
        print(self.rotation_measure_string)

    def ext_ini_dispersion_measure(self, archive=None):
        if archive is not None:
            self.ini_dispersion_measure = archive.get_dispersion_measure()
        else:
            self.ini_dispersion_measure = self.archive.get_dispersion_measure()
        self.dispersion_measure_string = "None"

    def ext_ini_rotation_measure(self, archive=None):
        if archive is not None:
            self.ini_rotation_measure = archive.get_rotation_measure()
        else:
            self.ini_rotation_measure = self.archive.get_rotation_measure()
        self.rotation_measure_string = "None"

    def ext_is_dedispersed(self, archive=None):
        if archive is not None:
            self.is_dedispersed = archive.get_dedispersed()
        else:
            self.is_dedispersed = self.archive.get_dedispersed()

    def ext_is_faraday_rotated(self, archive=None):
        if archive is not None:
            self.is_faraday_rotated = archive.get_faraday_corrected()
        else:
            self.is_faraday_rotated = self.archive.get_faraday_corrected()

    def ext_is_pol_calib(self, archive=None):
        if archive is not None:
            self.is_pol_calib = archive.get_poln_calibrated()
        else:
            self.is_pol_calib = self.archive.get_poln_calibrated()

    def ext_data_units(self, archive=None):
        if archive is not None:
            self.data_units = archive.get_scale()
        else:
            self.data_units = self.archive.get_scale()

    def ext_data_state(self, archive=None):
        if archive is not None:
            self.data_state = archive.get_state()
        else:
            self.data_state = self.archive.get_state()

    def ext_subint_duration(self, archive=None):
        if archive is not None:
            self.subint_duration = archive.get_subint_duration()
        else:
            self.subint_duration = self.archive.get_subint_duration()

    def ext_RFI20_85(self, archive=None):
        if archive is not None:
            freq_vec = archive.get_freqs()
            weights = archive.get_weights()
        else:
            freq_vec = self.archive.get_freqs()
            weights = self.archive.get_weights()
        weights /= np.max(weights)
        weights20_85 = weights[:, np.argmin(np.abs(freq_vec - 20)):np.argmin(np.abs(freq_vec - 85))]
        self.RFI20_85 = 100. * (1. - np.mean(weights20_85))

    def ext_RFI(self, archive=None):
        if archive is not None:
            weights = archive.get_weights()
        else:
            weights = self.archive.get_weights()
        weights /= np.max(weights)
        self.RFI = 100. * (1. - np.mean(weights))

    def ext_snr_range(self, archive=None, **kargs):
        if archive is not None:
            self.snr_range, self.snr_range_err = archive.snr_range(**kargs)
        else:
            self.snr_range, self.snr_range_err = self.archive.snr_range(**kargs)
        self.snr_range = float(self.snr_range)
        self.snr_range_err = float(self.snr_range_err)

    def ext_snr_norm(self, archive=None, **kargs):
        if archive is not None:
            self.snr_norm = archive.snr_norm(**kargs)
        else:
            self.snr_norm = self.archive.snr_norm(**kargs)

    def ext_snr_psrstat(self, archive=None, **kargs):
        if archive is not None:
            self.snr_psrstat = archive.snr_psrstat(**kargs)
        else:
            self.snr_psrstat = self.archive.snr_psrstat(**kargs)

    def ext_period(self, archive=None):
        if archive is not None:
            self.period = archive.get_period()
        else:
            self.period = self.archive.get_period()

    def ext_altaz(self, archive=None):
        if archive is not None:
            self.alt, self.az = archive.get_altaz()
        else:
            self.alt, self.az = self.archive.get_altaz()
        self.mean_alt = np.mean(self.alt)
        self.mean_az = np.mean(self.az)
        self.start_alt = self.alt[0]
        self.start_az = self.az[0]
        self.stop_alt = self.alt[-1]
        self.stop_az = self.az[-1]

    def insert_from_list(self, list_to_insert, db=True):
        insert_dico = {}
        for dico in list_to_insert:
            exec('self.ext_' + dico['func'] + '()')
            for local_variable in dico['local']:
                insert_dico[local_variable] = eval('self.' + local_variable)
        if (db):
            self.log.log("Insert new entry in the database", objet="Database")
            self.psrdb.insert(insert_dico)

    def update_from_list(self, list_to_update, db=True):
        update_dico = {}
        for dico in list_to_update:
            exec('self.ext_' + dico['func'] + '()')
            for local_variable in dico['local']:
                update_dico[local_variable] = eval('self.' + local_variable)
        if (db):
            self.log.log("Update an entry in the database", objet="Database")
            try:
                self.psrdb.update(update_dico)
            except UpdateException:
                self.log.error("No corresponding entry found to update database", objet="Database")

    def database_insert_ini(self, ar=None, db=True):
        if (ar is not None):
            self.select_archive(ar)
        key_to_insert = [{'func': 'version', 'local': ['version']},
                         {'func': 'source', 'local': ['source']},
                         {'func': 'telescope', 'local': ['telescope']},
                         {'func': 'receiver_name', 'local': ['receiver_name']},
                         {'func': 'basis', 'local': ['basis']},
                         {'func': 'backend_name', 'local': ['backend_name']},
                         {'func': 'nbin', 'local': ['nbin']},
                         {'func': 'nchan', 'local': ['nchan']},
                         {'func': 'nsubint', 'local': ['nsubint']},
                         {'func': 'subint_duration', 'local': ['subint_duration']},
                         {'func': 'mjd_start', 'local': ['mjd_start']},
                         {'func': 'isot_start', 'local': ['isot_start']},
                         {'func': 'obs_duration', 'local': ['obs_duration']},
                         {'func': 'npol', 'local': ['npol']},
                         {'func': 'centre_frequency', 'local': ['centre_frequency']},
                         {'func': 'bandwidth', 'local': ['bandwidth']},
                         {'func': 'chan_bw', 'local': ['chan_bw']},
                         {'func': 'ini_dispersion_measure', 'local': ['ini_dispersion_measure']},
                         {'func': 'dispersion_measure', 'local': ['dispersion_measure']},
                         {'func': 'ini_rotation_measure', 'local': ['ini_rotation_measure']},
                         # {'func': 'rotation_measure', 'local': ['rotation_measure']},
                         {'func': 'is_dedispersed', 'local': ['is_dedispersed']},
                         {'func': 'is_faraday_rotated', 'local': ['is_faraday_rotated']},
                         {'func': 'is_pol_calib', 'local': ['is_pol_calib']},
                         {'func': 'data_units', 'local': ['data_units']},
                         {'func': 'data_state', 'local': ['data_state']},
                         {'func': 'RFI20_85', 'local': ['RFI20_85']},
                         {'func': 'RFI', 'local': ['RFI']},
                         {'func': 'period', 'local': ['period']},
                         {'func': 'radec', 'local': ['ra', 'dec']},
                         {'func': 'altaz', 'local': ['mean_alt', 'mean_az', 'start_alt', 'start_az', 'stop_alt', 'stop_az']},
                         {'func': 'snr_range', 'local': ['snr_range', 'snr_range_err']},
                         {'func': 'snr_norm', 'local': ['snr_norm']},
                         {'func': 'snr_psrstat', 'local': ['snr_psrstat']}]
        self.insert_from_list(key_to_insert, db=db)

    def database_update_scrunch(self, ar=None, db=True):
        if (ar is not None):
            self.select_archive(ar)
        key_to_insert = [{'func': 'source', 'local': ['source']},
                         {'func': 'mjd_start', 'local': ['mjd_start']},
                         {'func': 'obs_duration', 'local': ['obs_duration']},
                         {'func': 'centre_frequency', 'local': ['centre_frequency']},
                         {'func': 'bandwidth', 'local': ['bandwidth']},
                         {'func': 'telescope', 'local': ['telescope']},
                         {'func': 'receiver_name', 'local': ['receiver_name']},
                         {'func': 'backend_name', 'local': ['backend_name']},

                         {'func': 'nbin', 'local': ['nbin']},
                         {'func': 'nchan', 'local': ['nchan']},
                         {'func': 'nsubint', 'local': ['nsubint']},
                         {'func': 'subint_duration', 'local': ['subint_duration']},
                         {'func': 'npol', 'local': ['npol']},
                         {'func': 'chan_bw', 'local': ['chan_bw']}]
        self.update_from_list(key_to_insert, db=db)

    def database_update_snr(self, ar=None, db=True):
        if (ar is not None):
            self.select_archive(ar)
        key_to_insert = [{'func': 'source', 'local': ['source']},
                         {'func': 'mjd_start', 'local': ['mjd_start']},
                         {'func': 'obs_duration', 'local': ['obs_duration']},
                         {'func': 'centre_frequency', 'local': ['centre_frequency']},
                         {'func': 'bandwidth', 'local': ['bandwidth']},
                         {'func': 'telescope', 'local': ['telescope']},
                         {'func': 'receiver_name', 'local': ['receiver_name']},
                         {'func': 'backend_name', 'local': ['backend_name']},

                         {'func': 'snr_range', 'local': ['snr_range', 'snr_range_err']},
                         {'func': 'snr_norm', 'local': ['snr_norm']},
                         {'func': 'snr_psrstat', 'local': ['snr_psrstat']}]
        self.update_from_list(key_to_insert, db=db)

    def database_update_dm(self, ar=None, db=True):
        if (ar is not None):
            self.select_archive(ar)
        key_to_insert = [{'func': 'source', 'local': ['source']},
                         {'func': 'mjd_start', 'local': ['mjd_start']},
                         {'func': 'obs_duration', 'local': ['obs_duration']},
                         {'func': 'centre_frequency', 'local': ['centre_frequency']},
                         {'func': 'bandwidth', 'local': ['bandwidth']},
                         {'func': 'telescope', 'local': ['telescope']},
                         {'func': 'receiver_name', 'local': ['receiver_name']},
                         {'func': 'backend_name', 'local': ['backend_name']},

                         {'func': 'dispersion_measure', 'local': ['dispersion_measure']}]
        self.update_from_list(key_to_insert, db=db)

    def database_update_rm(self, ar=None, db=True):
        if (ar is not None):
            self.select_archive(ar)
        key_to_insert = [{'func': 'source', 'local': ['source']},
                         {'func': 'mjd_start', 'local': ['mjd_start']},
                         {'func': 'obs_duration', 'local': ['obs_duration']},
                         {'func': 'centre_frequency', 'local': ['centre_frequency']},
                         {'func': 'bandwidth', 'local': ['bandwidth']},
                         {'func': 'telescope', 'local': ['telescope']},
                         {'func': 'receiver_name', 'local': ['receiver_name']},
                         {'func': 'backend_name', 'local': ['backend_name']},

                         {'func': 'rotation_measure', 'local': ['RM_mjd', 'RM', 'RM_err', 'RM_phase_refining', 'RM_phase_refining_err']}]

        self.update_from_list(key_to_insert, db=db)

    def get_metadata_output(self):
        # Print out metadata
        def output_from_dic(result):
            return result['obj'] + result['description'] + str(result['strvalue']) + "\n"

        def output_from_list(result_to_print, output, colors):
            for iobj in range(len(result_to_print)):
                output.append(result_to_print[iobj]['obj'] + result_to_print[iobj]['description'] + str(result_to_print[iobj]['strvalue']) + "\n")
                colors.append('black')
            return (output, colors)

        def color_new_dm():
            try:
                if (self.dispersion_measure_string != 'None'):
                    def dispersed_bin_ratio(ref_freq):
                        bw = np.mean(self.chan_bw)
                        if (ref_freq < 20):
                            ref_freq = 20
                        lastchan_max = ref_freq + bw / 2.
                        lastchan_min = ref_freq - bw / 2.
                        delta_dm = np.abs(self.ini_dispersion_measure - self.dispersion_measure)
                        bin_time = self.period / float(self.nbin)
                        ratio = 4.15e3 * delta_dm * (lastchan_min**-2 - lastchan_max**-2) / bin_time
                        # if ratio > 1 it's bad!!
                        return np.abs(ratio)
                    mid_freq = self.centre_frequency
                    bot_freq = self.bottom_frequency
                    if (bot_freq < 20):
                        bot_freq = 20
                    if(dispersed_bin_ratio(mid_freq) > 1):
                        return 'red'
                    elif(dispersed_bin_ratio(bot_freq) > 1):
                        return 'purple'
                    else:
                        return 'blue'
                else:
                    return 'black'
            except AttributeError:
                return 'black'

        def color_new_rm():
            try:
                delta_RM = np.abs(self.ini_rotation_measure - np.mean(self.RM))
                if (self.rotation_measure_string != 'None'):
                    if(delta_RM > 2):
                        return 'red'
                    elif(delta_RM > 1):
                        return 'purple'
                    else:
                        return 'blue'
                else:
                    return 'black'
            except AttributeError:
                return 'black'

        def color_elev():
            try:
                if (self.mean_alt < 10):
                    return 'red'
                elif (self.mean_alt < 20):
                    return 'purple'
                else:
                    return 'blue'
            except AttributeError:
                return 'black'

        def color_RFI(rfi):
            try:
                if (rfi > 45):
                    return 'red'
                elif (rfi > 30):
                    return 'purple'
                else:
                    return 'blue'
            except AttributeError:
                return 'black'

        output = []
        colors = []
        result_to_print = [{'obj': 'NenuPlot     ', 'description': 'V', 'strvalue': self.version},
                           {'obj': 'name         ', 'description': 'Source name                           ', 'strvalue': self.source},
                           {'obj': 'Start        ', 'description': str(self.isot_start) + ' ' * 15         , 'strvalue': str(np.round(self.mjd_start, decimals=4))},
                           {'obj': 'nbin         ', 'description': 'Number of pulse phase bins            ', 'strvalue': self.nbin},
                           {'obj': 'nchan        ', 'description': 'Number of frequency channels          ', 'strvalue': self.nchan},
                           {'obj': 'npol         ', 'description': 'Number of polarizations               ', 'strvalue': self.npol},
                           {'obj': 'nsubint      ', 'description': 'Number of sub-integrations            ', 'strvalue': self.nsubint},
                           {'obj': 'length       ', 'description': 'Observation duration (s)              ', 'strvalue': self.obs_duration},
                           {'obj': 'dm           ', 'description': 'Dispersion measure (pc/cm^3)          ', 'strvalue': ("%.6f" % self.ini_dispersion_measure)}]
        output, colors = output_from_list(result_to_print, output, colors)

        result_to_print = {'obj': 'new dm       ', 'description': 'refined Dispersion measure (pc/cm^3)  ', 'strvalue': self.dispersion_measure_string}
        output.append(output_from_dic(result_to_print))
        colors.append(color_new_dm())

        result_to_print = {'obj': 'rm           ', 'description': 'Rotation measure (rad/m^2)            ', 'strvalue': self.ini_rotation_measure}
        output.append(output_from_dic(result_to_print))
        colors.append('black')

        result_to_print = {'obj': 'new rm       ', 'description': 'refined Rotation measure (rad/m^2)    ', 'strvalue': self.rotation_measure_string}
        output.append(output_from_dic(result_to_print))
        colors.append(color_new_rm())

        result_to_print = [{'obj': 'period topo  ', 'description': 'Folding_period (s)                    ', 'strvalue': self.period},
                           {'obj': 'site         ', 'description': 'Telescope name                        ', 'strvalue': self.telescope},
                           {'obj': 'coord ra     ', 'description': 'Source coordinates (hms)              ', 'strvalue': self.ra},
                           {'obj': 'coord dec    ', 'description': 'Source coordinates (dms)              ', 'strvalue': self.dec},
                           {'obj': 'freq         ', 'description': 'Centre frequency (MHz)                ', 'strvalue': self.centre_frequency},
                           {'obj': 'bw           ', 'description': 'Bandwidth (MHz)                       ', 'strvalue': self.bandwidth},
                           {'obj': 'dmc          ', 'description': 'Dispersion corrected                  ', 'strvalue': self.is_dedispersed},
                           {'obj': 'rmc          ', 'description': 'Faraday Rotation corrected            ', 'strvalue': self.is_faraday_rotated},
                           {'obj': 'polc         ', 'description': 'Polarization calibrated               ', 'strvalue': self.is_pol_calib},
                           {'obj': 'rcvr:name    ', 'description': 'Receiver and Backend name             ', 'strvalue': ("%s(%s)" % (self.receiver_name, self.backend_name))},
                           {'obj': 'SNR(psrstat) ', 'description': 'Signal noise ratio                    ', 'strvalue': ("%.1f" % self.snr_psrstat)},
                           {'obj': 'SNR(range)   ', 'description': 'Signal noise ratio with snr.py        ', 'strvalue': string_error_formating(self.snr_range, self.snr_range_err)},
                           {'obj': 'SNR(norm)    ', 'description': 'SNR(range) but norm at noRFI,1h,90d   ', 'strvalue': ("%.1f" % self.snr_norm)}]
        output, colors = output_from_list(result_to_print, output, colors)

        try:
            result_to_print = {'obj': 'RFI 20-80MHz ', 'description': 'Radio Frequency Interferency (/100)   ', 'strvalue': ("%.2f" % self.RFI20_85)}
            output.append(output_from_dic(result_to_print))
            colors.append(color_RFI(self.RFI20_85))
        except AttributeError:
            pass

        result_to_print = {'obj': 'RFI          ', 'description': 'Radio Frequency Interferency (/100)   ', 'strvalue': ("%.2f" % self.RFI)}
        output.append(output_from_dic(result_to_print))
        colors.append(color_RFI(self.RFI))

        result_to_print = {'obj': 'elevation    ', 'description': 'Elevation (Start, Mean, End)           ', 'strvalue': ("(%.2f, %.2f, %.2f)" % (self.start_alt, self.mean_alt, self.stop_alt))}
        output.append(output_from_dic(result_to_print))
        colors.append(color_elev())

        return (output, colors)


if __name__ == "__main__":
    metadata = Metadata(verbose=True)
    from mypsrchive import psrchive_class
    ar = psrchive_class()
    # ar.MyArchive_load(['/databf2/nenufar-pulsar/DATA/B1508+55/PSR/B1508+55_D20220707T1701_59767_002409_0057_BEAM0.fits'], bscrunch=4, tscrunch=4)
    ar.MyArchive_load(['/home/lbondonneau/data/rmme/B1508+55_D20220707T1701_59767_002409_0057_BEAM0.fits'], bscrunch=1, tscrunch=1)
    print(ar.get_subint_duration())
    ar.clean(fast=True)
    metadata.select_archive(ar)
    metadata.database_insert_ini()
    print(metadata.get_metadata_output())
