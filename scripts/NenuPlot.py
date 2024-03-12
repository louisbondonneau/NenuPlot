import os
import sys
from sys import argv
import socket
import numpy as np
import argparse
from datetime import timedelta
import NenuPlot_module
from NenuPlot_module import (reduce_pdf,
                             Log_class,
                             Methode,
                             Metadata,
                             IncertException,
                             Config_Reader,
                             DM_fit_class)

from subprocess import check_output

HOSTNAME = socket.gethostname()

"""
This code provide a quicklook in pdf/png and many option:
    - frequency band selection -minfreq 1410 -maxfreq 1430
        grep channels in the requested frequency range
    - noclean to jump coast_guard
exp:
python /path_to_code/NenuPlot.py /path_to_file/*.ar

2020/03/10   add option nodefaraday because defaraday can cause psrsplit troubles
2020/04/20   fix option option metadata_out (previously away True)
2020/05/30   new options chanthresh and subintthresh for CoatGuard
2020/05/30   iterative_cleaner: bad_chan becomes more limiting 0.75 -> 0.5
2020/05/30   iterative_cleaner: flattenBP is no dependent at 20% from adjacent channels
2020/05/30   iterative_cleaner: default chanthresh 5 -> 3.5, default subintthresh 3.5 -> 3.8
2020/05/31   iterative_cleaner: bad_subint and bad_chan are now in option
2020/07/09   quicklookutil: add a try to use iers.conf (do not exist in old version of astropy)
2020/07/09   plotfunction: fix (bad normalisation in QUV when used timenorme and timenpolar options )
2020/08/05   quicklookutil: fix bug in load_some_chan_in_archive_data if the option -minchan was used with -freqappend
2020/08/05   quicklookutil: fix bu in the pdf reduction when multiple simulttaneous instance of the code in tthe same directory
2020/08/06   iterative_cleaner_light: fix in function auto_find_on_window (when the puls is on the  border)
2020/08/06   quicklookutil: same bug in auto_find_on_window_from_ar
2020/08/06   quicklookutil: if npol==1 nodefaraday is set to True
2020/09/20   NenuPlot: new option -fit_DM
2020/09/25   quicklookutil: add dm new in metadata
2020/09/25   quicklookutil: add version number un metadata
2020/09/25   quicklookutil: add Nenuplot.py commande line in metadata
2020/09/25   NenuPlot: revers option nodefaraday to defaraday (nodefaraday by default)
2021/03/26   NenuPlot: python3 update
2021/03/26   iterative_cleaner_light: python3 update
2021/03/26   quicklookutil: python3 update
2021/03/26   mysendmaim: python3 update using of email instead of mailx in mysendmail.py
2021/03/26   NenuPlot: reverse small_pdf, iterative and flat_cleaner (now True by default)
2021/03/26   NenuPlot: bad_chan default value pass from 0.5 to 0.8
2021/03/28   NenuPlot: fix file liste printing condition (---processed--File(s)---)
2021/04/26   iterative_cleaner_light: zapfirstsubint == True if nsubint > 2 it was 1 previously. But cleaning an obs with 1 integration is too challenging.
2021/05/08   DM_fit: set a new DM but do not dedisperse the observation
2021/06/17   quicklookutil: Added a condition to fix ra=0 (psrchive bug). serching ra and dec with psredit.
2021/06/17   DM_fit: update auto_rebin with a new function
2021/06/17   iterative_cleaner_light: Added a stop if the on-pulse does not change anymore.
2021/07/06   iterative_cleaner_light: implement option force_niter.
2021/09/07   iterative_cleaner_light: debug if only 1 integration one channel.
2021/10/16   quicklookutil: freqqapend is know automatic
2021/10/16   NenuPlot: rm freqqapend option
2021/10/24   mysendmail: update email import for py3
2021/10/24   iterative_cleaner_light: catch on few empty slice warning
2021/10/24   quicklookutil: dont dedisperse anymore if dm is default (in func load_some_chan_in_archive_data) it was the cause of a timing bug (dedisperse/dededisperse)
2022/03/14   quicklookutil: load_some_chan_in_archive_data the maxfreq exact value is now the last channel (the channel of value maxfreq was previously rm)
2022/04/08   plotfunction: color inversion for L and V (now it's like in pav -S)
2022/04/08   quicklookutil: caluclation of a normalised SNR (normilised by time, elevation and RFI)
2022/08/20   quicklookutil: directory liste (input files can now come from diff path)
2022/XX/XX   NenuPlot v3.0.0

BUG know:
    - do not a recognized file format while using -archive_out and -gui in python3.8
    - some time RA and DEC values can be 0 -> wrong elevation
"""

CONFIG_FILE = NenuPlot_module.__path__[0] + '/NenuPlot.conf'


class NenuPlot():
    def __init__(self, logname='Nenuplot', config_file=CONFIG_FILE, verbose=False):
        self.config_file = config_file
        self.log = Log_class(logname=logname, verbose=verbose)
        self.__init_configuration()
        self.args_parser()
        self.methode = Methode(log_obj=self.log)
        self.metadata = Metadata(verbose=self.args.verbose, log_obj=self.log)
        self.useful_RM = False

    def __init_configuration(self):
        nenuplot_config = Config_Reader(config_file=self.config_file, log_obj=self.log, verbose=False)
        self.version = nenuplot_config.get_config('NENUPLOT', 'version')
        self.verbose = nenuplot_config.get_config('NENUPLOT', 'verbose')
        # --- MAIN FEATURE ---
        self.cleaner_toggle = nenuplot_config.get_config('NENUPLOT', 'cleaner_toggle')
        self.DM_fit_toggle = nenuplot_config.get_config('NENUPLOT', 'DM_fit_toggle')
        self.RM_fit_toggle = nenuplot_config.get_config('NENUPLOT', 'RM_fit_toggle')
        self.autorebin_toggle = nenuplot_config.get_config('NENUPLOT', 'autorebin_toggle')
        self.database_toggle = nenuplot_config.get_config('NENUPLOT', 'database_toggle')
        self.mail_toggle = nenuplot_config.get_config('NENUPLOT', 'mail_toggle')
        # --- INPUT DATA MODIF ---
        self.pscrunch = nenuplot_config.get_config('NENUPLOT', 'pscrunch')
        self.bscrunch = nenuplot_config.get_config('NENUPLOT', 'bscrunch')
        self.tscrunch = nenuplot_config.get_config('NENUPLOT', 'tscrunch')
        self.fscrunch = nenuplot_config.get_config('NENUPLOT', 'fscrunch')
        self.defaraday = nenuplot_config.get_config('NENUPLOT', 'defaraday')
        self.minfreq = nenuplot_config.get_config('NENUPLOT', 'minfreq')
        self.maxfreq = nenuplot_config.get_config('NENUPLOT', 'maxfreq')
        # --- DATA MODIF AFTER CLEAN ---
        self.bscrunch_after = nenuplot_config.get_config('NENUPLOT', 'bscrunch_after')
        self.tscrunch_after = nenuplot_config.get_config('NENUPLOT', 'tscrunch_after')
        self.fscrunch_after = nenuplot_config.get_config('NENUPLOT', 'fscrunch_after')
        # --- PHASE/TIME OPT ---
        self.plot_timepolar = nenuplot_config.get_config('NENUPLOT', 'plot_timepolar')
        self.plot_timenorme = nenuplot_config.get_config('NENUPLOT', 'plot_timenorme')
        self.plot_threshold = nenuplot_config.get_config('NENUPLOT', 'plot_threshold')
        self.plot_flatband = nenuplot_config.get_config('NENUPLOT', 'plot_flatband')
        self.plot_figsize = nenuplot_config.get_config('NENUPLOT', 'plot_figsize')
        # --- METADATA OPT ---
        self.keep_initmetadata = nenuplot_config.get_config('NENUPLOT', 'keep_initmetadata')
        self.init_metadata_inDB = nenuplot_config.get_config('NENUPLOT', 'init_metadata_inDB')
        self.DM_fit_metadata_inDB = nenuplot_config.get_config('NENUPLOT', 'DM_fit_metadata_inDB')
        self.SNR_metadata_inDB = nenuplot_config.get_config('NENUPLOT', 'SNR_metadata_inDB')
        # --- OUPUT OPT ---
        self.output_dir = nenuplot_config.get_config('NENUPLOT', 'output_dir')
        self.archive_out = nenuplot_config.get_config('NENUPLOT', 'archive_out')
        self.PDF_out = nenuplot_config.get_config('NENUPLOT', 'PDF_out')
        self.PDF_dpi = nenuplot_config.get_config('NENUPLOT', 'PDF_dpi')
        self.PNG_out = nenuplot_config.get_config('NENUPLOT', 'PNG_out')
        self.PNG_dpi = nenuplot_config.get_config('NENUPLOT', 'PNG_dpi')
        self.mask_out = nenuplot_config.get_config('NENUPLOT', 'mask_out')
        self.metadata_out = nenuplot_config.get_config('NENUPLOT', 'metadata_out')
        self.RM_out = nenuplot_config.get_config('NENUPLOT', 'RM_out')
        # --- UPLOAD METADATA ---
        self.upload_metadata_toggle = nenuplot_config.get_config('NENUPLOT', 'upload_metadata_toggle')
        self.upload_metadata_hostname = nenuplot_config.get_config('NENUPLOT', 'upload_metadata_hostname')
        self.upload_metadata_username = nenuplot_config.get_config('NENUPLOT', 'upload_metadata_username')
        self.upload_metadata_dir = nenuplot_config.get_config('NENUPLOT', 'upload_metadata_dir')
        # --- UPLOAD MASK ---
        self.upload_mask_toggle = nenuplot_config.get_config('NENUPLOT', 'upload_mask_toggle')
        self.upload_mask_hostname = nenuplot_config.get_config('NENUPLOT', 'upload_mask_hostname')
        self.upload_mask_username = nenuplot_config.get_config('NENUPLOT', 'upload_mask_username')
        self.upload_mask_dir = nenuplot_config.get_config('NENUPLOT', 'upload_mask_dir')
        # --- UPLOAD PDF ---
        self.upload_PDF_toggle = nenuplot_config.get_config('NENUPLOT', 'upload_PDF_toggle')
        self.upload_PDF_hostname = nenuplot_config.get_config('NENUPLOT', 'upload_PDF_hostname')
        self.upload_PDF_username = nenuplot_config.get_config('NENUPLOT', 'upload_PDF_username')
        self.upload_PDF_dir = nenuplot_config.get_config('NENUPLOT', 'upload_PDF_dir')
        # --- UPLOAD PDF ---
        self.upload_PNG_toggle = nenuplot_config.get_config('NENUPLOT', 'upload_PNG_toggle')
        self.upload_PNG_hostname = nenuplot_config.get_config('NENUPLOT', 'upload_PNG_hostname')
        self.upload_PNG_username = nenuplot_config.get_config('NENUPLOT', 'upload_PNG_username')
        self.upload_PNG_dir = nenuplot_config.get_config('NENUPLOT', 'upload_PNG_dir')
        # --- UPLOAD ARCHIVE ---
        self.upload_archive_toggle = nenuplot_config.get_config('NENUPLOT', 'upload_archive_toggle')
        self.upload_archive_hostname = nenuplot_config.get_config('NENUPLOT', 'upload_archive_hostname')
        self.upload_archive_username = nenuplot_config.get_config('NENUPLOT', 'upload_archive_username')
        self.upload_archive_dir = nenuplot_config.get_config('NENUPLOT', 'upload_archive_dir')
        # --- CLEANER OPT ---
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

    def args_parser(self):
        parser = argparse.ArgumentParser(prog='NenuPlot', description="This code provide a quicklook in pdf/png and many option for PSRFITS folded files.",
                                         formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=160))
        parser.add_argument('-u', dest='path',
                            help="output path (default current directory)", default=self.output_dir)
        parser.add_argument('-o', dest='name',
                            help="output name (default is the first archive name)")
        parser.add_argument('-v', dest='verbose', action='store_true', default=self.verbose,
                            help="Verbose mode")
        parser.add_argument('-version', action='version', version='%(prog)s' + '%s' % (self.version))

        parser.add_argument('-gui', dest='gui', action='store_true', default=False,
                            help="Open the matplotlib graphical user interface")

        parser.add_argument('-clean', dest='cleaner_toggle', action='store_true', default=self.cleaner_toggle,
                            help="Clean archive (default is %d)" % self.cleaner_toggle)
        parser.add_argument('-noclean', dest='cleaner_toggle', action='store_false', default=self.cleaner_toggle)

        parser.add_argument('-fit_DM', dest='fit_DM', action='store_true', default=self.DM_fit_toggle,
                            help="will fit for a new DM value (default is %d)" % self.DM_fit_toggle)
        parser.add_argument('-nofit_DM', dest='fit_DM', action='store_false', default=self.DM_fit_toggle)

        parser.add_argument('-fit_RM', dest='fit_RM', action='store_true', default=self.RM_fit_toggle,
                            help="will fit for a new RM value (default is %d)" % self.RM_fit_toggle)
        parser.add_argument('-nofit_RM', dest='fit_RM', action='store_false', default=self.RM_fit_toggle)

        parser.add_argument('-fit_RM_window', dest='fit_RM_window', type=float, default=None,
                            help="will fit for a new RM in a specifique window in rad.m-2")

        parser.add_argument('-autorebin', dest='autorebin', action='store_true', default=self.autorebin_toggle,
                            help="Archive will be automaticaly rebin to an optimal bin number (default is %d)" % self.autorebin_toggle)
        parser.add_argument('-noautorebin', dest='autorebin', action='store_false', default=self.autorebin_toggle)

        parser.add_argument('-database', dest='database', action='store_true', default=self.database_toggle,
                            help="Metadata will be automaticaly sotre in a tiny database (default is %d)" % self.database_toggle)
        parser.add_argument('-nodatabase', dest='database', action='store_false', default=self.database_toggle)

        # mail
        parser.add_argument('-mail', dest='sendmail',
                            help="send the metadata and file by mail -mail '[aaa@bbb.zz, bbb@bbb.zz]' ", default=False)
        parser.add_argument('-mailtitle', dest='mailtitle',
                            help="modified the title of the mail", default='mail from NenuPlot')
        # subint selection
        parser.add_argument('-mintime', dest='mintime', type=float, default=None,
                            help="Start time in sec to extract from the archives")
        parser.add_argument('-maxtime', dest='maxtime', type=float, default=None,
                            help="End time in sec to extract from the archives")
        parser.add_argument('-minsub', dest='mintime', type=int, default=None,
                            help="Start subintegration to extract from the archives")
        parser.add_argument('-maxsub', dest='maxtime', type=int, default=None,
                            help="End subintegration to extract from the archives")
        # freq selection
        parser.add_argument('-minfreq', dest='minfreq', default=self.minfreq,
                            help="Minimum frequency to extract from the archives (default is %.4f MHz)" % self.minfreq)
        parser.add_argument('-maxfreq', dest='maxfreq', default=self.maxfreq,
                            help="Maximum frequency to extract from the archives (default is %.4f MHz)" % self.maxfreq)
        # mask input
        parser.add_argument('-mask_input', dest='mask_input',
                            help="path to a mask in input")
        parser.add_argument('-RM_input', dest='RM_input',
                            help="path to an RM csv file in input (mjd, RM, RM_err, Phase, Phase_err)")
        parser.add_argument('-b', dest='bscrunch', default=self.bscrunch,
                            help="time scrunch factor (before CoastGuard) (default is %d)" % self.bscrunch)
        parser.add_argument('-t', dest='tscrunch', default=self.tscrunch,
                            help="time scrunch factor (before CoastGuard) (default is %d)" % self.tscrunch)
        parser.add_argument('-f', dest='fscrunch', default=self.fscrunch,
                            help="frequency scrunch factor (before CoastGuard) (default is %d)" % self.fscrunch)
        parser.add_argument('-p', dest='pscrunch', action='store_true', default=self.pscrunch,
                            help="polarisation scrunch (before CoastGuard) (default is %d)" % self.pscrunch)
        parser.add_argument('-ba', dest='bscrunch_after', default=self.bscrunch_after,
                            help="bin scrunch factor (after CoastGuard) (default is %d)" % self.bscrunch_after)
        parser.add_argument('-ta', dest='tscrunch_after', default=self.tscrunch_after,
                            help="time scrunch factor (after CoastGuard) (default is %d)" % self.tscrunch_after)
        parser.add_argument('-fa', dest='fscrunch_after', default=self.fscrunch_after,
                            help="frequency scrunch factor (after CoastGuard) (default is %d)" % self.fscrunch_after)

        # output
        parser.add_argument('-archive_out', dest='archive_out', action='store_true', default=self.archive_out,
                            help="write an archive in output PATH/*.ar.clear (default is %d)" % self.archive_out)
        parser.add_argument('-noarchive_out', dest='archive_out', action='store_false', default=self.archive_out)

        parser.add_argument('-metadata_out', dest='metadata_out', action='store_true', default=self.metadata_out,
                            help="write a metadatafile in output PATH/*.metadata (default is %d)" % self.metadata_out)
        parser.add_argument('-nometadata_out', dest='metadata_out', action='store_false', default=self.metadata_out)

        parser.add_argument('-RM_out', dest='RM_out', action='store_true', default=self.RM_out,
                            help="write a RMfile in output PATH/*.csv (default is %d)" % self.RM_out)
        parser.add_argument('-noRM_out', dest='RM_out', action='store_false', default=self.RM_out)

        parser.add_argument('-mask_out', dest='mask_out', action='store_true', default=self.mask_out,
                            help="write a dat file containing the mask PATH/*.mask (default is %d)" % self.mask_out)
        parser.add_argument('-nomask_out', dest='mask_out', action='store_false', default=self.mask_out)

        parser.add_argument('-PDF_out', dest='PDF_out', action='store_true', default=self.PDF_out,
                            help="write a PDF file containing the quicklook PATH/*.pdf (default is %d)" % self.PDF_out)
        parser.add_argument('-noPDF_out', dest='PDF_out', action='store_false', default=self.PDF_out)

        parser.add_argument('-PNG_out', dest='PNG_out', action='store_true', default=self.PNG_out,
                            help="write a PNG file containing the quicklook PATH/*.png (default is %d)" % self.PNG_out)
        parser.add_argument('-noPNG_out', dest='PNG_out', action='store_false', default=self.PNG_out)

        # graphic option
        parser.add_argument('-Coherence', dest='stokes', action='store_false', default=True,
                            help="use XX and YY for phase/freq phase/time and spectrum (default is stokes)")
        parser.add_argument('-PDF_dpi', dest='PDF_dpi', default=self.PDF_dpi,
                            help="Dots per inch of the output PDF file (default is %d)" % self.PDF_dpi)
        parser.add_argument('-PNG_dpi', dest='PNG_dpi', default=self.PNG_dpi,
                            help="Dots per inch of the output PNG file (default is %d)" % self.PNG_dpi)

        # cleaner option
        parser.add_argument('-force_niter', dest='cleaner_max_iter', default=self.cleaner_max_iter,
                            help="force N iteration of cleaning (default is %d)" % self.cleaner_max_iter)

        parser.add_argument('-fast', dest='cleaner_fast', action='store_true', default=self.cleaner_fast,
                            help="Run only 3 iteration of the cleaner (default is %d)" % self.cleaner_fast)

        parser.add_argument('-chanthresh', dest='cleaner_chanthresh', default=self.cleaner_chanthresh,
                            help="chanthresh for loop 2 to X of CoastGuard (default is %d)" % (self.cleaner_chanthresh))
        parser.add_argument('-subintthresh', dest='cleaner_subintthresh', default=self.cleaner_subintthresh,
                            help="subintthresh for loop 2 to X of CoastGuard (default is %d)" % (self.cleaner_subintthresh))
        parser.add_argument('-first_chanthresh', dest='cleaner_first_chanthresh', default=self.cleaner_first_chanthresh,
                            help="chanthresh for loop 2 to X of CoastGuard (default is %d)" % (self.cleaner_first_chanthresh))
        parser.add_argument('-first_subintthresh', dest='cleaner_first_subintthresh', default=self.cleaner_first_subintthresh,
                            help="subintthresh for loop 2 to X of CoastGuard (default is %d)" % (self.cleaner_first_subintthresh))

        parser.add_argument('-bad_subint', dest='cleaner_bad_subint', default=self.cleaner_bad_subint,
                            help="bad_subint for CoastGuard (default is %f a subint is removed if masked part > %d percent)" % (self.cleaner_bad_subint, int(self.cleaner_bad_subint * 100)))
        parser.add_argument('-bad_chan', dest='cleaner_bad_chan', default=self.cleaner_bad_chan,
                            help="bad_chan for CoastGuard (default is %f a channel is removed if masked part > %d percent)" % (self.cleaner_bad_chan, int(self.cleaner_bad_chan * 100)))

        # plot option
        parser.add_argument('-timepolar', dest='plot_timepolar', action='store_true', default=self.plot_timepolar,
                            help="plot phase time polar")
        parser.add_argument('-timenorme', dest='plot_timenorme', action='store_true', default=self.plot_timenorme,
                            help="normilized in time the phase/time plot")
        parser.add_argument('-threshold', dest='plot_threshold', default=self.plot_threshold,
                            help="threshold on the ampl in dyn spect and phase/freq (default is False)")
        parser.add_argument('-rm', dest='rm', default=None,
                            help="defaraday with a new RM in rad.m-2 (default use the observation RM)")

        parser.add_argument('-defaraday', dest='defaraday', action='store_true', default=self.defaraday,
                            help="defaraday the signal (signal is not defaraday by default)")
        parser.add_argument('-nodefaraday', dest='defaraday', action='store_false', default=self.defaraday)

        parser.add_argument('-dm', dest='dm', default=None,
                            help="dedisperse with a new DM in pc.cm-3 (default use the observation DM)")

        parser.add_argument('-singlepulses_patch', dest='singlepulses_patch', action='store_true', default=False,
                            help="phasing patch when append multiple single pulses archive in frequency dirrection")
        parser.add_argument('-keep_initmetadata', dest='keep_initmetadata', action='store_true', default=self.keep_initmetadata,
                            help="keep initial metadata")

        # upload option
        parser.add_argument('-upload_metadata', dest='upload_metadata', action='store_true', default=self.upload_metadata_toggle,
                            help="upload metadata to %s@%s:%s (default is %d)" % (self.upload_metadata_username, self.upload_metadata_hostname,
                                                                                  self.upload_metadata_dir, self.upload_metadata_toggle))
        parser.add_argument('-upload_mask', dest='upload_mask', action='store_true', default=self.upload_mask_toggle,
                            help="upload mask to %s@%s:%s (default is %d)" % (self.upload_mask_username, self.upload_mask_hostname,
                                                                              self.upload_mask_dir, self.upload_mask_toggle))
        parser.add_argument('-upload_archive', dest='upload_archive', action='store_true', default=self.upload_archive_toggle,
                            help="upload archive to %s@%s:%s (default is %d)" % (self.upload_archive_username, self.upload_archive_hostname,
                                                                                 self.upload_archive_dir, self.upload_archive_toggle))
        parser.add_argument('-upload_PDF', dest='upload_PDF', action='store_true', default=self.upload_PDF_toggle,
                            help="upload PDF to %s@%s:%s (default is %d)" % (self.upload_PDF_username, self.upload_PDF_hostname,
                                                                             self.upload_PDF_dir, self.upload_PDF_toggle))

        parser.add_argument('-upload_PNG', dest='upload_PNG', action='store_true', default=self.upload_PNG_toggle,
                            help="upload PNG to %s@%s:%s (default is %d)" % (self.upload_PNG_username, self.upload_PNG_hostname,
                                                                             self.upload_PNG_dir, self.upload_PNG_toggle))

        parser.add_argument('INPUT_ARCHIVE', nargs='+', help="Path to the Archives")

        self.args = parser.parse_args()

        if (self.args.path == 'current'):
            self.args.path = os.getcwd() + '/'

        if (self.args.path[-1] != '/'):
            self.args.path = self.args.path + '/'
        self.log.set_dir(self.args.path)

        if(self.args.minfreq is not None):
            self.args.minfreq = float(self.args.minfreq)
        if(self.args.maxfreq is not None):
            self.args.maxfreq = float(self.args.maxfreq)
        if(self.args.mintime is not None):
            if isinstance(self.args.mintime, float):
                self.args.mintime = timedelta(seconds=float(self.args.mintime))
        if(self.args.maxtime is not None):
            if isinstance(self.args.maxtime, float):
                self.args.maxtime = timedelta(seconds=float(self.args.maxtime))

    def apply_mask(self):
        # ----------------APPLY mask------------------------------
        if (self.args.mask_input):
            if(self.args.verbose):
                self.log.log("Nenuplot: apply_mask start", objet='NenuPlot')
            self.methode.check_file_validity(self.args.mask_input)
            self.ar.apply_mask(np.genfromtxt(self.args.mask_input))

    def load_RM(self):
        if (self.args.RM_input):
            if(self.args.verbose):
                self.log.log("Nenuplot: load_RM from file start", objet='NenuPlot')
            self.ar.init_RM_fit()
            self.methode.check_file_validity(self.args.RM_input)
            self.ar.open_RM(self.args.RM_input)
            self.useful_RM = True

    def load_archive(self):
        # ----------------load archives------------------------------

        # self.ar = psrchive_class(verbos=self.args.verbose, log_obj=self.log)
        # filename = self.ar.MyArchive_load(self.args.INPUT_ARCHIVE,
        #                                   minfreq=self.args.minfreq,
        #                                   maxfreq=self.args.maxfreq,
        #                                   mintime=self.args.mintime,
        #                                   maxtime=self.args.maxtime,
        #                                   bscrunch=int(self.args.bscrunch),
        #                                   tscrunch=int(self.args.tscrunch),
        #                                   fscrunch=int(self.args.fscrunch),
        #                                   pscrunch=self.args.pscrunch,
        #                                   dm=self.args.dm,
        #                                   rm=self.args.rm,
        #                                   singlepulses_patch=self.args.singlepulses_patch,
        #                                   defaraday=self.args.defaraday)

        if (self.args.fit_RM) or (self.args.RM_input):
            from NenuPlot_module import RM_fit_class as psrchive_class
        elif (self.args.fit_DM):
            from NenuPlot_module import DM_fit_class as psrchive_class
        else:
            from NenuPlot_module import psrchive_class

        self.ar = psrchive_class(ar_name=self.args.INPUT_ARCHIVE, verbose=self.args.verbose, log_obj=self.log,
                                 minfreq=self.args.minfreq,
                                 maxfreq=self.args.maxfreq,
                                 mintime=self.args.mintime,
                                 maxtime=self.args.maxtime,
                                 bscrunch=int(self.args.bscrunch),
                                 tscrunch=int(self.args.tscrunch),
                                 fscrunch=int(self.args.fscrunch),
                                 pscrunch=self.args.pscrunch,
                                 dm=self.args.dm,
                                 rm=self.args.rm,
                                 singlepulses_patch=self.args.singlepulses_patch,
                                 defaraday=False)  # self.args.defaraday)

        if not (self.args.name):
            self.args.name = self.ar.name
        self.log.set_logname(self.args.name)

    def clear_archive(self):
        # ----------------CLEAR archive------------------------------

        if (self.args.cleaner_toggle):
            if(self.args.verbose):
                if(self.args.cleaner_fast):
                    self.log.log("Nenuplot: cleaning start with max iteration=%d" % 3, objet='NenuPlot')
                else:
                    self.log.log("Nenuplot: cleaning start with max iteration=%d" % self.args.cleaner_max_iter, objet='NenuPlot')
            self.ar.set_zapfirstsubint(self.cleaner_zapfirstsubint)
            self.ar.set_flat_cleaner(self.cleaner_flat_cleaner)

            self.ar.set_fast(self.args.cleaner_fast)
            self.ar.set_chanthresh(self.args.cleaner_chanthresh)
            self.ar.set_subintthresh(self.args.cleaner_subintthresh)
            self.ar.set_first_chanthresh(self.args.cleaner_first_chanthresh)
            self.ar.set_first_subintthresh(self.args.cleaner_first_subintthresh)
            self.ar.set_bad_subint(self.args.cleaner_bad_subint)
            self.ar.set_bad_chan(self.args.cleaner_bad_chan)
            self.ar.set_max_iter(self.args.cleaner_max_iter)

            self.ar.clean()

    def save_mask(self):
        # ----------------SAVE mask------------------------------
        if self.args.mask_out or self.args.upload_mask:
            if(self.args.verbose):
                self.log.log("Nenuplot: mask save to " + self.args.path + self.args.name + '.mask', objet='NenuPlot')
            np.savetxt(self.args.path + self.args.name + '.mask', self.ar.get_weights())
        if (self.args.upload_mask):
            source_file = self.args.path + self.args.name + '.mask'
            self.methode.rsync(source_file, self.upload_mask_username, self.upload_mask_hostname, self.upload_mask_dir)
            if not self.args.mask_out:
                self.methode.remove(source_file)

    def DM_fit(self):
        # ----------------DM fit------------------------------
        if (self.ar.get_nchan() >= 10) and (self.args.fit_DM):
            if(self.args.verbose):
                self.log.log("Nenuplot: DM_fit start", objet='NenuPlot')
            new_dm, dm_err = DM_fit_class.DM_fit(self.ar, verbose=self.args.verbose, ncore=8)
            # new_dm, dm_err = self.ar.DM_fit(verbose=self.args.verbose, ncore=8)
            self.ar.set_dispersion_measure(new_dm)
            self.args.name = self.args.name + '_DMfit'
            self.metadata_DM()

    def RM_fit(self):
        # ----------------RM fit------------------------------
        if (self.ar.get_nchan() >= 10) and (self.args.fit_RM):
            if(self.args.verbose):
                self.log.log("Nenuplot: RM_fit start", objet='NenuPlot')
            if(self.args.fit_RM_window):
                self.ar.rm_window = self.args.fit_RM_window
            self.ar.init_RM_fit()
            self.ar.RM_reduction(only_bestbin=False, sum_stokes_bin=False, QU_fit=True)
            self.ar.RM_refining(sum_stokes_bin=True)
            self.ar.RM_interpolate_result()
            self.args.name = self.args.name + '_RMfit'
            if (self.args.RM_out):
                self.ar.save_RM(name=self.args.path + self.args.name + '.csv')
            self.metadata_RM()

    def auto_rebin(self):
        if (self.args.autorebin):
            self.ar.auto_rebin()

    def scrunch_after(self):
        if (self.needplot) or (self.args.archive_out):
            if (self.args.bscrunch_after > 1):
                if(self.ar.get_nbin() / int(self.args.bscrunch_after) < 8):
                    self.args.bscrunch_after = int(self.ar.get_nbin() / 8)
                self.ar.mybscrunch(int(self.args.bscrunch_after))
            if (self.args.tscrunch_after > 1):
                self.ar.mytscrunch(int(self.args.tscrunch_after))
            if (self.args.fscrunch_after > 1):
                self.ar.myfscrunch(int(self.args.fscrunch_after))

    def save_archive(self):
        if (self.args.archive_out):
            if (self.args.cleaner_toggle):
                self.ar.unload(self.args.path + self.args.name + '.ar.clear')
                if(self.args.verbose):
                    self.log.log("Nenuplot: unload archive to " + self.args.path + self.args.name + '.ar.clear', objet='NenuPlot')
            else:
                self.ar.unload(self.args.path + self.args.name + '.ar')
                if(self.args.verbose):
                    self.log.log("Nenuplot: unload archive to " + self.args.path + self.args.name + '.ar', objet='NenuPlot')

    def apply_auxRM(self):
        if (self.ar.get_npol() > 1) and (self.args.defaraday):
            if (self.args.fit_RM) or (self.args.RM_input):
                if (self.useful_RM):
                    out_name_tmp = self.args.path + self.args.name + '.tmp'
                    ini_RM = self.ar.get_rotation_measure()
                    self.ar.set_rotation_measure(0)  # otherwise it is summed with auxRM
                    self.log.log("Nenuplot: unload tmp archive to " + self.args.path + self.args.name + '.tmp', objet='NenuPlot')
                    self.ar.unload(out_name_tmp)
                    cmd = "psredit -c int:ext=+aux -c int:aux:rm=%f  -m %s" % (np.nanmean(self.ar.interp_RM_refining), out_name_tmp)
                    self.log.log("psredit cmd: %s" % cmd)
                    output = check_output(cmd, shell=True).decode("utf-8")
                    cmd = 'psredit'
                    for isub in range(self.ar.get_nsubint()):
                        if (self.args.fit_RM):
                            cmd = cmd + " -c int[%d]:aux:rm=%f" % (isub, self.ar.interp_RM_refining[isub])
                        else:  # (self.args.RM_input)
                            cmd = cmd + " -c int[%d]:aux:rm=%f" % (isub, self.ar.RM_file_interp[isub])
                    cmd = cmd + " -m %s " % (out_name_tmp)
                    self.log.log("psredit cmd: %s" % cmd)
                    output = check_output(cmd, shell=True).decode("utf-8")
                    from NenuPlot_module import RM_fit_class as psrchive_class
                    self.ar.set_Archive(psrchive_class(ar_name=out_name_tmp, verbose=False, log_obj=self.log,
                                                       defaraday=True))

    def archive_defaraday(self):
        if (self.ar.get_npol() > 1) and (self.args.defaraday):
            self.ar.defaraday()

    def archive_centre(self):
        self.ar.centre_max_bin()

    def metadata_ini(self):
        if(self.args.verbose):
            self.log.log("Nenuplot: extract metadata", objet='NenuPlot')
        self.metadata.select_archive(self.ar)
        try:
            self.metadata.database_insert_ini(db=self.args.database)
        except IncertException as e:
            self.log.error("Can not add this new entry to the database because of \"%s\"" % str(e.args), objet='Database')
            self.args.database = False

    def metadata_SNR(self):
        if(self.args.verbose):
            self.log.log("Nenuplot: update SNR metadata", objet='NenuPlot')
        self.metadata.database_update_snr(self.ar, db=self.args.database)
        self.metadata_list, self.metadata_colors = self.metadata.get_metadata_output()

    def metadata_DM(self):
        if(self.args.verbose):
            self.log.log("Nenuplot: update DM metadata", objet='NenuPlot')
        self.metadata.database_update_dm(self.ar, db=self.args.database)
        self.metadata_list, self.metadata_colors = self.metadata.get_metadata_output()

    def metadata_RM(self):
        if (len(self.ar.scrunch_subint_RM_refining) > 1):
            if(self.args.verbose):
                self.log.log("Nenuplot: update RM metadata or database", objet='NenuPlot')
            self.metadata.database_update_rm(self.ar, db=self.args.database)
            self.metadata_list, self.metadata_colors = self.metadata.get_metadata_output()
            self.useful_RM = True
        else:
            if(self.args.verbose):
                self.log.log("Nenuplot: no useful RM value to uptade metadata or database", objet='NenuPlot')
            self.useful_RM = False

    def main(self):
        if (self.args.gui is False) and (self.args.sendmail is False) and (self.args.PDF_out is False) and (self.args.PNG_out is False) and (self.args.upload_PDF is False):
            self.needplot = False
        else:
            self.needplot = True

        self.load_archive()

        if (self.args.gui):
            # print('GUI MODE')
            import matplotlib
            import matplotlib.pyplot as plt
            from NenuPlot_module import PlotArchive
        else:
            # print('PDF MODE')
            import matplotlib
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            from NenuPlot_module import PlotArchive

        if (self.needplot):
            try:
                fig = plt.figure(figsize=eval(self.plot_figsize))
            except RuntimeError:
                self.log.error("You probably try to oppen the matplotlib graphical inerface (-gui) while ssh connection is not in -X mode", objet='NenuPlot')
                exit(0)
            plt.subplots_adjust(top=0.975, bottom=0.055,
                                left=0.085, right=0.925,
                                hspace=0.280, wspace=0.06)
            matplotlib.rcParams.update({'font.size': 8})

        self.apply_mask()
        self.clear_archive()
        self.save_mask()
        self.scrunch_after()

        self.metadata_ini()

        self.DM_fit()

        if (self.args.keep_initmetadata):
            self.metadata_SNR()

        self.auto_rebin()

        if not (self.args.keep_initmetadata):
            self.metadata_SNR()

        self.RM_fit()
        self.load_RM()

        self.apply_auxRM()

        self.archive_defaraday()

        self.save_archive()
        self.archive_centre()

        # ----------------metadata in ax0------------------------------
        if (self.needplot):
            ax0 = plt.subplot2grid((5, 5), (0, 2), colspan=3, rowspan=2, frameon=False)

        if (self.needplot):
            plot_ar = PlotArchive(ar=self.ar)
            plot_ar.color_text(ax0, 0, 1, self.metadata_list, self.metadata_colors,
                               horizontalalignment='left',
                               verticalalignment='top',
                               fontdict={'family': 'DejaVu Sans Mono'},  # monospace
                               size=8,
                               wrap=True)
            ax0.axes.get_xaxis().set_visible(False)
            ax0.axes.get_yaxis().set_visible(False)

            if (self.args.plot_timepolar):
                ax2 = plt.subplot2grid((5, 4), (2, 0), colspan=1, rowspan=1)
                ax21 = plt.subplot2grid((5, 4), (2, 1), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
                ax22 = plt.subplot2grid((5, 4), (2, 2), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
                ax23 = plt.subplot2grid((5, 4), (2, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
                if(self.args.verbose):
                    self.log.log("Nenuplot: plot phase_time with polar", objet='NenuPlot')
                plot_ar.phase_time(ax2, timenorme=self.args.plot_timenorme, pol=0, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                plot_ar.phase_time(ax21, timenorme=self.args.plot_timenorme, pol=1, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                plot_ar.phase_time(ax22, timenorme=self.args.plot_timenorme, pol=2, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                plot_ar.phase_time(ax23, timenorme=self.args.plot_timenorme, pol=3, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                ax21.axes.get_yaxis().set_visible(False)
                ax22.axes.get_yaxis().set_visible(False)
                ax23.axes.get_yaxis().set_visible(False)

            else:
                # ----------------phase_time in ax2------------------------------
                ax2 = plt.subplot2grid((5, 5), (2, 0), colspan=2, rowspan=1)
                if(self.args.verbose):
                    self.log.log("Nenuplot: plot phase_time without polar", objet='NenuPlot')
                plot_ar.phase_time(ax2, timenorme=self.args.plot_timenorme, stokes=self.args.stokes, threshold=self.args.plot_threshold)

            # ----------------search for onpulse------------------------------------
            left_onpulse, righ_onpulse = self.ar.get_on_window(safe_fraction=1 / 16.)

            # ----------------dynaspect_onpulse in ax8------------------------------
            ax8 = plt.subplot2grid((5, 2), (4, 0), colspan=1, rowspan=1)
            if(self.args.verbose):
                self.log.log("Nenuplot: plot dynaspect_onpulse", objet='NenuPlot')
            plot_ar.dynaspect_onpulse(ax8, left_onpulse=left_onpulse, righ_onpulse=righ_onpulse, threshold=self.args.plot_threshold)

            # ----------------dynaspect_bandpass in ax9------------------------------
            ax9 = plt.subplot2grid((5, 2), (4, 1), colspan=1, rowspan=1, sharey=ax8, sharex=ax8)
            if(self.args.verbose):
                self.log.log("Nenuplot: plot dynaspect_bandpass", objet='NenuPlot')
            plot_ar.dynaspect_bandpass(ax9, left_onpulse=left_onpulse, righ_onpulse=righ_onpulse,
                                       flatband=self.plot_flatband, threshold=self.args.plot_threshold)
            ax9.yaxis.tick_right()
            ax9.yaxis.set_ticks_position('both')
            ax9.yaxis.set_label_position("right")

            if not (self.useful_RM):
                self.ar.tscrunch()  # no more need time dim
                plot_ar.set_Archive(self.ar)

            if not (self.args.plot_timepolar) and not (self.useful_RM):
                # ----------------bandpass in ax3------------------------------
                ax3 = plt.subplot2grid((5, 5), (2, 2), colspan=3, rowspan=1)
                if(self.args.verbose):
                    self.log.log("Nenuplot: plot bandpass", objet='NenuPlot')
                plot_ar.bandpass(ax3, mask=True, rightaxis=True)

            if not (self.args.plot_timepolar) and (self.useful_RM):
                if(self.args.verbose):
                    self.log.log("Nenuplot: plot phase/time polarisation", objet='NenuPlot')
                ax31 = plt.subplot2grid((10, 5), (4, 2), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
                ax32 = plt.subplot2grid((10, 5), (4, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
                ax33 = plt.subplot2grid((10, 5), (4, 4), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
                plot_ar.phase_time(ax31, timenorme=self.args.plot_timenorme, pol=1, stokes=self.args.stokes,
                                   threshold=self.args.plot_threshold, nsub=32, nbin=128)
                plot_ar.phase_time(ax32, timenorme=self.args.plot_timenorme, pol=2, stokes=self.args.stokes,
                                   threshold=self.args.plot_threshold, nsub=32, nbin=128)
                plot_ar.phase_time(ax33, timenorme=self.args.plot_timenorme, pol=3, stokes=self.args.stokes,
                                   threshold=self.args.plot_threshold, nsub=32, nbin=128, rightaxis=True)
                ax31.axes.get_yaxis().set_visible(False)
                ax32.axes.get_yaxis().set_visible(False)

                # self.ar.tscrunch()  # no more need time dim
                # plot_ar.set_Archive(self.ar)

                if(self.args.verbose):
                    self.log.log("Nenuplot: plot phase/freq polarisation", objet='NenuPlot')
                ax34 = plt.subplot2grid((10, 5), (5, 2), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
                ax35 = plt.subplot2grid((10, 5), (5, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
                ax36 = plt.subplot2grid((10, 5), (5, 4), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
                plot_ar.phase_freq(ax34, pol=1, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold, nchan=32, nbin=128)
                plot_ar.phase_freq(ax35, pol=2, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold, nchan=32, nbin=128)
                plot_ar.phase_freq(ax36, pol=3, flatband=self.plot_flatband, stokes=self.args.stokes,
                                   threshold=self.args.plot_threshold, nchan=32, nbin=128, rightaxis=True)
                ax34.axes.get_yaxis().set_visible(False)
                ax35.axes.get_yaxis().set_visible(False)

            # ----------------phase_freq I Q U V in ax4 to ax7------------------------------
            if self.args.stokes:
                ax4 = plt.subplot2grid((5, 5), (3, 0), colspan=2, rowspan=1, sharex=ax2, sharey=ax8)
                ax5 = plt.subplot2grid((5, 5), (3, 2), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
                ax6 = plt.subplot2grid((5, 5), (3, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
                ax7 = plt.subplot2grid((5, 5), (3, 4), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
            else:
                ax4 = plt.subplot2grid((5, 4), (3, 0), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)

            if(self.args.verbose):
                self.log.log("Nenuplot: plot phase_freq", objet='NenuPlot')
            plot_ar.phase_freq(ax4, pol=0, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold)

            if (self.args.fit_RM) or (self.args.RM_input):
                if (self.useful_RM):
                    ax5 = plt.subplot2grid((10, 5), (6, 2), colspan=3, rowspan=1)
                    ax6 = plt.subplot2grid((10, 5), (7, 2), colspan=3, rowspan=1)
                    plot_ar.RM_vs_time(ax5, rightaxis=True)
                    plot_ar.PA_vs_time(ax6, rightaxis=True)
                else:
                    ax5 = plt.subplot2grid((5, 5), (3, 2), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
                    ax6 = plt.subplot2grid((5, 5), (3, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
                    ax7 = plt.subplot2grid((5, 5), (3, 4), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
                    plot_ar.phase_freq(ax5, pol=1, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                    ax5.axes.get_yaxis().set_visible(False)
                    plot_ar.phase_freq(ax6, pol=2, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                    ax6.axes.get_yaxis().set_visible(False)
                    plot_ar.phase_freq(ax7, pol=3, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                    ax7.axes.get_yaxis().set_visible(False)
            else:
                if (self.ar.get_npol() > 1):
                    plot_ar.phase_freq(ax5, pol=1, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                    ax5.axes.get_yaxis().set_visible(False)
                    plot_ar.phase_freq(ax6, pol=2, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                    ax6.axes.get_yaxis().set_visible(False)
                    plot_ar.phase_freq(ax7, pol=3, flatband=self.plot_flatband, stokes=self.args.stokes, threshold=self.args.plot_threshold)
                    ax7.axes.get_yaxis().set_visible(False)
                else:
                    ax5 = plt.subplot2grid((5, 5), (3, 2), colspan=3, rowspan=1)
                    plot_ar.zaped_bandpass(ax5, rightaxis=True)

            # ----------------profil in ax1------------------------------
            ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=2, rowspan=2, sharex=ax2)

            ax1.set_xlim([0, 1])

            if(self.args.verbose):
                self.log.log("Nenuplot: plot profil", objet='NenuPlot')
            plot_ar.profil(ax1, stokes=self.args.stokes)

            plt.draw()

            if(self.args.PDF_out) or (self.args.sendmail) or (self.args.upload_PDF):
                if(self.args.verbose):
                    self.log.log("Nenuplot: savefig to " + self.args.path + self.args.name + '.pdf', objet='NenuPlot')
                plt.savefig(self.args.path + self.args.name + '.pdf', dpi=int(self.args.PDF_dpi), format='pdf')
                reduce_pdf(self.args.path, self.args.name + '.pdf', dpi=int(self.args.PDF_dpi), log_obj=self.log)

            if(self.args.PNG_out) or (self.args.sendmail) or (self.args.upload_PNG):
                if(self.args.verbose):
                    self.log.log("Nenuplot: savefig to " + self.args.path + self.args.name + '.png', objet='NenuPlot')
                plt.savefig(self.args.path + self.args.name + '.png', dpi=int(self.args.PNG_dpi), format='png')
        self.metadata_string = ''.join(self.metadata_list)
        self.metadata_string = self.metadata_string + '\n' + '---------------------processed--File(s)------------------------'
        if (self.args.sendmail) or (self.args.metadata_out):
            for file in self.args.INPUT_ARCHIVE:
                self.metadata_string = self.metadata_string + '\n' + file

        self.metadata_string = self.metadata_string + '\n' + '---------------------command-to-repeat-------------------------'
        self.metadata_string = self.metadata_string + '\n' + 'python ' + " ".join(argv)

        if not (self.args.archive_out):
            self.metadata_string = self.metadata_string + ' -archive_out'

        if (self.needplot):
            if (self.args.sendmail):
                # mysendmail.sendMail(['aa@bb.com', 'bb@ff.com'], "New observation "+args.name, metadata, [args.path+args.name+'.pdf'])
                str_mail = "New observation " + str(self.ar.get_source())
                if (self.args.mailtitle):
                    str_mail = str_mail + ' ' + self.args.mailtitle
                self.methode.sendMail(self.args.sendmail, str_mail, self.metadata_string, [self.args.path + self.args.name + '.png'])

        if(self.args.metadata_out) or (self.args.upload_metadata):
            with open(self.args.path + self.args.name + '.metadata', "w") as text_file:
                text_file.write("%s" % self.metadata_string)

        if (self.args.upload_metadata):
            source_file = self.args.path + self.args.name + '.metadata'
            self.methode.rsync(source_file, self.upload_metadata_username, self.upload_metadata_hostname, self.upload_metadata_dir)

        if (self.needplot):
            if (self.args.upload_PDF):
                source_file = self.args.path + self.args.name + '.pdf'
                self.methode.rsync(source_file, self.upload_PDF_username, self.upload_PDF_hostname, self.upload_PDF_dir)
            if (self.args.upload_PNG):
                source_file = self.args.path + self.args.name + '.png'
                self.methode.rsync(source_file, self.upload_PNG_username, self.upload_PNG_hostname, self.upload_PNG_dir)

            if (self.args.upload_PDF) and not (self.args.PDF_out):
                self.methode.remove(self.args.path + self.args.name + '.pdf')

            if (self.args.upload_PNG) and not (self.args.PNG_out):
                self.methode.remove(self.args.path + self.args.name + '.png')

            if (self.args.upload_metadata) and not (self.args.metadata_out):
                self.methode.remove(self.args.path + self.args.name + '.metadata')

            if (self.args.gui):
                plt.show()

        if (self.args.upload_archive):
            pass

        # rm temporary file used to apply RMaux
        if (self.args.fit_RM) or (self.args.RM_input):
            if (self.useful_RM):
                self.methode.remove(self.args.path + self.args.name + '.tmp')


if __name__ == "__main__":
    nenuplot_obj = NenuPlot(verbose=True)

    nenuplot_obj.main()
