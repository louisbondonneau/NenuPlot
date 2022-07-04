import sys, os
from sys import argv
import socket
import psrchive as psr
import numpy as np
import iterative_cleaner_light
import quicklookutil
import argparse
import mysendmail
import uploadto
import DM_fit_lib

HOSTNAME=socket.gethostname()

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

BUG know:
    - do not a recognized file format while using -arout and -gui in python3.8
"""

version = '03.11.02' # 8 characters

parser = argparse.ArgumentParser(description="This code will.")
parser.add_argument('-u', dest='path',
                    help="output path (default current directory)")
parser.add_argument('-o', dest='name',
                    help="output name (default is archive name)")
parser.add_argument('-v', dest='verbose', action='store_true', default=False,
                    help="Verbose mode")
#freq selection
parser.add_argument('-minfreq', dest='minf', default=0,
                    help="Minimum frequency to extract from the archives, no default 0 MHz exepted for nenufar at 20 MHz")
parser.add_argument('-maxfreq', dest='maxf', default=2**24,
                    help="Maximum frequency to extract from the archives, no default 2^24 MHz exepted for nenufar at 87 MHz")
#mask input
parser.add_argument('-mask', dest='mask',
                    help="mask in input")

parser.add_argument('-b', dest='bscrunch', default=1,
                    help="time scrunch factor (before CoastGuard)")
parser.add_argument('-t', dest='tscrunch', default=1,
                    help="time scrunch factor (before CoastGuard)")
parser.add_argument('-f', dest='fscrunch', default=1,
                    help="frequency scrunch factor (before CoastGuard)")
parser.add_argument('-p', dest='pscrunch', action='store_true', default=False,
                    help="polarisation scrunch (before CoastGuard)")
parser.add_argument('-ba', dest='bscrunch_after', default=1,
                    help="bin scrunch factor (after CoastGuard)")
parser.add_argument('-ta', dest='tscrunch_after', default=1,
                    help="time scrunch factor (after CoastGuard)")
parser.add_argument('-fa', dest='fscrunch_after', default=1,
                    help="frequency scrunch factor (after CoastGuard)")

parser.add_argument('-fit_DM', dest='fit_DM', action='store_true', default=False,
                    help="will fit for a new DM value (at your own risk)")
parser.add_argument('-noautorebin', dest='noautorebin', action='store_true', default=False,
                    help="Archive will NOT be automaticaly rebin to an optimal bins number")

#mail
parser.add_argument('-mail', dest='sendmail',
                    help="send the metadata and file by mail -mail [aaa@bbb.zz, bbb@bbb.zz] ")
parser.add_argument('-mailtitle', dest='mailtitle',
                    help="modified the title of the mail")

#output
parser.add_argument('-arout', dest='arout', action='store_true', default=False,
                    help="write an archive in output PATH/*.ar.clear")
parser.add_argument('-metadata_out', dest='metadata_out', action='store_true', default=False,
                    help="copy the metadatafile in a directory")
parser.add_argument('-maskout', dest='maskout', action='store_true', default=False,
                    help="write a dat file containing the mask PATH/*.mask")
parser.add_argument('-uploadpdf', dest='uploadpdf', action='store_true', default=False,
                    help="upload the pdf/png file")

#graphic option
parser.add_argument('-gui', dest='gui', action='store_true', default=False,
                    help="Open the matplotlib graphical user interface")
parser.add_argument('-nostokes', dest='nostokes', action='store_true', default=False,
                    help="use XX and YY for phase/freq phase/time and spectrum")
parser.add_argument('-dpi', dest='dpi', default=400,
                    help="Dots per inch of the output file (default 400 for pdf and 96 for png)") 
parser.add_argument('-nosmall_pdf', dest='nosmall_pdf', action='store_true', default=False,
                    help="reduction of the pdf size using ghostscript")
parser.add_argument('-nopdf', dest='nopdf', action='store_true', default=False,
                    help="do not sauve the pdf/png file")
parser.add_argument('-png', dest='png', action='store_true', default=False,
                    help="output in png")

#cleaner
parser.add_argument('-noclean', dest='noclean', action='store_true', default=False,
                    help="Do not run coastguard")
parser.add_argument('-force_niter', dest='force_niter', default=False,
                    help="force N iteration of cleaning")
parser.add_argument('-noiterative', dest='iterative', action='store_false', default=True,
                    help="Run the iterative cleaner")
parser.add_argument('-noflat_cleaner', dest='flat_cleaner', action='store_false', default=True,
                    help="flat bandpass for the RFI mitigation")
parser.add_argument('-chanthresh', dest='chanthresh', default=6.0,
                    help="chanthresh for loop 2 to X of CoastGuard (default is 6 or 3 if -flat_cleaner)")
parser.add_argument('-subintthresh', dest='subintthresh', default=3.0,
                    help="subintthresh for loop 2 to X of CoastGuard (default 3)")
parser.add_argument('-bad_subint', dest='bad_subint', default=0.9,
                    help="bad_subint for CoastGuard (default is 0.9 a subint is removed if masked part > 90 percent)")
parser.add_argument('-bad_chan', dest='bad_chan', default=0.8,
                    help="bad_chan for CoastGuard (default is 0.8 a channel is removed if masked part > 80 percent)")

#plot option
parser.add_argument('-timepolar', dest='timepolar', action='store_true', default=False,
                    help="plot phase time polar")
parser.add_argument('-timenorme', dest='timenorme', action='store_true', default=False,
                    help="normilized in time the phase/time plot")
parser.add_argument('-threshold', dest='threshold',
                    help="threshold on the ampl in dyn spect and phase/freq")
parser.add_argument('-rm', dest='rm', default=0.0,
                    help="defaraday with a new rm in rad.m-2")
parser.add_argument('-defaraday', dest='defaraday', action='store_true', default=False,
                    help="defaraday the signal (signal is not defaraday by default)")
parser.add_argument('-dm', dest='dm', default=0.0,
                    help="dedisperse with a new dm in pc.cm-3")
parser.add_argument('-noflat', dest='noflat', action='store_true', default=False,
                    help="Do not flat the bandpass for plots")
parser.add_argument('-nozapfirstsubint', dest='nozapfirstsubint', action='store_true', default=False,
                    help="Do not zap the first subint")
parser.add_argument('-singlepulses_patch', dest='singlepulses_patch', action='store_true', default=False,
                    help="phasing patch when append multiple single pulses archive in frequency dirrection")
parser.add_argument('-initmetadata', dest='initmetadata', action='store_true', default=False,
                    help="keep initial metadata")
                    
parser.add_argument('INPUT_ARCHIVE', nargs='+', help="Path to the Archives")

args = parser.parse_args()



if (args.gui):
    import matplotlib
    import matplotlib.pyplot as plt
    import plotfunc
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotfunc

def main():
    ar_name = args.INPUT_ARCHIVE
    if (args.flat_cleaner == True) and (args.chanthresh == 6.0):
        args.chanthresh = 4.0
    
    if (args.iterative):
        fast = False
    else:
        fast = True
    
    if (args.noflat):
        flatband = False
    else:
        flatband = True
    
    if (args.nostokes):
        stokes = False
    else:
        stokes = True
    
    if (args.threshold):
        threshold = float(args.threshold)
    else:
        threshold = False
    
    if (args.png):
        form = 'png'
        if (args.dpi == 400):
            args.dpi = 96
    else:
        form = 'pdf'
    
    if (args.nozapfirstsubint):
        zapfirstsubint = False
    else:
        zapfirstsubint = True
    
    if (args.gui == False) and (args.sendmail == False) and (args.nopdf == True) and (args.png == False):
        needplot = False
    else:
        needplot = True

    # ----------------load archives------------------------------
    ar, filename, initTFB = quicklookutil.load_some_chan_in_archive_data(ar_name,
                                                                initmetadata=args.initmetadata,
                                                                minfreq=float(args.minf),
                                                                maxfreq=float(args.maxf),
                                                                bscrunch=int(args.bscrunch),
                                                                tscrunch=int(args.tscrunch),
                                                                fscrunch=int(args.fscrunch),
                                                                pscrunch=args.pscrunch,
                                                                dm=float(args.dm),
                                                                rm=float(args.rm),
                                                                singlepulses_patch=args.singlepulses_patch,
                                                                defaraday=args.defaraday)
    
    if not (args.path):
        args.path = ''
    elif (args.path[-1] != '/'):
        args.path = args.path+'/'
    if not (args.name):
        args.name = filename
    

    fig = plt.figure(figsize=(10, 11))
    plt.subplots_adjust(top=0.97, bottom=0.06,
                        left=0.1, right=0.915,
                        hspace=0.355, wspace=0.1)
    
    
    # ----------------CLEAR archive------------------------------
    if not (args.noclean) and not (args.mask):
        if(args.verbose): print("Nenuplot: leaning start")
        ar = iterative_cleaner_light.clean(ar,
                                           zapfirstsubint=zapfirstsubint,
                                           fast=fast,
                                           flat_cleaner=args.flat_cleaner,
                                           chanthresh=float(args.chanthresh),
                                           subintthresh=float(args.subintthresh),
                                           bad_subint=float(args.bad_subint),
                                           bad_chan=float(args.bad_chan),
                                           forceiter=args.force_niter)
        if args.maskout:
            if(args.verbose): print("Nenuplot: mask save to "+args.path+args.name+'.mask')
            np.savetxt(args.path+args.name+'.mask', ar.get_weights())

    if (needplot):
        if(args.verbose): print("Nenuplot: bandpass_filter start %f to %f MHz" %(float(args.minf), float(args.maxf)))
        ar = quicklookutil.bandpass_filter(ar, minfreq=float(args.minf), maxfreq=float(args.maxf))
    
    if (args.mask) and not (args.noclean):
        if(args.verbose): print("Nenuplot: apply_mask start")
        quicklookutil.apply_mask(ar, args.mask)
    
    if (ar.get_nchan() >= 10) and (args.fit_DM):
        if(args.verbose): print("Nenuplot: DM_fit start")
        ar, new_dm, dm_err = DM_fit_lib.DM_fit(ar, verbose=args.verbose, ncore=8, autorebin=(not args.noautorebin))
        args.name = args.name+'_DMfit'
    
    if (needplot): 
        if (args.bscrunch_after > 1):
            if(ar.get_nbin()/int(args.bscrunch_after) < 8):
                args.bscrunch_after = int(ar.get_nbin()/8)
            ar.bscrunch(int(args.bscrunch_after))
        if (args.tscrunch_after > 1):
            ar.tscrunch(int(args.tscrunch_after))
        if (args.fscrunch_after > 1):
            ar.fscrunch(int(args.fscrunch_after))
    
    if (args.arout):
        #if not (args.freqappend): ar.update_centre_frequency()
        ar.dededisperse()
        if (ar.get_npol() > 1) and (args.defaraday): ar.defaraday()
        if not (args.noclean):
            ar.unload(args.path+args.name+'.ar.clear')
            if(args.verbose): print("Nenuplot: unload archive to "+args.path+args.name+'.ar.clear')
        else:
            ar.unload(args.path+args.name+'.ar')
            if(args.verbose): print("Nenuplot: unload archive to "+args.path+args.name+'.ar')

    ar.centre_max_bin()
    # ----------------HEADER in ax0------------------------------
    if (needplot): ax0 = plt.subplot2grid((5, 5), (0, 2), colspan=3, rowspan=2, frameon=False)
    
    if(args.verbose): print("Nenuplot: extract metadata")
    metadata = quicklookutil.print_metadata(ar, initTFB=initTFB, initmetadata=args.initmetadata, version=version)
    
    if (needplot): 
        ax0.text(0, 1, metadata,
                 horizontalalignment='left',
                 verticalalignment='top',
                 fontdict={'family': 'DejaVu Sans Mono'}, #monospace
                 size=8,
                 wrap=True)
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        
        if (args.timepolar):
            ax2 = plt.subplot2grid((5, 4), (2, 0), colspan=1, rowspan=1)
            ax21 = plt.subplot2grid((5, 4), (2, 1), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
            ax22 = plt.subplot2grid((5, 4), (2, 2), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
            ax23 = plt.subplot2grid((5, 4), (2, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax2)
            if(args.verbose): print("Nenuplot: plot phase_time with polar")
            plotfunc.phase_time(ar, ax2, timenorme=args.timenorme, pol=0, stokes=stokes, threshold=threshold)
            plotfunc.phase_time(ar, ax21, timenorme=args.timenorme, pol=1, stokes=stokes, threshold=threshold)
            plotfunc.phase_time(ar, ax22, timenorme=args.timenorme, pol=2, stokes=stokes, threshold=threshold)
            plotfunc.phase_time(ar, ax23, timenorme=args.timenorme, pol=3, stokes=stokes, threshold=threshold)
            ax21.axes.get_yaxis().set_visible(False)
            ax22.axes.get_yaxis().set_visible(False)
            ax23.axes.get_yaxis().set_visible(False)
            
        else:
            # ----------------phase_time in ax2------------------------------
            ax2 = plt.subplot2grid((5, 5), (2, 0), colspan=2, rowspan=1)
            if(args.verbose): print("Nenuplot: plot phase_time without polar")
            plotfunc.phase_time(ar, ax2, timenorme=args.timenorme, stokes=stokes, threshold=threshold)
        
        # ----------------search for onpulse------------------------------------
        
        left_onpulse, righ_onpulse = quicklookutil.auto_find_on_window_from_ar(ar, safe_fraction = 1/16.)
        
        # ----------------dynaspect_onpulse in ax8------------------------------
        ax8 = plt.subplot2grid((5, 2), (4, 0), colspan=1, rowspan=1)
        if(args.verbose): print("Nenuplot: plot dynaspect_onpulse")
        plotfunc.dynaspect_onpulse(ar, ax8, left_onpulse=left_onpulse, righ_onpulse=righ_onpulse, threshold=threshold)
        
        # ----------------dynaspect_bandpass in ax9------------------------------
        ax9 = plt.subplot2grid((5, 2), (4, 1), colspan=1, rowspan=1, sharey=ax8, sharex=ax8)
        if(args.verbose): print("Nenuplot: plot dynaspect_bandpass")
        plotfunc.dynaspect_bandpass(ar, ax9, left_onpulse=left_onpulse, righ_onpulse=righ_onpulse, flatband=flatband, threshold=threshold)
        ax9.yaxis.tick_right()
        ax9.yaxis.set_ticks_position('both')
        ax9.yaxis.set_label_position("right")
        
        ar.tscrunch()  # no more need time dim
        
        if not (args.timepolar):
            # ----------------bandpass in ax3------------------------------
            ax3 = plt.subplot2grid((5, 5), (2, 2), colspan=3, rowspan=1)
            if(args.verbose): print("Nenuplot: plot bandpass")
            plotfunc.bandpass(ar, ax3, mask=True)
            ax3.yaxis.tick_right()
            ax3.yaxis.set_ticks_position('both')
            ax3.yaxis.set_label_position("right")
        
        # ----------------phase_freq I Q U V in ax4 to ax7------------------------------
        if (stokes):
            ax4 = plt.subplot2grid((5, 5), (3, 0), colspan=2, rowspan=1, sharex=ax2, sharey=ax8)
            ax5 = plt.subplot2grid((5, 5), (3, 2), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
            ax6 = plt.subplot2grid((5, 5), (3, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
            ax7 = plt.subplot2grid((5, 5), (3, 4), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
        else:
            ax4 = plt.subplot2grid((5, 4), (3, 0), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
            ax5 = plt.subplot2grid((5, 4), (3, 1), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
            ax6 = plt.subplot2grid((5, 4), (3, 2), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
            ax7 = plt.subplot2grid((5, 4), (3, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
        
        if(args.verbose): print("Nenuplot: plot phase_freq")
        plotfunc.phase_freq(ar, ax4, pol=0, flatband=flatband, stokes=stokes, threshold=threshold)
        
        if (ar.get_npol() > 1):
            plotfunc.phase_freq(ar, ax5, pol=1, flatband=flatband, stokes=stokes, threshold=threshold)
            ax5.axes.get_yaxis().set_visible(False)
        
            plotfunc.phase_freq(ar, ax6, pol=2, flatband=flatband, stokes=stokes, threshold=threshold)
            ax6.axes.get_yaxis().set_visible(False)
        
            plotfunc.phase_freq(ar, ax7, pol=3, flatband=flatband, stokes=stokes, threshold=threshold)
            ax7.axes.get_yaxis().set_visible(False)
        else:
            ax5 = plt.subplot2grid((5, 5), (3, 2), colspan=3, rowspan=1)
            plotfunc.zaped_bandpass(ar, ax5)
            ax5.yaxis.tick_right()
            ax5.yaxis.set_ticks_position('both')
            ax5.yaxis.set_label_position("right")
            #ax5.axes.get_yaxis().set_visible(False)
        
        # ----------------profil in ax1------------------------------
        ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=2, rowspan=2, sharex=ax2)
        
        ax1.set_xlim([0, 1])
        
        if(args.verbose): print("Nenuplot: plot profil")
        plotfunc.profil(ar, ax1, stokes=stokes)
        
        
        plt.draw()
        
        
        if(args.verbose): print("Nenuplot: savefig to "+args.path+args.name+'.'+form)
        plt.savefig(args.path+args.name+'.'+form, dpi=int(args.dpi), format=form)
        if (form=='pdf') and not (args.nosmall_pdf):
            quicklookutil.reduce_pdf(args.path, args.name+'.'+form, dpi=int(args.dpi))
    
    metadata =  metadata+'\n'+'---------------------processed--File(s)------------------------'
    if (args.sendmail) or (args.metadata_out):
        for file in ar_name:
            metadata =  metadata+'\n'+file
    
    metadata =  metadata+'\n'+'---------------------command-to-repeat-------------------------'
    metadata = metadata+'\n'+'python2.7 '+" ".join(argv)
    if not (args.arout): metadata = metadata+' -arout'
    
    if (needplot):
        if (args.sendmail):
            #mysendmail.sendMail(['aa@bb.com', 'bb@ff.com'], "New observation "+args.name, metadata, [args.path+args.name+'.pdf'])
            str_mail = "New observation "+str(ar.get_source())
            if (args.mailtitle):
                str_mail = str_mail+' '+args.mailtitle
            mysendmail.sendMail(args.sendmail, str_mail, metadata, [args.path+args.name+'.'+form])
    
    if(args.metadata_out):
        with open(args.path+args.name+'.metadata', "w") as text_file:
            text_file.write("%s" % metadata)

    if (needplot):
        if (args.uploadpdf):
            if (HOSTNAME=='nancep3'):
                uploadto.uptodatabf2(args.path+args.name+'.'+form, '/quicklook/')
        
        if (args.nopdf):
            if (os.path.isfile(args.path+args.name+'.'+form)):
                # print(args.path+args.name+'.'+form)
                os.remove(args.path+args.name+'.'+form)
        
        if (args.gui):
            plt.show()

if __name__ == "__main__":
    main()

