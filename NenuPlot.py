import sys, os
import socket
HOSTNAME=socket.gethostname()
if (HOSTNAME=='undysputedbk1') or (HOSTNAME=='undysputedbk2'):
    sys.path = ['', '/usr/local/lib/python2.7/dist-packages/mpld3-0.3.1.dev1-py2.7.egg', '/home/louis/Pulsar/presto/python', '/home/louis/Pulsar/presto/lib/python', '/home/louis/Pulsar/lib/python2.7/site-packages', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', '/usr/lib/python2.7/dist-packages/wx-3.0-gtk2', '/home/louis/quicklook_NenuFAR2']
import psrchive as psr
import numpy as np
import iterative_cleaner_light
import quicklookutil
import argparse
import mysendmail
import uploadto

"""
This code provide a quicklook in pdf/png and many option:
    - frequency band selection -minfreq 1410 -maxfreq 1430
        grep channels in the requested frequency range
    - noclean to jump coast_guard
exp:
python /path_to_code/MeerPlot.py '/path_to_file/'

2020/03/10   add option nodefaraday because defaraday can cause psrsplit troubles
2020/04/20   fix option option metadata_out (previously away True)
2020/05/30   new options chanthresh and subintthresh for CoatGuard
2020/05/30   iterative_cleaner: bad_chan becomes more limiting 0.75 -> 0.5
2020/05/30   iterative_cleaner: flattenBP is no dependent at 20% from adjacent channels
2020/05/30   iterative_cleaner: default chanthresh 5 -> 3.5, default subintthresh 3.5 -> 3.8
2020/05/31   iterative_cleaner: bad_subint and bad_chan are now in option
2020/07/09   quicklookutil: add a try to use iers.conf (do not exist in old version of astropy)
2020/07/09   plotfunction: fix (bad normalisation in QUV when used timenorme and timenpolar options )
"""


parser = argparse.ArgumentParser(description="This code will.")
parser.add_argument('-u', dest='path',
                    help="output path (default current directory)")
parser.add_argument('-o', dest='name',
                    help="output name (default is archive name)")
parser.add_argument('-minfreq', dest='minf', default=0,
                    help="Minimum frequency to extract from the archives, no default 0 MHz exepted for nenufar at 20 MHz")
parser.add_argument('-maxfreq', dest='maxf', default=2**24,
                    help="Maximum frequency to extract from the archives, no default 2^24 MHz exepted for nenufar at 87 MHz")
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

parser.add_argument('-v', dest='verbose', action='store_true', default=False,
                    help="Verbose mode")
parser.add_argument('-mail', dest='sendmail',
                    help="send the metadata and file by mail -mail [aaa@bbb.zz, bbb@bbb.zz] ")
parser.add_argument('-mailtitle', dest='mailtitle',
                    help="modified the title of the mail")
parser.add_argument('-metadata_out', dest='metadata_out', action='store_true', default=False,
                    help="copy the metadatafile in a directory")
parser.add_argument('-gui', dest='gui', action='store_true', default=False,
                    help="Open the matplotlib graphical user interface")
parser.add_argument('-arout', dest='arout', action='store_true', default=False,
                    help="write an archive in output PATH/*.ar.clear")
parser.add_argument('-maskout', dest='maskout', action='store_true', default=False,
                    help="write a dat file containing the mask PATH/*.mask")
parser.add_argument('-uploadpdf', dest='uploadpdf', action='store_true', default=False,
                    help="upload the pdf/png file")
parser.add_argument('-nopdf', dest='nopdf', action='store_true', default=False,
                    help="do not sauve the pdf/png file")
parser.add_argument('-nostokes', dest='nostokes', action='store_true', default=False,
                    help="do not transforme to nostokes for phase/freq")
parser.add_argument('-dpi', dest='dpi', default=400,
                    help="Dots per inch of the output file (default 400 for pdf and 96 for png)") 
parser.add_argument('-small_pdf', dest='small_pdf', action='store_true', default=False,
                    help="reduction of the pdf size using ghostscript")
parser.add_argument('-png', dest='png', action='store_true', default=False,
                    help="output in png")

#cleaner
parser.add_argument('-noclean', dest='noclean', action='store_true', default=False,
                    help="Do not run coastguard")
parser.add_argument('-iterative', dest='iterative', action='store_true', default=False,
                    help="Run the iterative cleaner")
parser.add_argument('-flat_cleaner', dest='flat_cleaner', action='store_true', default=False,
                    help="flat bandpass for the RFI mitigation")
parser.add_argument('-chanthresh', dest='chanthresh', default=6,
                    help="chanthresh for loop 2 to X of CoastGuard (default is 6 or 3.5 if -flat_cleaner)")
parser.add_argument('-subintthresh', dest='subintthresh', default=4,
                    help="subintthresh for loop 2 to X of CoastGuard (default 3)")
parser.add_argument('-bad_subint', dest='bad_subint', default=0.9,
                    help="bad_subint for CoastGuard (default is 0.9 a subint is removed if masked part > 90 percent)")
parser.add_argument('-bad_chan', dest='bad_chan', default=0.5,
                    help="bad_chan for CoastGuard (default is 0.5 a channel is removed if masked part > 50 percent)")


parser.add_argument('-timepolar', dest='timepolar', action='store_true', default=False,
                    help="plot phase time polar")
parser.add_argument('-timenorme', dest='timenorme', action='store_true', default=False,
                    help="normilized in time the phase/time plot")
parser.add_argument('-threshold', dest='threshold',
                    help="threshold on the ampl in dyn spect and phase/freq")
parser.add_argument('-rm', dest='rm', default=0.0,
                    help="defaraday with a new rm in rad.m-2")
parser.add_argument('-nodefaraday', dest='nodefaraday', action='store_true', default=False,
                    help="do not defaraday the signal (signal is defaraday by default)")
parser.add_argument('-dm', dest='dm', default=0.0,
                    help="dedisperse with a new dm in pc.cm-3")
parser.add_argument('-noflat', dest='noflat', action='store_true', default=False,
                    help="Do not flat the bandpass for plots")
parser.add_argument('-nozapfirstsubint', dest='nozapfirstsubint', action='store_true', default=False,
                    help="Do not zap the first subint")
parser.add_argument('-freqappend', dest='freqappend', action='store_true', default=False,
                    help="append multiple archive in frequency dirrection")
parser.add_argument('-singlepulses_patch', dest='singlepulses_patch', action='store_true', default=False,
                    help="phasing patch when append multiple single pulses archive in frequency dirrection")
parser.add_argument('-initmetadata', dest='initmetadata', action='store_true', default=False,
                    help="keep initial metadata")
                    
parser.add_argument('INPUT_ARCHIVE', nargs='+', help="Path to the Archives")

args = parser.parse_args()
ar_name = args.INPUT_ARCHIVE



if (args.gui):
    import matplotlib
    import matplotlib.pyplot as plt
    import plotfunc
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotfunc

if (args.flat_cleaner == True) and (args.chanthresh == 6):
    args.chanthresh = 3.5

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
                                                            freqappend=args.freqappend,
                                                            singlepulses_patch=args.singlepulses_patch,
                                                            nodefaraday=args.nodefaraday)

if not (args.path):
    args.path = './'
if not (args.name):
    args.name = filename

fig = plt.figure(figsize=(10, 11))
plt.subplots_adjust(top=0.97, bottom=0.06,
                    left=0.1, right=0.915,
                    hspace=0.355, wspace=0.1)


# ----------------CLEAR archive------------------------------
if not (args.noclean) and not (args.mask):
    ar = iterative_cleaner_light.clean(ar,
                                       zapfirstsubint=zapfirstsubint,
                                       fast=fast,
                                       flat_cleaner=args.flat_cleaner,
                                       chanthresh=float(args.chanthresh),
                                       subintthresh=float(args.subintthresh),
                                       bad_subint=float(args.bad_subint),
                                       bad_chan=float(args.bad_chan))
    if args.maskout:
        np.savetxt(args.path+'/'+args.name+'.mask', ar.get_weights())

ar = quicklookutil.bandpass_filter(ar, minfreq=float(args.minf), maxfreq=float(args.maxf))

if (args.mask) and not (args.noclean):
    quicklookutil.apply_mask(ar, args.mask)

if (args.bscrunch_after > 1):
    if(ar.get_nbin()/int(args.bscrunch_after) < 8):
        args.bscrunch_after = int(ar.get_nbin()/8)
    ar.bscrunch(int(args.bscrunch_after))
if (args.tscrunch_after > 1):
    ar.tscrunch(int(args.tscrunch_after))
if (args.fscrunch_after > 1):
    ar.fscrunch(int(args.fscrunch_after))

if (args.arout):
    ar.update_centre_frequency()
    ar.dededisperse()
    if (ar.get_npol() > 1) and not (args.nodefaraday): ar.defaraday()
    if not (args.noclean):
        ar.unload(args.path+'/'+args.name+'.ar.clear')
    else:
        ar.unload(args.path+'/'+args.name+'.ar')
ar.centre_max_bin()

# ----------------HEADER in ax0------------------------------
ax0 = plt.subplot2grid((5, 5), (0, 2), colspan=3, rowspan=2, frameon=False)

metadata = quicklookutil.print_metadata(ar, initTFB=initTFB, initmetadata=args.initmetadata)

if(args.metadata_out):
    with open(args.path+'/'+args.name+'.metadata', "w") as text_file:
        text_file.write("%s" % metadata)
    

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
    plotfunc.phase_time(ar, ax2, timenorme=args.timenorme, pol=0, threshold=threshold)
    plotfunc.phase_time(ar, ax21, timenorme=args.timenorme, pol=1, stokes=stokes, threshold=threshold)
    plotfunc.phase_time(ar, ax22, timenorme=args.timenorme, pol=2, stokes=stokes, threshold=threshold)
    plotfunc.phase_time(ar, ax23, timenorme=args.timenorme, pol=3, stokes=stokes, threshold=threshold)
    ax21.axes.get_yaxis().set_visible(False)
    ax22.axes.get_yaxis().set_visible(False)
    ax23.axes.get_yaxis().set_visible(False)
    
else:
    # ----------------phase_time in ax2------------------------------
    ax2 = plt.subplot2grid((5, 5), (2, 0), colspan=2, rowspan=1)
    plotfunc.phase_time(ar, ax2, timenorme=args.timenorme, threshold=threshold)

# ----------------search for onpulse------------------------------------

left_onpulse, righ_onpulse = quicklookutil.auto_find_on_window_from_ar(ar, safe_fraction = 1/16.)

# ----------------dynaspect_onpulse in ax8------------------------------
ax8 = plt.subplot2grid((5, 2), (4, 0), colspan=1, rowspan=1)
plotfunc.dynaspect_onpulse(ar, ax8, left_onpulse=left_onpulse, righ_onpulse=righ_onpulse, threshold=threshold)

# ----------------dynaspect_bandpass in ax9------------------------------
ax9 = plt.subplot2grid((5, 2), (4, 1), colspan=1, rowspan=1, sharey=ax8, sharex=ax8)
plotfunc.dynaspect_bandpass(ar, ax9, left_onpulse=left_onpulse, righ_onpulse=righ_onpulse, flatband=flatband, threshold=threshold)
ax9.yaxis.tick_right()
ax9.yaxis.set_ticks_position('both')
ax9.yaxis.set_label_position("right")

ar.tscrunch()  # no more need time dim

if not (args.timepolar):
    # ----------------bandpass in ax3------------------------------
    ax3 = plt.subplot2grid((5, 5), (2, 2), colspan=3, rowspan=1)
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

plotfunc.profil(ar, ax1)


plt.draw()


plt.savefig(args.path+'/'+args.name+'.'+form, dpi=int(args.dpi), format=form)
if (form=='pdf') and (args.small_pdf):
    quicklookutil.reduce_pdf(args.path+'/', args.name+'.'+form, dpi=int(args.dpi))
    
if (args.sendmail):
    metadata =  metadata+'\n'+'----------------------------------------------------'
    for file in ar_name:
        metadata =  metadata+'\n'+file
    #mysendmail.sendMail(['aa@bb.com', 'bb@ff.com'], "New observation "+args.name, metadata, [args.path+args.name+'.pdf'])
    str_mail = "New observation "+str(ar.get_source())
    if (args.mailtitle):
        str_mail = str_mail+' '+args.mailtitle
    mysendmail.sendMail_sub(args.sendmail, str_mail, metadata, [args.path+args.name+'.'+form])

if (args.uploadpdf):
    if (HOSTNAME=='undysputedbk1') or (HOSTNAME=='undysputedbk2')  or (HOSTNAME=='nancep3'):
        uploadto.uptodatabf2(args.path+args.name+'.'+form, '/quicklook/')

if (args.nopdf):
    if (os.path.isfile(args.path+args.name+'.'+form)):
        print(args.path+args.name+'.'+form)
        os.remove(args.path+args.name+'.'+form)

if (args.gui):
    plt.show()
