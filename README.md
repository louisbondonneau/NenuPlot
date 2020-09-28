
python NenuPlot.py -flat_cleaner -metadata_out -iterative -arout  /MyPATH_to_archive/archive.ar



              input:
                    path to archive data files
              output:
                    archive.metadata      metadata ASCII file
                    archive.ar.clear      cleaned archive
                    archive.pdf           quicklook in pdf


              optional arguments:
                    -h, --help            show this help message and exit
                    -u PATH               output path (default in current directory)
                    -o NAME               output name (default is archive name)
                    -minfreq MINF         Minimum frequency to extract from the archives, no
                                          default 0 MHz exepted for nenufar at 20 MHz
                    -maxfreq MAXF         Maximum frequency to extract from the archives, no
                                          default 2^24 MHz exepted for nenufar at 87 MHz
                    -mask MASK            mask in input
                    -b BSCRUNCH           time scrunch factor (before CoastGuard)
                    -t TSCRUNCH           time scrunch factor (before CoastGuard)
                    -f FSCRUNCH           frequency scrunch factor (before CoastGuard)
                    -p                    polarisation scrunch (before CoastGuard)
                    -ba BSCRUNCH_AFTER    bin scrunch factor (after CoastGuard)
                    -ta TSCRUNCH_AFTER    time scrunch factor (after CoastGuard)
                    -fa FSCRUNCH_AFTER    frequency scrunch factor (after CoastGuard)
                    -fit_DM               will fit for a new DM value (at your own risk)
                    -v                    Verbose mode
                    -mail SENDMAIL        send the metadata and file by mail -mail [aaa@bbb.zz,
                                          bbb@bbb.zz]
                    -mailtitle MAILTITLE  modified the title of the mail
                    -metadata_out         copy the metadatafile in a directory
                    -gui                  Open the matplotlib graphical user interface
                    -arout                write an archive in output PATH/*.ar.clear
                    -maskout              write a dat file containing the mask PATH/*.mask
                    -uploadpdf            upload the pdf/png file
                    -nopdf                do not sauve the pdf/png file
                    -nostokes             do not transforme to nostokes for phase/freq
                    -dpi DPI              Dots per inch of the output file (default 400 for pdf
                                          and 96 for png)
                    -small_pdf            reduction of the pdf size using ghostscript
                    -png                  output in png
                    -noclean              Do not run coastguard
                    -iterative            Run the iterative cleaner
                    -flat_cleaner         flat bandpass for the RFI mitigation
                    -chanthresh CHANTHRESH
                                          chanthresh for loop 2 to X of CoastGuard (default is 6
                                          or 3.5 if -flat_cleaner)
                    -subintthresh SUBINTTHRESH
                                          subintthresh for loop 2 to X of CoastGuard (default 3)
                    -bad_subint BAD_SUBINT
                                          bad_subint for CoastGuard (default is 0.9 a subint is
                                          removed if masked part > 90 percent)
                    -bad_chan BAD_CHAN    bad_chan for CoastGuard (default is 0.5 a channel is
                                          removed if masked part > 50 percent)
                    -timepolar            plot phase time polar
                    -timenorme            normilized in time the phase/time plot
                    -threshold THRESHOLD  threshold on the ampl in dyn spect and phase/freq
                    -rm RM                defaraday with a new rm in rad.m-2
                    -defaraday            defaraday the signal (signal is not defaraday by
                                          default)
                    -dm DM                dedisperse with a new dm in pc.cm-3
                    -noflat               Do not flat the bandpass for plots
                    -nozapfirstsubint     Do not zap the first subint
                    -freqappend           append multiple archive in frequency dirrection
                    -singlepulses_patch   phasing patch when append multiple single pulses
                                          archive in frequency dirrection
                    -initmetadata         keep initial metadata

