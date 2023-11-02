
# sourcing of the corresponding python environment using /home/vkondratiev/.psr-latest.bashrc and numpy scipy==1.2.3 astropy astroplan matplotlib lmfit tinydb pyfits
source /cep/lofar/pulsar/NenuPlot_DIR/python_env/bin/activate
# applying NenuPlot_v4 on an archive
python /cep/lofar/pulsar/NenuPlot_DIR/NenuPlot_v4/NenuPlot.py -fit_DM -fit_RM -defaraday -archive_out -minfreq 20 -maxfreq 80 /databf2/nenufar-pulsar/LT03/2023/01/B0950+08_D20230109T0217*.fits -b 4 -t 6
# get out this python environment
deactivate


