#!/usr/bin/env python3
import os
import re
import numpy as np

# time liba
from astropy.time import Time
from datetime import datetime

from .methode_class import Methode


class Log_class(object):
    def __init__(self, logname=None, verbose=False, print_log=False):
        self.set_print(print_log)
        if(verbose):
            self.set_print(True)
            self.verbose = verbose
        time_now = Time.now()
        if logname is None:
            logname = 'NONAME_'
        self.string_start = time_now.isot.replace('-', '').replace(':', '').replace('T', '_').split('.')[0]
        self.logname = logname + '_' + self.string_start
        self.log_dir = ''  # current dir if error at start
        self.log_length = 20
        self.methode = Methode(log_obj=self, verbose=False)

    def set_dir(self, directory):
        try:
            self.methode.check_file_validity(self.log_dir + self.logname + '.log')
            self.methode.movefile(self.log_dir + self.logname + '.log', str(directory) + self.logname + '.log')
        except NameError:
            pass
        try:
            self.methode.check_file_validity(self.log_dir + self.logname + '.warning')
            self.methode.movefile(self.log_dir + self.logname + '.warning', str(directory) + self.logname + '.warning')
        except NameError:
            pass
        try:
            self.methode.check_file_validity(self.log_dir + self.logname + '.error')
            self.methode.movefile(self.log_dir + self.logname + '.error', str(directory) + self.logname + '.error')
        except NameError:
            pass
        self.log_dir = str(directory)

    def set_logname(self, logname):
        try:
            self.methode.check_file_validity(self.log_dir + self.logname + '.log')
            self.methode.movefile(self.log_dir + self.logname + '.log', self.log_dir + logname + '_' + self.string_start + '.log')
        except NameError:
            pass
        try:
            self.methode.check_file_validity(self.log_dir + self.logname + '.warning')
            self.methode.movefile(self.log_dir + self.logname + '.warning', self.log_dir + logname + '_' + self.string_start + '.warning')
        except NameError:
            pass
        try:
            self.methode.check_file_validity(self.log_dir + self.logname + '.error')
            self.methode.movefile(self.log_dir + self.logname + '.error', self.log_dir + logname + '_' + self.string_start + '.error')
        except NameError:
            pass
        self.logname = logname + '_' + self.string_start

    def set_print(self, res):
        self.print_log = bool(res)

    def __string_formating(self, msg, objet='LOG', timing=True):
        msg = msg.strip('\r').strip('\n').split('\n')
        string = []
        if timing is True:
            time_string = self.__timing_string()
        for imsg in range(len(msg)):
            if timing is True:
                msg[imsg] = time_string + ' ' + msg[imsg]
            string_tmp = "%s: %" + str(self.log_length - len(objet) + len(msg[imsg])) + "s"
            string.append(string_tmp % (objet, msg[imsg]))
        return string

    def log(self, msg, objet='LOG', timing=True):
        string = self.__string_formating(msg, objet=objet, timing=timing)
        with open(self.log_dir + self.logname + '.log', 'a') as log_file:
            for istring in string:
                # FIXIT
                # print(istring, file=log_file)
                log_file.write(istring + "\n")
                if(self.print_log):
                    print('LOG: ' + istring)

    def warning(self, msg, objet='WARNING', timing=True):
        self.log(msg, objet=objet, timing=timing)
        string = self.__string_formating(msg, objet=objet, timing=timing)
        with open(self.log_dir + self.logname + '.warning', 'a') as warning_file:
            for istring in string:
                # FIXIT
                # print(istring, file=warning_file)
                warning_file.write(istring + "\n")

    def error(self, msg, objet='ERROR', timing=True):
        self.log(msg, objet=objet, timing=timing)
        string = self.__string_formating(msg, objet=objet, timing=timing)
        with open(self.log_dir + self.logname + '.error', 'a') as error_file:
            for istring in string:
                # FIXIT
                # print(istring, file=error_file)
                error_file.write(istring + "\n")

    def __timing_string(self):
        time_string = datetime.now()
        mili = time_string.strftime("%f")[:3]
        time_string = time_string.strftime("%Y-%m-%d %H:%M:%S.") + mili
        return time_string

    def filter(self, msg, objet='Filter', timing=True):
        msg = msg.strip('\r').strip('\n')
        if (re.search(' e:', msg.lower())) or (re.search('err', msg.lower())):
            self.error(msg, objet=objet, timing=timing)
        elif (re.search(' w:', msg.lower())) or (re.search('warn', msg.lower())):
            self.warning(msg, objet=objet, timing=timing)
        else:
            self.log(msg, objet=objet, timing=timing)

    def sort_log(self):
        self.__sort_file(self.log_dir + self.logname + '.log')

    def __sort_file(self, file):
        log_file = open(file, "r")
        time_list = []
        listed_file = []
        for line in log_file:
            line_tmp = line.strip('\r').strip('\n')
            listed_file.append(line_tmp)
            if (line_tmp != ''):
                iso_str = line_tmp[22:22 + 23].replace('/', '-')
                iso_str = np.asarray(iso_str.split(' '))
                try:
                    iso_str = iso_str[(iso_str != '')]
                    if (len(iso_str) >= 2):
                        iso_str = iso_str[0] + ' ' + iso_str[1]
                        time_obj = Time(iso_str, format='iso', scale='utc')
                        time_list.append(time_obj.unix)
                    else:
                        time_list.append(0)
                except Exception as e:
                    self.warning('%s while converting \'%s\' as iso in line \'%s\'' % (e, line_tmp[22:22 + 23], line_tmp), objet='LOG')
                    time_list.append(0)
            else:
                time_list.append(0)
        log_file.close()

        os.remove(file)
        with open(file, 'a') as log_file:
            for i in np.argsort(time_list):
                # FIXIT
                # print(listed_file[i], file=log_file)
                print(listed_file[i])
