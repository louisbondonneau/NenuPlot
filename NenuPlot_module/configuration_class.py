

# !/opt/python3/bin/python3
# -*- coding: utf-8 -*-
import os
import re

# from NenuPlot.log_class import Log_class
from .log_class import Log_class
from .methode_class import Methode


class Config_Reader():
    def __init__(self, config_file='/NenuPlot.conf', log_obj=None, verbose=False):
        self.verbose = verbose
        if log_obj is None:
            self.log = Log_class()
        else:
            self.log = log_obj
        self.config_file = config_file
        self.methode = Methode(log_obj=self.log)
        self.methode.check_file_validity(self.config_file)
        if (self.verbose):
            self.log.log('Read configuration from :%s' % (self.config_file), objet='CONFIG_READER')
        self.dico = {}
        config_file_obj = open(self.config_file, "r")
        for line in config_file_obj:
            if not re.search('^;', line):
                if re.search('^\[', line):
                    last_sector = line.strip('\n').strip(' ').strip('[').rstrip(']')
                    self.dico[last_sector] = {}
                elif re.search("=", line):
                    line = line.strip('\n').strip(' ').split('=')
                    obj = line[0].strip(' ')
                    result = line[1].strip(' ').strip('\'')
                    self.dico[last_sector][obj] = result
                else:
                    self.log.error("do not understand :\"" + line + '\"', objet='CONFIG_READER')
        config_file_obj.close()

    def get_config(self, sector, obj):  # dico['MR']['LOG_FIRE']
        '''  get an object from CONFIG_FILE
        Arguments:
            object = PREFIX PREFIX_DATA LOG_FIRE IP PORT
            sector = PATH LOG BACKEND MR POINTAGE_AUTO_SERVICE
                     BACKEND_AUTO_SERVICE POINTAGE_LISTEN_SERVICE'''
        try:
            try:
                if not re.search(".", self.dico[sector][obj]):
                    int(self.dico[sector][obj])
                    return int(self.dico[sector][obj])
            except ValueError:
                pass

            try:
                if re.search(".", self.dico[sector][obj]):
                    float(self.dico[sector][obj])
                    return float(self.dico[sector][obj])
            except ValueError:
                pass

            if (self.dico[sector][obj] == 'True'):
                return True
            elif (self.dico[sector][obj] == 'False'):
                return False
            if (self.dico[sector][obj] == 'None'):
                return None
        except KeyError:
            self.log.error("can not find %s in [%s] in %s" % (obj, sector, self.config_file) , objet='CONFIG_READER')
            exit(0)
        return str(self.dico[sector][obj])
