import os
import numpy as np
from tinydb import TinyDB, Query

from log_class import Log_class
from methode_class import Methode
from configuration_class import Config_Reader

"""
# keys ############# def ##########
source          char        
sourceB         char        
sourceJ         char        

site             char       
mjd              float        
duration         float (sec)     
nbin             int        
nchan            int        
nsubint          int       
center_freq      float (MHz)        
bw               float (MHz)  

dm_init          float (pc.cm-3)        
dm_fit           float (pc.cm-3)    

snr_range
snr_norm
snr_psrchive
w10
w50          

rm_init          float (rad.m-2)              
rm_value        list(float) (rad.m-2)          
rm_err           list(float) (rad.m-2)         
rm_mjd           list(float) (rad.m-2)     

flux_value
flux_err
flux_freq
flux_bw
flux_duration
flux_snr
flux_w10
flux_w50
flux_tsky
flux_tinst
flux_effarea
flux_gain_corr

path_fits

IF NenuFAR
    path_parset
    path_log

##################################
"""

DATABASE = 'psrdatabase'

CONFIG_FILE = os.path.dirname(os.path.realpath(__file__)) + '/' + 'NenuPlot.conf'


class Psrdb():
    def __init__(self, logname=DATABASE, config_file=CONFIG_FILE, log_obj=None, verbose=False):
        self.verbose = verbose
        if log_obj is None:
            self.log = Log_class(logname=logname, verbose=verbose)
        else:
            self.log = log_obj
        self.methode = Methode(log_obj=self.log)
        self.config_file = config_file
        self.methode.check_file_validity(self.config_file)
        self.__init_config__()
        self._init_key_liste()

    def __init_config__(self):
        self.config = Config_Reader(config_file=self.config_file, log_obj=self.log)
        self.set_db_name(self.config.get_config('PSRDB', 'db_name'))
        self.round_float = self.config.get_config('PSRDB', 'round_float')

        requested_key = self.config.get_config('PSRDB', 'requested_key').split(',')
        self.requested_key = []
        for key in requested_key:
            self.requested_key.append(key.strip(' '))

    def set_db_name(self, db_name):
        self.db_name = str(db_name)
        self.db = TinyDB(self.db_name)

    def _init_key_liste(self):
        self.key = []
        for obs in self.db:
            for item in obs:
                self.key.append(item)
        self.key = np.unique(np.array(self.key))

    def insert(self, new_entry, force=False):
        if not (self._valide_entry(new_entry)) and not force:
            self.log.error("Error can not insert this new entry %s" % (new_entry), objet=DATABASE)
            raise IncertException('Unvalide entry')
        # TODO verif (mjd in past) & (freq > 0) & duplicates(same time, duration, freq, bw and site)
        self.db.insert(new_entry)

    def update(self, entry, force=False):
        query = self._searching_query(entry)
        if (len(self.db) != 0):
            result = self.search(query=query)[self.requested_key[0]]
        else:
            self.log.error("An update can't be the first entry in the database %s" % (self.db_name), objet=DATABASE)
            raise UpdateException('No entry to update')
            raise IncertException
        if (len(result) == 0):
            self.log.error("Can not update %s because the entry does not exist" % (entry), objet=DATABASE)
            raise UpdateException('No corresponding entry found')
        elif (len(result) > 1):
            self.log.error("Can not update %s because it corresponds to several (%d) entry" % (entry, len(result)), objet=DATABASE)
            raise UpdateException('Multiple corresponding entry found')
        if not (self._valide_update(entry)) and not force:
            self.log.error("Error can not update this entry %s" % (entry), objet=DATABASE)
            raise UpdateException('Unvalide entry')
        # TODO verif (mjd in past) & (freq > 0) & duplicates(same time, duration, freq, bw and site)
        self.db.update(entry, query)

    def _searching_query(self, new_entry):
        PSR = Query()
        query = "(PSR.source != '')"
        for key in self.requested_key:
            # query = query & (eval('Query().' + key) == new_entry[key])
            if (isinstance(new_entry[key], str)):
                query = query + (" & (PSR.%s == '%s')" % (key, new_entry[key]))
            if (isinstance(new_entry[key], float)):
                # entry = np.round(float(new_entry[key]), decimals=int(self.round_float))
                # query = query + (" & (PSR.%s == %f)" % (key, entry))
                entry_ceil = np.ceil(float(new_entry[key]) * (10**int(self.round_float))) / (10**int(self.round_float))
                entry_floor = np.floor(float(new_entry[key]) * (10**int(self.round_float))) / (10**int(self.round_float))
                query = query + (" & (PSR.%s >= %f) & (PSR.%s <= %f)" % (key, entry_floor, key, entry_ceil))
            if (isinstance(new_entry[key], int)):
                query = query + (" & (PSR.%s == %d)" % (key, new_entry[key]))
        # if (self.verbose):
        #     self.log.log("verbose - query:%s" % (query), objet=DATABASE)
        #     self.log.log("verbose - database len:%s" % (len(self.db)), objet=DATABASE)
        return eval(query)

    def _valide_entry(self, new_entry):
        missing_key = 0
        for key in self.requested_key:
            try:
                new_entry[key]
            except KeyError:
                missing_key += 1
                self.log.warning('Error key %s is missing in %s' % (key, new_entry), objet=DATABASE)

        if (missing_key == 0):
            # TODO search if entry already existe comparing with requested_key
            query = self._searching_query(new_entry)
            if (len(self.db) != 0):
                result = self.search(query=query)[self.requested_key[0]]
                if (self.verbose):
                    print(self.search(query=query))
                    print(self.db.all())
                    self.log.log("verbose - len(result):%d" % (len(result)), objet=DATABASE)
                if (len(result) > 0):
                    self.log.warning("entry %s already exist %d time in the database %s" % (str(new_entry), len(result), self.db_name), objet=DATABASE)
                    return False
            else:
                self.log.warning("This is the first entry in the database %s" % (self.db_name), objet=DATABASE)
            return True
        else:
            return False

    def _valide_update(self, new_entry):
        missing_key = 0
        for key in self.requested_key:
            try:
                new_entry[key]
            except KeyError:
                missing_key += 1
                self.log.warning('Error key %s is missing in %s' % (key, new_entry), objet=DATABASE)

        if (missing_key == 0):
            # TODO search if entry already existe comparing with requested_key
            query = self._searching_query(new_entry)
            if (len(self.db) != 0):
                result = self.search(query=query)[self.requested_key[0]]
                if (self.verbose):
                    self.log.log("verbose - len(result):%d" % (len(result)), objet=DATABASE)
                if (len(result) == 0):
                    self.log.warning("No coresponding entry in the database %s in %s" % (str(new_entry), self.db_name), objet=DATABASE)
                    return False
                elif (len(result) == 1):
                    return True
                else:
                    self.log.warning("entry %s already exist %d time in the database %s" % (str(new_entry), len(result), self.db_name), objet=DATABASE)
                    return False

            else:
                self.log.warning("There is no entry in the database %s" % (self.db_name), objet=DATABASE)
            return False
        else:
            return False

    def search(self, query=Query().source != ''):
        res = {}
        result = self.db.search(query)
        self._init_key_liste()
        for item in self.key:
            res[item] = []
            for uniq_result in result:
                try:
                    res[item].append(uniq_result[item])
                except KeyError:
                    res[item].append(None)
        return (res)


class IncertException(Exception):
    def __init__(self, message):
        super(IncertException, self).__init__(message)


class UpdateException(Exception):
    def __init__(self, message):
        super(UpdateException, self).__init__(message)


if __name__ == "__main__":
    PSR = Psrdb(verbose=False)
    PSR.log.set_print(True)
    PSR.set_db_name('test_db.json')
    # PSR.db.truncate()
    PSR.insert({'source': 'B0809+74', 'mjd_start': 59800.234656, 'dm_init': 10.25, 'rm_init': 10.25,
                'dm_fit': 10.249999, 'obs_duration': 3600, 'centre_frequency': 60.1, 'bandwidth': 75, 'telescope': 'NenuFAR',
                'receiver_name': 'LaNewba', 'backend_name': 'LUPPI'})
    PSR.insert({'source': 'B0809+74', 'mjd_start': 59801.234656, 'dm_init': 10.25, 'rm_init': 10.25,
                'dm_fit': 10.249999, 'obs_duration': 3600, 'centre_frequency': 60.1, 'bandwidth': 75, 'telescope': 'NenuFAR',
                'receiver_name': 'LaNewba', 'backend_name': 'LUPPI'})
    PSR.insert({'source': 'B0809+74', 'mjd_start': 59900.837389, 'dm_init': 10.25, 'dm_fit': 10.24988,
                'obs_duration': 3600, 'centre_frequency': 60.1, 'bandwidth': 75, 'telescope': 'NenuFAR',
                'receiver_name': 'LaNewba', 'backend_name': 'LUPPI'})
    PSR.insert({'source': 'B1508+55', 'mjd_start': 59960.345566, 'obs_duration': 3600,
                'centre_frequency': 60.1, 'bandwidth': 75, 'telescope': 'NenuFAR',
                'receiver_name': 'LaNewba', 'backend_name': 'LUPPI'})
    try:
        PSR.insert({'source': 'B1508+55', 'mjd_start': 59960.345566, 'obs_duration': 3600,
                    'centre_frequency': 60.1, 'bandwidth': 75, 'telescope': 'NenuFAR',
                    'receiver_name': 'LaNewba', 'backend_name': 'LUPPI'})
    except IncertException:
        pass

    query = (Query().source == 'B0809+74') & (Query().mjd_start >= 59800) & (Query().rm_init != None)

    print(PSR.search(query=query)['mjd_start'])
    print(PSR.search(query=query)['rm_init'])
