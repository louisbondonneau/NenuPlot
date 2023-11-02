# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import socket
import subprocess
import signal

# from NenuPlot.log_class import Log_class


UPLOAD = 'upload file'
MAIL = 'send mail'


class Methode():
    def __init__(self, log_obj=None, verbose=True):
        self.verbose = verbose
        if log_obj is None:
            from log_class import Log_class
            self.log = Log_class()
        else:
            self.log = log_obj

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -------------------------------- CHEKER -------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def check_file_validity(self, file_):
        ''' Check whether the a given file exists, readable and is a file '''
        if not os.access(file_, os.F_OK):
            if (self.verbose):
                self.log.error("File '%s' does not exist" % (file_))
            raise NameError(0, "File '%s' does not exist" % (file_))
        if not os.access(file_, os.R_OK):
            if (self.verbose):
                self.log.error("File '%s' not readable" % (file_))
            raise NameError(1, "File '%s' not readable" % (file_))
        if os.path.isdir(file_):
            if (self.verbose):
                self.log.error("File '%s' is a directory" % (file_))
            raise NameError(2, "File '%s' is a directory" % (file_))

    def check_directory_validity(self, dir_):
        ''' Check whether the a given file exists, readable and is a file '''
        if not os.access(dir_, os.F_OK):
            if (self.verbose):
                self.log.error("Directory '%s' does not exist" % (dir_))
            raise NameError(3, "Directory '%s' does not exist" % (dir_))
        if not os.access(dir_, os.R_OK):
            if (self.verbose):
                self.log.error("Directory '%s' not readable" % (dir_))
            raise NameError(4, "Directory '%s' not readable" % (dir_))
        if not os.path.isdir(dir_):
            if (self.verbose):
                self.log.error("'%s' is not a directory" % (dir_))
            raise NameError(5, "'%s' is not a directory" % (dir_))

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # ----------------------------- FILE MANAGER ----------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def remove(self, file_):
        ''' Check whether the a given file exists, readable and is a file '''
        self.check_file_validity(file_)
        os.remove(file_)

    def copyfile(self, file_, target):
        ''' Check whether the a given file exists, readable and is a file '''
        self.check_file_validity(file_)
        self.check_directory_validity(os.path.dirname(target))
        shutil.copyfile(file_, target)
        self.check_file_validity(target)

    def movefile(self, file_, target):
        ''' Check whether the a given file exists, readable and is a file '''
        self.check_file_validity(file_)
        if os.path.isdir(target):
            if (target[-1] != '/'):
                target = target + '/'
            self.check_directory_validity(target)
            target = target + os.path.basename(file_)
        elif os.path.isfile(target):
            self.check_file_validity(target)
        shutil.move(file_, target)
        self.check_file_validity(target)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # ---------------------------------- MAIL -------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def attach_file(self, msg, nom_fichier):
        if os.path.isfile(nom_fichier):
            piece = open(nom_fichier, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((piece).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "piece; filename= %s" % os.path.basename(nom_fichier))
            msg.attach(part)

    def sendMail(self, msg_to, subject, text, files=[]):
        msg = MIMEMultipart()
        msg['From'] = socket.gethostname() + '@obs-nancay.fr'
        msg['To'] = msg_to.strip("[").strip("]")
        msg['Subject'] = subject
        msg.attach(MIMEText(text))
        if (len(files) > 0):
            for ifile in range(len(files)):
                self.attach_file(msg, files[ifile])
                # print(files[ifile])
        mailserver = smtplib.SMTP('localhost')
        # mailserver.set_debuglevel(1)
        mailserver.sendmail(msg['From'], msg['To'].split(','), msg.as_string())
        mailserver.quit()
        self.log.log('Send a mail: \"%s\"" to %s' % (subject, msg['To']), objet=MAIL)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -------------------------------- UPLOAD ------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def rsync(self, source_file, target_user, target_host, target_directory):
        dest_path = "%s@%s:%s" % (target_user, target_host, target_directory)
        cmd = ["rsync", "-av", source_file, dest_path]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            self.log.log('rsync success: [%s]' % (' '.join(cmd)), objet=UPLOAD)
        except subprocess.CalledProcessError as e:
            # la commande a retourné un code d'erreur
            self.log.error("rsync error: {}".format(e.returncode), objet=UPLOAD)
            self.log.error("rsync error: {}".format(e.output.decode('utf-8')), objet=UPLOAD)
        except Exception as err:
            self.log.error('rsync error: [{}]'.format(err), objet=UPLOAD)

    def scp(self, source_file, target_user, target_host, target_directory):
        cmd = "chmod 664 %s && " % (source_file)
        cmd = cmd + "scp -p %s %s@%s:%s && " % (source_file, target_user, target_host, target_directory)
        proc = subprocess.Popen(cmd, shell=True)
        try:
            # self.log.log('Start Command: [%s]' % (cmd), objet='scp')
            stdout_data, stderr_data = proc.communicate(timeout=900)
            if proc.returncode != 0:
                self.log.error(
                    "%r failed, status code %s stdout %r stderr %r" % (
                        cmd, proc.returncode,
                        stdout_data, stderr_data), objet=UPLOAD)
            else:
                self.log.log('scp success: [%s]' % (cmd), objet=UPLOAD)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as err:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            self.log.error('scp error: [{}]'.format(err), objet=UPLOAD)
        except Exception as err:
            self.log.error('scp error: [{}]'.format(err), objet=UPLOAD)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -------------------------------- UPLOAD ------------------------------------
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


def string_error_formating(value, error):
    if (np.isnan(value)):
        return ("nan")
    elif (np.isnan(error)):
        return ("%.6f(nan)" % value)
    else:
        dec = 1 - int(np.log10(float(error)))
        if dec < 0:
            dec = 0
        err = float(10**dec) * error
        if (err >= 10) and (dec > 0):
            err /= 10
            dec -= 1
        err = int(np.ceil(err))
        if (err == 10) and (dec > 0):
            err = 1
            dec -= 1
        string = "%." + str(dec) + "f(" + str(err) + ')'
        return (string % np.round(value, decimals=dec))


def smoothGaussian(original_curve, size=0.07):
    sigma = float(len(original_curve)) * float(size) / 2. / 2.3548
    x = np.linspace(0, len(original_curve), len(original_curve))
    phase = float(len(original_curve)) / 2.
    gaussian = np.exp(-((x - phase) / sigma)**2 / 2)
    gaussian /= np.sum(gaussian)
    result = np.convolve(original_curve, gaussian, 'same')
    return result


def mad(data, axis='all'):
    if(axis == 'all'):
        data = np.nanmedian(np.abs(data - np.nanmedian(data)))
    else:
        if (axis != 0):
            axis_array = range(np.size(np.shape(data)))
            data = data.transpose(np.roll(axis_array, -axis))
        data = np.nanmedian(np.abs(data - np.ones(np.shape(data)) * np.nanmedian(data, axis=0)), axis=0)
        if (axis != 0):
            axis_array = range(np.size(np.shape(data)))
            data = data.transpose(np.roll(axis_array, axis))
    return data

# def smoothGaussian(list, size=0.07):
#    degree = int(np.ceil(size * np.size(list) / 4.))
#    window = degree * 2 - 1
#    weight = np.array([1.0] * window)
#    weightGauss = []
#    for i in range(window):
#        i = i - degree + 1
#        frac = i / float(window)
#        gauss = 1 / (np.exp((4 * (frac))**2))
#        weightGauss.append(gauss)
#    weight = np.array(weightGauss) * weight
#    smoothed = [0.0] * (len(list))
#    for i in range(len(smoothed) - window):
#        smoothed[i + int(window / 2)] = sum(np.array(list[i:i + window]) * weight) / sum(weight)
#    return np.asarray(smoothed)
