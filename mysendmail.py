import smtplib
import os
import subprocess
import sys
#from email.MIMEMultipart import MIMEMultipart
#from email.MIMEText import MIMEText
#from email.MIMEBase import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import socket

def sendMail_sub(to, subject, text, files=[]):
    attach = ' -a %s' % files[0]
    if len(files) > 1:
        for i in range(len(files-1)):
            attach = attach + ' -a %s' % files[i+1]
        
    cmd="""echo "%s" | mail -r undysputed@obs-nancay.fr -s '%s' %s %s""" % (text, subject, attach, ' '.join(str(to).strip('[').strip(']').split(',')))
    #print("\n"+cmd+"\n")
    p=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output, errors = p.communicate()
    print(errors,output)

def attach_file(msg, nom_fichier):
    piece = open(nom_fichier, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((piece).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "piece; filename= %s" % os.path.basename(nom_fichier))
    msg.attach(part)

def sendMail(to, subject, text, files=[]):
    msg = MIMEMultipart()
    msg['From'] = socket.gethostname()+'@obs-nancay.fr'
    msg['To'] = str(to).strip('[').strip(']')
    msg['Subject'] = subject
    msg.attach(MIMEText(text))
    if (len(files) > 0):
    	for ifile in range(len(files)):
            attach_file(msg, files[ifile])
            print(files[ifile])
    mailserver = smtplib.SMTP('localhost')
    #mailserver.set_debuglevel(1)
    mailserver.sendmail( msg['From'], msg['To'].split(','), msg.as_string())
    mailserver.quit()

#if __name__ == '__main__':
#    sendMail("[adresse1@cnrs-orleans.fr,adresse2@gmail.com]", "hello","cheers", ['/home/moi/test1.txt','/home/moi/test2.txt'])
