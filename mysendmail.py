import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEImage import MIMEImage
from email.MIMEText import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders
import os
import subprocess
import sys

def sendMail(to, subject, text, files=[],server="smtp.gmail.com"):
    assert type(to)==list
    assert type(files)==list
    fro = "undysputed.mailbox@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fro
    msg['To'] = COMMASPACE.join(to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach( MIMEText(text) )

    for file in files:
        if '.png' in file:
            part = MIMEImage(file,_subtype="png")
            part.set_payload( open(file,"rb").read() )
            Encoders.encode_base64(part)
	    part.add_header('Content-Disposition', 'attachment;filename="%s"' % os.path.basename(file))

	else:
            part = MIMEBase('application', "octet-stream")
            part.set_payload( open(file,"rb").read() )
            Encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment;filename="%s"' % os.path.basename(file))
        msg.attach(part)

    if 'cnrs-orleans' in server: 
        smtp = smtplib.SMTP(server, 25)
    if 'smtp.gmail.com' in server: 
        print("smtplib.SMTP")
        smtp = smtplib.SMTP(server, 587)
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login("undysputed.mailbox", "nenufarpulsar")
    #smtp.set_debuglevel(1)
    print("smtp.sendmail")
    smtp.sendmail(fro, to, msg.as_string() )
    print("smtp.close")
    smtp.close()



def sendMail_sub(to, subject, text, files=[]):
    attach = ' -a %s' % files[0]
    if len(files) > 1:
        for i in range(len(files-1)):
            attach = attach + ' -a %s' % files[i+1]
        
    cmd="""echo "%s" | mail -r undysputed@obs-nancay.fr -s '%s' %s %s""" % (text, subject, attach, ' '.join(str(to).strip('[').strip(']').split(',')))
    #print("\n"+cmd+"\n")
    p=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output, errors = p.communicate()
    print errors,output
    
    
    
    

#if __name__ == '__main__':
#    sendMail(["louis.bondonneau@cnrs-orleans.fr"], "hello","cheers", [''])
