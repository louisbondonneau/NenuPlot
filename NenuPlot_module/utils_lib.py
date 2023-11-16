import os
from os.path import basename
import os.path
from subprocess import check_output, CalledProcessError

# try:
#    iers.conf.auto_max_age = None
#    iers.conf.auto_download = False
#    iers.IERS.iers_table = iers.IERS_A.open('/home/lbondonneau/lib/python/astropy/utils/iers/data/finals2000A.all')
# except:
#    print('WARNING: Can not use iers.conf probably due to the astropy version')


def reduce_pdf(pdf_path, pdf_name, dpi=400, log_obj=None):
    """Function to reduce the size of a pdf using Ghostscript
    """
    try:
        commande = 'gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -r' + \
            str(int(dpi)) + ' -dNOPAUSE -dQUIET -dBATCH -sOutputFile=' + pdf_path + pdf_name + 'TMPfile.tmp ' + pdf_path + pdf_name
        # TODO try/catch on subprocess
        output = check_output(commande, shell=True)
        commande = 'mv ' + pdf_path + pdf_name + 'TMPfile.tmp ' + pdf_path + pdf_name
        output = check_output(commande, shell=True)
    except CalledProcessError:
        log_obj.warning("WARNING: Command \'gs\' not found, but can be installed with:", objet='PDF_reduction')
        log_obj.warning("         sudo apt install ghostscript", objet='PDF_reduction')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ------------------------------ HTML stuff -----------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def PNG_to_html_img_tag(png_file):
    """Function to convert a PNG file in html code.
       Input:
            png_file: a .png file
    """
    data_uri = open(png_file, 'rb').read().encode('base64').replace('\n', '')
    img_tag = '<div ><img src="data:image/png;base64,{0}" width="640" alt="Computer Hope"></div>'.format(data_uri)
    img_tag = '<h1>' + basename(png_file).split('.')[0] + '</h1>' + img_tag
    return img_tag


def ASCII_to_html(ascii_file):
    """Function to convert a txt file in html code.
       Input:
            ascii_file: a .txt file
    """
    data_uri = open(ascii_file, 'rb').read().replace('\n',
                                                     '</pre><pre class="tab">')
    html = ('<h1>%s</h1><body><pre class="tab">%s</pre></body>' %
            (basename(ascii_file).split('.')[0], data_uri))
    return html


def make_html_code(html_body):
    """Function to finalize the html code.
       Input:
            html_body: string containing the html body of the code
    """
    html_code = ('<!DOCTYPE html>%s<html></html>' % (html_body))
    return html_code


def all_to_html(WORKDIR):
    """Function to convert and stack all txt an PNG file in an html code

       Input:
            WORKDIR: path to the working directory

    """
    html_body = ''
    if(os.path.isdir(WORKDIR)):
        for the_file in sorted(os.listdir(WORKDIR)):
            file_path = os.path.join(WORKDIR, the_file)
            try:
                if os.path.isfile(file_path):
                    print(file_path)
                    if (file_path.split('.')[1] == 'png'):
                        html_body = html_body + PNG_to_html_img_tag(file_path)
                    if (file_path.split('.')[1] == 'txt'):
                        html_body = html_body + ASCII_to_html(file_path)
            except Exception as e:
                print(e)
    html_code = make_html_code(html_body)
    return html_code