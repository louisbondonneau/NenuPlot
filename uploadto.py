import subprocess

def uptodatabf2(path_from, path_to):
    cmd="""rsync -av -e \"ssh \"  %s nfrplsobs@databf2dt:/databf2/nenufar-pulsar/%s""" % (path_from, path_to)
    p=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output, errors = p.communicate()
    print errors,output
