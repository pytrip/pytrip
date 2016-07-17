import sys
import os
import tarfile

if sys.version_info >= (3,):
    import urllib.request as urllib2
else:
    import urllib2


def get_files():
    # get plans from https://neptun.phys.au.dk/~bassler/TRiP/
    bname = "TST003"
    fname = bname + ".tar.gz"
    if not os.path.exists(fname):
        datafile = urllib2.urlopen(
            "https://neptun.phys.au.dk/~bassler/TRiP/" + fname)
        with open(fname, 'wb') as output:
            output.write(datafile.read())
    if not os.path.exists(bname):
        tfile = tarfile.open(fname, 'r:gz')
        tfile.extractall(".")
