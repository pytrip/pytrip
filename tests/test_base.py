import sys
import os
import tarfile

if sys.version_info >= (3,):
    import urllib.request as urllib2
else:
    import urllib2


def get_files():
    # get plans from https://neptun.phys.au.dk/~bassler/TRiP/
    dirname = "tests"
    bname = "TST003"
    fname = bname + ".tar.gz"
    fname_path = os.path.join(dirname, fname)
    if not os.path.exists(fname_path):
        datafile = urllib2.urlopen("https://neptun.phys.au.dk/~bassler/TRiP/" + fname)
        with open(fname_path, 'wb') as output:
            output.write(datafile.read())
    if not os.path.exists(os.path.join(dirname, bname)):
        tfile = tarfile.open(fname_path, 'r:gz')
        tfile.extractall(dirname)
