import os


def get_files():
    """
    get plans from https://neptun.phys.au.dk/~bassler/TRiP/
    """
    dirname = os.path.join('tests', 'res')
    bname = "TST003"
    return os.path.join(dirname, bname)
