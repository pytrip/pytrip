import os
import sys
import pytest


def image_type(filename):
    """
    Determine the type of image file.
    
    This function replaces imghdr.what() which was removed in Python 3.13.
    For Python < 3.13, it uses imghdr. For Python 3.13+, it checks file signatures directly.
    
    Args:
        filename: Path to the image file
        
    Returns:
        Image type as string (e.g., 'png', 'jpeg') or None if not recognized
    """
    if sys.version_info < (3, 13):
        import imghdr
        return imghdr.what(filename)
    # For Python 3.13+, check file signature directly
    with open(filename, 'rb') as f:
        header = f.read(32)
    
    # PNG signature
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    # JPEG signature
    if header.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    # GIF signature
    if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        return 'gif'
    # BMP signature
    if header.startswith(b'BM'):
        return 'bmp'
    
    return None


@pytest.fixture(scope='module')
def ctx_corename():
    return os.path.join('tests', 'res', 'TST003', 'tst003000')


@pytest.fixture(scope='module', params=["", ".ctx", ".hed", ".CTX", ".HED", ".hed.gz", ".ctx.gz"])
def ctx_allnames(ctx_corename):
    return ctx_corename


@pytest.fixture(scope='module')
def dos_filename():
    return os.path.join('tests', 'res', 'TST003', 'tst003001.dos.gz')


@pytest.fixture(scope='module')
def let_filename():
    return os.path.join('tests', 'res', 'TST003', 'tst003001.dosemlet.dos.gz')


@pytest.fixture(scope='module')
def vdx_filename():
    return os.path.join('tests', 'res', 'TST003', 'tst003000.vdx')


@pytest.fixture(scope='module')
def rst_filename():
    return os.path.join('tests', 'res', 'TST003', 'tst003001.rst')


def exists_and_nonempty(filename):
    """check if file exists and its size is greater than 1 byte"""
    return os.path.exists(filename) and os.path.getsize(filename) > 1
