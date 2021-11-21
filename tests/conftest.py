import os
import pytest


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
