import os
import pytest


@pytest.fixture(scope='module')
def ctx_corename():
    return os.path.join('tests', 'res', 'TST003', 'tst003000')


@pytest.fixture(scope='module', params=["", ".ctx", ".hed", ".CTX", ".HED", ".hed.gz", ".ctx.gz"])
def ctx_allnames(ctx_corename):
    return ctx_corename


@pytest.fixture(scope='module')
def vdx_filename():
    return os.path.join('tests', 'res', 'TST003', 'tst003000.vdx')


@pytest.fixture(scope='module')
def rst_filename():
    return os.path.join('tests', 'res', 'TST003', 'tst003001.rst')
