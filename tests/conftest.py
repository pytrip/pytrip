import os
import pytest


@pytest.fixture(scope='module')
def test_ctx_corename():
    return os.path.join('tests', 'res', 'TST003', 'tst003000')


@pytest.fixture(scope='module', params=["", ".ctx", ".hed", ".CTX", ".HED", ".hed.gz", ".ctx.gz"])
def test_ctx_allnames(test_ctx_corename):
    return test_ctx_corename
