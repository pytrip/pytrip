import importlib
import logging

import pytest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

entry_points = [
    ('pytrip.utils.rst_plot', 'main'),
    ('pytrip.utils.dvhplot', 'main'),
    ('pytrip.utils.rst2sobp', 'main'),
    ('pytrip.utils.trip2dicom', 'main'),
    ('pytrip.utils.dicom2trip', 'main'),
    ('pytrip.utils.gd2dat', 'main'),
    ('pytrip.utils.gd2agr', 'main'),
    ('pytrip.utils.bevlet2oer', 'main'),
    ('pytrip.utils.spc2pdf', 'main'),
    #    ('pytrip.utils.cubeslice', 'main')
]


@pytest.mark.parametrize("option_name", ["version", "help"])
@pytest.mark.parametrize("module_name,function_name", entry_points)
def test_call_cmd_option(module_name, function_name, option_name):
    with pytest.raises(SystemExit) as e:
        logger.info("Catching {:s}".format(str(e)))
        module = importlib.import_module(module_name)
        getattr(module, function_name)(['--' + option_name])
        assert e.code == 0


@pytest.mark.parametrize("module_name,function_name", entry_points)
def test_call_no_arguments(module_name, function_name):
    with pytest.raises(SystemExit) as e:
        logger.info("Catching {:s}".format(str(e)))
        module = importlib.import_module(module_name)
        getattr(module, function_name)([])
        assert e.code == 2
