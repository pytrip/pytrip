from distutils.core import setup
setup(name='pytrip',
      description='Python scripts for TRiP and virtuos',
      author='Niels Bassler',
      author_email='bassler@phys.au.dk',
      url='http://aptg-trac.phys.au.dk/pytrip/',
      packages=['pytrip','pytrip/res'],
      package_data={'pytrip': ['pygtktrip.glade','data/*.dat']},
      scripts=['gtrip','pytrip/gd2dat.py','pytrip/bevlet2oer.py'],
      version='0.1_svn',
      )
