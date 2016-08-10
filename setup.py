
from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, Extension
from setuptools.command.install import install

####################################################################
#                    CONFIGURATION
####################################################################

# do the build/install
setup(
      name="gns_flyspec",
      version="0.1",
      description="automated flyspec flux calculator",
      long_description="automated flyspec flux calculator",
      author="Nial Peters",
      author_email="nonbiostudent@hotmail.com",
      url="",
      license="GPL v3",
      package_dir={'':'src'},
      packages=['gns_flyspec'],
      entry_points={'console_scripts': ['flyspec-png = gns_flyspec.png_loader_script:main',
                                              'flyspec-realtime = gns_flyspec.main_script:main',
                                              'flyspec-ui = gns_flyspec.main_script:main_no_realtime',
                                              'flyspec-to-csv = gns_flyspec.binary_to_csv:main']}
      )
