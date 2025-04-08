This repo contains the STIS_pipeline_functions.py file that contains all of the framework for our uniform analysis of HST/STIS low resolution data as part of the SHEL archival project. Also contained are two Jupyter notebooks per each of the four test planets, one labeled -paper that contains the exact methods used for testing in the SHEL paper, and one labeled -examples that contains a more basic rundown of how to use the functions.

You need the following packages:

numpy

glob

pickle

scipy

pandas

matplotlib

seaborn

astropy

juliet

barycorrpy (requires astroquery=0.4.6)

transitspectroscopy

batman

lmfit

dynesty

as well as the limb darkening files from https://zenodo.org/records/6344946.
