[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7223895.svg)](https://doi.org/10.5281/zenodo.7223895)

# Multianalyzer
Rebinning of powder diffraction data taken with multiple analyzer crystals and a large area detector

This project is about the rebinning of data comming from the [high resolution powder diffraction beamline (ID22) at the ESRF](https://www.esrf.fr/id22). 
It implements the math which are described in [J. Appl. Cryst. (2021). 54, 1088-1099](https://doi.org/10.1107/S1600576721005288).

There are two implementations, Cython parallelized with OpenMP and OpenCL running on GPU.
The later is faster. 

The main executable is `id22rebin` which takes a BLISS-HDF5 file with ROI-collections in it (as generated on the ID22 beamline) and rebins the intensities 
of the diffeent ROI, after calculating their precise position.
The geometry is described in an output file from TOPAS.
Intensities are grouped per analyzer crystal and per column (depth dependent) and need to be normalized.   

## Installation
`pip install multianalyzer`

## Development version:
```
git clone https://github.com/kif/multianalyzer
cd multianalyzer
pip install -r requirements.txt
pip install .
```

The tests are only validating input file reading. 