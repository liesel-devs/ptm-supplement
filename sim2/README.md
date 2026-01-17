# Simulation study

This directory contains the code for the simulation study reported in Section 4 of the paper.

It contains one directory for every model. For each model, the parameters are defined in
a `params.csv` file, and a single model run is started by executing `launch.py`, passing
the job directory and the row of `params.csv` to be used. For example, to run the 
PTM model using the parameters defined in the first row of `params.csv`, you call:

```
python sim2/008-ptm/launch.py sim2/008-ptm 0
```

The model code in each subdirectory saves its output to an `out` directory, for example `010-ptm/out`. The data is collected by concatenating the output dataframes. It is included in `scaling/analysis/data`. The data analysis is conducted in `scaling/analysis/analyse.R`.

## Data must be downloaded manually before running code

To run simulation study code, download the data from Zenodo and replace the directory 
`sim2/data` with the extracted content of `data.zip`.

Zenodo record: https://doi.org/10.5281/zenodo.18202553


