# Scaling experiments

This directory contains the code for the experiments on the scaling of runtime with increasing sample sizes reported in Section 4.2 and Figure 7.

It contains one directory for every model. For each model, a shell script `scaling.sh` is used to run the experiments. For some of the models, the parameters are defined in a `params.csv` file, for others they are passed directly in the shell script.

The model code in each subdirectory saves its output to an `out` directory, for example `010-ptm/out`. The data is collected by concatenating the output dataframes. It is included in `scaling/analysis/data`. The data analysis is conducted in `scaling/analysis/analyse.R`.
