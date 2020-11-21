# The MK regression

## Synopsis

The MK regression is a hybrid of the McDonald-Kreitman (MK) test and the generalized linear regression to infer genomic features responsible for positive selection. Unlike previous MK-based methods that can only analyze one genomic feature at a time, the MK regression can analyze multiple genomic features simultaneously to disentangle direct and indirect effects on the rate of adaptation (ω<sub>α</sub> ).

## Requirements

The MK regression is implemented in Python 3 with TensorFlow 2, NumPy, SciPy, and Pandas. It has been extensively tested in the following environment.

- python 3.7.7
- TensorFlow 2.1.0
- numpy 1.18.1
- scipy 1.4.1
- pandas 1.0.3

## Quick guide

### Input files

The MK regression requires two tab-separated files, one for functional sites and the other for neutral sites. An example file of functional sites is as follows,
```
div_label    poly_label    feature_1    feature_2
0.0          0.0           0.621        0.778
1.0          0.0           0.356        0.132
0.0          1.0           0.019        1.074
0.0          0.0           0.443        -1.359
...
...
```
in which the first two columns are binary indicators of interspecies divergence and intraspecies polymorphism, respectively, followed by one or more genomic features.

Similarly, an example file of neutral sites is as follows,
```
div_label    poly_label
1.0          0.0
0.0          0.0
0.0          1.0
0.0          1.0
...
...
```
in which the first two columns are binary indicators of interspecies divergence and intraspecies polymorphism, respectively.

### Model fitting

**Step** 0: You can obtain the arguments of the MK regression.
```
python MKRegression.py --help
```

The following arguments are available in the MK regression.
```
optional arguments:
  -h, --help           show this help message and exit
  -n NEUTRAL_FILE      input file of neutral sites
  -f FOREGROUND_FILE   input file of functional sites
  -p PARAMETER_FILE    output file of estimated parameters
  -o OMEGA_ALPHA_FILE  output file of site-wise omega alpha (optional)
```

**Step** 1: As an example, you may obtain compressed input files for the MK regression paper from [Penn State's ScholarSphere](https://scholarsphere.psu.edu/resources/409ab824-65cf-40de-97e5-fc22dce9ad64). 

**Step** 2: Uncompress gzipped files.
```
gunzip chimp_0D_sites.tsv.gz   # functional sites
gunzip chimp_4D_sites.tsv.gz   # neutral sites
```

**Step** 3: Fit the MK regression model.
```
python MKRegression.py -n chimp_4D_sites.tsv  -f chimp_0D_sites.tsv -p estimated_parameter.tsv > log_likelihood.txt 
```
You should get two output files, estimated_parameter.tsv and log_likelihood.txt, which include the estimated regression coefficients and the log likelihood of the whole dataset, respectively.
