# The MK regression

## Synopsis
The MK regression is a hybrid of the McDonald-Kreitman (MK) test and the generalized linear model to infer genomic features responsible for positive selection. Unlike previous MK-based methods that can only analyze one genomic feature at a time, the MK regression can analyze multiple genomic simultaneously to disentangle direct and indirect effects on the rate of adaptation.

## Requirements
The MK regression is implemented in Python 3 with TensorFlow 2, NumPy, SciPy, and Pandas. It has been extensively tested in the following environment.

- python 3.7.7
- TensorFlow 2.1.0
- numpy 1.18.1
- scipy 1.4.1
- pandas 1.0.3
