# Image reconstruction with pytorch

The goal of this project is to create missing projection using 2 input projections (previous and next projection of targeted missing projection) ,obtained from DICOM file, using neural network, which will be implemented in Pytorch

## Installation

It's adviced to create fresh Python 3.9 virtual environment. Install required packages with following command:

```
$pip install -r requirements.txt
```

## Usage

### Step 0

Acquire DICOM files, which will be used for traning and testing the neural network(NN). Put the DICOM file for which you want to generate missing projection in separate directory and don't use it during neural network training.

### Step 1

You need to create a neural network model. To achive this, you need to run `piq_training.py`.

> Don't forget to specify correct path to your training DICOM files.

> You can always tweak the hyperparameters for training to achive better results.

### Step 2

Test your newly created NN model with `baseline_test.py`.

> Don't forget to specify correct path for your NN model and test DICOM file

### Step 3 

Generate missing projections by executing `generating_dicom.py`.

> Don't forget to specify correct path for your NN model and test DICOM file

If you want to generate missing projections with baseline model for testing purposes, you need to comment the line 43 and uncomment the line 44 inside `generating_dicom.py`.
