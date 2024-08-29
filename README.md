# CBSuiTe---Calling CNV from WGBS
> CBSuiTe is a deep learning based software that performs CNV call predictions on WGBS data using read depth sequences.
> CBSuiTe can predict CNV on both germline and somatic data.

## Contents 

- [Installation](#installation)
- [Run CBSuiTe](#run-cbsuite)
- [Parameters](#parameters)
- [Citations](#citations)
- [License](#license)

## Installation

- CBSuiTe is written in python3 and can be run directly after installing the provided environment.

### Requirements

You can directly use ``cbsuite_environment.yml`` file to initialize conda environment with requirements installed:

```shell
$ conda env create --name cbsuite -f cbsuite_environment.yml
$ conda activate cbsuite
```
or install following dependencies:
* Python >= 3.8
* Torch >= 1.7.1
* Numpy
* Pandas
* Tqdm
* Scikit-learn
* Einops
* Samtools
  
## Run CBSuiTe
### Step 0: Install Conda and set up your environment
See details in [Installation](#installation).
### Step 1: Run preprocess script
CBSuiTe need to obtain read depth and methylation level information from ``bam`` files and convert to ``npy`` files.
You can simply run ``preprocess.sh`` by following commands:
```shell
$ source preprocess.sh
```
### Step 2: Run call script
Then you can predict CNV with CBSuiTe pretrained model.
```shell
$ source callCNV.sh
```


## Parameters
You can adjust these parameters in ``callCNV.sh`` to choose the way you want to extract cnv.

### Required Arguments

#### -m, --model
- If you want to use pretrained CBSuiTe weights choose one of the options: 
  (i) germlilne  (ii) somatic.
- Or you can use your own trained model by giving path/to/your/model.pt

#### -i, --input
- Relative or direct input directory path which stores input files(npy files) for CBSuiTe.
  
#### -o, --output
- Relative or direct output directory path to write CBSuiTe output file.

#### -n, --normalize
- Relative or direct path for mean&std stats of read depth values to normalize. These values are obtained precalculated from the training dataset.


### Optional Arguments

#### -g, --gpu
- Whether using GPU. 1 means using GPU, 0 means using CPU
  
#### -bs, --batch_size
- Batch size used to perform CNV call on the samples.
  
#### -v, --version
- Check the version of CBSuiTe.

#### -h, --help
- See help page.

## Citations
