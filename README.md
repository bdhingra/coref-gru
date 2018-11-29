# Coref-GRU

This is the code for reproducing the NAACL 2018 paper [Neural Models for Reasoning over Multiple Mentions using Coreference](http://aclweb.org/anthology/N18-2007).

The model architectures and preprocessing details are slightly different for the LAMBADA and WikiHop datasets. Hence they are in separate branches of this repository. The `master` branch holds the code for WikiHop dataset, and the `lambada` branch holds the code for LAMBADA dataset.

## Prerequisites

- Python 2.7
- tensorflow_gpu==1.3.0
- tensorflow==1.11.0 (CPU only)
- numpy==1.13.3
- Maybe more, just use `pip install` if you run into problems.

## Data and Pretrained Models

1. Download the pre-processed data with coreference annotations for both datasets from [here](http://curtis.ml.cmu.edu/datasets/coref-gru/datasets). To preprocess some other data, you can follow the steps in this [Codalab bundle](https://worksheets.codalab.org/bundles/0xa32632e302a24e05821fd021055e5f79/) which was used for official evaluation on WikiHop.
2. Extract the tar archives into separate folders and create symbolic links to these locations. On Unix this can be done by `ln -s <path_to_lambada_directory> lambada` and `ln -s <path_to_wikihop_directory> wikihop`.
3. Download the pre-processed Glove embeddings from [here](http://curtis.ml.cmu.edu/datasets/preprocessed_glove/) and create a symbolic link to this as well: `ln -s <path_to_glove_file> glove`.

If you want to run predictions using pre-trained models, download these from [here](http://curtis.ml.cmu.edu/datasets/coref-gru/models).

NOTE: The validation and test sets in the pre-processed data above only contain a subset of all the examples in the released versions. The remaining ones either could not be pre-processed reliably, or (in the case of LAMBADA) did not fit the reading comprehension framework. This means the accuracy numbers printed during training and testing need to be multiplied with a constant factor to obtain the actual accuracies. For LAMBADA validation set this is 3951 / 4869, for the test set it is 4166 / 5153, and for the Wikihop dev set it is 5033 / 5129.

## Training / Testing

The main script for running is `run.py`.
```
$ python run.py --help
usage: run.py [-h] [--mode MODE] [--nlayers NLAYERS] [--dataset DATASET]
              [--seed SEED] [--save_path SAVE_PATH] [--reload]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           run mode - (0-train only, 1-test only, 2-val only)
                        (default: 0)
  --nlayers NLAYERS     Number of reader layers (default: 3)
  --dataset DATASET     Location of training, test and validation files.
                        (default: wikihop)
  --seed SEED           Seed for different experiments with same settings
                        (default: 1)
  --save_path SAVE_PATH
                        Location of output logs and model checkpoints.
                        (default: None)
  --reload
```

To train a model from scratch:
```
python run.py --dataset lambada/wikihop
```
Model hyperparameters can be set in `config.py`. On LAMBADA, 1 epoch takes approximately 10 hours. The accuracy at the end of 2 epochs is close to 69%. On WikiHop, 1 epoch taked approximately 15 hours. The accuracy at the end of 2 epochs is close to 56% and after 4 epochs it is close to 57%. Note that these numbers are on a subset of the data, to get the actual accuracy you need to multiply them by the factors listed above.

To test an already trained model on the test set.
```
python run.py --mode 1 --dataset lambada/wikihop --save_path <top_level_directory_with_saved_model>
```
To test on the dev set pass in `--mode 2`.

## Contributors

If you use this code please cite the following:

Bhuwan Dhingra, Qiao Jin, Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov.
Neural Models for Reasoning over Multiple Mentions Using Coreference. NAACL-HLT (2) 2018: 42-48
```
@inproceedings{DBLP:conf/naacl/DhingraJYCS18,
  author    = {Bhuwan Dhingra and
               Qiao Jin and
               Zhilin Yang and
               William W. Cohen and
               Ruslan Salakhutdinov},
  title     = {Neural Models for Reasoning over Multiple Mentions Using Coreference},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               NAACL-HLT, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 2 (Short
               Papers)},
  pages     = {42--48},
  year      = {2018},
  crossref  = {DBLP:conf/naacl/2018-2},
  url       = {https://aclanthology.info/papers/N18-2007/n18-2007},
  timestamp = {Wed, 30 May 2018 15:04:59 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/naacl/DhingraJYCS18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
