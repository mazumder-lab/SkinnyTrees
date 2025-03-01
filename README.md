# SkinnyTrees

This is our implementation of End-to-end Feature Selection Approach for Learning Skinny Trees as described in our manuscript.

[End-to-end Feature Selection Approach for Learning Skinny Trees](https://arxiv.org/abs/2310.18542) by Shibal Ibrahim, Kayhan Behdin, Rahul Mazumder

## Installation
This code uses tensorflow 2.9, scikit-learn, numpy, pandas, matplotlib. Please make sure tensorflow installation is done properly
such that it recognizes GPU device.
 
## Proposed Models
* `SkinnyTrees`: End-to-end Feature Selection Approach for Learning Skinny Trees with Group L0-L2.

## Running Code
Scripts folder contains different bash scripts for running SkinnyTrees on synthetic dataset as well as real-world datasets.

For example, SkinnyTrees can be run for different hyperparameters as follows:
```bash
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'churn' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 2000 --version 1 --tuning_seed 0 --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata
```

## Citing SkinnyTrees
If you find our repository useful in your research, please consider citing the following paper.

```
@InProceedings{pmlr-v207-ibrahim24,
  title={End-to-end Feature Selection Approach for Learning Skinny Trees},
  author={Shibal Ibrahim and Kayhan Behdin and Rahul Mazumder},
  booktitle={Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  year={2024},
  series={Proceedings of Machine Learning Research},
  month={25--27 Apr},
  publisher={PMLR},
}

```


