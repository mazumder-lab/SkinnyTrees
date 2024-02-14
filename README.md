# SkinnyTrees

This is our implementation of End-to-end Feature Selection Approach for Learning Skinny Trees as described in our manuscript.

[End-to-end Feature Selection Approach for Learning Skinny Trees](https://arxiv.org/abs/2310.18542) by Shibal Ibrahim, Kayhan Behdin, Rahul Mazumder

## Installation
This code uses tensorflow 2.0, scikit-learn, numpy, pandas, matplotlib. Please make sure tensorflow installation is done properly
such that it recognizes GPU device.
 
## Proposed Models
* `SkinnyTrees`: End-to-end Feature Selection Approach for Learning Skinny Trees with Group L0-L2.

## Running Code
Scripts folder contains different bash scripts for running SkinnyTrees on synthetic dataset as well as real-world datasets for different seeds.

For example, SkinnyTrees can be run for different hyperparameters on one seed as follows:
```bash
/home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/src/elaani/elaani_census.py --load_directory '/home/gridsan/shibal/elaan/Census-Data' --seed 1 --relative_penalty 1.0 --grid_search 'reduced' --run_first_round --version 1 --eval_criteria 'mse' --logging
```

SkinnyTrees can be run on synthetic data for different hyperparameters on one seed as follows:
```bash
/home/gridsan/shibal/.conda/envs/aoas/bin/python /home/gridsan/shibal/elaan/src/elaanh/elaanh_synthetic.py  --dataset 'synthetic' --dist 'normal' --correlation 0.5 --seed $SLURM_ARRAY_TASK_ID --train_size 100 --version 1 --r 1.0 --Ki 10 --Kij 5
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


