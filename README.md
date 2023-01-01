## Model Fusion in Medical Image Segmentation: U-Net, Transformer and Domain-Shift
### Install Packages
```bash
# make sure the current directory is the project root directory
pip install -r requirements.txt
pip install -e .
```
To test whether installation is successful, please run at project root directory:
```bash
python scripts/test_install.py
```
### Set up Datasets
Please download [M&Ms](https://polybox.ethz.ch/index.php/s/ZmUDpfT8EwFpzte) and [CIFAR10](https://polybox.ethz.ch/index.php/s/rl1NUo9lDQYnCbS) 
datasets. Unzip them and fill in at the corresponding entries in`configs/general_configs.yml`.

### Run Experiments
We support custom model weights and pre-trained weights. Pre-trained weights are automatically downloaded when you run
the corresponding script. 
1. Run U-Net experiments with pre-trained weights:
   ```bash
   python scripts/evaluate_fused_u_nets_run.py --vendor A --save_dir "./" --exp_name "domain_generalization" --num_retrain_epochs 1 --ensemble_step 0.7 --square_factor "1/5" --retrain_fraction 0.1
   ```
   With custom weights (e.g.):
   ```bash
   python scripts/evaluate_fused_u_nets_run.py --vendor A --save_dir "./" --exp_name "domain_generalization" --num_retrain_epochs 1 --ensemble_step 0.7 --square_factor "1/5" --retrain_fraction 0.1 --model1_path "./seg_logs/2022_11_25_18_42_15_582775/" --model2_path "./seg_logs/2022_12_11_22_07_06_449724/"
   ```
1. Run ViT experiments with pre-trained weights:
   ```bash
   python scripts/evaluate_fused_vit_run.py --save_dir "./" --exp_name "data_parallel" --num_retrain_epochs 1 --ensemble_step 0.7 --square_factor "1/5" --retrain_fraction 0.1
   ```
   With custom weights (e.g.):
   ```bash
   python scripts/evaluate_fused_vit_run.py --save_dir "./" --exp_name "data_parallel" --num_retrain_epochs 1 --ensemble_step 0.7 --square_factor "1/5" --retrain_fraction 0.1 --model1_path "./clf_logs/2022_12_13_23_55_00_073244" --model2_path "./clf_logs/2022_12_13_23_55_00_086375"
   ```
