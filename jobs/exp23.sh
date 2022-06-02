#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --output=output23.out
# Get the resources we need
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
pip install -r requirements.txt --user
cd examples
python run_finetune_sec_adapter_finbert.py --data_dirs ../data/real_input_sec/k_fold_1,../data/real_input_sec/k_fold_2,../data/real_input_sec/k_fold_3,../data/real_input_sec/k_fold_4,../data/real_input_sec/k_fold_5 --finbert_path ../pretrained_models/FinBERT/ --kpi_model_path ../pretrained_models/xgboost_model_all_features.json --percentage_change_type percentage_change_robust --task_name sec_regressor --output_dir ../output --do_train --num_train_epochs 100 --learning_rate 5e-5 --max_grad_norm 1 --grouped_params --train_batch_size 64 --eval_batch_size 64 --max_seq_length 64 --type_text mda_sentences
