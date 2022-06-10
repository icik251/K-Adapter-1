#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=8GB
#SBATCH --output=output50.out
# Get the resources we need
# Example for final training of a model on the final data
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
pip install -r requirements.txt --user
cd examples
python run_finetune_sec_adapter_finbert.py --data_dirs ../data/real_input_sec_train_test/final_data_train_test --finbert_path ../pretrained_models/FinBERT/ --kpi_model_path ../pretrained_models/xgboost_model_final.json --meta_sec_adapter ../pretrained_models/sec-adapter/pytorch_model.bin --percentage_change_type percentage_change_robust --task_name sec_regressor --output_dir ../output --do_train --num_train_epochs 30 --learning_rate 5e-5 --max_grad_norm 1 --train_batch_size 64 --eval_batch_size 64 --max_seq_length 256 --type_text mda_paragraphs --is_adapter --is_kpi_loss --final_epoch some_int --comment RNN_architect_1
