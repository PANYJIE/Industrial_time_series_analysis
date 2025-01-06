export CUDA_VISIBLE_DEVICES=0

model_name=STDNet

#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_192 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --c_out 7 \
#  --d_model 256 \
#  --des 'Exp' \
#  --itr 5 \
#  --dropout 0.2 \
#  --batch_size 64 \
#  --learning_rate 0.0001 \
#  --loss MAE \
#  --n_layers 3 * [2] \
#  --n_mlp_units 3 * [[256, 256]] \
#  --weight_decay 1e-4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.2 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3
#  --n_layers 3 * [3] \
#  --n_mlp_units 3 * [[256, 256, 256]] \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.2 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3
#  --n_layers 3 * [3] \
#  --n_mlp_units 3 * [[256, 256, 256]] \




  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336_2 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3
#  --n_layers 3 * [3] \
#  --n_mlp_units 3 * [[256, 256, 256]] \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720_2 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3
#  --n_layers 3 * [3] \
#  --n_mlp_units 3 * [[256, 256, 256]] \


