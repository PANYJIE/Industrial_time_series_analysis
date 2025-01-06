export CUDA_VISIBLE_DEVICES=0

model_name=MSNet9
exp_index=1




python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 862 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.2 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-4 \
  --train_epochs 20 \
  --lradj type3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 862 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.2 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-4 \
  --train_epochs 20 \
  --lradj type3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --c_out 862 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.2 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-4 \
  --train_epochs 20 \
  --lradj type3


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 862 \
  --d_model 128 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.2 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-4 \
  --train_epochs 20 \
  --lradj type3