export CUDA_VISIBLE_DEVICES=0

model_name=STDNet
exp_index=1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path trainjzl.csv \
  --model_id Exchange_96_96_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --pred_len 24 \
  --c_out 35 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3 \
  --train_epochs 20 \
  --lradj type3

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path trainjzl.csv \
  --model_id Exchange_96_96_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --pred_len 12 \
  --c_out 35 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3 \
  --train_epochs 20 \
  --lradj type3

    python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path trainjzl.csv \
  --model_id Exchange_96_96_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --pred_len 6 \
  --c_out 35 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3 \
  --train_epochs 20 \
  --lradj type3

    python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path trainjzl.csv \
  --model_id Exchange_96_96_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --pred_len 3 \
  --c_out 35 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3 \
  --train_epochs 20 \
  --lradj type3

  