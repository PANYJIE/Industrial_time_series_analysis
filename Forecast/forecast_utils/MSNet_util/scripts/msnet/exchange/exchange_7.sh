export CUDA_VISIBLE_DEVICES=0

model_name=MSNet12
exp_index=7




  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 8 \
  --d_model 128 \
  --des 'Exp' \
  --itr 2 \
  --dropout 0.2 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-4\
  --train_epochs 20 \
  --lradj type3

