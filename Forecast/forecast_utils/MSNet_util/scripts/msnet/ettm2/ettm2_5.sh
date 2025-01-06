export CUDA_VISIBLE_DEVICES=0

model_name=MSNet5
exp_index=5




  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96_$exp_index \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.4 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-4 \
  --train_epochs 10 \
  --lradj type3