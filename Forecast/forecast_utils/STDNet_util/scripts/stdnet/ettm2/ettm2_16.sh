export CUDA_VISIBLE_DEVICES=0

model_name=STDNet
exp_index=16


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720_$exp_index \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.2 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 2e-3 \
  --train_epochs 20 \
  --lradj type3 










