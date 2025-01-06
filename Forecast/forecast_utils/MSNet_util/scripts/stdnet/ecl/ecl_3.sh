
export CUDA_VISIBLE_DEVICES=0

model_name=STDNet
exp_index=3



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 321 \
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
  
