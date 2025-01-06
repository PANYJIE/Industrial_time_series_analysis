
export CUDA_VISIBLE_DEVICES=0

model_name=STDNet
exp_index=1



python -u run_ab_T.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_ab_T_96_96_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 321 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-4 \
  --train_epochs 20 \
  --lradj type3



python -u run_ab_T.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_ab_T_96_192_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 321 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-4
  --train_epochs 20 \
  --lradj type3


python -u run_ab_T.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_ab_T_96_336_$exp_index \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --c_out 321 \
  --d_model 256 \
  --des 'Exp' \
  --itr 7 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 5e-4 \
  --train_epochs 20 \
  --lradj type3

python -u run_ab_T.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_ab_T_96_720_$exp_index \
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
  --weight_decay 5e-4 \
  --train_epochs 20 \
  --lradj type3
