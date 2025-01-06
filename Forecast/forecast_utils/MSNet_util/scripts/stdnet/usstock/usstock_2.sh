export CUDA_VISIBLE_DEVICES=0

model_name=STDNet
exp_index=2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/usstock/ \
  --data_path us_stock.csv \
  --model_id usstock_96_96_$exp_index \
  --model $model_name \
  --data custom2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 31 \
  --d_model 256 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-2 \
  --train_epochs 20 \
  --lradj type3

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/usstock/ \
  --data_path us_stock.csv \
  --model_id usstock_96_192_$exp_index \
  --model $model_name \
  --data custom2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 31 \
  --d_model 256 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-2 \
  --train_epochs 20 \
  --lradj type3


  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/usstock/ \
  --data_path us_stock.csv \
  --model_id usstock_96_336_$exp_index \
  --model $model_name \
  --data custom2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --c_out 31 \
  --d_model 256 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-2 \
  --train_epochs 20 \
  --lradj type3


  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/usstock/ \
  --data_path us_stock.csv \
  --model_id usstock_96_720_$exp_index \
  --model $model_name \
  --data custom2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 31 \
  --d_model 256 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-2 \
  --train_epochs 20 \
  --lradj type3
