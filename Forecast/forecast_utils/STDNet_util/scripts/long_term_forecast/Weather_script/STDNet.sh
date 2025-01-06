export CUDA_VISIBLE_DEVICES=0

model_name=STDNet



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
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


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
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



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
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

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
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




python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 7 \
  --d_model 256 \
  --des 'Exp' \
  --itr 5 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss MAE \
  --weight_decay 1e-3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
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

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
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
