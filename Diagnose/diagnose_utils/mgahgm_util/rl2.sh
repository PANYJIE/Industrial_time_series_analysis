conda activate nzf_deepfib
cd /home/rl/DATA/nzf1/mtag_game_2.0/
nohup python -u train.py --dataset SWAT --mask_rate 0.5 --lookback 100 --gamma 0.8 --gru_hid_dim 300 --fc_hid_dim 300 --recon_hid_dim 300 --epochs 15 --init_lr 0.001 --forecast_loss_fn sce --recon_loss_fn sce --val_split 0.1 > test1.log 2>&1 &