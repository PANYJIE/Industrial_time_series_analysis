import utils
from model import Encoder
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import RUL_Dataset
from utils import EarlyStopping
from copy import deepcopy
import os
import argparse
import csv
from evaluation import score_func
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--save_dir', type=str, default='./checkpoint/FD001/base')
	parser.add_argument('--dataset', type=str, default='FD001')
	parser.add_argument('--lr', type=float, default=0.0005)
	parser.add_argument('--device', type=str, default='cuda:0')
	args = parser.parse_args()

	# ------------------------------ DATA -----------------------------------
	dataset = args.dataset
	save_folder = args.save_dir
	os.makedirs(save_folder, exist_ok=True)

	sensors = ['s_2','s_3', 's_4','s_7','s_8','s_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17','s_20', 's_21']

	sequence_length = 30
	# smoothing intensity
	alpha = 0.1
	batch_size = 128
	# max RUL
	threshold = 125
	# Load Dataset
	x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset, sensors, sequence_length, alpha, threshold)
	tr_dataset = RUL_Dataset(x_train, y_train)
	val_dataset = RUL_Dataset(x_val, y_val)
	test_dataset = RUL_Dataset(x_test, y_test)

 	# Load Loader
	tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


	# --------------------------------------- MODEL ----------------------------------------
	timesteps = x_train.shape[1]
	input_dim = x_train.shape[2]
	intermediate_dim = 300
	latent_dim = 2
	epochs = 100
	device = args.device
	lr = args.lr
	early_stopping_with_loss = False

	encoder = Encoder().to(device)

	# ---------------------------- Optimizer and Early Stopping ----------------------------
	optimizer = torch.optim.Adam(list(encoder.parameters()), lr=lr)
	early = EarlyStopping(patience=10)


	# --------------------------------- Train and Validation --------------------------------
	for epoch in range(epochs):
		# Train
		encoder.train()

		tr_loss = 0.
		for tr_x, tr_y in tr_loader:
			tr_x, tr_y = tr_x.to(device), tr_y.to(device)
			optimizer.zero_grad()
			out = encoder(tr_x).view(-1)
			rmse_loss = torch.sqrt(F.mse_loss(out, tr_y) + 1e-6)
			loss = rmse_loss
			loss.backward()
			optimizer.step()

			tr_loss += loss.item() / len(tr_loader)


		# Validation
		encoder.eval()
		val_loss = 0.
		val_rmse = 0.
		for val_x, val_y in val_loader:
			val_x, val_y = val_x.to(device), val_y.to(device)

			with torch.no_grad():
				out = encoder(val_x).view(-1)

			rmse_loss = torch.sqrt(F.mse_loss(out, val_y) + 1e-6)
			loss = rmse_loss

			val_loss += loss / len(val_loader)
			val_rmse += rmse_loss.item() / len(val_loader)

		print('Epoch %d : tr_loss %.2f, val_loss %.2f, val_rmse %.2f' %(epoch, tr_loss, val_loss, val_rmse))
		param_dict = {'encoder': deepcopy(encoder.state_dict())}


		# Early Stopping
		if early_stopping_with_loss:
			early(val_loss, param_dict)
		else:
			early(val_rmse, param_dict)

		if early.early_stop == True:
			break


	# Save Best Model
	torch.save(early.model, os.path.join(save_folder, 'best_model.pt'))


	# --------------------------------- Test --------------------------------
	encoder.load_state_dict(early.model['encoder'])

	encoder.eval()


	test_loss = 0.
	test_rmse = 0.
	test_score = 0.
	test_mse_sum = 0.0  # 初始化MSE总和
	mse_all = 0.0
	all_out_lists = []
	for test_x, test_y in test_loader:
		test_x, test_y = test_x.to(device), test_y.to(device)

		with torch.no_grad():
			out = encoder(test_x).view(-1)

		rmse_loss = torch.sqrt(F.mse_loss(out, test_y) + 1e-6)
		mse_a = rmse_loss * rmse_loss
		loss = rmse_loss
		test_loss += loss / len(test_loader)
		mse_all += mse_a.item()
		test_rmse = torch.sqrt(torch.tensor(mse_all / len(test_loader)))
		test_score += score_func(test_y, out)

		all_out_lists.append(out.tolist())

	csv_file_path = "./result/FD004.csv"
	out_list = out.tolist()
	with open(csv_file_path, "a", newline="") as csvfile:
		writer = csv.writer(csvfile)
		for out_list in all_out_lists:
			writer.writerow(out_list)

	print('Final Result : test loss %.2f, test_rmse %.2f, test_score %.2f' %(test_loss, test_rmse,test_score))
	with open(os.path.join(save_folder, 'result.txt'), 'w') as f:
		f.writelines('Final Result : test loss %.2f, test_rmse %.2f,test_score %.2f' %(test_loss, test_rmse,test_score))