import torch
device=torch.device('cuda'if torch.cuda.is_available()else'cpu')

def score_func(y_true, y_hat):
  subs = y_hat - y_true
  if subs < 0:
    res =  torch.exp(-subs/10).item()-1
  else:
    res =  torch.exp(subs/13).item()-1
  return res