import numpy as np
from sklearn.metrics import ndcg_score


def hit_att(ascore, labels, ps=[100, 150]):
	# Calculate hit metrics
	# Ascore: Scoring results for two-dimensional matrices
	# Labels: Real indicators of two-dimensional matrices
	# Ps=[100, 150]: percentage list
	res = {}
	for p in ps:
		hit_score = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])  # np.argsort return sorting index array from large to small
			if l:	 # Indexing elements for True
				size = round(p * len(l) / 100)
				a_p = set(a[:size])
				intersect = a_p.intersection(l)		# How many are selected
				hit = len(intersect) / len(l)
				hit_score.append(hit)
		res[f'Hit@{p}%'] = np.mean(hit_score)
	return res


def ndcg(ascore, labels, ps=[100, 150]):
	res = {}
	for p in ps:
		ndcg_scores = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			labs = list(np.where(l == 1)[0])
			if labs:
				k_p = round(p * len(labs) / 100)
				try:
					# Calculate the normalized cumulative loss gain (NDCG) score, the higher the better
					hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k=k_p)
				except Exception as e:
					return {}
				ndcg_scores.append(hit)
		res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
	return res



