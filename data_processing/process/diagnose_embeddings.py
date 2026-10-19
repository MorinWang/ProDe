
import torch
import numpy as np


emb_path = "data/Movie_noise_user0.1/item_embedding_with_desc.pt"
embeddings = torch.load(emb_path)

valid_embs = embeddings[1:]
norms = torch.norm(valid_embs, p=2, dim=1)
idx1, idx2 = 100, 200
dist_euclidean = torch.cdist(valid_embs[idx1:idx1+1], valid_embs[idx2:idx2+1], p=2).item()
emb1_norm = valid_embs[idx1] / torch.norm(valid_embs[idx1])
emb2_norm = valid_embs[idx2] / torch.norm(valid_embs[idx2])
cosine_sim = torch.dot(emb1_norm, emb2_norm).item()


np.random.seed(42)
sample_size = 1000
indices = np.random.choice(len(valid_embs), size=sample_size, replace=False)
sample_embs = valid_embs[indices]


batch_size = 100
all_dists = []
for i in range(0, len(sample_embs), batch_size):
    batch = sample_embs[i:i+batch_size]
    dists = torch.cdist(batch, sample_embs, p=2)
    all_dists.append(dists)

all_dists = torch.cat(all_dists, dim=0)

triu_indices = torch.triu_indices(len(sample_embs), len(sample_embs), offset=1)
pairwise_dists = all_dists[triu_indices[0], triu_indices[1]]

pass
