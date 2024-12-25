import numpy as np

nlags = np.arange(10)
valid_mask = nlags == 5
# safe_nlags = np.where(valid_mask, nlags, 0)
results = np.zeros((6, 6))
results[valid_mask] = nlags[valid_mask]

print(results)