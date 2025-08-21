import os, numpy as np
os.makedirs("src/data/anchor", exist_ok=True)
x = np.random.randn(256, 64).astype("float32")
np.save("src/data/anchor/anchor_x.npy", x)
print("Saved src/data/anchor/anchor_x.npy with shape", x.shape)
