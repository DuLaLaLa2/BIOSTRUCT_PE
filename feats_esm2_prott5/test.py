import numpy as np

data = np.load(r"seq1422_concat_esm2_prott5\\1422\\0000_6e0o_A.npz")
print(data.files)
print(str(data["seq"]))
print(data["seq_feat"].shape)
