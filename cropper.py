import numpy as np

data = np.load("/Users/arnlois/data/code/sub-20191030-07_ses-20191030_ophys_calcium.npz")

calcium = data["calcium"]
calcium_seg = data["calcium_seg"]
calcium_labels = data["calcium_labels"]

cropped_calcium = calcium[:, 0:82, 25:95, 0:20]

#, Z, Y, X

# Save everything (segmentation and labels unchanged)
np.savez("ASA_Sample1",
         calcium=cropped_calcium,
         calcium_seg=calcium_seg,
         calcium_labels=calcium_labels)