import numpy as np
import napari
"""This code is to inspect 4d calcium imaging data and visualize them in napari"""

data = np.load("/Users/arnlois/data/code/sub-20190928-13_ses-20190928_ophys_calcium.npz")   
calcium = data["calcium"]
calcium_seg = data["calcium_seg"]        

print("Calcium Shape:", calcium.shape)      
print(calcium_seg.shape)

t=[1,2,3]

points = []
for t, seg in enumerate(calcium_seg):
    if seg is not None: 
        for neuron in seg:
            points.append([t, neuron[0], neuron[1], neuron[2]])  # time, x, y, z
points = np.array(points)

viewer = napari.view_image(calcium.astype(float), 
                           name="Calcium Volume", 
                           )

# viewer.add_points(points, size=5, face_color='orange', name="Neurons")


if __name__ == '__main__':
    napari.run()

print("4d image is loaded succesfully")