from pynwb import NWBHDF5IO

nwb_path = "/Users/arnlois/000981/Hermaphrodites/sub-20220327-h2/sub-20220327-h2_ses-20220327_ophys.nwb"
io = NWBHDF5IO(nwb_path, 'r')
nwbfile = io.read()

neuropal_neurons = nwbfile.processing["NeuroPAL"]["NeuroPALSegmentation"]["NeuroPALNeurons"]
voxel_mask = neuropal_neurons.voxel_mask[:]  # Coordinates
id_labels = neuropal_neurons.ID_labels[:]     # Neuron names

print(id_labels)