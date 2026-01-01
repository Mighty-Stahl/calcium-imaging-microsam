"""Essential first step to identify where calcium imaging 
and NeuroPAL are stored respectively. 
Different nwb files may have different keys, hence this is important. """
from pynwb import NWBHDF5IO

io = NWBHDF5IO("/Users/arnlois/000981/Hermaphrodites/sub-20220327-h2/sub-20220327-h2_ses-20220327_ophys.nwb", "r")
nwb = io.read()

print(nwb.acquisition.keys())
series = nwb.acquisition["CalciumImageSeries"]

print("Rate:", series.rate)
print("Data shape:", series.data.shape)

