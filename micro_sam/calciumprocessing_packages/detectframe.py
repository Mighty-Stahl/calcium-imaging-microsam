fps = 2.67

stim_time = 342.0
stim_frame = int(stim_time * fps)

baseline = 10.0
response = 15  # seconds

start = int((stim_time - baseline) * fps)
end = int((stim_time + response) * fps)  

result = int((end - start) * fps)
x = int((stim_time - (stim_time - baseline)) * fps)


print("Stimulus Frame: " + str(x))

