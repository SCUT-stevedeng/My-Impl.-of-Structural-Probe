import h5py

with h5py.File("dependency_train_data.h5", "r") as f:
    ds = f['word_depth']['X']
    print(ds.shape)