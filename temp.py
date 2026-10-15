import h5py
import doctest
with h5py.File("data/test_last_6_mean_states.h5", "r") as f:
    shape = f['sentence_001943']['hidden_states']
    length = f['sentence_001943']['id_in_tree']
    print(shape)
    print(length)