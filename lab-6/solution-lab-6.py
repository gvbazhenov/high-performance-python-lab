
import numpy as np
import pickle
import sys

from mpi4py import MPI

### define functions for transformation

def construct_spectrogram(tt, yy, window_positions_range, window_width=4.0 * 2 * np.pi):
    tt_size = len(tt) // 2
    spectrogram = np.zeros((len(window_positions_range), tt_size), dtype=np.float32)

    for i, window_position in enumerate(window_positions_range):
        window_function = np.exp(-(tt - window_position) ** 2 / (2 * window_width**2))
        yy_window = (yy * window_function)

        values = np.abs(np.fft.fft(yy_window)) ** 2
        spectrogram[i, :] = values[:tt_size]

    return np.log(1 + spectrogram).reshape(-1)

### setup mpi environment

comm = MPI.COMM_WORLD
start = MPI.Wtime()

rank = comm.Get_rank()
size = comm.Get_size()
root = 0

### read parameters of signal

with open(f'tt.pkl', 'rb') as f:
    tt = pickle.load(f)

with open(f'yy.pkl', 'rb') as f:
    yy = pickle.load(f)

tt_size = len(tt) // 2

### define parameters for transformation

n_window_steps = 1000 if len(sys.argv) < 2 else int(sys.argv[1])
window_width = 4.0 * 2 * np.pi if len(sys.argv) < 3 else int(sys.argv[2])

window_positions = np.linspace(-30 * 2 * np.pi, 30 * 2 * np.pi, n_window_steps, dtype=np.float32)
frequencies = np.fft.fftfreq(len(yy), d=(tt[1] - tt[0]) / (2 * np.pi))[:tt_size]

### assign transformation parts between processes

window_positions_count = int(len(window_positions) / size)
window_positions_range = window_positions[rank * window_positions_count:(rank + 1) * window_positions_count]
spectrogram_complete = np.empty(tt_size * len(window_positions), dtype=np.float32) if rank == root else None

### perform transformation in every process

spectrogram_range = construct_spectrogram(tt.astype(np.float32), yy.astype(np.float32), window_positions_range, window_width)

### gather transformation results in root process

comm.Gather(spectrogram_range, spectrogram_complete, root)
end = MPI.Wtime()

if rank == root:
    with open(f'positions-{size}.pkl', 'wb') as f:
        pickle.dump(window_positions, f)
    
    with open(f'frequencies-{size}.pkl', 'wb') as f:
        pickle.dump(frequencies, f)

    with open(f'spectrogram-{size}.pkl', 'wb') as f:
        pickle.dump(spectrogram_complete.reshape(len(window_positions), tt_size).T, f)
    
    with open(f'time-{size}.pkl', 'wb') as f:
        pickle.dump(end - start, f)
