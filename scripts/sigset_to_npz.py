import numpy as np
import signalset

sigset = signalset.SignalSet()
np.savez("../audition/Data/TIMIT.npz", *sigset.data)