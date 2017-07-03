import numpy as np
import argparse
import signalset
import matchingpursuit


parser = argparse.ArgumentParser(description='Fit model')
parser.add_argument('--n_dict', '-n', type=int, default=32)
parser.add_argument('--min_spike', '-m', type=float, default=0.1)
parser.add_argument('--max_iter', type=float, default=1000)
parser.add_argument('--data', type=str, default="../audition/Data/TIMIT.npz")
parser.add_argument('--savefile', '-s',
                    default="../audition/Results/MPonTIMIT.pickle")
parser.add_argument('--ntrials', '-t', default=1000, type=int)
parser.add_argument('--learn_rate', '-l', default=100, type=float)
parser.add_argument('--load', dest='load', action='store_true')
parser.set_defaults(load=False)
args = parser.parse_args()

n_dict = args.n_dict
min_spike = args.min_spike
max_iter = args.max_iter
data = args.data
learn_rate = args.learn_rate

if isinstance(data, str):
    if data.endswith(".npz"):
        data = np.load(data)
        data_list = []
        for key, val in data.items():
            data_list.append(val)
        data = data_list

sigset = signalset.SignalSet(data=data, seg_length=16000)

mp = matchingpursuit.MatchingPursuer(data=sigset,
                                     max_iter=max_iter,
                                     nunits=n_dict,
                                     min_spike=min_spike,
                                     use_gammachirps=False,
                                     learn_rate=learn_rate,
                                     paramfile=args.savefile)

if args.load:
    mp.load(args.savefile)
mp.learn_rate = learn_rate
mp.min_spike = min_spike
mp.max_iter = max_iter
mp.save()

mp.train(ntrials=args.ntrials)
mp.save()
