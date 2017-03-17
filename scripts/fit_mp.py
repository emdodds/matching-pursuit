import numpy as np
import argparse
import signalset
import matchingpursuit


parser = argparse.ArgumentParser(description='Fit model')
parser.add_argument('--n_dict', '-n', type=int, default=200)
parser.add_argument('--min_spike', '-m', type=float, default=0.1)
parser.add_argument('--max_iter', type=float, default=1000)
parser.add_argument('--data', type=str, default="../audition/Data/TIMIT/")
parser.add_argument('--savefile', '-s', default="../audition/Results/MPonTIMIT.pickle")
parser.add_argument('--ntrials', '--t', default=1000)
args = parser.parse_args()

n_dict = args.n_dict
min_spike = args.min_spike
max_iter = args.max_iter
data = args.data

sigset = signalset.SignalSet(data=data)

mp = matchingpursuit.MatchingPursuer(data=sigset,
                    max_iter=max_iter,
                    nunits=n_dict,
                    min_spike=min_spike,
                    use_gammachirps=False,
                    paramfile = args.savefile)

mp.train(ntrials=args.ntrials)