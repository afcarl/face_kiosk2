from __future__ import print_function
import argparse
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
parser.add_argument("npy_file")
parser.add_argument("idx_file")
parser.add_argument("--n_neighbors", type=int, default=1)
parser.add_argument("--algorithm", choices=['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto')
args = parser.parse_args()

nn = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm=args.algorithm)
nn.fit(X)
