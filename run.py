import train
import argparse
import os
import numpy as np
import random
import json
import config

# parse arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', dest='mode', type=int, default=0,
                    help='run mode - (0-train only, 1-test only, 2-val only)')
parser.add_argument('--nlayers', dest='nlayers', type=int, default=3,
                    help='Number of reader layers')
parser.add_argument('--dataset', dest='dataset', type=str, default='lambada',
                    help='Location of training, test and validation files.')
parser.add_argument('--seed', dest='seed', type=int, default=1,
                    help='Seed for different experiments with same settings')
parser.add_argument('--save_path', dest='save_path', type=str, default=None,
                    help='Location of output logs and model checkpoints.')
parser.add_argument('--reload', dest='reload_', action='store_true')
parser.set_defaults(reload_=False)
args = parser.parse_args()
cmd = vars(args)
params = config.params
for k, v in cmd.iteritems():
    if k not in params or v is not None: params[k] = v

np.random.seed(params['seed'])
random.seed(params['seed'])

# save directory
if params["save_path"] is not None:
    save_path = params["save_path"]
else:
    save_path = 'output/' + params['dataset']
print "storing params to " + save_path
if not os.path.exists(save_path): os.makedirs(save_path)
if not os.path.exists(os.path.join(save_path, "params.json")):
    json.dump(params, open(os.path.join(save_path, "params.json"), "w"))

# train
if params['mode']==0:
    train.main(save_path, params)
# test
elif params['mode']==1:
    train.main(save_path, params, mode='test')
elif params['mode']==2:
    train.main(save_path, params, mode='validation')
