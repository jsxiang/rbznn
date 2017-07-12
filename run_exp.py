import os
import itertools
import subprocess as sp


all_options = [
    ['--n_layers', 1, 2, 3, 4],
    ['--n_LSTM', 1, 2, 3, 4],
    ['--n_units', 16, 32, 64, 128, 256, 512],
    ['--n_LSTM_units', 64, 128, 256, 32, 16],
    #['--batch_size', 128, 256, 512, 64, 32],
    #['--dropout', 0.1, 0.2, 0.5, 0.7],
    ['--learning_rate', 1e-4, 1e-5, 1e-3],
    ['--reg', 1e-1, 1, 1e-2, 1e-3, 1e-4]
    
    ]

option_names = [e[0] for e in all_options]
option_options = [e[1:] for e in all_options]

base_name = '_'.join(['%s'%(name[2:]) for name in option_names])
for options in itertools.product(*option_options):
    print(options)
    cmd = ['python', 'rbzlstm_YFSIVc.py']
    exp_name = '_'.join(['%s'%(o) for o in options]) + '_' + base_name
    #print('EXP_NAME', exp_name)
    cmd += ['--name', exp_name]
    for k, v in zip(option_names, options):
        cmd += [k, str(v)]
    print('RUNNING', cmd)
    sp.call(cmd)
