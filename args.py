import argparse

parser = argparse.ArgumentParser(description='RBZ NN')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.2)

parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_units', type=int, default=100)
parser.add_argument('--n_LSTM_units', type=int, default=100)
parser.add_argument('--n_LSTM', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)

parser.add_argument('--restore', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--dataset-size', type=int, default=100, metavar='D',
                    help='input batch size for training (default: 100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = True
