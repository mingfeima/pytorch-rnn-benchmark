import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
import subprocess

steps = 100 # nb of steps in loop to average perf
nDryRuns = 100 # nb of warmup steps

parser = argparse.ArgumentParser(description='PyTorch Convnet Benchmark')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='use cuda')
parser.add_argument("--disable-mkldnn", action='store_true', default=False,
                    help='disable mkldnn')
parser.add_argument('--inference', action='store_true', default=False,
                    help='run inference only')
parser.add_argument('--time_step', type=int, default=50,
                    help='time step')
parser.add_argument('--input_size', type=int, default=500,
                    help='input size')
parser.add_argument('--hidden_size', type=int, default=500,
                    help='hidden size')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--layers', type=int, default=1,
                    help='number of layers');
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='use bidirectional rnn')
parser.add_argument('--model_type', type=str, default='lstm',
                    choices=['lstm', 'gru'],
                    help='Type of RNN models, Options are [lstm|gru]')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

### rnn parameters
L = args.layers
T = args.time_step
N = args.batch_size
I = args.input_size
H = args.hidden_size
D = 2 if args.bidirectional else 1

### i wrote torch._C._set_mkldnn_enabled with MKLDNN RNN in the same PR
### the flag is used to turn on/off MKLDNN functionality
if hasattr(torch._C, '_set_mkldnn_enabled'):
    if args.disable_mkldnn:
        torch._C._set_mkldnn_enabled(False)
else:
    if not args.cuda:
        print("You are runnning on PyTorch without torch._C._set_mkldnn_enabled,",
              "perhaps MKLDNN RNN API is not integrated!")

if args.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = True
    kernel_name = 'cudnn'
else:
    kernel_name = 'cpu   ' if args.disable_mkldnn else 'mkldnn'

def _time():
    if args.cuda:
        torch.cuda.synchronize()

    return time.time()

if args.model_type == 'gru':
    hx = torch.randn(L*D, N, H).type(dtype)
    rnn = nn.GRU
else:
    hx = (torch.randn(L*D, N, H).type(dtype), torch.randn(L*D, N, H).type(dtype))
    rnn = nn.LSTM

x = torch.randn(T, N, I).type(dtype)
model = rnn(I, H, L, bidirectional=args.bidirectional)

if args.inference:
    model.eval()
else:
    model.train()

if args.cuda:
    model.cuda()

for i in range(nDryRuns):
    y, _ = model(x, hx)
    if not args.inference:
        y.mean().backward()

time_fwd, time_bwd = 0, 0

for i in range(steps):
    t1 = _time()
    y, _ = model(x, hx)
    t2 = _time()
    time_fwd = time_fwd + (t2 - t1)
    if not args.inference:
        y.mean().backward()
        t3 = _time()
        time_bwd = time_bwd + (t3 - t2)

time_fwd_avg = time_fwd / steps * 1000
time_bwd_avg = time_bwd / steps * 1000
time_total = time_fwd_avg + time_bwd_avg
sps = N / time_total * 1000

print("%s: [T,N,I,H] = [%d,%d,%d,%d], time: %.2f (ms), SPS: %.2f" % (kernel_name, T, N, I, H, time_total, sps))
