import torch
from torch import nn
from torch.utils import mkldnn as mkldnn_utils
import numpy as np

N = 10
T = 35
I = 200
H = 250
L = 2

def compare(t1, t2, msg):
    ret = np.allclose(t1.detach().numpy(), t2.detach().numpy(), atol=1e-06)
    print(msg, "pass" if ret else "fail")


def test_lstm_forward(layer=L, bidirectional=True, bias=True):
    print("##### lstm forward #####", "layer = ", layer,
          ", bidirectional = ", bidirectional, ", bias = ", bias)
    D = 2 if bidirectional else 1
    input = torch.randn(T, N, I)
    h0 = torch.randn(layer*D, N, H)
    c0 = torch.randn(layer*D, N, H)

    rnn = nn.LSTM(I, H, layer, bidirectional=bidirectional, bias=bias)
    rnn.eval()
    rnn_mkldnn = mkldnn_utils.to_mkldnn(rnn)

    output1, hn1 = rnn(input, (h0, c0))
    hy1, cy1 = hn1
    output2, hn2 = rnn_mkldnn(input, (h0, c0))
    hy2, cy2 = hn2

    compare(output1, output2, "output: ")
    compare(hy1, hy2, "hy: ")
    compare(cy1, cy2, "cy: ")


def test_gru_forward(layer=L, bidirectional=True, bias=True):
    print("##### gru forward #####", "layer = ", layer,
          ", bidirectional = ", bidirectional, ", bias = ", bias)
    D = 2 if bidirectional else 1
    input = torch.randn(T, N, I)
    h0 = torch.randn(layer*D, N, H)

    rnn = nn.GRU(I, H, layer, bidirectional=bidirectional, bias=bias)
    rnn.eval()
    rnn_mkldnn = mkldnn_utils.to_mkldnn(rnn)

    output1, hy1 = rnn(input, h0)
    output2, hy2 = rnn_mkldnn(input, h0)

    compare(output1, output2, "output: ")
    compare(hy1, hy2, "hy: ")



for bidirectional in [True, False]:
    for bias in [True, False]:
        for layer in [1, L]:
            test_lstm_forward(layer=layer, bidirectional=bidirectional, bias=bias)
            test_gru_forward(layer=layer, bidirectional=bidirectional, bias=bias)


