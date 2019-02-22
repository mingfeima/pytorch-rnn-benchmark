#!/bin/sh

# script for benchmarking single batch inference mkldnn RNN performance
# tesing both single socket and single thread performance

# uncomment the following line to get mkldnn trace log
# export MKLDNN_VERBOSE=2

###########################################################
# RNN configs
N=1
T=15
L=2
I=250
H=200
###########################################################

ARGS=""
ARGS="$ARGS --inference"
echo -e "\n### inference mode "

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "\n### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME"
echo -e "### using $PREFIX\n"

### single socket test
echo -e "\n### using OMP_NUM_THREADS=$CORES"
OMP_NUM_THREADS=$CORES $PREFIX python -u benchmark.py $ARGS --time_step=$T --batch_size=$N --input_size=$I --hidden_size=$H
OMP_NUM_THREADS=$CORES $PREFIX python -u benchmark.py $ARGS --time_step=$T --batch_size=$N --input_size=$I --hidden_size=$H --disable-mkldnn

### single thread test
echo -e "\n### using OMP_NUM_THREADS=1"
OMP_NUM_THREADS=1 $PREFIX python -u benchmark.py $ARGS --time_step=$T --batch_size=$N --input_size=$I --hidden_size=$H
OMP_NUM_THREADS=1 $PREFIX python -u benchmark.py $ARGS --time_step=$T --batch_size=$N --input_size=$I --hidden_size=$H --disable-mkldnn

### single thread test
echo -e "\n### using OMP_NUM_THREADS=1"
echo -e "### using layer=$T"
OMP_NUM_THREADS=1 $PREFIX python -u benchmark.py $ARGS --time_step=1 --layers=$T --batch_size=$N --input_size=$I --hidden_size=$H
OMP_NUM_THREADS=1 $PREFIX python -u benchmark.py $ARGS --time_step=1 --layers=$T --batch_size=$N --input_size=$I --hidden_size=$H --disable-mkldnn

### single thread test
echo -e "\n### using OMP_NUM_THREADS=1"
echo -e "### using layer=1"
OMP_NUM_THREADS=1 $PREFIX python -u benchmark.py $ARGS --time_step=1 --layers=1 --batch_size=$N --input_size=$I --hidden_size=$H
OMP_NUM_THREADS=1 $PREFIX python -u benchmark.py $ARGS --time_step=1 --layers=1 --batch_size=$N --input_size=$I --hidden_size=$H --disable-mkldnn
