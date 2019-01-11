#!/bin/sh

# script for benchmarking single batch inference mkldnn RNN performance

# uncomment the following line to get mkldnn trace log
# export MKLDNN_VERBOSE=2


ARGS=""
ARGS="$ARGS --inference"
echo -e "\n### inference mode "

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "\n### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

time_steps="35 50"
batch_sizes="1"
hidden_sizes="300 500 800 1000"

for ts in $time_steps; do
    for bs in $batch_sizes; do
        for hs in $hidden_sizes; do
            python -u benchmark.py $ARGS --time_step=$ts --batch_size=$bs --input_size=$hs --hidden_size=$hs
            python -u benchmark.py $ARGS --time_step=$ts --batch_size=$bs --input_size=$hs --hidden_size=$hs --disable-mkldnn
        done
    done
done


