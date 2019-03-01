#!/bin/sh

# uncomment the following line to get mkldnn trace log
# export MKLDNN_VERBOSE=2


ARGS=""
if [[ "$1" == "--inference" ]]; then
    ARGS="$ARGS --inference"
    echo -e "\n### inference mode "
    shift
else
    echo -e "\n### training mode"
fi

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
batch_sizes="32 64"
hidden_sizes="500 800"

for ts in $time_steps; do
    for bs in $batch_sizes; do
        for hs in $hidden_sizes; do
            python -u benchmark.py $ARGS --time_step=$ts --batch_size=$bs --input_size=$hs --hidden_size=$hs
        done
    done
done
