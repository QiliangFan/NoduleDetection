#!/bin/bash

log_dir=$(cd `dirname $0`; pwd)"/log"
if [ ! -d $log_dir ];
then
    mkdir $log_dir
fi

echo "log dir: "$log_dir
tensorboard --logdir $log_dir --host 10.10.1.220 --port 8888
