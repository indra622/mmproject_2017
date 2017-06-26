#!/usr/bin/env sh
set -e

./build/tools/caffe test --model=examples/mmproject/train_val100_46_11.prototxt --weights=/home/splab/hosung/data2/mmproject/label/snapshot/SOM_iter_50000.caffemodel $@
