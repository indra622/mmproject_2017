#!/usr/bin/env sh
set -e
./build/tools/caffe train --solver=models/mmproject/solver100_46_11.prototxt $@ -gpu all
