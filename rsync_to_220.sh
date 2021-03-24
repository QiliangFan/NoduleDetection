#!/bin/bash

project_dir=$(cd $(dirname $0);pwd)/
echo $project_dir

rsync -av $project_dir maling@10.10.1.220:/home/maling/fanqiliang/projects/NoduleDetection