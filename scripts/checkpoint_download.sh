#!/usr/bin/env bash
gdown --id 1QgmA8FHDv-oDcBJXeQ8Ql99WAcb0tnyj -O checkpoints/pretrained.zip
unzip -q checkpoints/pretrained.zip -d checkpoints
rm checkpoints/pretrained.zip