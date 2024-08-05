#!/bin/bash

python3 generator_main.py

python3 road_generator.py

python3 model_weights.py

python3 road_weights.py

python3 segmented_concat.py

python3 merge_imgs.py

echo "all scripts ran successfully"
