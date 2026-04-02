import numpy as np
import json
import matplotlib.pyplot as plt
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate RGB-D association file')
parser.add_argument('filename', type=str, help='Base filename (without extension) for input JSON file in support_files/')
args = parser.parse_args()

# Construct input and output file paths
input_file = f'support_files/{args.filename}.json'
output_file = f'support_files/{args.filename}_associated.txt'

# Check if input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

# Load JSON file
filenames = json.load(open(input_file))
rgbs = filenames['rgb']
depths = filenames['depth']

# Write association file
f = open(output_file, 'w')
for rgb, depth in zip(rgbs, depths):
    _, filename_rgb = os.path.split(rgb)
    time_rgb = filename_rgb[:-4]
    filename_rgb = 'rgb/' + filename_rgb
    _, filename_depth = os.path.split(depth)
    time_depth = filename_depth[:-4]
    filename_depth = 'depth/' + filename_depth
    line = time_rgb + ' ' + filename_rgb + ' ' + time_depth + ' ' + filename_depth + '\n'
    f.write(line)
'''for i in range(1508):
    time = (i+1)/3.0
    line = str(time) + ' rgb/' + str(i+1) + '.png ' + str(time) + ' depth/' + str(i+1) + '.png' + '\n'
    f.write(line)'''
f.close()

print(f"Association file created: {output_file}")