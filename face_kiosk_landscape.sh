#!/bin/sh

python kiosk.py --capture_device 1 --detector opencv --fullscreen --canvas_width 1280 --canvas_height 720 --face_count 5 --n_neighbors 5 --show_distance 0 $*
