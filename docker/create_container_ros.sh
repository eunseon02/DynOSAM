#!/usr/bin/env bash

### EDIT THIS TO WHEREVER YOU'RE STORING YOU DATA ###
# folder should exist before you mount it
LOCAL_DATA_FOLDER=/media/eunseon/T7/datasets
LOCAL_RESULTS_FOLDER=~/results/
LOCAL_DYNO_SAM_FOLDER=~/Code/src/MR/
# LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=~/Code/src/third_party_dynosam/
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=None
bash create_container_base.sh ros1:latest mr $LOCAL_DATA_FOLDER $LOCAL_RESULTS_FOLDER $LOCAL_DYNO_SAM_FOLDER $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER
