#!/usr/bin/env bash

echo "Submitting a Cloud ML Engine job..."

REGION="us-central"
MODEL_NAME="ImageClassifier" # change to your model name

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
EPOCHS=100
BATCH_SIZE=16
BASE_PATH=gs://bigdata-allanbatista-com-br/ml/image-classifier/${CURRENT_DATE}/

JOB_NAME=train_${MODEL_NAME}_${CURRENT_DATE}
JOB_DIR=${BASE_PATH}train/${CURRENT_DATE}/job/

gcloud ml-engine jobs submit training ${JOB_NAME} \
        --scale-tier=BASIC_GPU \
        --job-dir=${JOB_DIR} \
        --runtime-version=1.10 \
        --region=${REGION} \
        --module-name=image-classifier.task2 \
        --package-path=image-classifier  \
        -- \
        --epochs=${EPOCHS} \
        --batch-size=${BATCH_SIZE} \
        --current-date=${CURRENT_DATE} \
        --base-path=${BASE_PATH}


echo "to see training progress"
echo "$ tensorboard --port 8080 --logdir ${BASE_PATH}tensorboards/"

# notes:
# use --packages instead of --package-path if gcs location
# add --reuse-job-dir to resume training