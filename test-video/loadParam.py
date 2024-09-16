import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "run1"
MODEL_NAME  =   "windowseg"
VIDEO_PATH = "./"

JOB_FOLDER  =   os.path.join(VIDEO_PATH, JOB_ID)
TRAINED_MDL_PATH    =   os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE          =   15
LR                  =   1e-4
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = True
NUM_WORKERS  =   3