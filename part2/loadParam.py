import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "run2"
MODEL_NAME  =   "windowseg"
DS_PATH     =   "/home/manoj/blender/flyingWindows/og/"
OUT_PATH    =   "/home/manoj/outputs/windowseg/"

JOB_FOLDER  =   os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH    =   os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE          =   8
LR                  =   1e-4
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = True
NUM_WORKERS  =   1