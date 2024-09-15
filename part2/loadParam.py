import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "run1"
MODEL_NAME  =   "windowseg"
DS_PATH = os.path.join(HOME_PATH, "window_detection", "DLG2_p2", "Dataset")
OUT_PATH = os.path.join(HOME_PATH, "window_detection", "DLG2_p2", "Outputs")

JOB_FOLDER  =   os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH    =   os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE          =   3
LR                  =   1e-4
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = True
NUM_WORKERS  =   1