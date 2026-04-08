import os

class Config:
    DATA_BASE    = "/content/drive/MyDrive/anomaly_project/data/kaggle/working/data"
    NORMAL_DIR   = DATA_BASE + "/normal"
    FIGHT_DIR    = DATA_BASE + "/fighting"
    ACCIDENT_DIR = DATA_BASE + "/accident"
    THEFT_DIR    = DATA_BASE + "/theft"
    BASE_DIR     = "/content/drive/MyDrive/anomaly_project"
    CHECKPOINT   = BASE_DIR + "/checkpoints/best_model_v2.pth"
    CLF_CKPT     = BASE_DIR + "/checkpoints/classifier.pth"
    REPORT_DIR   = BASE_DIR + "/reports"
    THUMB_DIR    = BASE_DIR + "/thumbnails"
    FRAME_H      = 128
    FRAME_W      = 128
    SEQ_LEN      = 16
    STRIDE       = 4
    TRAIN_SPLIT  = 0.85
    LATENT_DIM   = 128
    LSTM_CH      = 32
    LR           = 1e-4
    BATCH_SIZE   = 8
    EPOCHS       = 40
    SSIM_WEIGHT  = 0.15
    THRESHOLD_K  = 3
    CLASSES      = ["Normal", "Fighting", "Accident", "Theft"]
    NUM_CLASSES  = 4
