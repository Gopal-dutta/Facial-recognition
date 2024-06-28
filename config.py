import os


class Config:
    RAW_DATA_PATH = os.path.join('data', 'Raw_img')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')
    DATASET_PATH = os.path.join('data', 'datasets')
    MODEL_CHECKPOINT_PATH = os.path.join('models', 'checkpoints')
    FINAL_MODEL_PATH = os.path.join('models', 'final')
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    LFW_ALLNAMES_FILE = os.path.join('Dataset', 'lfw_allnames.csv')
