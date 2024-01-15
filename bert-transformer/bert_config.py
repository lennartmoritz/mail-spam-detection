from easydict import EasyDict as edict

# Defining some key variables that will be used later on in the training
DEFAULT_CFG = edict({
    "MAX_LEN": 256,
    "TRAIN_BATCH_SIZE": 16, # 8
    "VALID_BATCH_SIZE": 128, # 4
    "EPOCHS": 20,
    "LEARNING_RATE": 1e-05,
})