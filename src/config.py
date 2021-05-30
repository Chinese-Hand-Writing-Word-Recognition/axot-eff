# data_root = "/home/user/code/OCR-YS/data"
# data_root = "/home/user/code/axot/data"
data_root = "/home/user/code/data_cut"
cut_root = "/home/user/code/data_cut"
save_dir = "../save_model"

pretrained_path = f"{save_dir}/eff_multichannel_87574.pth"
# pretrained_path = ""

## dataloader
batch_size = 64
shuffle = True
img_size = 224
num_workers = 6
multi_channel = False

## training
EPOCHS = 100
learning_rate = 1e-1
weight_decay = 0
dropout = 0.2
threshold = 0

## transforms
base_level = 0

