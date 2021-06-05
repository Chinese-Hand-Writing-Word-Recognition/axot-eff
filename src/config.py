# data_root = "/home/user/code/OCR-YS/data"
data_root = "/home/user/code/axot/data"
# data_root = "/home/user/code/data_cut"
cut_root = "/home/user/code/data_cut"
save_dir = "../save_model"
save_name = "eff_ori128_t_d9.pth"

pretrained_path = f"{save_dir}/eff_ori128_t_d9_f94550.pth"
# pretrained_path = ""

## dataloader
batch_size = 128
shuffle = True
img_size = 128
num_workers = 4
multi_channel = False

## training
EPOCHS = 600
learning_rate = 1e-5
weight_decay = 3e-4
dropout = 0.9
threshold = 0.9

## transforms
base_level = 0

