import sys
sys.path.append('..')

import torch
from kandinsky3 import get_T2I_pipeline

device_map = torch.device('cuda:0')
dtype_map = {
    'unet': torch.float32,
    'text_encoder': torch.float16,
    'movq': torch.float32,
}

t2i_pipe = get_T2I_pipeline(
    device_map, dtype_map,
)
res = t2i_pipe("A cute corgi lives in a house made out of sushi.")

res[0]