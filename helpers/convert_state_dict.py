# convert saved cudadict to cpu
import torch


saved = "/nrs/branson/kwaki/outputs/20180205_3dconv/networks/53200/network.pt"

state_dict = torch.load(saved)

for key in state_dict.keys():
    temp = state_dict[key]
    temp = temp.cpu()
    state_dict[key] = temp
    import pdb; pdb.set_trace()