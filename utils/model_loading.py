import warnings
from collections import OrderedDict

import torch
from torch.nn import DataParallel

use_cuda = torch.cuda.is_available()

def load_model(model, bin_file, use_dataparallel):
    """
    Given a model instance, loads the weights from bin_file. Handles cuda & DataParallel stuff.
    """
    print(bin_file, use_dataparallel, use_cuda)
    if not use_cuda and use_dataparallel:
        warnings.warn("Cuda not available. Model can not be made Data Parallel.")

    state_dict = torch.load(bin_file, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    if use_cuda:
        for k,v in state_dict.items():
            if k[:7] == 'module.':
                if use_dataparallel:
                    model = DataParallel(model)
                    break
                else:
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
        model = model.cuda()
    else:
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

    if len(new_state_dict.keys()) > 0:
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    return model
