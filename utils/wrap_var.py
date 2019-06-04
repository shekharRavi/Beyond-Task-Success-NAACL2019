import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def to_var(x, volatile=False):
    """Short summary.

    Parameters
    ----------
    x : type
        Input tensor.
    volatile : type
        Flag for the variable has to be volatile or not.

    Returns
    -------
    Variable
        Varible wrapped around the tensor

    """

    if use_cuda:
        x = x.cuda()
    return Variable(x, volatile= volatile)