from .bbrbm import BBRBM
from .gbrbm import GBRBM
from .reg_rbm import RegRBM

# default RBM
RBM = BBRBM

__all__ = [RBM, BBRBM, GBRBM, RegRBM]
#__all__ = [RBM, BBRBM, GBRBM]

