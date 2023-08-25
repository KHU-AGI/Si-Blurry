# import torch
import time
from configuration import config
from datasets import *
from methods.er_baseline import ER
from methods.clib import CLIB
from methods.L2P import L2P
from methods.rainbow_memory import RM
from methods.Finetuning import FT
from methods.ewc import EWCpp
from methods.lwf import LwF
from methods.ours import Ours
from methods.dualprompt import DualPrompt

# torch.backends.cudnn.enabled = False
methods = { "er"    : ER, 
            "clib"  : CLIB,
            "L2P"   : L2P,
            "rm"    : RM,
            "lwf"   : LwF,
            "Finetuning"    :FT,
            "ewc++" : EWCpp,
            "ours"  : Ours,
            "DualPrompt"    : DualPrompt }

def main():
    # Get Configurations
    args = config.base_parser()
    print(args)
    trainer = methods[args.mode](**vars(args))

    trainer.run()

if __name__ == "__main__":
    main()
    time.sleep(60)
