# import torch
import time
from configuration import config
from datasets import *
from methods.er_baseline import ER
from methods.clib import CLIB
from methods.rainbow_memory import RM
from methods.finetuning import FT
from methods.ewcpp import EWCpp
from methods.lwf import LwF
from methods.mvp import MVP
from methods.dualprompt import DualPrompt

# torch.backends.cudnn.enabled = False
methods = { "er"    : ER, 
            "clib"  : CLIB,
            "rm"    : RM,
            "lwf"   : LwF,
            "Finetuning"    :FT,
            "ewc++" : EWCpp,
            "mvp"   : MVP,
            "dualprompt"    : DualPrompt
            }

def main():
    # Get Configurations
    args = config.base_parser()
    print(args)
    trainer = methods[args.mode](**vars(args))

    trainer.run()

if __name__ == "__main__":
    main()
    time.sleep(60)
