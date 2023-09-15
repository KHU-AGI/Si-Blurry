from .clib import CLIB
from .er_baseline import ER
from .ewcpp import EWCpp
from .finetuning import FT
from .lwf import LwF
from .rainbow_memory import RM
from .mvp import MVP

__all__ = [
    "CLIB",
    "ER",
    "EWCpp",
    "FT",
    "LwF",
    "RM",
    "MVP",
]

def get_method(name):
    name = name.lower()
    try:
        return {
            "clib": CLIB,
            "er": ER,
            "ewcpp": EWCpp,
            "ft": FT,
            "lwf": LwF,
            "rm": RM,
            "mvp": MVP,
        }[name]
    except KeyError:
        raise NotImplementedError(f"Method {name} not implemented")