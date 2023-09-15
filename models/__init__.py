# The codes in this directory were from https://github.com/drimpossible/GDumb/tree/master/src/models
import timm
from .dualprompt import DualPrompt
from .l2p import L2P
from .mvp import MVP

__all__ = [
    "DualPrompt",
    "l2p",
    "vision_trainsfomer"
]

def get_model(name):
    name = name.lower()
    try:
        if 'vit' in name:
            name = name.split('_')[1]
            model = timm.create_model(f'vit_{name}_patch16_224', pretrained=True)
            return (model, 224)
        elif 'resnet' in name:
            model = timm.create_model(name, pretrained=True)
            return (model, 224)
        else:
            return {
                "dualprompt": (DualPrompt, 224),
                "l2p": (l2p, 224),
                "mvp": (MVP, 224),
            }[name]
    except KeyError:
        raise NotImplementedError(f"Model {name} not implemented")