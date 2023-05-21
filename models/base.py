from .vgg import VGG16
from .AlexNet import AlexNet


def build_model(model, act, dataset):
    if model.lower() == 'vgg16':
        model = VGG16(act, dataset)
    elif model.lower() == 'alexnet':
        model = AlexNet(act, dataset)
    return model
