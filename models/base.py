from .vgg import VGG16


def build_model(model, act, dataset):
    if model.lower() == 'vgg16':
        model = VGG16(act, dataset)
    return model
