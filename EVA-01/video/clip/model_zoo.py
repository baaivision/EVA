import os


def get_model_path(ckpt):
    if os.path.isfile(ckpt):
        return ckpt
    else:
        print('not found pretrained model in {}'.format(ckpt))
        raise FileNotFoundError
