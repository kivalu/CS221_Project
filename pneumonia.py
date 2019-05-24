from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from sklearn.metrics import confusion_matrix
arch = resnet34

PATH = 'data/chest_xray'
normal_example = os.listdir(f'{PATH}train/NORMAL')[0]
pneumonia_example = os.listdir(f'{PATH}train/PNEUMONIA')[0]

normal_img = plt.imread(f'{PATH}train/NORMAL/{normal_example}')
plt.imshow(normal_img)

# def get_data(sz, bs):
#     # data augmentation
#     tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=2)
#     data = ImageClassifierData.from_paths(PATH, bs=bs, tfms=tfms, val_name='val, test_name="test", test_with_labels=True)
#     return data
