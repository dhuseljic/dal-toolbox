from . import mnist
from . import fashion_mnist
from . import cifar
from . import svhn
from . import imagenet
from . import imagenet_subsets
# TODO(dhuseljic): from . import tiny_imagenet

from .food import Food101
from .mnist import build_mnist
from .fashion_mnist import build_fashionmnist
from .cifar import CIFAR10, CIFAR100, CIFAR10C, CIFAR10Contrastive, CIFAR10Plain, CIFAR100Contrastive, CIFAR100Plain, \
    CIFAR10SimCLR, CIFAR10WithoutNormalization, CIFAR100SimCLR
from .svhn import SVHN, SVHNContrastive, SVHNPlain, SVHNSimCLR
from .imagenet import ImageNet, ImageNetContrastive, ImageNetPlain, ImageNetDINO
from .imagenet_subsets.imagenet_subset import ImageNet50, ImageNet50Contrastive, ImageNet50Plain, ImageNet50DINO, \
    ImageNet100, ImageNet100Contrastive, ImageNet100Plain, ImageNet100DINO, ImageNet200, ImageNet200Contrastive, \
    ImageNet200Plain, ImageNet200DINO
