from . import mnist
from . import fashion_mnist
from . import cifar
from . import svhn
from . import imagenet
from . import imagenet_subsets
# TODO(dhuseljic): from . import tiny_imagenet

from .mnist import build_mnist
from .fashion_mnist import build_fashionmnist
from .cifar import CIFAR10, CIFAR100, CIFAR10C, CIFAR10Contrastive, CIFAR10Plain, CIFAR100Contrastive, CIFAR100Plain
from .svhn import SVHN, SVHNContrastive, SVHNPlain
from .imagenet import ImageNet, ImageNetContrastive, ImageNetPlain
from .imagenet_subsets.imagenet_subset import ImageNet50, ImageNet50Contrastive, ImageNet50Plain, ImageNet100, \
    ImageNet100Contrastive, ImageNet100Plain, ImageNet200, ImageNet200Contrastive, ImageNet200Plain
