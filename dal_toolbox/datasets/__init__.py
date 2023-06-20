from . import mnist
from . import fashion_mnist
from . import cifar
from . import svhn
from . import imagenet
# TODO(dhuseljic): from . import tiny_imagenet

from .mnist import build_mnist
from .fashion_mnist import build_fashionmnist
from .cifar import CIFAR10, CIFAR100, CIFAR10C, CIFAR10Contrastive, CIFAR10Plain
from .svhn import SVHN
from .imagenet import ImageNet
