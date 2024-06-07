from . import mnist
from . import fashion_mnist
from . import cifar
from . import svhn
from . import imagenet
from . import imagenet_subsets
from . import stl10

from .stl10 import STL10
from .snacks import Snacks
from .dtd import DTD
from .food import Food101
from .flowers import Flowers102
from .country211 import Country211
from .tiny_imagenet import TinyImageNet
from .stanford_dogs import StanfordDogs

from .cifar import CIFAR10, CIFAR100, CIFAR10C
from .cifar import CIFAR10SimCLR, CIFAR10WithoutNormalization, CIFAR100SimCLR
from .cifar import CIFAR10Contrastive, CIFAR10Plain, CIFAR100Contrastive, CIFAR100Plain
from .svhn import SVHN, SVHNSimCLR
from .svhn import SVHNContrastive, SVHNPlain
from .imagenet import ImageNet, ImageNetContrastive, ImageNetPlain, ImageNetDINO
from .imagenet_subsets.imagenet_subset import ImageNet50, ImageNet50Contrastive, ImageNet50Plain, ImageNet50DINO, \
    ImageNet100, ImageNet100Contrastive, ImageNet100Plain, ImageNet100DINO, ImageNet200, ImageNet200Contrastive, \
    ImageNet200Plain, ImageNet200DINO


from .mnist import build_mnist
from .fashion_mnist import build_fashionmnist
