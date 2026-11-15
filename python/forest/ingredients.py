
import os
import torch
import torchvision
from tools.image_folders import ImageFolder #, Dataset
from tools.cifar_10 import CIFAR10

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset, Dataset, DataLoader
import numpy as np
import random
import math 

from tools.Constants import CIFAR_mean, CIFAR_std, Imagenet_mean, Imagenet_std

class Ingredient:

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)) -> None:

        self.args, self.setup = args, setup
        traindir = None
        valdir = None
        self.num_classes = 10

        if args.dataset == 'Imagenette':
            traindir = os.path.join(args.datapath, 'imagenette2/train')
            valdir = os.path.join(args.datapath, 'imagenette2/val')
        elif args.dataset == 'Imagewoof':
            traindir = os.path.join(args.datapath, 'imagewoof2/train')
            valdir = os.path.join(args.datapath, 'imagewoof2/val')
        
        self.data_mean = (0.485, 0.456, 0.406)
        self.data_std = (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(mean=self.data_mean,std=self.data_std)

        # Class Dictionary for CIFAR10
        if args.dataset == "CIFAR10":
            self.classDict = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer',
             5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        elif args.dataset == "CIFAR2":
            self.classDict = {0:'Machine', 1:'Animal'} # Machine , Animal
        else:
            self.classDict = None

        if traindir:
            self.trainset = ImageFolder(
            traindir,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                normalize,
            ]))

            self.validationset = ImageFolder(
            valdir,
                transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                normalize,
            ]))
        else:
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor()])

            # [S0-C repair 2026-09-13] test transform normalized to the training
            # input representation ([0,1], ToTensor only). Upstream had
            # ToTensor+ImageNet-Normalize here while training stays in [0,1] --
            # the train/eval mismatch identified in the 2026-09-11 audit.
            test_transforms = transforms.Compose([transforms.ToTensor()])

            if args.no_augment:
                train_transformations = test_transforms

            self.trainset = CIFAR10(train=True, transform = train_transforms)
            self.validationset = CIFAR10(train = False, transform = test_transforms)

        # [S0-C repair 2026-09-13] perturbation space marker: CIFAR trains in
        # [0,1] (unit), ImageNette/Woof train in ImageNet-normalized space.
        # witch.initialize_delta uses this to project eps=16 with the correct
        # per-space std/mean (upstream applied ImageNet std to the CIFAR
        # branch, an effective pixel bound ~4.4x the declared eps).
        self.delta_space = 'unit' if traindir is None else 'normalized'
        self.trainloader = DataLoader(self.trainset, batch_size=args.batchsize, shuffle=True)
        self.testloader = DataLoader(self.validationset, batch_size=args.batchsize, shuffle=False)
        
        

    def set_global_seed(self, seed):
        print("Setting seed as {}".format(seed))
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed(seed + 2)
        torch.cuda.manual_seed_all(seed + 3)
        np.random.seed(seed)
        return
    
    def initialize_attack_setup(self):
        if self.args.seed is not None:  # upstream falsy bug: seed 0 was silently discarded
            self.set_global_seed(self.args.seed)
        else:
            seed = np.random.randint(10000111)
            self.set_global_seed(seed)

        avail_classes = np.arange(self.num_classes)
        [self.target_class, self.poison_class] = np.random.choice(avail_classes, replace=False, size=2)
        camou_class = self.target_class

        # Choose Target

        target_indices = self.validationset.get_index(self.target_class)
        target_index = []
        target_index.append(np.random.choice(target_indices))

        self.targetset = Subset(self.validationset, target_index)
        self.targetloader = torch.utils.data.DataLoader(self.targetset)

        # [S0-I entry adaptation 2026-09-11] ImageFolder (Imagenette/Woof) has no
        # remove(); on the CIFAR branch remove() is dead code (mask computed, never
        # assigned), so the target image remains in the validation set either way.
        # Skipping here reproduces the CIFAR branch's actual behavior.
        if hasattr(self.validationset, 'remove'):
            self.validationset.remove([target_index])
        if self.classDict:
            print("Target image is chosen with ID {} from class {} ({})".format(target_index, self.classDict[self.target_class], self.target_class))
        else:
            print("Target image is chosen with ID {} from class {}".format(target_index, self.target_class))
        # Choose Poison Images:

        poison_index = []

        poison_index = self.trainset.get_index(self.poison_class)

        number_poisons = math.floor(self.args.pbudget * len(self.trainset))

        if self.classDict:
            print("{} poison images are chosen from class {} ({})".format(number_poisons, self.classDict[self.poison_class], self.poison_class))
        else:
            # [S0-I fix 2026-09-11] upstream else-branch had 3 slots / 2 args (only
            # reachable on non-CIFAR datasets); dropped the "({})" slot.
            print("{} poison images are chosen from class {}".format(number_poisons, self.poison_class))

        if number_poisons > len(poison_index):
            number_poisons = len(poison_index)
            print("Poison budget is over the maximum limit and set to {}".format(number_poisons))

        poison_index = np.random.choice(poison_index, number_poisons, replace=False)
        self.poison_dict = {}

        for index, val in enumerate(poison_index):
            self.poison_dict[val] = index

        self.poisonset = Subset(self.trainset, poison_index)
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=20, drop_last=False)


        # Choose Camouflage Images:
        camou_index = self.trainset.get_index(camou_class)
        number_camous = math.floor(self.args.cbudget * len(self.trainset))

        if self.classDict:
            print("{} camouflage images are chosen from class {} ({})".format(number_camous, self.classDict[self.poison_class], camou_class))
        else:
            print("{} camouflage images are chosen from class {}".format(number_camous, camou_class))

        if number_camous > len(camou_index):
            number_camous = len(camou_index)
            print("Camouflage budget is over the maximum limit and set to {}".format(number_camous))

        camou_index = np.random.choice(camou_index, number_camous, replace=False)

        self.camou_dict = {}

        for index, val in enumerate(camou_index):
            self.camou_dict[val] = index

        self.camouset = Subset(self.trainset, camou_index)
        self.camouloader = torch.utils.data.DataLoader(self.camouset, batch_size=20, drop_last=False)

        return
    