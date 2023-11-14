from PIL import Image
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import cv2
import os
import random
import itertools
import tarfile
from tqdm import tqdm
import urllib.request
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader

class BiscuitDataset(Dataset):
    def __init__(self, root_path='../data', is_train=True,
                 resize=128, cropsize=128):
        self.root_path = root_path
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.biscuit_folder_path = os.path.join('IndustryBiscuit_Folders')
        self.x, self.y = self.load_dataset_folder()
        self.transform_x = T.Compose([T.Resize(resize),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor()])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)


        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []

        img_dir = os.path.join(self.biscuit_folder_path, phase)
        gt_dir = os.path.join(self.biscuit_folder_path, phase, 'ok')

        img_types = sorted(os.listdir(img_dir), reverse=True)
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.jpg')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'ok':
                y.extend([1] * len(img_fpath_list))
            else:
                y.extend([0] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.jpg')
                                 for img_fname in img_fname_list]

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)
    
def download_class_biscuit(opt):
    # there is only one class so we hardcoded the pos_class
    opt.pos_class = "biscuit"
    opt.input_name = "biscuit_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(opt.pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    scale = opt.size_image
    pos_class = opt.pos_class
    num_images = opt.num_images

    def imsave(img, i):
        if not os.path.exists("Input/Images"):
            os.makedirs("Input/Images")
        # transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((scale, scale))])
        # im = transform(img)
        img = (img) * 255
        npimg = img.numpy().astype(np.uint8)
        npimg = np.transpose(npimg, (1, 2, 0))
        im = Image.fromarray(npimg)
        # im = Image.fromarray(img.astype(np.uint8)).resize((scale,scale))
        im.save("Input/Images/biscuit_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")
        im = cv2.imread("Input/Images/biscuit_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")
        H, S, V = cv2.split(cv2.cvtColor((im), cv2.COLOR_RGB2HSV))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_V = clahe.apply(V)
        im = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
        im = cv2.resize(im, (scale,scale), interpolation=cv2.INTER_AREA)

        cv2.imwrite("Input/Images/biscuit_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png", im)

    if opt.mode == "train":
        trainset = BiscuitDataset(is_train=True)  # images of size 224, 224
        trainloader = DataLoader(trainset, batch_size=len(trainset), pin_memory=True)
        dataiter = iter(trainloader)
        images, _ = next(dataiter)
        dicty = {}
        if opt.random_images_download == False:
            count_images,step_index = 0,0
            for i in range(images.shape[0]):
                t = images[i]
                imsave(t, count_images)
                dicty[count_images] = i
                count_images += 1
                if count_images == num_images: step_index +=1
                if step_index == opt.index_download: break
                if count_images == num_images and step_index != opt.index_download: count_images=0
            training_images = list(dicty.values())

        else:
            random_index = random.sample(range(0, images.shape[0]), opt.num_images)
            training_images = list(random_index)
            for i in range(len(training_images)):
                index = training_images[i]
                t = images[index]
                imsave(t,i)
        print("training imgaes: ", training_images)


        if opt.add_jiggle_transformation:
            genertator0 = itertools.product((0,), (False, True), (-1, 1, 0), (-1,), (0,), (0,))
            genertator1 = itertools.product((0,), (False, True), (0, 1), (0, 1), (0, 1, 2, 3), (0,))
            genertator2 = itertools.product((0,), (False, True), (0,), (0,), (0, 1, 2, 3), (1,))
            genertator3 = itertools.product((0,), (False, True), (-1,), (1, 0), (0,), (0,))
            genertator4 = itertools.product((0,), (False,), (1, -1), (0,), (0,), (1,))
            genertator5 = itertools.product((0,), (False,), (0,), (1, -1), (0,), (1,))
            genertator = itertools.chain(genertator0, genertator1, genertator2, genertator3, genertator4, genertator5)
        else:
            genertator0 = itertools.product((0,), (False, True), (-1, 1, 0), (-1,), (0,))
            genertator1 = itertools.product((0,), (False, True), (0, 1), (0, 1), (0, 1, 2, 3))
            genertator2 = itertools.product((1,), (False, True), (0,), (0,), (0, 1, 2, 3))
            genertator3 = itertools.product((0,), (False, True), (-1,), (1, 0), (0,))
            genertator4 = itertools.product((1,), (False,), (1, -1), (0,), (0,))
            genertator5 = itertools.product((1,), (False,), (0,), (1, -1), (0,))
            genertator = itertools.chain(genertator0, genertator1, genertator2, genertator3, genertator4, genertator5)
        lst = list(genertator)
        random.shuffle(lst)
        path_transform = "TrainedModels/" + str(opt.input_name)[:-4]
        if os.path.exists(path_transform) == False:
            os.mkdir(path_transform)
        np.save(path_transform + "/transformations.npy", lst)

    biscuit_testset = BiscuitDataset(is_train=False)
    biscuit_loader = DataLoader(biscuit_testset, batch_size=len(biscuit_testset), pin_memory=True)
    (biscuit_data, biscuit_targets) = next(iter(biscuit_loader))
    biscuit_target = biscuit_targets.numpy()
    biscuit_data = biscuit_data.numpy()
    path_test = "biscuit_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(num_images)
    if os.path.exists(path_test) == False:
        os.mkdir(path_test)
    np.save(path_test + "/biscuit_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", biscuit_data)
    np.save(path_test + "/biscuit_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", biscuit_target)

    opt.input_name = "biscuit_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    return opt.input_name
