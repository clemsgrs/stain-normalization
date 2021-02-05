import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def default_image_loader(path):
    return Image.open(path).convert('RGB')

def get_transform(p):
    transform_list = []
    if p.resize_or_crop == 'resize_and_crop':
        osize = [p.loadSize, p.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(p.cropSize))
    elif p.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(p.cropSize))
    elif p.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, p.cropSize)))
    elif p.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, p.loadSize)))
        transform_list.append(transforms.RandomCrop(p.cropSize))

    if p.isTrain and not p.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


def CreateDataset(p):
    dataset = None
    if p.dataset_mode == 'aligned':
        dataset = AlignedDataset(p)
    elif p.dataset_mode == 'unaligned':
        dataset = UnalignedDataset(p)
    elif p.dataset_mode == 'single':
        dataset = SingleDataset(p)
    else:
        raise ValueError(f'dataset {p.dataset_mode} not recognized.')

    print(f'dataset {dataset.name()} was created')
    return dataset

def CreateDataLoader(p):
    data_loader = CustomDatasetDataLoader(p)
    print(data_loader.name())
    return data_loader

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, img_loader=default_image_loader):

        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError(f'Found 0 images in: {root}\n Supported image extensions are: {",".join(IMG_EXTENSIONS)}'))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.img_loader = img_loader

    def __getitem__(self, index):
        
        path = self.imgs[index]
        img = self.img_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class SingleDataset(data.Dataset):
    
    def __init__(self, p):
        
        super(SingleDataset, self).__init__()
        self.p = p
        self.root = p.dataroot
        self.transform = get_transform(p)

        self.dir_A = os.path.join(p.dataroot)
        self.A_paths = make_dataset(self.dir_A)
        # self.A_paths = sorted(self.A_paths)

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        if self.p.which_direction == 'BtoA':
            input_nc = self.p.output_nc
        else:
            input_nc = self.p.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'A_path': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'


class UnalignedDataset(data.Dataset):

    def __init__(self, p):
        
        super(UnalignedDataset, self).__init__()
        self.p = p
        self.root = p.dataroot
        self.dir_A = os.path.join(p.dataroot, p.phase + 'A')
        self.dir_B = os.path.join(p.dataroot, p.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(p)

    def __getitem__(self, index):
        
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.p.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform(A_img)
        B = self.transform(B_img)

        if self.p.which_direction == 'BtoA':
            input_nc = self.p.output_nc
            output_nc = self.p.input_nc
        else:
            input_nc = self.p.input_nc
            output_nc = self.p.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        
        return {'A': A, 'B': B,
                'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


class AlignedDataset(data.Dataset):

    def __init__(self, p):
        
        super(AlignedDataset, self).__init__()
        self.p = p
        self.root = p.dataroot
        self.dir_AB = os.path.join(p.dataroot, p.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(p.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.p.cropSize - 1))
        h_offset = random.randint(0, max(0, h - self.p.cropSize - 1))

        A = AB[:, h_offset:h_offset + self.p.cropSize,
               w_offset:w_offset + self.p.cropSize]
        B = AB[:, h_offset:h_offset + self.p.cropSize,
               w + w_offset:w + w_offset + self.p.cropSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.p.which_direction == 'BtoA':
            input_nc = self.p.output_nc
            output_nc = self.p.input_nc
        else:
            input_nc = self.p.input_nc
            output_nc = self.p.output_nc

        if (not self.p.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_path': AB_path, 'B_path': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


class CustomDatasetDataLoader():
    
    def __init__(self, p):
        self.p = p
        self.dataset = CreateDataset(p)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=p.batchSize,
            shuffle=not p.serial_batches,
            num_workers=int(p.nThreads))

    def name(self):
        return 'CustomDatasetDataLoader'

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.p.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.p.max_dataset_size:
                break
            yield data