from inspect import trace
import os

# import imageio
from PIL import Image
import yaml
import torch
import torchvision
from torch.utils.data.dataset import Subset
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip, Resize, ToTensor, RandomResizedCrop, RandomCrop, Normalize)
# import misc
import socket
import numpy as np
import random
# print(socket.gethostname())
from torch.utils.data import Dataset, DataLoader
import io
import pdb
import torch.distributed as dist
import webdataset as wds
import json
Image.MAX_IMAGE_PIXELS = 1000000000
# import clip  # pylint: disable=import-outside-toplevel

def read_label_dict():
    data = np.loadtxt('utils/imagenet_labels.txt', str, delimiter='$')
    label_dict = {}

    for d in data:
        label, name = d.split(':')
        label = int(label)
        name = name[2:-2].split(',')
        name = [n.strip() for n in name if len(n) >0]
        label_dict[label] = name
    return label_dict


def my_split_by_worker(urls):
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        return urls[wi.id::wi.num_workers]

def my_split_by_node(urls):
    node_id, node_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
    return urls[node_id::node_count]

def create_webdataset(
    urls,
    image_transform,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
    nsample=None,
    imagenet=True,
    resolution=None,
    tokenizer=None
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    
      # pylint: disable=import-outside-toplevel


    # dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue)#, repeat=True, resampled=True)
    # dataset = wds.WebDataset(urls, handler=wds.handlers.warn_and_continue, repeat=True, resampled=True, splitter=my_split_by_worker, nodesplitter=my_split_by_node)
    dataset = wds.WebDataset(urls, handler=wds.handlers.warn_and_continue, resampled=True)
    # tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]

    def filter_dataset(item):
        # if enable_text and caption_key not in item:
            # return False
        if enable_text:
            if (caption_key not in item and 'cls' not in item):
                return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        
        if resolution is not None:
            print(item.keys(),'item in filter')
            meta = item["json"].decode("utf-8")
            meta = json.loads(meta)
            
            if meta["WIDTH"][0] is None or meta["HEIGHT"][0] is None:
                return False
            if meta["WIDTH"][0] < resolution or meta["HEIGHT"][0] < resolution:
                return False
        return True

    filtered_dataset = dataset.select(filter_dataset)
    # print('!!!')
    # if imagenet:
        # imnt_dict = read_label_dict()

    def preprocess_dataset(item):
        output = {}
        # print(item["__key__"])
        if enable_image:
            image_data = item[image_key]
            if item["__key__"].startswith('laion'):
                image = pilimg_from_base64(image_data)
            else:
                image = Image.open(io.BytesIO(image_data))
                image = image.convert('RGB') 
            image_tensor = image_transform(image)
            output["image_filename"] = item["__key__"]
            output["image_tensor"] = image_tensor
            if item["__key__"].startswith('imagenet'):
                output["label"] = int(item["cls"])
        if enable_text:
            if item["__key__"].startswith('imagenet'):
                label = int(item["cls"])
                names = imnt_dict[label]
                name = np.random.choice(names)
                caption = f'photo of a {name}'
            
            elif item["__key__"].startswith('ffhq'):
                names = ['person', 'human']
                name = np.random.choice(names)
                caption = f'photo of a {name}'

            else:
                text = item[caption_key]
                caption = text.decode("utf-8")
            
            if tokenizer is not None:
                tokenized_text = tokenizer(caption,
                                padding="max_length",
                                # padding="do_not_pad",
                                truncation=True,
                                max_length=tokenizer.model_max_length,
                                return_tensors="pt",).input_ids
                
                # tokenized_text = tokenizer.pad(
                #     {"input_ids": [tokenized_text]},
                #     padding="max_length",
                #     max_length=tokenizer.model_max_length,
                #     return_tensors="pt",
                # ).input_ids
                output["text_tokens"] = tokenized_text
            output["text"] = caption

        if enable_metadata:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            output["metadata"] = metadata
        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue).with_length(int(nsample)).shuffle(2000)#.with_epoch(100)
    # transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.reraise_exception).with_length(int(nsample)).shuffle(25000)#.with_epoch(100)
    return transformed_dataset


def create_wbd(args, load_txt=True, tokenizer=None):
    tar_list = []
    n_samples = 0

    if 'laion-aes' in args.data_list:
        ts = os.listdir(os.path.join(args.root_path, 'laion_aes_wbd'))
        ts = [t for t in ts if t.endswith('tar')]
        ts = [os.path.join(args.root_path, 'laion_aes_wbd', t) for t in ts]
        tar_list.extend(ts)

        print('laion-aes loaded')
        n_samples += 11131263

    if 'gcc' in args.data_list:
        folders = ['gcc3m_shards', 'gcc12m_shards']
        for f in folders:
            if os.path.exists(os.path.join(args.root_path, 'cross_domain', 'local_data', f)):
                ts = os.listdir(os.path.join(args.root_path, 'cross_domain', 'local_data', f))
                ts = [t for t in ts if t.endswith('tar')]
                ts = [os.path.join(args.root_path, 'cross_domain', 'local_data', f, t) for t in ts]
                tar_list.extend(ts)

        print('gcc 15m loaded')
        n_samples += 14.44e6

    if 'imagenet' in args.data_list:
        ts = os.listdir(os.path.join(args.data_path, 'imagenet_wbd'))
        ts = [t for t in ts if t.endswith('tar')]
        ts = [os.path.join(args.data_path, 'imagenet_wbd', t) for t in ts]
        tar_list.extend(ts)

        print('imagenet loaded')
        n_samples += 1273455
    
    if 'ffhq' in args.data_list:
        ts = os.listdir(os.path.join(args.data_path, 'ffhq_wbd'))
        ts = [t for t in ts if t.endswith('tar')]
        ts = [os.path.join(args.data_path, 'ffhq_wbd', t) for t in ts]
        tar_list.extend(ts)

        print('ffhq loaded')
        n_samples += 69487
    
    if 'laion5p' in args.data_list:
        ts = os.listdir(os.path.join(args.root_path, 'laion5plus'))
        ts = [t for t in ts if t.endswith('tar')]
        ts = [os.path.join(args.root_path, 'laion5plus', t) for t in ts]
        tar_list.extend(ts)

        ts = os.listdir(os.path.join(args.root_path, 'laion5p'))
        ts = [t for t in ts if t.endswith('tar')]
        ts = [os.path.join(args.root_path, 'laion5p', t) for t in ts]
        tar_list.extend(ts)

        print('laion5p loaded')
        n_samples += 11131263

    print(f'Found {len(tar_list)} tar files and {n_samples} samples.')
    np.random.shuffle(tar_list)

    # transform_train = Compose([Resize(args.img_size), RandomCrop(args.img_size), RandomHorizontalFlip(p=0.5), ToTensor()])
    # transform_train = Compose([Resize(args.img_size), RandomCrop(args.img_size), ToTensor()])
    size = int(args.img_size)
    transform_train = Compose([Resize(size), 
                               CenterCrop(size), 
                               ToTensor(),
                               Normalize([0.5], [0.5])])

    # resolution = size//2 if args.sr_scale != 1 else None

    dataset_train = create_webdataset(
        tar_list,
        transform_train,
        enable_text=load_txt,
        enable_image=True,
        image_key="jpg",
        caption_key="txt",
        enable_metadata=False,
        cache_path=None,
        nsample=n_samples,
        resolution=None,
        tokenizer=tokenizer,
    )

    # local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0

    # print(f'local rank {local_rank} / global rank {dist.get_rank()} \
    # successfully build train dataset')
    # print(f'world size {dist.get_world_size()}')

    # num_tasks = misc.get_world_size()
    # global_rank = misc.get_rank()

    # indices = np.arange(misc.get_rank(), len(dataset_train), misc.get_world_size())

    # if args.distributed:
    #     sampler_train = SubsetRandomSampler(indices)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,# sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        # prefetch_factor=10,
    )

    return train_loader

class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        preprocess,
        input_dataset,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        cache_path=None,
    ):
        self.batch_size = batch_size
        dataset = create_webdataset(
            input_dataset,
            preprocess,
            enable_text=enable_text,
            enable_image=enable_image,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "webdataset")

    def __iter__(self):
        for batch in self.dataloader:
            yield batch


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        # img = imageio.imread(self.folder+path)
        # img = Image.open(self.folder+path)
        img = Image.open(os.path.join(self.folder, path))
        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> channels first
        return img

    def __len__(self):
        return len(self.image_paths)


class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None, transforms=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)
        self.transforms = transforms
    def __getitem__(self, index):
        # print(self.dataset[index][0].shape)
        data = self.dataset[index][0]
        if self.transforms:
            return self.transforms(data)
        else:
            return data
        # return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)

    def __len__(self):
        return self.length


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_default_dataset_paths():
    with open("datasets.yml") as yaml_file:
        read_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    paths = {}
    for i in range(len(read_data)):
        paths[read_data[i]["dataset"]] = read_data[i]["path"]

    return paths


def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_datasets(
    dataset_name,
    img_size,
    get_val_dataset=False,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    random=False
):
    # transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
    # transform = Compose([Resize(img_size), CenterCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor()])
    # transform = Compose([Resize(img_size), RandomCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor()])
    # transform_with_flip = Compose([Resize(img_size), CenterCrop(img_size), RandomHorizontalFlip(p=1.0), ToTensor()])
    if random:
        # transform = Compose([Resize(int(img_size)), CenterCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor(), Normalize([0.5], [0.5]),])
        transform = Compose([Resize(int(img_size)), CenterCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor()])
    else:
        # transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor(), Normalize([0.5], [0.5]),])
        transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
    # transform_with_flip = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
    transform_with_flip = transform

    print(transform)
    
    default_paths = get_default_dataset_paths()

    if dataset_name in default_paths:
        dataset_path = default_paths[dataset_name]
    elif dataset_name == "custom":
        if custom_dataset_path:
            dataset_path = custom_dataset_path
        else:
            raise ValueError("Custom dataset selected, but no path provided")
    else:
        raise ValueError(f"Invalid dataset chosen: {dataset_name}. To use a custom dataset, set --dataset \
            flag to 'custom'.")

    hostname = socket.gethostname()
    if not (hostname == 'ubuntu' or hostname.startswith('Qlab')):
        dataset_path = custom_dataset_path

    if dataset_name == "churches":
        train_dataset = torchvision.datasets.LSUN(
            dataset_path,
            classes=["church_outdoor_train"],
            transform=None
        )
        train_dataset = NoClassDataset(train_dataset, transforms=transform)
        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                dataset_path,
                classes=["church_outdoor_train"],
                transform=None,
            )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                dataset_path,
                classes=["church_outdoor_val"],
                transform=transform
            )

    elif dataset_name == "bedrooms":
        train_dataset = torchvision.datasets.LSUN(
            dataset_path,
            classes=["bedroom_train"],
            transform=None,
        )
        train_dataset = NoClassDataset(train_dataset, transforms=transform)
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                dataset_path,
                classes=["bedroom_val"],
                transform=transform,
            )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                dataset_path,
                classes=["bedroom_train"],
                transform=transform_with_flip,
            )

    elif dataset_name == "ffhq":
        ann_file = 'maps.txt'
        prefix =  "images1024x1024.zip@/"

        # transform_train = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
        # transform_train = Compose([Resize(int(img_size*1.05)), CenterCrop(img_size), ToTensor()])
        transform_train = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
        # transform_train = Compose([Resize(int(img_size*1.05)), RandomCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor()])
        print(transform_train)
        cache_mode ='part'

        # train_dataset = CachedImageFolder_ori(custom_dataset_path, ann_file, prefix, transform_train,
                                    # cache_mode=cache_mode)
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(custom_dataset_path, 'images256'),
            transform=transform_train,
        )

        val_dataset = None
    
    elif dataset_name == "celeba":
        ann_file = 'maps.txt'
        prefix =  "images1024x1024.zip@/"

        transform_train = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])

        cache_mode ='part'

        # train_dataset = CachedImageFolder_ori(custom_dataset_path, ann_file, prefix, transform_train,
                                    # cache_mode=cache_mode)
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(custom_dataset_path),
            transform=transform_train,
        )

        val_dataset = None
    
    elif dataset_name == "imagenet":

        transform_train = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])

        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(custom_dataset_path, 'train'),
            transform=transform_train,
        )

        val_dataset = None

    elif dataset_name == "custom":
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transform,
        )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.ImageFolder(
                dataset_path,
                transform=transform_with_flip,
            )

        if get_val_dataset:
            train_dataset, val_dataset = train_val_split(train_dataset, train_val_split_ratio)
            if get_flipped:
                train_dataset_flip, _ = train_val_split(train_dataset_flip, train_val_split_ratio)

    if get_flipped:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_flip])

    if not get_val_dataset:
        val_dataset = None

    return train_dataset, val_dataset


def get_data_loaders(
    dataset_name,
    img_size,
    batch_size,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    num_workers=1,
    drop_last=True,
    shuffle=True,
    get_val_dataloader=False,
    distributed=False,
    random=False,
    args=None, 
    
):


    if dataset_name in ['bedrooms', 'churches']:
        train_dataset, val_dataset = get_datasets(
            dataset_name,
            img_size,
            get_flipped=get_flipped,
            get_val_dataset=get_val_dataloader,
            train_val_split_ratio=train_val_split_ratio,
            custom_dataset_path=custom_dataset_path,
            random=random
        )

        if distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            
        else:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)
            
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=num_workers,
            sampler=sampler_train,
            # shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=drop_last
        )

    elif dataset_name == 'ffhq':
        ann_file = 'maps.txt'
        prefix =  "images1024x1024.zip@/"

        transform_train = Compose([Resize(img_size), CenterCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor()])

        cache_mode = 'part' if distributed else 'no'

        dataset_train = CachedImageFolder_ori(custom_dataset_path, ann_file, prefix, transform_train,
                                    cache_mode=cache_mode)

        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        # sampler_train = torch.utils.data.DistributedSampler(
            # dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, 
        # )
        indices = np.arange(misc.get_rank(), len(dataset_train), misc.get_world_size())

        if distributed:
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        
        train_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker
        )
    
    elif dataset_name == 'celeba':
        ann_file = 'maps.txt'
        prefix =  "data1024x1024.zip@/"

        if img_size == 1024:
            transform_train = Compose([RandomHorizontalFlip(p=0.5), ToTensor()])
        else:
            transform_train = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])

        cache_mode = 'part' if distributed else 'no'

        dataset_train = CachedImageFolder_ori(custom_dataset_path, ann_file, prefix, transform_train,
                                    cache_mode=cache_mode)

        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        # sampler_train = torch.utils.data.DistributedSampler(
            # dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, 
        # )
        indices = np.arange(misc.get_rank(), len(dataset_train), misc.get_world_size())

        if distributed:
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        
        train_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker
        )
    
    elif dataset_name == 'laion-art':
        
        tar_list = os.listdir(custom_dataset_path)
        tar_list = [t for t in tar_list if t.endswith('.tar')]
        tar_list = [os.path.join(custom_dataset_path, t) for t in tar_list]

        tar_list = custom_dataset_path+'/{00000..00807}.tar'
        print(tar_list)

        transform_train = Compose([Resize(img_size), RandomCrop(img_size), 
                                   RandomHorizontalFlip(p=0.5), 
                                   ToTensor(),
                                   Normalize([0.5], [0.5])]) #ADDED BY TW

        # train_loader = WebdatasetReader(
        #     transform_train,
        #     tar_list,
        #     batch_size,
        #     num_workers,
        #     enable_text=True,
        #     enable_image=True,
        #     enable_metadata=True,
        # )
        
        
        dataset_train = create_webdataset(
            tar_list,
            transform_train,
            enable_text=True,
            enable_image=True,
            image_key="jpg",
            caption_key="txt",
            enable_metadata=False,
            cache_path=None,
        )

        local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0

        print(f'local rank {local_rank} / global rank {dist.get_rank()} \
        successfully build train dataset')
        print(f'world size {dist.get_world_size()}')
        # dc_collate = partial(collate, samples_per_gpu=batch_size)
        # train_len = len(dataset_train)
        # init_fn = partial(worker_init_fn, num_workers=num_workers, rank=dist.get_rank(), seed=dist.get_rank())
        # train_loader = wds.WebLoader(
        #     dataset_train.batched(batch_size, dc_collate, partial=False),
        #     batch_size=None,
        #     shuffle=False,
        #     num_workers=num_workers,
        #     pin_memory=True,
        #     persistent_workers=num_workers > 0,
        #     worker_init_fn=init_fn)
        # train_loader = wds.WebLoader(dataset_train.batched(batch_size),
            # num_workers=num_workers,
            # pin_memory=True,)
        # train_loader = train_loader.ddp_equalize(train_len // batch_size)

        # train_nbatches = max(1, train_len // (batch_size * dist.get_world_size()))
        # print(train_nbatches, train_len, batch_size)
        # train_loader = (train_loader.with_epoch(train_nbatches).with_length(train_nbatches))

        # exit()
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        # sampler_train = torch.utils.data.DistributedSampler(
            # dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, 
        # )
        indices = np.arange(misc.get_rank(), len(dataset_train), misc.get_world_size())

        if distributed:
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        
        train_loader = torch.utils.data.DataLoader(
            dataset_train,# sampler=sampler_train,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker
        )

    elif dataset_name == 'laion-aes':
        transform_train = Compose([Resize(img_size), RandomCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor()])

        dataset_train = LAIONdataset(custom_dataset_path, transform_train)

        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=False)
        indices = np.arange(misc.get_rank(), len(dataset_train), misc.get_world_size())
        train_sampler = SubsetSampler(indices)

        if args.seqdata:
            train_loader = torch.utils.data.DataLoader(
                dataset_train,
                sampler=train_sampler,
                batch_size=batch_size,
                # num_workers=num_workers,
                num_workers=1,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=seed_worker,
                # shuffle=True
                shuffle=False
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset_train,
                # sampler=train_sampler,
                batch_size=batch_size,
                # num_workers=num_workers,
                num_workers=1,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=seed_worker,
                shuffle=True
                # shuffle=False
            )

    else:
        ann_file = 'train_map.txt'
        prefix =  "train.zip@/"

        # if args.debug:
        hostname = socket.gethostname()
        if (hostname == 'ubuntu' or hostname.startswith('Qlab')):
            ann_file = 'val_map.txt'
            prefix =  "val.zip@/"
            # ann_file = 'train_map.txt'
            # prefix =  "train.zip@/"

        # transform_train = Compose([Resize(img_size), CenterCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor()])
        if distributed:
            # transform_train = Compose([RandomResizedCrop(size=img_size), RandomHorizontalFlip(p=0.5), ToTensor()])
            transform_train = Compose([Resize(img_size), RandomCrop(img_size), RandomHorizontalFlip(p=0.5), ToTensor()])
        else:
            transform_train = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])

        print(transform_train)
        cache_mode = 'part' if distributed else 'no'
        dataset_train = CachedImageFolder_ori(custom_dataset_path, ann_file, prefix, transform_train,
                                    cache_mode=cache_mode)

        
        # sampler_train = SubsetRandomSampler(indices)
        if distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            # sampler_train = torch.utils.data.DistributedSampler(
                # dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, 
            # )
            indices = np.arange(misc.get_rank(), len(dataset_train), misc.get_world_size())
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        
        train_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker
        )


        if get_val_dataloader:
            ann_file = 'val_map.txt'
            prefix =  "val.zip@/"

            transform_val = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])

            print(transform_val)
            cache_mode = 'part' if distributed else 'no'
            dataset_val = CachedImageFolder_ori(custom_dataset_path, ann_file, prefix, transform_val,
                                        cache_mode=cache_mode)

            val_loader = torch.utils.data.DataLoader(
                dataset_val, #sampler=sampler_train,
                batch_size=100,
                num_workers=1,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=seed_worker
            )

    if get_val_dataloader:

        # sampler_val = torch.utils.data.SequentialSampler(val_dataset)

        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset, 
        #     num_workers=num_workers,
        #     sampler=sampler_val,
        #     # shuffle=shuffle,
        #     batch_size=batch_size,
        #     pin_memory=True,
        #     drop_last=drop_last
        # )
        val_loader = val_loader
    else:
        val_loader = None

    return train_loader, val_loader
