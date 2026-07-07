# dataset_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT_PATH = './dataset'

class DatasetLoader(Dataset):
    def __init__(self, dataset_name='cub', phase='train', size=84, transform=None, root_dir=ROOT_PATH):
        """
        dataset_name: e.g. 'cub', 'dog', 'car'.
        phase: 'train', 'valid', or 'test'
        """
        
        if dataset_name == 'cub':
            name = "CUB_200_2011/Split"
        elif dataset_name == 'dog':
            name = "Stanford_Dogs/Split"
        elif dataset_name == 'car':
            name = "Stanford_Cars/Split"
            
        data_folder = os.path.join(root_dir, name)
        image_folder = os.path.join(data_folder, phase)

        self.data = []
        self.label = []

        class_folders = sorted(os.listdir(image_folder))
        for label, class_folder in enumerate(class_folders):
            class_path = os.path.join(image_folder, class_folder)
            image_names = os.listdir(class_path)
            for image_name in image_names:
                img_path = os.path.join(class_path, image_name)
                self.data.append(img_path)
                self.label.append(label)

        if transform is None:
            if phase == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((100, 100)),
                    transforms.Pad(50),
                    transforms.CenterCrop(size),
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((100, 100)),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = Image.open(self.data[i]).convert('RGB')
        return self.transform(image), self.label[i]
