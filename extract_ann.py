import os
import pandas as pd
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms, utils

# from pandas_profiling import ProfileReport

FOLDER_KEY='folder'
FILENAME_KEY='filename'
WIDTH_KEY='width'
HEIGHT_KEY='height'
CLASS_NAME_KEY='class_name'
XMIN_KEY='xmin'
YMIN_KEY='ymin'
XMAX_KEY='xmax'
YMAX_KEY='ymax'
XML_PATH_KEY = 'xml_path'
HEADERS = [FOLDER_KEY, FILENAME_KEY, XML_PATH_KEY, WIDTH_KEY, HEIGHT_KEY, CLASS_NAME_KEY, YMIN_KEY, XMIN_KEY, YMAX_KEY, XMAX_KEY]

def get_xml_tags(xml_dir):
    df_data = []
    for root,dirs,files in os.walk(xml_dir):
        for f in files:
            if f.startswith('.'):
                continue
            xml_path = os.path.join(root,f)
            try:
                xml_obj = open(xml_path)
                doc = ET.parse(xml_obj)
            except:
                print("parsing xml file: %s failed!"%xml_path)
                continue

            folder = doc.findtext('folder')
            filename = doc.findtext('filename')
            width = int(doc.findtext('size/width'))
            height = int(doc.findtext('size/height'))
            for item in doc.iterfind('object'):
                class_name = item.findtext('name')
                xmin = int(item.findtext('bndbox/xmin'))
                ymin = int(item.findtext('bndbox/ymin'))
                xmax = int(item.findtext('bndbox/xmax'))
                ymax = int(item.findtext('bndbox/ymax'))
                df_data.append([folder, filename, xml_path,width,height,class_name,ymin,xmin,ymax,xmax])
    df = pd.DataFrame(data=df_data, columns=HEADERS)
    return df


class DogImagesDataset(Dataset):
    def __init__(self, xml_dir, img_dir, transform=None):
        self.data_frame = get_xml_tags(xml_dir)
        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir,
                                'n' + self.data_frame.loc[idx, FOLDER_KEY] + '-' + self.data_frame.loc[idx, CLASS_NAME_KEY],
                                self.data_frame.loc[idx, FILENAME_KEY] + '.jpg')
        image = Image.open(img_path)

        metadata = list(self.data_frame.loc[idx])
        sample = {'image': image, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample

def show_some_images(images):
    plt.figure()

    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.show()

def bbox_crop(sample):
    metadata = sample['metadata']
    # Crop per the BBox, and make sure it's square, then resize to be the same size:
    bboxWidth = metadata[9] - metadata[7]
    bboxHeight = metadata[8] - metadata[6]
    size = max(bboxWidth, bboxHeight)
    image = sample['image'].crop((metadata[7], metadata[6], metadata[7] + size, metadata[6] + size)).resize((224, 224))
    return {'image': image, 'metadata': metadata}

orig_to_tensor = transforms.ToTensor()
def to_tensor(sample):
    image_tensor = orig_to_tensor(sample['image'])
    return {'image': image_tensor, 'metadata': sample['metadata'] }

data_transform = transforms.Compose([
    bbox_crop,
    to_tensor
])

if __name__ == '__main__':
    dog_dataset = DogImagesDataset(xml_dir='Annotation', img_dir='Images', transform=data_transform)
    dataloader = DataLoader(dog_dataset, batch_size=4, shuffle=True, num_workers=4)
    _, sample_batch = next(enumerate(dataloader))

    show_some_images(sample_batch['image'])
    print('finish')





