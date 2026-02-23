import os
import json
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from .data_utils import pre_question

class vqa_dataset(Dataset):
    def __init__(self, annotation_file, image_dir):
        self.annotation = json.load(open(os.path.join("annotation", annotation_file), 'r'))
        self.image_dir = image_dir
        self.transforms = transforms.Compose([transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path_name = self.annotation[index]["image_name"]
        question = pre_question(self.annotation[index]["question"])
        question_id = self.annotation[index]["question_id"]
        image_path = os.path.join(self.image_dir, image_path_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        return image_path_name.split(".")[0], image, question, question_id
    
class vqa_imagefolder_dump_attention(Dataset):
    def __init__(self, image_dir=None, clean_image_dir=None, attacked_image_dir=None, images_root=None):
        self.samples = []
        self.transforms = transforms.Compose([transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                ])
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # New mode: recursively read all images under images_root and keep relative path.
        if images_root is not None:
            for root, _, files in os.walk(images_root):
                for image_name in sorted(files):
                    ext = os.path.splitext(image_name)[1].lower()
                    if ext not in image_exts:
                        continue
                    image_path = os.path.join(root, image_name)
                    rel_path = os.path.relpath(image_path, images_root)
                    # Keep folder structure in output by returning relative path.
                    image_name_with_rel = rel_path
                    is_clean = 0 if "clean" in image_name.lower() else 1
                    self.samples.append((image_name_with_rel, image_path, is_clean))
            if len(self.samples) == 0:
                raise ValueError(f"No images found recursively under images_root={images_root}")
            return
        # Backward compatible mode: read from <image_dir>/image and infer clean/attack by filename.
        if image_dir is not None:
            base_image_dir = os.path.join(image_dir, "image")
            for image_name in sorted(os.listdir(base_image_dir)):
                image_path = os.path.join(base_image_dir, image_name)
                if not os.path.isfile(image_path):
                    continue
                is_clean = 0 if "clean" in image_name else 1
                self.samples.append((image_name, image_path, is_clean))
            return

        # New mode: explicit directories for clean and attacked images.
        if clean_image_dir is not None:
            for image_name in sorted(os.listdir(clean_image_dir)):
                image_path = os.path.join(clean_image_dir, image_name)
                if not os.path.isfile(image_path):
                    continue
                self.samples.append((f"clean_{image_name}", image_path, 0))
        if attacked_image_dir is not None:
            for image_name in sorted(os.listdir(attacked_image_dir)):
                image_path = os.path.join(attacked_image_dir, image_name)
                if not os.path.isfile(image_path):
                    continue
                self.samples.append((f"attacked_{image_name}", image_path, 1))

        if len(self.samples) == 0:
            raise ValueError("No images found. Provide --image_dir or --clean_image_dir/--attacked_image_dir.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_name, image_path_name, is_clean = self.samples[index]
        image = Image.open(image_path_name).convert('RGB')
        image = self.transforms(image)
        return image_name, image, is_clean
