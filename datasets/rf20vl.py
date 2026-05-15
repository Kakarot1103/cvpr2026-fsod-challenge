r""" RF-20VL few-shot object detection dataset.

Deterministic, class-ordered iteration over all (query_image, class) pairs.
Each item returns the query image with ALL bboxes of that class,
plus ALL support images and bboxes from the train split for the same class.
"""
import os
import json
from collections import defaultdict

import torch
import PIL.Image as Image
from torch.utils.data import Dataset


class DatasetRF20VL(Dataset):

    def __init__(self, datapath, subset, query_split="test", transform=None):
        self.datapath = datapath
        self.subset = subset
        self.query_split = query_split
        self.transform = transform

        self.subset_dir = os.path.join(datapath, subset)

        # Load annotations
        train_data = self._load_json(os.path.join(self.subset_dir, "train", "_annotations.coco.json"))
        query_data = self._load_json(os.path.join(self.subset_dir, query_split, "_annotations.coco.json"))

        # Build image lookups
        self.train_imgs = {img["id"]: img for img in train_data["images"]}
        self.query_imgs = {img["id"]: img for img in query_data["images"]}

        # Build category map (skip id=0 placeholder)
        self.categories = {cat["id"]: cat["name"] for cat in train_data["categories"] if cat["id"] != 0}

        # Build query index: (cat_id, img_id) -> [bbox, ...]
        self.query_bboxes = defaultdict(list)
        for ann in query_data["annotations"]:
            if ann["category_id"] != 0:
                self.query_bboxes[(ann["category_id"], ann["image_id"])].append(ann["bbox"])

        # Build support index: cat_id -> [(img_id, file_name, [bbox, ...]), ...]
        train_bboxes_by_class = defaultdict(lambda: defaultdict(list))
        for ann in train_data["annotations"]:
            if ann["category_id"] != 0:
                train_bboxes_by_class[ann["category_id"]][ann["image_id"]].append(ann["bbox"])

        self.support_info = {}
        for cat_id, img_dict in train_bboxes_by_class.items():
            self.support_info[cat_id] = []
            for img_id in sorted(img_dict.keys()):
                img_info = self.train_imgs[img_id]
                self.support_info[cat_id].append((img_id, img_info["file_name"], img_dict[img_id]))

        # Build ordered index: [(cat_id, img_id), ...] sorted by cat_id then img_id
        self.index = sorted(self.query_bboxes.keys(), key=lambda x: (x[0], x[1]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cat_id, img_id = self.index[idx]

        # Load query image
        query_img_info = self.query_imgs[img_id]
        query_img_path = os.path.join(self.subset_dir, self.query_split, query_img_info["file_name"])
        query_img = Image.open(query_img_path).convert("RGB")

        # Query bboxes
        query_bboxes = torch.tensor(self.query_bboxes[(cat_id, img_id)], dtype=torch.float32)

        # Load all support images
        support_items = self.support_info.get(cat_id, [])
        support_imgs = []
        support_bboxes = []
        support_img_paths = []
        for supp_img_id, supp_file_name, supp_bboxes in support_items:
            supp_path = os.path.join(self.subset_dir, "train", supp_file_name)
            support_imgs.append(Image.open(supp_path).convert("RGB"))
            support_bboxes.append(torch.tensor(supp_bboxes, dtype=torch.float32))
            support_img_paths.append(supp_path)

        # Apply transforms and rescale bboxes accordingly
        if self.transform is not None:
            orig_qw, orig_qh = query_img.size
            query_img = self.transform(query_img)
            if isinstance(query_img, Image.Image):
                new_qw, new_qh = query_img.size
            else:
                new_qh, new_qw = query_img.shape[-2:]
            sx, sy = new_qw / orig_qw, new_qh / orig_qh
            query_bboxes = self._rescale_bboxes(query_bboxes, sx, sy)

            rescaled_support_bboxes = []
            for i, supp_img in enumerate(support_imgs):
                orig_sw, orig_sh = supp_img.size
                support_imgs[i] = self.transform(supp_img)
                if isinstance(support_imgs[i], Image.Image):
                    new_sw, new_sh = support_imgs[i].size
                else:
                    new_sh, new_sw = support_imgs[i].shape[-2:]
                ssx, ssy = new_sw / orig_sw, new_sh / orig_sh
                rescaled_support_bboxes.append(self._rescale_bboxes(support_bboxes[i], ssx, ssy))
            support_bboxes = rescaled_support_bboxes
        else:
            from torchvision.transforms.functional import pil_to_tensor
            query_img = pil_to_tensor(query_img).float() / 255.0
            support_imgs = [pil_to_tensor(img).float() / 255.0 for img in support_imgs]

        if len(support_imgs) == 0:
            support_imgs = []

        return {
            "query_img": query_img,
            "query_bboxes": query_bboxes,
            "query_img_path": query_img_path,
            "query_img_size": (query_img_info["width"], query_img_info["height"]),

            "support_imgs": support_imgs,
            "support_bboxes": support_bboxes,
            "support_img_paths": support_img_paths,

            "class_id": cat_id,
            "category": self.categories[cat_id],
            "subset": self.subset,
        }

    @staticmethod
    def _rescale_bboxes(bboxes, sx, sy):
        """Rescale [x, y, w, h] bboxes by scale factors."""
        if bboxes.numel() == 0:
            return bboxes
        scaled = bboxes.clone()
        scaled[:, 0] *= sx
        scaled[:, 1] *= sy
        scaled[:, 2] *= sx
        scaled[:, 3] *= sy
        return scaled

    @staticmethod
    def _load_json(path):
        with open(path, "r") as f:
            return json.load(f)


def get_all_subsets(datapath):
    """Return sorted list of valid subset directory names."""
    subsets = []
    for name in sorted(os.listdir(datapath)):
        sub_dir = os.path.join(datapath, name)
        if os.path.isdir(sub_dir) and os.path.isfile(
            os.path.join(sub_dir, "train", "_annotations.coco.json")
        ):
            subsets.append(name)
    return subsets


if __name__ == "__main__":
    datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    subsets = get_all_subsets(datapath)
    print(f"Found {len(subsets)} subsets\n")

    # Test on first subset
    subset = subsets[0]
    for split in ["valid", "test"]:
        ds = DatasetRF20VL(datapath, subset, query_split=split)
        print(f"Subset: {subset}")
        print(f"  Split: {split}, Length: {len(ds)}")
        print(f"  Categories: {ds.categories}")

        if len(ds) > 0:
            sample = ds[0]
            print(f"  First item: class_id={sample['class_id']}, category={sample['category']}")
            print(f"    query_bboxes shape: {sample['query_bboxes'].shape}")
            print(f"    num_support: {len(sample['support_imgs'])}")

            last = ds[-1]
            print(f"  Last item: class_id={last['class_id']}, category={last['category']}")
        print()
