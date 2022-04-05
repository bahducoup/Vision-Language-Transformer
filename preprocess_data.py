from __future__ import annotations
from collections import namedtuple
from h5py import File, Dataset, string_dtype
from typing import Dict, List

import numpy as np

ImageFeature = namedtuple("ImageFeature", "last second_to_last third_to_last id idx")


class PreprocessedDataset:
    """id is the image name"""

    def __init__(
        self,
        f: File,
        last: Dataset,
        second_to_last: Dataset,
        third_to_last: Dataset,
        ids: Dataset,
    ) -> None:
        self._f = f
        mode = f.mode
        self._last = last
        self._second_to_last = second_to_last
        self._third_to_last = third_to_last
        self._ids = ids.asstr() if mode == "r" else ids
        self._id_lookup = {}
        if mode == "r":
            self._id_lookup = {id: i for i, id in enumerate(self._ids)}

    def __del__(self):
        self._f.close()

    def write_item(
        self,
        id: str,
        last: np.ndarray,
        second_to_last: np.ndarray,
        third_to_last: np.ndarray,
    ):
        """one item at a time!"""
        if id not in self._id_lookup:
            idx = len(self._id_lookup)
            self._id_lookup[id] = idx
            self._ids[idx] = id
            self._last[idx] = last
            self._second_to_last[idx] = second_to_last
            self._third_to_last[idx] = third_to_last
    
    def get_item_by_idx(self, idx: int):
        return ImageFeature(self._last[idx], self._second_to_last[idx], self._third_to_last[idx], self._ids[idx], idx)

    def get_item(self, id: str):
        if id not in self._id_lookup:
            raise ValueError("id {id} doesn't exist")
        idx = self._id_lookup[id]
        return self.get_item_by_idx(idx)
    
    @property
    def id_lookup(self):
        return self._id_lookup
    
    def print_vitals(self):
        print(f"unique images #: {len(self._id_lookup)}")
        print(f"file mode: {self._f.mode}")

    @property
    def file_attrs(self):
        return self._f.attrs

    @classmethod
    def dataset_names(cls) -> List[str]:
        return ["last", "second_to_last", "third_to_last", "ids"]

    @classmethod
    def create_new(cls, file_path: str, sample_num: int) -> PreprocessedDataset:
        """raises exception if file already exists"""
        f = File(file_path, "w")
        datasets = []
        dims = [
            (sample_num, 13, 13, 1024),
            (sample_num, 26, 26, 512),
            (sample_num, 52, 52, 256),
        ]
        for i, name in enumerate(cls.dataset_names()[:3]):
            datasets.append(f.create_dataset(name, dims[i]))
        datasets.append(f.create_dataset("ids", (sample_num,), dtype=string_dtype()))
        return cls(
            f,
            *datasets
        )

    @classmethod
    def read_from_h5_file(cls, file_path: str, mode="r") -> PreprocessedDataset:
        f = File(file_path, mode)
        names = cls.dataset_names()
        return cls(f, *[f[n] for n in names])
    
    def compress(self):
        size= len(self.id_lookup)
        compressed_data = self.create_new("compressed.hdf5", size)
        compressed_data._id_lookup = self._id_lookup.copy()
        compressed_data._last[:] = self._last[:size]
        compressed_data._second_to_last[:] = self._second_to_last[:size]
        compressed_data._third_to_last[:] = self._third_to_last[:size]
        return compressed_data

if __name__ == "__main__":
    data = PreprocessedDataset.read_from_h5_file("data/val_darknet.hdf5")
    print(len(data._ids))
    print(data._ids[:10])
    print(len(np.unique(data._ids)))
