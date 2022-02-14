"""
stac-ml
"""
import importlib
import requests
import io

__version__ = "0.1.0"


def load_model(item, asset_key="model"):
    """
    Load a model using the information in a STAC item.
    """
    import torch
    module_name, cls_name = item.properties["torch:model_class"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    model = cls(**item.properties["torch:model_options"])
    
    r = requests.get(item.assets[asset_key].href)
    sink = io.BytesIO(r.content)
    sink.seek(0)
    model.load_state_dict(torch.load(sink, map_location="cpu"))
    
    return model


# class STACDataset(torch.utils.data.Dataset):
#     def __init__(self, items, transformers=None):
#         self.items = items
#         self.transformers = transformers or []
#         self.vrt = stac_vrt.build_vrt(items.to_dict()["features"], data_type="Byte")
#         self._slices = dask.array.core.slices_from_chunks(self.ds.data.chunks)

#     @property
#     def ds(self):
#         ds = rioxarray.open_rasterio(self.vrt, chunks={"x": 2048, "y": 2048})
#         for transformer in self.transformers:
#             ds = transformer(ds)
#         return ds

#     def __len__(self):
#         return self.ds.data.npartitions
    
#     def __getitem__(self, idx):
#         ds = self.ds[self._slices[idx]]
#         for transformer in self.transformers:
#             ds = transformer(ds)
#         return ds
