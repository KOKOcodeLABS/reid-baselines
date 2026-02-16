from .base import BaseReIDDataset
from .registry import register_dataset


@register_dataset("duke")
class DukeMTMC(BaseReIDDataset):
    def _parse_camid(self, filename):
        # Example: 0002_c1_f0044158.jpg
        cam_part = filename.split("_")[1]  # c1
        camid = int(cam_part[1])
        return camid