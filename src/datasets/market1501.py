from .base import BaseReIDDataset
from .registry import register_dataset


@register_dataset("market1501")
class Market1501(BaseReIDDataset):
    def _parse_camid(self, filename):
        # Example: 0002_c1s1_000451_03.jpg
        cam_part = filename.split("_")[1]  # c1s1
        camid = int(cam_part[1])  # extract camera number
        return camid