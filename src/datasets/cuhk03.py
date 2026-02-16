from .base import BaseReIDDataset
from .registry import register_dataset


@register_dataset("cuhk03")
class CUHK03(BaseReIDDataset):
    def _parse_camid(self, filename):
        # Example: 1_001_1_01.png
        parts = filename.split("_")
        camid = int(parts[2])
        return camid
