import timm
from .abstract_models import ImageEncoder


class EfficientNetB0(ImageEncoder):
    def __init__(self, in_channels: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, in_chans=in_channels)

    def forward(self, x):
        return self.model(x)

