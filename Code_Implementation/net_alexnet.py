import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Any
import algorithm_wrap

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(
            self,
            bits: int,
            fun_flag: int,
            gpuid: int,
            quant_mode: str,
            net_mode: str,
            reserve_interval: int,
            alpha: float,
            beta: float,
            csv_path: str,
            csv_name: str,
            num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self._bits = bits
        self._fun_flag = fun_flag
        self._gpuid = gpuid
        self._quant_mode = quant_mode
        self._net_mode = net_mode
        self._reserve_interval = reserve_interval
        self._alpha = alpha
        self._beta = beta
        self._csv_path = csv_path
        self._csv_name = csv_name
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features[0:6](x)

        net_structure = []
        net_structure.append(self.features[6:])
        net_structure.append(self.avgpool)
        net_structure.append(torch.flatten)
        net_structure.append(self.classifier)

        dorefa_clip_value = 0.9

        output = algorithm_wrap.channel_wise_bit_allocation_algorithm(
            self._fun_flag, self._bits, self._gpuid,
            x, net_structure, self._quant_mode, self._net_mode,
            self._reserve_interval, self._alpha, self._beta,
            self._csv_path, self._csv_name,
            dorefa_clip_value
        )

        return output[0]


def alexnet(bits, fun_flag, gpuid, quant_mode, net_mode, reserve_interval, alpha, beta, csv_path, csv_name, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(bits, fun_flag, gpuid, quant_mode, net_mode, reserve_interval, alpha, beta, csv_path, csv_name, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
