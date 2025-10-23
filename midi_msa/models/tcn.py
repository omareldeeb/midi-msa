from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class TCNOutput:
    beat_output: torch.Tensor
    downbeat_output: torch.Tensor
    segment_output: torch.Tensor
    function_outputs: torch.Tensor

class TCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout_rate: float,
    ):
        super(TCNBlock, self).__init__()

        self.residual_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            padding="same"
        )

        # padding_1 = dilation * (kernel_size - 1) // 2
        self.dc1 = nn.Sequential(
            # nn.ZeroPad2d((padding_1, padding_1, 0, 0)),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, kernel_size),
                dilation=(1, dilation),
                padding="same"
            )
        )

        # padding_2 = (dilation * 2) * (kernel_size - 1) // 2
        self.dc2 = nn.Sequential(
            # nn.ZeroPad2d((padding_2, padding_2, 0, 0)),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, kernel_size),
                dilation=(1, dilation * 2),
                padding="same"
            )
        )

        self.skip_connection = nn.Sequential(
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=(1, 1),
                padding="same"
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.residual_conv(x)

        x1 = self.dc1(x)
        x2 = self.dc2(x)

        x = torch.cat((x1, x2), dim=1)
        x = self.skip_connection(x)

        return res + x, x


class TCN(nn.Module):
    INPUT_CHANNELS: int = 3
    CONV_FILTERS: int = 20
    CONV_KERNEL_SIZE: Tuple[int, int] = (5, 5)
    CONV_DROPOUT_RATE: float = 0.15
    CONV_POOL_SIZE: Tuple[int, int] = (5, 1)
    FREQUENCY_CONV_KERNEL_SIZE: Tuple[int, int] = (10, 1)
    TCN_LAYERS: int = 5
    TCN_KERNEL_SIZE: int = 5

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        conv_filters: int = CONV_FILTERS,
        conv_kernel_size: Tuple[int, int] = CONV_KERNEL_SIZE,
        conv_dropout_rate: float = CONV_DROPOUT_RATE,
        conv_pool_size: Tuple[int, int] = CONV_POOL_SIZE,
        frequency_conv_kernel_size: Tuple[int, int] = FREQUENCY_CONV_KERNEL_SIZE,
        tcn_layers: int = TCN_LAYERS,
        tcn_kernel_size: int = TCN_KERNEL_SIZE,
        segment_function_vocab: List[str] = None,
        **kwargs
    ):
        super(TCN, self).__init__()

        self.frontend = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=conv_filters,
                kernel_size=conv_kernel_size,
                padding="same"
            ),
            nn.ELU(),
            nn.Dropout(conv_dropout_rate),
            nn.MaxPool2d(conv_pool_size),

            # "Moving the “frequency only” convolution in between the two 3 × 3 convolutions as shown in Figure 1,
            # enables the network to better capture harmonic content across a wider frequency range instead of detecting
            # local changes in smaller regions of the spectrogram only and then later aggregating them."
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=frequency_conv_kernel_size,
                padding="same"
            ),
            nn.ELU(),
            nn.Dropout(conv_dropout_rate),
            nn.MaxPool2d(conv_pool_size),

            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=conv_kernel_size,
                padding="same"
            ),
            nn.ELU(),
            nn.Dropout(conv_dropout_rate),
            nn.MaxPool2d(conv_pool_size),
        )

        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_layers):
            self.tcn_layers.append(
                TCNBlock(
                    channels=conv_filters,
                    kernel_size=tcn_kernel_size,
                    dilation=2 ** i,
                    dropout_rate=conv_dropout_rate
                )
            )

        self.beat_output = nn.Sequential(
            nn.Dropout(conv_dropout_rate),
            nn.Linear(conv_filters, 1),
        )
        self.downbeat_output = nn.Sequential(
            nn.Dropout(conv_dropout_rate),
            nn.Linear(conv_filters, 1),
        )
        self.segment_boundary_output = nn.Sequential(
            nn.Dropout(conv_dropout_rate),
            nn.Linear(conv_filters, 1),
        )
        self.segment_function_output = nn.Sequential(
            nn.Dropout(conv_dropout_rate),
            nn.Linear(conv_filters, len(segment_function_vocab)),
        )

        # Init confidences
        self.beat_output[-1].bias.data.fill_(-torch.log(torch.tensor(1 / 0.05 - 1)))
        self.downbeat_output[-1].bias.data.fill_(-torch.log(torch.tensor(1 / 0.0125 - 1)))
        self.segment_boundary_output[-1].bias.data.fill_(-torch.log(torch.tensor(1 / 0.01 - 1)))

    def forward(self, x: torch.Tensor) -> TCNOutput:
        N, C, F, T = x.shape
        x = self.frontend(x)

        for layer in self.tcn_layers:
            x, _ = layer(x)

        x = x.squeeze(-2).permute(0, 2, 1)  # (N, C, 1, T) -> (N, T, C)
        beat_output = None# self.beat_output(x).permute(0, 2, 1).squeeze(-2)
        downbeat_output = None# self.downbeat_output(x).permute(0, 2, 1).squeeze(-2)
        segment_output = self.segment_boundary_output(x).permute(0, 2, 1).squeeze(-2)
        function_outputs = self.segment_function_output(x).permute(0, 2, 1)

        return TCNOutput(
            beat_output=beat_output,
            downbeat_output=downbeat_output,
            segment_output=segment_output,
            function_outputs=function_outputs
        )