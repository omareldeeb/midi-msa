from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TCNOutput:
    beat_output: torch.Tensor
    downbeat_output: torch.Tensor
    segment_output: torch.Tensor
    function_outputs: torch.Tensor
    segment_embeddings: torch.Tensor


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
            # nn.Dropout2d(dropout_rate),  # Do we want Dropout2d here?
            nn.Dropout(dropout_rate),
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


class TCNFrontend(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: Tuple[int, int],
        conv_dropout_rate: float,
        conv_pool_size: Tuple[int, int],
        frequency_conv_kernel_size: Tuple[int, int],
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding="same"
            ),
            nn.ELU(),
            nn.Dropout(conv_dropout_rate),
            # nn.AvgPool2d(conv_pool_size),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(5, 1),
                stride=(5, 1),
                padding=0,
                groups=out_channels
            ),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=frequency_conv_kernel_size,
                padding="same"
            ),
            nn.ELU(),
            nn.Dropout(conv_dropout_rate),
            # nn.AvgPool2d(conv_pool_size),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(5, 1),
                stride=(5, 1),
                padding=0,
                groups=out_channels
            ),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding="same"
            ),
            nn.ELU(),
            nn.Dropout(conv_dropout_rate),
            # nn.AvgPool2d(conv_pool_size),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(5, 1),
                stride=(5, 1),
                padding=0,
                groups=out_channels
            ),
        )

    def forward(self, x):
        return self.net(x)  # (1, conv_filters, time)


class TCN(nn.Module):
    PIANO_ROLL_CHANNELS: int = 3
    CONV_FILTERS: int = 20
    CONV_KERNEL_SIZE: Tuple[int, int] = (5, 5)
    DROPOUT_RATE: float = 0.15
    CONV_POOL_SIZE: Tuple[int, int] = (5, 1)
    FREQUENCY_CONV_KERNEL_SIZE: Tuple[int, int] = (12, 1)
    TCN_LAYERS: int = 2
    TCN_KERNEL_SIZE: int = 5

    def __init__(
        self,
        piano_roll_channels: int = PIANO_ROLL_CHANNELS,
        conv_filters: int = CONV_FILTERS,
        conv_kernel_size: Tuple[int, int] = CONV_KERNEL_SIZE,
        dropout_rate: float = DROPOUT_RATE,
        conv_pool_size: Tuple[int, int] = CONV_POOL_SIZE,
        frequency_conv_kernel_size: Tuple[int, int] = FREQUENCY_CONV_KERNEL_SIZE,
        tcn_layers: int = TCN_LAYERS,
        tcn_kernel_size: int = TCN_KERNEL_SIZE,
        segment_function_vocab: List[str] = None,
        use_sslm_near: bool = True,
        use_sslm_far: bool = True,
        **kwargs
    ):
        super(TCN, self).__init__()

        self.segment_function_vocab = segment_function_vocab
        self.use_sslm_near = use_sslm_near
        self.use_sslm_far = use_sslm_far

        self.piano_roll_frontend = TCNFrontend(
            in_channels=piano_roll_channels,
            out_channels=conv_filters,
            conv_kernel_size=conv_kernel_size,
            conv_dropout_rate=dropout_rate,
            conv_pool_size=conv_pool_size,
            frequency_conv_kernel_size=frequency_conv_kernel_size,
        )

        if self.use_sslm_near:
            self.sslm_near_frontend = TCNFrontend(
                in_channels=1,
                out_channels=conv_filters,
                conv_kernel_size=conv_kernel_size,
                conv_dropout_rate=dropout_rate,
                conv_pool_size=conv_pool_size,
                frequency_conv_kernel_size=frequency_conv_kernel_size,
            )
        if self.use_sslm_far:
             self.sslm_far_frontend = TCNFrontend(
                in_channels=1,
                out_channels=conv_filters,
                conv_kernel_size=conv_kernel_size,
                conv_dropout_rate=dropout_rate,
                conv_pool_size=conv_pool_size,
                frequency_conv_kernel_size=frequency_conv_kernel_size,
            )
        
        frontend_out_channels = conv_filters
        if self.use_sslm_near:
            frontend_out_channels += conv_filters
        if self.use_sslm_far:
            frontend_out_channels += conv_filters

        self.frontend_projection = nn.Conv2d(
            in_channels=frontend_out_channels,
            out_channels=conv_filters,
            kernel_size=(1, 1),
            padding=0
        )

        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_layers):
            self.tcn_layers.append(
                TCNBlock(
                    channels=conv_filters,
                    kernel_size=tcn_kernel_size,
                    dilation=2 ** i,
                    dropout_rate=dropout_rate
                )
            )

        self.beat_output = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters, 1),
        )
        self.downbeat_output = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters, 1),
        )
        self.segment_boundary_output = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters, 1),
        )
        self.segment_function_output = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters, len(segment_function_vocab)),
        )

        self.segment_embedding = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters, 32)
        )

        # Init confidences
        self.beat_output[-1].bias.data.fill_(-torch.log(torch.tensor(4 - 1)))
        self.downbeat_output[-1].bias.data.fill_(-torch.log(torch.tensor(16 - 1)))
        self.segment_boundary_output[-1].bias.data.fill_(-torch.log(torch.tensor(16*12 - 1)))

        class_probs = torch.tensor([
            1 / len(segment_function_vocab) for _ in range(len(segment_function_vocab))], dtype=torch.float32)
        with torch.no_grad():
            self.segment_function_output[-1].bias.copy_(torch.log(class_probs))

    def forward(self, x: torch.Tensor, sslm_near: Optional[torch.Tensor], sslm_far: Optional[torch.Tensor]) -> TCNOutput:
        N, C, F, T = x.shape
        x = self.piano_roll_frontend(x)    # (1, n_filters, 1, T)

        if self.use_sslm_near and sslm_near is not None:
            sslm_near = self.sslm_near_frontend(sslm_near)
            x = torch.cat((x, sslm_near), dim=1)
        if self.use_sslm_far and sslm_far is not None:
            sslm_far = self.sslm_far_frontend(sslm_far)
            x = torch.cat((x, sslm_far), dim=1)

        if self.use_sslm_near or self.use_sslm_far:
            x = self.frontend_projection(x)

        for layer in self.tcn_layers:
            x, _ = layer(x)

        x = x.squeeze(-2).permute(0, 2, 1)  # (N, C, 1, T) -> (N, T, C)
        beat_output = self.beat_output(x).permute(0, 2, 1).squeeze(-2)
        downbeat_output = self.downbeat_output(x).permute(0, 2, 1).squeeze(-2)
        segment_output = self.segment_boundary_output(x).permute(0, 2, 1).squeeze(-2)
        function_outputs = self.segment_function_output(x).permute(0, 2, 1)

        segment_embeddings = self.segment_embedding(x).permute(0, 2, 1)  # (N, 32, T)

        return TCNOutput(
            beat_output=beat_output,
            downbeat_output=downbeat_output,
            segment_output=segment_output,
            function_outputs=function_outputs,
            segment_embeddings=segment_embeddings,
        )

    def compute_predictions_for_visualization(
        self,
        output: TCNOutput,
        measure_ticks: torch.Tensor,
        threshold: float = 0.5,
        function_activation: str = "softmax",
    ) -> Tuple[np.ndarray, np.ndarray]:
        measure_ticks_np = measure_ticks.long().squeeze(0).cpu().numpy()
        pred_boundary_probs = torch.sigmoid(output.segment_output).squeeze(0).cpu().numpy()
        if function_activation == "sigmoid":
            pred_function_probs = torch.sigmoid(output.function_outputs).squeeze(0).cpu().numpy()
        else:
            pred_function_probs = torch.softmax(output.function_outputs, dim=1).squeeze(0).cpu().numpy()

        pred_boundary_ticks = []
        pred_labels = []
        for i, measure_tick in enumerate(measure_ticks_np):
            measure_left = 0
            if i - 1 >= 0:
                measure_left = measure_ticks_np[i - 1]
            window_left = (measure_tick - measure_left) // 2

            measure_right = pred_boundary_probs.shape[-1]
            if i + 1 < len(measure_ticks_np):
                measure_right = measure_ticks_np[i + 1]
            window_right = (measure_right - measure_tick) // 2

            probs = pred_boundary_probs[measure_tick - window_left: measure_tick + window_right]
            if len(probs) > 0:
                max_prob = np.max(probs)
                if max_prob >= threshold:
                    pred_function_probs_window = pred_function_probs[:, measure_left:measure_right]
                    prob_sums = np.sum(pred_function_probs_window, axis=-1)
                    if len(prob_sums) > 0:
                        pred_function_index = prob_sums.argmax()
                        pred_boundary_ticks.append(measure_tick)
                        pred_labels.append(pred_function_index)

        return np.array(pred_boundary_ticks), np.array(pred_labels)

    def compute_predictions(
        self,
        output: TCNOutput,
        measure_ticks: torch.Tensor,
        threshold: float = 0.5,
        function_activation: str = "softmax",
    ) -> Tuple[np.ndarray, np.ndarray]:
        measure_ticks_np = measure_ticks.long().squeeze(0).cpu().numpy()
        pred_boundary_probs = torch.sigmoid(output.segment_output).squeeze(0).cpu().numpy()
        if function_activation == "sigmoid":
            pred_function_probs = torch.sigmoid(output.function_outputs).squeeze(0).cpu().numpy()
        else:
            pred_function_probs = torch.softmax(output.function_outputs, dim=1).squeeze(0).cpu().numpy()
        total_ticks = pred_boundary_probs.shape[-1]

        # First, find all segments
        pred_boundary_ticks = [0]  # start with a segment at tick 0
        for i, measure_tick in enumerate(measure_ticks_np):
            if measure_tick >= total_ticks:
                break
            if measure_tick == 0:
                continue

            measure_left = 0
            if i - 1 >= 0:
                measure_left = measure_ticks_np[i - 1]
            window_left = (measure_tick - measure_left) // 2

            measure_right = pred_boundary_probs.shape[-1]
            if i + 1 < len(measure_ticks_np):
                measure_right = measure_ticks_np[i + 1]
            window_right = (measure_right - measure_tick) // 2

            probs = pred_boundary_probs[measure_tick - window_left: measure_tick + window_right]
            if len(probs) > 0:
                max_prob = np.max(probs)
                if max_prob >= threshold:
                    pred_boundary_ticks.append(measure_tick)

        # Then, label each segment
        pred_labels = []
        for i, t in enumerate(pred_boundary_ticks):
            if i < len(pred_boundary_ticks) - 1:
                next_t = pred_boundary_ticks[i + 1]
            else:
                next_t = total_ticks
            # next_t = min(8, next_t)
            segment_probs = pred_function_probs[:, t: next_t]
            if segment_probs.shape[-1] > 0:
                pred_function_index = np.sum(segment_probs, axis=-1).argmax()
                pred_labels.append(pred_function_index)

        return np.array(pred_boundary_ticks), np.array(pred_labels)
