from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
            nn.MaxPool2d(conv_pool_size),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=frequency_conv_kernel_size,
                padding="same"
            ),
            nn.ELU(),
            nn.Dropout(conv_dropout_rate),
            nn.MaxPool2d(conv_pool_size),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding="same"
            ),
            nn.ELU(),
            nn.Dropout(conv_dropout_rate),
            nn.MaxPool2d(conv_pool_size),
        )

    def forward(self, x):
        return self.net(x)


class TCN(nn.Module):
    INPUT_CHANNELS: int = 3
    CONV_FILTERS: int = 20
    CONV_KERNEL_SIZE: Tuple[int, int] = (5, 5)
    CONV_DROPOUT_RATE: float = 0.15
    CONV_POOL_SIZE: Tuple[int, int] = (5, 1)
    FREQUENCY_CONV_KERNEL_SIZE: Tuple[int, int] = (10, 1)
    TCN_LAYERS: int = 2
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

        self.frontend = TCNFrontend(
            in_channels=input_channels,
            out_channels=conv_filters,
            conv_kernel_size=conv_kernel_size,
            conv_dropout_rate=conv_dropout_rate,
            conv_pool_size=conv_pool_size,
            frequency_conv_kernel_size=frequency_conv_kernel_size,
        )

        self.sslm_near_frontend = TCNFrontend(
            in_channels=1,
            out_channels=conv_filters,
            conv_kernel_size=conv_kernel_size,
            conv_dropout_rate=conv_dropout_rate,
            conv_pool_size=conv_pool_size,
            frequency_conv_kernel_size=frequency_conv_kernel_size,
        )

        self.sslm_far_frontend = TCNFrontend(
            in_channels=1,
            out_channels=conv_filters,
            conv_kernel_size=conv_kernel_size,
            conv_dropout_rate=conv_dropout_rate,
            conv_pool_size=conv_pool_size,
            frequency_conv_kernel_size=frequency_conv_kernel_size,
        )
        
        frontend_out_channels = conv_filters * 3  # 3 channels: spectrogram + 2 SSLM
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

    def forward(self, x: torch.Tensor, sslm_near: Optional[torch.Tensor], sslm_far: Optional[torch.Tensor]) -> TCNOutput:
        N, C, F, T = x.shape
        x = self.frontend(x)    # (1, 20, 1, T)

        if sslm_near is not None:
            sslm_near = self.sslm_near_frontend(sslm_near)
            x = torch.cat((x, sslm_near), dim=1)
        if sslm_far is not None:
            sslm_far = self.sslm_far_frontend(sslm_far)
            x = torch.cat((x, sslm_far), dim=1)

        x = self.frontend_projection(x)

        for layer in self.tcn_layers:
            x, _ = layer(x)

        x = x.squeeze(-2).permute(0, 2, 1)  # (N, C, 1, T) -> (N, T, C)
        beat_output = self.beat_output(x).permute(0, 2, 1).squeeze(-2)
        downbeat_output = self.downbeat_output(x).permute(0, 2, 1).squeeze(-2)
        segment_output = self.segment_boundary_output(x).permute(0, 2, 1).squeeze(-2)
        function_outputs = self.segment_function_output(x).permute(0, 2, 1)

        return TCNOutput(
            beat_output=beat_output,
            downbeat_output=downbeat_output,
            segment_output=segment_output,
            function_outputs=function_outputs
        )

    def compute_predictions(
        self,
        outputs: TCNOutput,
        ticks_per_beat: int,
        boundary_threshold: float = 0.0,
        local_maxima_filter_size: int = 97,  # 4 * 24 + 1
        window_past_beats: float = 12.0,
        window_future_beats: float = 12.0
    ) -> Dict[str, np.ndarray]:
        """
        Compute segment boundaries and labels from model outputs.

        Args:
            outputs: TCNOutput object containing model predictions
            ticks_per_beat: MIDI ticks per beat resolution (e.g., 4, 48)
            boundary_threshold: Threshold for boundary detection (default: 0.0)
            local_maxima_filter_size: Filter size for local maxima detection (default: 97)
            window_past_beats: Past window size in beats for peak picking (default: 12.0)
            window_future_beats: Future window size in beats for peak picking (default: 12.0)

        Returns:
            Dictionary containing:
                - 'boundaries': Array of boundary times in ticks
                - 'segments': Array of (start, end) tuples in ticks
                - 'labels': Array of predicted segment function indices
                - 'label_probs': Array of label probabilities for each segment
        """
        # Process segment boundaries
        segment_prob = torch.sigmoid(outputs.segment_output.squeeze())

        # Apply local maxima
        prob_sections, _ = self._local_maxima(segment_prob, filter_size=local_maxima_filter_size)

        # Peak picking
        boundary_candidates = self._peak_picking(
            prob_sections.detach().cpu().numpy(),
            window_past=int(window_past_beats * ticks_per_beat),
            window_future=int(window_future_beats * ticks_per_beat)
        )
        boundary = boundary_candidates > boundary_threshold

        # Convert to ticks
        duration_ticks = len(prob_sections)
        # pred_boundary_ticks = self._event_frames_to_ticks(boundary, ticks_per_frame=1)
        pred_boundary_ticks = np.where(boundary)[0]

        # Add start and end if necessary
        if len(pred_boundary_ticks) == 0 or pred_boundary_ticks[0] != 0:
            pred_boundary_ticks = np.insert(pred_boundary_ticks, 0, 0)

        if pred_boundary_ticks[-1] != duration_ticks - 1:
            pred_boundary_ticks = np.append(pred_boundary_ticks, duration_ticks - 1)

        # Create segments
        pred_segments = np.stack([pred_boundary_ticks[:-1], pred_boundary_ticks[1:]]).T

        # Predict labels for each segment
        pred_boundary_indices = np.flatnonzero(boundary)
        # Remove first boundary at 0 to avoid empty first segment
        if len(pred_boundary_indices) > 0 and pred_boundary_indices[0] == 0:
            pred_boundary_indices = pred_boundary_indices[1:]
        # Remove last boundary if it's at or past the end to avoid empty last segment
        if len(pred_boundary_indices) > 0 and pred_boundary_indices[-1] >= duration_ticks - 1:
            pred_boundary_indices = pred_boundary_indices[:-1]

        # Split function probabilities by segment boundaries
        function_probs = torch.softmax(outputs.function_outputs, dim=1).detach().cpu().numpy()

        if len(pred_boundary_indices) > 0:
            prob_segment_function = np.split(function_probs, pred_boundary_indices, axis=-1)
        else:
            prob_segment_function = [function_probs]

        # Calculate mean probability for each segment and get labels
        pred_labels = []
        pred_label_probs = []
        for p in prob_segment_function:
            if p.size > 0:
                mean_probs = p.mean(axis=-1).squeeze()
                pred_labels.append(mean_probs.argmax())
                pred_label_probs.append(mean_probs)
            else:
                # Handle empty segments
                pred_labels.append(0)
                pred_label_probs.append(np.zeros(outputs.function_outputs.shape[1]))

        return {
            'boundaries': pred_boundary_ticks,
            'segments': pred_segments,
            'labels': np.array(pred_labels),
            'label_probs': np.array(pred_label_probs)
        }

    @staticmethod
    def _local_maxima(tensor, filter_size=41):
        """Find local maxima in a tensor."""
        assert len(tensor.shape) in (1, 2), 'Input tensor should have 1 or 2 dimensions'
        assert filter_size % 2 == 1, 'Filter size should be an odd number'

        original_shape = tensor.shape
        if len(original_shape) == 1:
            tensor = tensor.unsqueeze(0)

        # Pad the input array with the minimum value
        padding = filter_size // 2
        padded_arr = torch.nn.functional.pad(tensor, (padding, padding), mode='constant', value=-torch.inf)

        # Create a rolling window view of the padded array
        rolling_view = padded_arr.unfold(1, filter_size, 1)

        # Find the indices of the local maxima
        center = filter_size // 2
        local_maxima_mask = torch.eq(rolling_view[:, :, center], torch.max(rolling_view, dim=-1).values)
        local_maxima_indices = local_maxima_mask.nonzero()

        # Initialize a new PyTorch tensor with zeros and the same shape as the input tensor
        output_arr = torch.zeros_like(tensor)

        # Set the local maxima values in the output tensor
        output_arr[local_maxima_mask] = tensor[local_maxima_mask]

        output_arr = output_arr.reshape(original_shape)

        return output_arr, local_maxima_indices

    @staticmethod
    def _peak_picking(boundary_activation, window_past=12, window_future=6):
        """Peak picking algorithm for boundary detection."""
        # Find local maxima using a sliding window
        window_size = window_past + window_future
        assert window_size % 2 == 0, 'window_past + window_future must be even'
        window_size += 1

        # Pad boundary_activation
        boundary_activation_padded = np.pad(boundary_activation, (window_past, window_future), mode='constant')
        max_filter = sliding_window_view(boundary_activation_padded, window_size)
        local_maxima = (boundary_activation == np.max(max_filter, axis=-1)) & (boundary_activation > 0)

        # Compute strength values by subtracting the mean of the past and future windows
        past_window_filter = sliding_window_view(boundary_activation_padded[:-(window_future + 1)], window_past)
        future_window_filter = sliding_window_view(boundary_activation_padded[window_past + 1:], window_future)
        past_mean = np.mean(past_window_filter, axis=-1)
        future_mean = np.mean(future_window_filter, axis=-1)
        strength_values = boundary_activation - ((past_mean + future_mean) / 2)

        # Get boundary candidates and their corresponding strength values
        boundary_candidates = np.flatnonzero(local_maxima)
        strength_values = strength_values[boundary_candidates]

        strength_activations = np.zeros_like(boundary_activation)
        strength_activations[boundary_candidates] = strength_values

        return strength_activations
    