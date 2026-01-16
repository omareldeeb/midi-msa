#!/usr/bin/env python3
"""
Interactive visualization script for TCN Dataset.
Shows piano rolls, segment boundaries, and labels.
Use arrow keys or buttons to navigate through the dataset.
"""

import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import torch

from midi_msa.data.tcn_dataset import TCNMidiDataset
from midi_msa.data.label_preprocessor import LABEL_MAP
from midi_msa.models.tcn import TCN


class DatasetVisualizer:
    def __init__(self, dataset, target_ticks_per_beat=4, model=None, device='cpu', segment_function_vocab=None):
        self.dataset = dataset
        self.target_ticks_per_beat = target_ticks_per_beat
        self.current_idx = 0
        self.info_text_obj = None  # Store reference to info text for cleanup
        self.model = model
        self.segment_function_vocab = segment_function_vocab
        self.device = device

        # Set model to eval mode if provided
        if self.model is not None:
            self.model.eval()
            self.model.to(device)

        # Create figure with subplots
        self.fig = plt.figure(figsize=(18, 6), dpi=150)
        self.fig.suptitle('TCN Dataset Visualizer', fontsize=14, fontweight='bold')

        # Create grid spec for better layout (removed segment labels subplot)
        gs = self.fig.add_gridspec(6, 1, height_ratios=[3, 3, 3, 1, 2, 0.3], hspace=0.3)
        # gs = self.fig.add_gridspec(1, 1, height_ratios=[1], hspace=0.3)

        # Main piano roll plot
        self.ax_piano = self.fig.add_subplot(gs[0])
        self.ax_piano.set_title('Piano Roll with Segment Labels')
        self.ax_piano.set_ylabel('MIDI Pitch')

        self.ax_sslm_near = self.fig.add_subplot(gs[1])
        self.ax_sslm_near.set_title('SSLM Near Features')
        self.ax_sslm_near.set_ylabel('Feature Dimension')

        self.ax_sslm_far = self.fig.add_subplot(gs[2])
        self.ax_sslm_far.set_title('SSLM Far Features')
        self.ax_sslm_far.set_ylabel('Feature Dimension')

        # Segment boundaries plot
        self.ax_boundaries = self.fig.add_subplot(gs[3], sharex=self.ax_piano)
        self.ax_boundaries.set_title('Segment Boundaries (Activation)')
        self.ax_boundaries.set_ylabel('Activation')
        self.ax_boundaries.set_ylim(-0.1, 1.1)
        self.ax_boundaries.set_xlabel('Time (ticks)')

        # Functional label activations plot
        self.ax_functions = self.fig.add_subplot(gs[4], sharex=self.ax_piano)
        self.ax_functions.set_title('Functional Label Activations')
        self.ax_functions.set_ylabel('Label')
        self.ax_functions.set_xlabel('Time (ticks)')

        # Navigation buttons
        self.ax_buttons = self.fig.add_subplot(gs[5])
        self.ax_buttons.axis('off')

        # Create button axes
        button_width = 0.1
        button_height = 0.05
        button_y = 0.01

        ax_prev = plt.axes([0.3, button_y, button_width, button_height])
        ax_next = plt.axes([0.6, button_y, button_width, button_height])

        self.btn_prev = Button(ax_prev, 'Previous (←)')
        self.btn_next = Button(ax_next, 'Next (→)')

        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)

        # Keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Display first sample
        self.update_plot()

        plt.show()

    def on_key_press(self, event):
        """Handle keyboard navigation."""
        if event.key == 'left':
            self.prev_sample(None)
        elif event.key == 'right':
            self.next_sample(None)

    def prev_sample(self, event):
        """Navigate to previous sample."""
        self.current_idx = (self.current_idx - 1) % len(self.dataset)
        self.update_plot()

    def next_sample(self, event):
        """Navigate to next sample."""
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        self.update_plot()

    def update_plot(self):
        """Update all plots with current sample."""
        # Get sample from dataset
        sample = self.dataset[self.current_idx]
        file_id = self.dataset.midi_file_ids[self.current_idx]

        # Extract data
        piano_roll = sample["piano_roll"].numpy()  # Shape: (3, 128, time_frames)
        sslm_near = sample.get("sslm_near", None)
        sslm_far = sample.get("sslm_far", None)
        segment_activation = sample.get("segment_activation", None)
        segment_labels = sample.get("segment_label_activations", None)
        measure_ticks = sample.get("measure_ticks", None)

        # Merge piano roll channels for visualization (max across channels)
        piano_roll_merged = piano_roll.max(axis=0)  # Shape: (128, time_frames)

        time_frames = piano_roll_merged.shape[1]

        # Clear previous plots
        self.ax_piano.clear()
        self.ax_boundaries.clear()
        self.ax_functions.clear()

        # Plot 1: Piano Roll with segment labels as rectangles

        # First, draw segment label rectangles at the top
        if segment_labels is not None:
            num_classes = len(self.dataset.segment_function_vocab)
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

            # Draw colored rectangles for each segment at the top (MIDI notes 115-128)
            segment_height_start = 115
            segment_height_end = 128

            for class_idx in range(num_classes):
                mask = segment_labels == class_idx
                if mask.any():
                    regions = self._find_contiguous_regions(mask)
                    for start, end in regions:
                        # Draw rectangle
                        from matplotlib.patches import Rectangle
                        rect = Rectangle(
                            (start, segment_height_start),
                            end - start,
                            segment_height_end - segment_height_start,
                            facecolor=colors[class_idx],
                            edgecolor='white',
                            linewidth=1,
                            alpha=0.7,
                            zorder=10
                        )
                        self.ax_piano.add_patch(rect)

                        # Add text label in the middle of the segment
                        segment_center = (start + end) / 2
                        segment_label = self.dataset.segment_function_vocab[class_idx]

                        # Only add text if segment is wide enough
                        segment_width = end - start
                        if segment_width > time_frames * 0.02:  # At least 2% of total width
                            self.ax_piano.text(
                                segment_center,
                                (segment_height_start + segment_height_end) / 2,
                                segment_label,
                                ha='center',
                                va='center',
                                fontsize=8,
                                fontweight='bold',
                                color='white',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5),
                                zorder=11
                            )

        # Now plot the piano roll
        self.ax_piano.imshow(
            piano_roll_merged,
            aspect='auto',
            origin='lower',
            cmap='magma',
            interpolation='nearest',
            extent=[0, time_frames, 0, 128]
        )
        self.ax_piano.set_ylabel('MIDI Pitch')
        self.ax_piano.set_title(f'Piano Roll with Segment Labels - Sample {self.current_idx}/{len(self.dataset)-1}: {file_id}')

        # Plot SSLM features if available
        self.ax_sslm_near.clear()
        if sslm_near is not None:
            sslm_near_np = sslm_near[0].numpy().T  # Shape: (time_frames, feature_dim)
            self.ax_sslm_near.imshow(
                sslm_near_np.T,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='nearest',
                extent=[0, sslm_near_np.shape[0], 0, sslm_near_np.shape[1]]
            )
            self.ax_sslm_near.set_ylabel('Feature Dimension')
        else:
            self.ax_sslm_near.text(0.5, 0.5, 'No SSLM Near Features', ha='center', va='center', transform=self.ax_sslm_near.transAxes)
        
        self.ax_sslm_far.clear()
        if sslm_far is not None:
            sslm_far_np = sslm_far[0].numpy().T  # Shape: (time_frames, feature_dim)
            self.ax_sslm_far.imshow(
                sslm_far_np.T,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='nearest',
                extent=[0, sslm_far_np.shape[0], 0, sslm_far_np.shape[1]]
            )
            self.ax_sslm_far.set_ylabel('Feature Dimension')
        else:
            self.ax_sslm_far.text(0.5, 0.5, 'No SSLM Far Features', ha='center', va='center', transform=self.ax_sslm_far.transAxes)

        # Add measure lines if available
        if measure_ticks is not None:
            for tick in measure_ticks:
                self.ax_piano.axvline(x=tick, color='cyan', alpha=0.3, linestyle='--', linewidth=0.5)

        # Plot segment boundaries on piano roll
        boundary_positions = []
        if segment_activation is not None:
            boundary_positions = np.where(segment_activation > 0.5)[0]
            for pos in boundary_positions:
                self.ax_piano.axvline(x=pos, color='white', alpha=0.8, linestyle='-', linewidth=2)

        # Plot 2: Segment Boundaries (Activation)
        if segment_activation is not None:
            self.ax_boundaries.plot(
                range(time_frames),
                segment_activation,
                color='red',
                linewidth=1.5,
                label='Ground Truth Boundaries'
            )
            self.ax_boundaries.fill_between(
                range(time_frames),
                segment_activation,
                alpha=0.3,
                color='red'
            )

            # Mark exact boundary positions
            for pos in boundary_positions:
                self.ax_boundaries.axvline(x=pos, color='red', alpha=0.5, linestyle='--', linewidth=1)

        # Compute and plot model predictions if model is provided
        if self.model is not None:
            with torch.no_grad():
                # Get piano roll input and move to device
                piano_roll_input = sample["piano_roll"].unsqueeze(0).to(torch.float32).to(self.device)  # Add batch dimension
                sslm_near_input = sample.get("sslm_near", None)
                sslm_far_input = sample.get("sslm_far", None)
                if sslm_near_input is not None:
                    sslm_near_input = sslm_near_input.unsqueeze(0).to(torch.float32).to(self.device)
                if sslm_far_input is not None:
                    sslm_far_input = sslm_far_input.unsqueeze(0).to(torch.float32).to(self.device)

                # Get model predictions
                output = self.model(piano_roll_input, sslm_near=sslm_near_input, sslm_far=sslm_far_input)

                # Store output for later use in functional label visualization
                self.current_output = output

                predicted_boundary_ticks, predicted_label_indices = self.model.compute_predictions_for_visualization(output=output, measure_ticks=measure_ticks)
                print(f"Predicted labels: {[self.segment_function_vocab[idx] for idx in predicted_label_indices]}")

                # Get segment boundary predictions and apply sigmoid
                segment_pred = torch.sigmoid(output.segment_output).squeeze(0).cpu().numpy()

                # Plot predictions
                self.ax_boundaries.plot(
                    range(len(segment_pred)),
                    segment_pred,
                    color='blue',
                    linewidth=1.5,
                    label='Model Predictions',
                    linestyle='--'
                )

                # Mark predicted boundaries (threshold at 0.5)
                pred_boundary_positions = np.where(segment_pred > 0.5)[0]
                for pos in pred_boundary_positions:
                    self.ax_boundaries.axvline(x=pos, color='blue', alpha=0.3, linestyle=':', linewidth=1)
                    # Also mark on piano roll
                    self.ax_piano.axvline(x=pos, color='cyan', alpha=0.6, linestyle=':', linewidth=2)

        self.ax_boundaries.set_ylabel('Activation')
        self.ax_boundaries.set_ylim(-0.1, 1.1)
        self.ax_boundaries.set_xlabel(f'Time (ticks @ {self.target_ticks_per_beat} ticks/beat)')
        self.ax_boundaries.grid(True, alpha=0.3)
        self.ax_boundaries.legend(loc='upper right')

        # Plot 3: Functional Label Activations
        if self.segment_function_vocab is not None and self.model is not None and hasattr(self, 'current_output'):
            num_classes = len(self.segment_function_vocab)

            # Get function outputs and apply softmax to get probabilities
            function_probs = torch.softmax(self.current_output.function_outputs, dim=1).squeeze(0).cpu().numpy()
            # Shape: (num_classes, time_frames)

            # Generate colors for each functional label
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

            # Plot each functional label as a separate curve
            for class_idx in range(num_classes):
                label_name = self.segment_function_vocab[class_idx]
                probs = function_probs[class_idx, :]

                self.ax_functions.plot(
                    range(time_frames),
                    probs,
                    color=colors[class_idx],
                    linewidth=1.5,
                    label=label_name,
                    alpha=0.8
                )

            # Set axis properties
            self.ax_functions.set_ylabel('Probability')
            self.ax_functions.set_xlabel(f'Time (ticks @ {self.target_ticks_per_beat} ticks/beat)')
            self.ax_functions.set_ylim(-0.05, 1.05)
            self.ax_functions.grid(True, alpha=0.3)
            self.ax_functions.legend(loc='upper right', fontsize=8, ncol=2)
        else:
            # If no model predictions, just show a message
            self.ax_functions.text(
                0.5, 0.5,
                'No Functional Label Predictions\n(Model required)',
                ha='center',
                va='center',
                transform=self.ax_functions.transAxes,
                fontsize=10
            )
            self.ax_functions.set_ylabel('Probability')
            self.ax_functions.set_xlabel(f'Time (ticks @ {self.target_ticks_per_beat} ticks/beat)')

        # Remove old info text if it exists
        if self.info_text_obj is not None:
            self.info_text_obj.remove()

        # Add info text
        info_text = f"Sample {self.current_idx + 1}/{len(self.dataset)}\n"
        info_text += f"File: {file_id}\n"
        info_text += f"Duration: {time_frames} ticks ({time_frames / self.target_ticks_per_beat:.1f} quarter notes)\n"
        info_text += f"Piano roll shape: {piano_roll.shape}\n"

        if segment_activation is not None:
            num_boundaries = len(boundary_positions)
            info_text += f"Number of segments: {num_boundaries}\n"

        # self.info_text_obj = self.fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                                        #   verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Update canvas
        self.fig.canvas.draw()

    def _find_contiguous_regions(self, mask):
        """Find contiguous True regions in boolean mask."""
        regions = []
        in_region = False
        start = 0

        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i))
                in_region = False

        # Close final region if needed
        if in_region:
            regions.append((start, len(mask)))

        return regions


def main():
    parser = argparse.ArgumentParser(description='Visualize TCN Dataset')
    parser.add_argument('--midi-dir', type=str, required=True,
                       help='Directory containing MIDI files')
    parser.add_argument('--sslm-dir', type=str, default=None,
                       help='Directory containing SSLM files (if using SSLM features)')
    parser.add_argument('--annotation-dir', type=str, required=True,
                       help='Directory containing annotation JSON files')
    parser.add_argument('--midi-files', type=str, nargs='+', default=None,
                       help='Specific MIDI file IDs to visualize (optional)')
    parser.add_argument('--split-file', type=str, default=None,
                       help='JSON file defining dataset splits. Will load validation files only.')
    parser.add_argument('--target-ticks-per-beat', type=int, default=4,
                       help='Target ticks per beat (default: 4)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of samples to load (for faster startup)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint for visualization (optional)')

    args = parser.parse_args()

    # Determine which MIDI files to load
    midi_files = args.midi_files
    if args.split_file:
        print(f"Loading validation files from split: {args.split_file}")
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
            midi_files = splits.get('val', [])
            print(f"Found {len(midi_files)} validation files")

    # Create dataset with same parameters as training
    print("Loading TCN Dataset...")
    label_map = list(set(LABEL_MAP.values()))
    label_map.sort()
    dataset = TCNMidiDataset(
        midi_dir=args.midi_dir,
        sslms_dir=args.sslm_dir,
        annotation_dir=args.annotation_dir,
        midi_files=midi_files,
        target_ticks_per_beat=args.target_ticks_per_beat,
        segment_function_vocab=label_map,
        compute_beats=False,
        compute_downbeats=False,
        compute_segments=True,
        instrument_overtones=True,
        separate_drums=True
    )

    # Limit samples if requested
    if args.max_samples and args.max_samples < len(dataset):
        print(f"Limiting to first {args.max_samples} samples")
        dataset.midi_file_ids = dataset.midi_file_ids[:args.max_samples]

    print(f"Loaded {len(dataset)} samples")
    print(f"Segment vocabulary ({len(dataset.segment_function_vocab)} classes):")
    for i, func in enumerate(dataset.segment_function_vocab):
        print(f"  {i}: {func}")

    # Load model if checkpoint is provided
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.checkpoint:
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        print(f"Using device: {device}")

        # Ensure we have a segment function vocab
        if dataset.segment_function_vocab is None:
            raise ValueError("Dataset must have segment_function_vocab to load model")

        # Create model with same vocab as dataset
        model = TCN(segment_function_vocab=dataset.segment_function_vocab, conv_filters=20, tcn_layers=5, tcn_kernel_size=3)
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        # Extract model state dict from checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}, fold {checkpoint.get('fold', 'unknown')}")
        else:
            # Fallback: assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        print("Model loaded successfully!")

    # Create visualizer
    print("\nStarting interactive visualizer...")
    print("Controls:")
    print("  - Use arrow keys (← →) or buttons to navigate")
    print("  - Close window to exit")
    if model is not None:
        print("  - Blue dashed line shows model predictions")
        print("  - Red line shows ground truth boundaries")

    DatasetVisualizer(dataset, target_ticks_per_beat=args.target_ticks_per_beat,
                      model=model, device=device, segment_function_vocab=dataset.segment_function_vocab)


if __name__ == '__main__':
    main()
