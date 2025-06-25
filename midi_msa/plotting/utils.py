import matplotlib.pyplot as plt


def plot_piano_roll(piano_roll, start_tick, num_ticks, label=None, markers_ticks=None):
    if markers_ticks is None:
        markers_ticks = []
    plt.figure(figsize=(12, 6))
    # 3 plots, one for the main notes, one for the overtones, and one for the drums

    x_min = start_tick
    x_max = start_tick + num_ticks
    y_min = 0
    y_max = piano_roll.shape[-2]
    extent = (x_min, x_max, y_min, y_max)

    plt.subplot(3, 1, 1)
    for marker in markers_ticks:
        if marker >= x_min and marker <= x_max:
            plt.axvline(marker, color="red", linestyle="--")
    plt.imshow(piano_roll[0, :, start_tick:start_tick + num_ticks], aspect="auto", origin="lower", cmap="viridis", extent=extent)

    plt.subplot(3, 1, 2)
    plt.imshow(piano_roll[1, :, start_tick:start_tick + num_ticks], aspect="auto", origin="lower", cmap="viridis", extent=extent)

    plt.subplot(3, 1, 3)
    plt.imshow(piano_roll[2, :, start_tick:start_tick + num_ticks], aspect="auto", origin="lower", cmap="viridis", extent=extent)

    plt.xlabel("Time (ticks)")
    plt.ylabel("Note")
    plt.title(label)