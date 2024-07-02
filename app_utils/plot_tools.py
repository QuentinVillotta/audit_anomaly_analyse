import io
import numpy as np

# Function to save plots as PNG and return bytes
def save_plot_as_png(fig, format="eps", dpi=1000):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    return buf


def freedman_diaconis_rule(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1/3)
    bins = int((data.max() - data.min()) / bin_width)
    return bins
