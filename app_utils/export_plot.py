import io

# Function to save plots as PNG and return bytes
def save_plot_as_png(fig, format="eps", dpi=1000):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    return buf