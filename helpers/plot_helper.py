"""Plotting helpers."""
import plotly
import plotly.graph_objs as go


def line_plot(data, names=['name'], title='title',
              filename='filename.html'):
    """Create a line plot from an array."""
    traces = []
    # num_plots is -1 the shape because the first column are the
    # x values
    num_plots = data.shape[1] - 1
    for i in range(num_plots):
        trace = go.Scatter(
            x=data[:, 0],
            y=data[:, i + 1],
            mode='lines+markers',
            name=names[i])
        traces.append(trace)

    plotly.offline.plot({
        "data": traces,
        "layout": go.Layout(title=title)
    }, filename=filename, auto_open=False)
