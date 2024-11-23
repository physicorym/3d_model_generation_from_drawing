
import numpy as np
import plotly.graph_objects as go


def show_voxel_mesh(binary_voxel_mesh_array, n_points_to_show=5*10**3, plot_width=600, plot_height=400):
    coords = np.where(binary_voxel_mesh_array)
    ids = np.random.choice(np.arange(0, len(coords[0])), n_points_to_show)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=coords[0][ids],
        y=coords[1][ids],
        z=coords[2][ids],
        mode='markers',
        marker=dict(
            size=5,
            symbol='square',
            color='rgba(0, 0, 255, 0.5)',
        )
    )])
    fig.update_layout(
        autosize=False,
        width=plot_width, height=plot_height,
        scene = dict(
            xaxis = dict(nticks=4, range=[0, 100],),
            yaxis = dict(nticks=4, range=[0, 100],),
            zaxis = dict(nticks=4, range=[0, 100],)
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )
    fig.show()
