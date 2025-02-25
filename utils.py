import torch
import enum
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def triangles_to_edges(faces):
    # Collect edges from triangles
    edges = torch.cat([faces[:, 0:2],
                        faces[:, 1:3],
                        torch.stack([faces[:, 2], faces[:, 0]], dim=1)], dim=0)
    # Those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # Sort & pack edges as single torch.int64
    receivers = torch.min(edges, dim=1)[0]
    senders = torch.max(edges, dim=1)[0]
    packed_edges = torch.stack([senders, receivers], dim=1).to(torch.int64)
    # Remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0).to(torch.int32)
    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
    # Create two-way connectivity
    return torch.cat([senders, receivers], dim=0), torch.cat([receivers, senders], dim=0)

def rect_to_edges(faces):
    # Collect edges from triangles
    edges = torch.cat([faces[:, 0:2],
                      faces[:, 1:3],
                      faces[:, 2:4],
                      torch.stack([faces[:, 3], faces[:, 0]], dim=1)],dim=0)

    # Those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # Sort & pack edges as single torch.int64
    receivers = torch.min(edges, dim=1)[0]
    senders = torch.max(edges, dim=1)[0]
    packed_edges = torch.stack([senders, receivers], dim=1).to(torch.int64)
    # Remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0).to(torch.int32)
    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
    # Create two-way connectivity
    return torch.cat([senders, receivers], dim=0), torch.cat([receivers, senders], dim=0)

# This class defines the node type you want to classify
# For example, internal nodes/surface nodes/nodes under loading/fixed nodes, you can adjust this function based on your problem
class NodeType(enum.IntEnum):
    internal = 0
    impact = 1
    SIZE = 2

def normalize(to_normalize, mean_vec, std_vec):
    return (to_normalize - mean_vec) / std_vec

def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize * std_vec + mean_vec

def get_stats(data_list):
    '''
    Method for normalizing processed datasets. Given  the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    '''

    #mean and std of the node features are calculated
    mean_vec_x = torch.zeros(data_list[0]["x"].shape[1:])
    std_vec_x = torch.zeros(data_list[0]["x"].shape[1:])

    #mean and std of the edge features are calculated
    mean_vec_edge = torch.zeros(data_list[0]["edge_attr"].shape[1:])
    std_vec_edge = torch.zeros(data_list[0]["edge_attr"].shape[1:])

    #mean and std of the output parameters are calculated
    mean_vec_y = torch.zeros(data_list[0]["stress"].shape[1:])
    std_vec_y = torch.zeros(data_list[0]["stress"].shape[1:])

    #Define the maximum number of accumulations to perform such that we do
    #not encounter memory issues
    max_accumulations = 1e6

    #Define a very small value for normalizing to
    eps = torch.tensor(1e-8)

    #Define counters used in normalization
    num_accs_x = 0
    num_accs_edge = 0
    num_accs_y = 0

    #Iterate through the data in the list to accumulate statistics
    for dp in data_list:

        mean_vec_x += torch.sum(dp["x"], dim=0)
        std_vec_x += torch.sum(dp["x"]**2, dim=0)
        num_accs_x += dp["x"].shape[0]

        mean_vec_edge += torch.sum(dp["edge_attr"], dim=0)
        std_vec_edge += torch.sum(dp["edge_attr"]**2, dim=0)
        num_accs_edge += dp["edge_attr"].shape[0]

        mean_vec_y += torch.sum(dp["stress"], dim=0)
        std_vec_y += torch.sum(dp["stress"]**2, dim=0)
        num_accs_y += dp["stress"].shape[0]

        if num_accs_x > max_accumulations or num_accs_edge > max_accumulations or num_accs_y > max_accumulations:
            break

    mean_vec_x = mean_vec_x / num_accs_x
    std_x = torch.sqrt(std_vec_x / num_accs_x - mean_vec_x**2)
    std_x = torch.nan_to_num(std_x, nan=0.0)   
    std_vec_x = torch.maximum(std_x, eps)


    mean_vec_edge = mean_vec_edge / num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge / num_accs_edge - mean_vec_edge**2), eps)

    mean_vec_y = mean_vec_y / num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y / num_accs_y - mean_vec_y**2), eps)

    mean_std_list = [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y]

    return mean_std_list
    
def plot_losses(losses, test_losses, epoch, test_interval, save_path):
    f = plt.figure()
    plt.title('Losses Plot')
    plt.plot(losses, label="training loss" )
    x=np.array(list(range(len(test_losses))))*test_interval
    plt.plot(x.tolist(),test_losses, label="test loss")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{save_path}/Losses_{epoch}.png") 
    plt.legend()
    plt.show()
    
def plot_test_r2(test_r2, epoch, test_interval, save_path):
    f = plt.figure()
    plt.title('R2 Plot')
    x=np.array(list(range(len(test_r2))))*test_interval
    plt.plot(x.tolist(),test_r2, label="test R2")
    
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.savefig(f"{save_path}/R2_{epoch}.png") 
    plt.legend()
    plt.show()

def plot_dist(test_distribution, save_path):
    a = test_distribution["pos"][:, 0]
    b = test_distribution["pos"][:, 1]
    c = test_distribution["pos"][:, 2]
    d = test_distribution["faces"][:, 0]
    e = test_distribution["faces"][:, 1]
    f = test_distribution["faces"][:, 2]
    fig = make_subplots(
    rows=1, cols=3, horizontal_spacing=0.01,
    specs=[[{'type': 'scene'},{'type': 'scene'},{'type': 'scene'}]])

    color="Jet"
    
    maxx=test_distribution["data_stress"].max().item()
    minn=test_distribution["data_stress"].min().item()
    scale_=0.9
    y_=0.5
    
    # adding surfaces to subplots.
    fig.add_trace(
        go.Mesh3d(
            x=a,y=b,z=c,
            colorbar_title='stress',
            cmax=maxx,
            cmin=minn,
            colorscale=color,
            intensity=test_distribution["data_stress"].numpy().tolist(),
            i=d,j=e,k=f,name='y1',showscale=True, colorbar=dict(x=0.29, y=y_, len=scale_)),row=1, col=1)
    
    fig.add_trace(
        go.Mesh3d(
            x=a,y=b,z=c,
            colorbar_title='stress',
            cmax=maxx,
            cmin=minn,
            colorscale=color,
            intensity=test_distribution["prediction_unnormalize"].numpy().tolist(),
            i=d,j=e,k=f,name='y1',showscale=True, colorbar=dict(x=0.63, y=y_, len=scale_)),row=1, col=2)
    
    fig.add_trace(
        go.Mesh3d(
            x=a,y=b,z=c,
            colorbar_title='stress',
            cmax=maxx,
            cmin=-maxx,
            colorscale=color,
            intensity=(test_distribution["data_stress"]-test_distribution["prediction_unnormalize"]).numpy().tolist(),
            i=d,j=e,k=f,name='y1',showscale=True, colorbar=dict(x=0.97, y=y_, len=scale_)),row=1, col=3)
    
    
    fig.update_scenes(aspectmode='data')
    fig.update_traces(lightposition_x=0, lightposition_y=0, lightposition_z=2)
    fig.update_scenes(camera_eye_x=0, camera_eye_y=0, camera_eye_z=2)
    fig.update_scenes(xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False)
    fig.update_scenes(xaxis_title_text='', yaxis_title_text='',zaxis_title_text='')
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    
    fig.update_layout(height=400, width=1000, title_text="Side By Side Subplots", showlegend=False)
    fig.write_image(f"{save_path}.png") 
    # fig.show()