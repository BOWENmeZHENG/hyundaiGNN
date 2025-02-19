import pandas as pd
import numpy as np
import torch
import time
import os
import copy
from torch_geometric.data import DataLoader
import net
import utils
from tqdm import trange
from sklearn.metrics import r2_score


def train(dataset, device, stats_list, args):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''
    utils.seed_everything(args.seed)
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss', 'test_r2'])

    model_name = f"s_{args.seed}_nl_{args.num_layers}_bs_{args.batch_size}" + \
                f"_hd_{args.hidden_dim}_ep_{args.epochs}_wd_{args.weight_decay}" + \
                f"_lr_{args.lr}_shuff_{args.shuffle}_tr_{args.train_size}" + \
                f"_te_{args.test_size}_time_{time.strftime('%Y-%b-%d-%H-%M-%S')}"
    
    # torch_geometric DataLoaders are used for handling the data of lists of graphs
    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=args.shuffle)

    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = stats_list
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = mean_vec_x.to(device), std_vec_x.to(device), mean_vec_edge.to(device), std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device)

    # build model
    num_node_features = dataset[0]["x"].shape[1]  # 6
    num_edge_features = dataset[0]["edge_attr"].shape[1] # 4
    num_out_features = 1 # the stress variables have the shape of 1

    model = net.MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_out_features, args).to(device)
    scheduler, opt = net.build_optimizer(args, model.parameters())

    # train
    losses, test_losses, test_r2 = [], [], []
    best_test_loss = np.inf
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        model.train()
        total_loss = 0
        num_loops = 0
        for batch in loader:
            batch = batch.to(device)
            # DataBatch(x=[52941, 6], edge_index=[2, 315078], edge_attr=[315078, 4], cells=[104570, 3], stress=[52941, 1], top2_index=[787], outerspoke_index=[39334], top30_index_new=[11013], name=[1], batch=[52941], ptr=[2])
            opt.zero_grad()
            # Still needs both inner and outer surfaces to predict
            # Only use outer surfaces in loss function
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge) # Pred is all stress, torch.Size([32771, 1])
            # extracting pred for outer surface
            # loss = model.loss(pred, batch["stress"], mean_vec_y, std_vec_y)
            outerspoke_index = batch["outerspoke_index"]
            loss = model.loss(pred[outerspoke_index], batch["stress"][outerspoke_index], mean_vec_y, std_vec_y)
            loss.backward()   
            opt.step()

            total_loss += loss.item()
            num_loops += 1

        total_loss /= num_loops
        losses.append(total_loss)
        
        if epoch % args.test_interval == 0:
            test_loss, r2_avg = test(test_loader, device, model, mean_vec_x, std_vec_x, mean_vec_edge,
                              std_vec_edge, mean_vec_y, std_vec_y)
            test_losses.append(test_loss.item())
            test_r2.append(r2_avg)
            # saving model
            if not os.path.isdir(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            result = os.path.join(args.checkpoint_dir, model_name + '.csv')
            df.to_csv(result, index=False)
            #save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)
            
            print(f"train loss: {round(total_loss, 2)}  test loss: {round(test_loss.item(), 2)}")
        df = df._append({'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1], 'test_r2': test_r2[-1]}, ignore_index=True)

        if (epoch + 1) % args.plot_interval == 0:
            utils.plot_losses(losses, test_losses, epoch, args.test_interval)
            utils.plot_test_r2(test_r2, epoch, args.test_interval)

        if args.save_best_model:
            PATH = os.path.join(args.checkpoint_dir, model_name + '.pt')
            torch.save(best_model.state_dict(), result)

    return test_losses, test_r2, losses, best_model, best_test_loss, test_loader

def test(loader, device, test_model, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y):
    loss = 0
    r2_list = []
    num_loops = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            prediction_unnormalize = utils.unnormalize(pred, mean_vec_y, std_vec_y)
            # r2 = r2_score(data["stress"].detach().cpu().numpy(), prediction_unnormalize.detach().cpu().numpy())
            outerspoke_index = data["outerspoke_index"]
            r2 = r2_score(data["stress"][outerspoke_index].detach().cpu().numpy(), prediction_unnormalize[outerspoke_index].detach().cpu().numpy())
            r2_list.append(r2)
            # loss += test_model.loss(pred, data["stress"], mean_vec_y, std_vec_y)
            loss += test_model.loss(pred[outerspoke_index], data["stress"][outerspoke_index], mean_vec_y, std_vec_y)             
        num_loops += 1
    r2_avg = sum(r2_list) / len(r2_list)
    print(f"\n{r2_avg}")
    return loss / num_loops, r2_avg
