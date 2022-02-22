from collections import defaultdict
import os
import math
import time
import datetime
import random
import torch
import argparse
import yaml
import logging
import torch.nn.functional as F
import numpy as np
import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from models import Generator, Discriminator, LinearClassifier
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, sigmoid
from GraphRicciCurvature.OllivierRicci import OllivierRicci

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="default.yaml")
parser.add_argument("--testprop", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--dim", type=int, default=16)
class Config(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

class Model():
    '''
    ANGEL overall model, include training and evaluation process.
    '''
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.graph, self.nodelist, self.edgelist = data['train_graph'], data['train_nodes'], data['train_edges']
        orc_G = OllivierRicci(self.graph, alpha=0.5, verbose="TRACE")
        orc_G.compute_ricci_curvature()
        self.graph = orc_G.G.copy()
        self.generator, self.discriminator = Generator(config), Discriminator(config)
        # Optimizers
        if config.lr_reduce_freq == None:
            config.lr_reduce_freq = config.max_epochs
        self.optimizer_g = getattr(optimizers, config.optimizer)(params=self.generator.parameters(), lr=config.lr_g, weight_decay=config.weight_decay_g)
        self.lr_scheduler_g = torch.optim.lr_scheduler.StepLR(self.optimizer_g, step_size=int(config.lr_reduce_freq), gamma=float(config.gamma))
        self.optimizer_d = getattr(optimizers, config.optimizer)(params=self.discriminator.parameters(), lr=config.lr_d, weight_decay=config.weight_decay_d)
        self.lr_scheduler_d = torch.optim.lr_scheduler.StepLR(self.optimizer_d, step_size=int(config.lr_reduce_freq), gamma=float(config.gamma))
        pass


    def random_walk(self, node_id):
        walk = []
        p = node_id
        while len(walk) < self.config.walk_len:
            if p not in self.graph or len(list(self.graph.neighbors(p))) == 0:
                break
            p = random.choice(list(self.graph.neighbors(p)))
            walk.append(p)
        return walk


    def distortion(self, u, pos_u, pos_fake_emb):
        '''
        Return L2 norm between true distortion and fake distortion.
        '''
        g_emb = self.generator.embedding
        d_emb = self.discriminator.embedding
        nodes_list = np.array([u, pos_u]).T
        distortion_t, distortion_f = [], []
        for i, (x, y) in enumerate(nodes_list):
            try:
                if self.graph.has_edge(x, y):
                    dlt = self.generator.manifold.sqdist(d_emb[x], d_emb[y])
                    dlf = self.generator.manifold.sqdist(pos_fake_emb[x], d_emb[y])
                    distortion_t.append(abs(self.graph[x][y]['ricciCurvature'] / dlt))
                    distortion_f.append(abs(self.graph[x][y]['ricciCurvature'] / dlf))
            except KeyError:
                continue
        dt, df = torch.cat(distortion_t), torch.cat(distortion_f)
        return (dt - df).norm()


    def train_gen(self):
        self.generator.train()
        self.discriminator.eval()
        info = {
            "Training time": 0.0,
            "G_Loss": 0.0, "fake_loss": 0.0, "dt_loss": 0.0
        }
        train_time = 0
        batch_count = math.floor(len(self.nodelist) / self.config.g_batch_size)
        for index in range(batch_count):
            pos_node_ids = []
            pos_node_neighbor_ids = []

            for node_id in self.nodelist[index * self.config.g_batch_size : (index + 1) * self.config.g_batch_size]:
                for k in range(self.config.walk_num):
                    walk = self.random_walk(node_id)
                    for t in walk:
                        pos_node_ids.append(node_id)
                        pos_node_neighbor_ids.append(t)

            start = time.time()

            node_fake_embedding = self.generator(pos_node_ids)
            pos_node_embedding = self.generator.embedding.index_select(0, torch.tensor(pos_node_ids).to(self.config.device))
            neighbor_fake_embedding = self.generator(pos_node_neighbor_ids)

            # 计算distortion
            if self.config.lmda > 0:
                dt_loss = self.distortion(pos_node_ids, pos_node_neighbor_ids, neighbor_fake_embedding)
            else: 
                dt_loss = torch.tensor(0.)
            self.optimizer_g.zero_grad()
            fake_loss = F.binary_cross_entropy(self.discriminator(pos_node_embedding, node_fake_embedding), 
                                                    torch.ones(len(pos_node_ids)).to(self.config.device) * self.config.label_smooth)

            batch_loss = fake_loss + self.config.lmda * dt_loss
            batch_loss.backward()
            self.optimizer_g.step()
            self.lr_scheduler_g.step()

            info["G_Loss"] += batch_loss.data.item() / batch_count
            info["fake_loss"] += fake_loss.data.item() / batch_count
            info["dt_loss"] += dt_loss.data.item() / batch_count
            train_time += time.time() - start
        
        info["Training time"] = train_time
        return info


    def train_dis(self):
        self.discriminator.train()
        self.generator.eval()
        info = {
            "Training time": 0.0,
            "D_Loss": 0.0, "pos_loss": 0.0, "neg_loss": 0.0, "fake_loss": 0.0, 
        }
        # Mini batch training
        start = time.time()
        batch_count = math.floor(len(self.nodelist) / self.config.d_batch_size)
        for index in range(batch_count):
            pos_node_ids = []
            pos_node_neighbor_ids = []
            neg_node_neighbor_ids = []

            for node_id in self.nodelist[index * self.config.d_batch_size : (index + 1) * self.config.d_batch_size]:
                for k in range(self.config.walk_num):  
                    walk = self.random_walk(node_id)
                    for t in walk:
                        pos_node_ids.append(node_id)
                        pos_node_neighbor_ids.append(t)
                        neg = random.choice(self.nodelist)
                        neg_node_neighbor_ids.append(neg)

            # generate fake node()
            node_fake_embedding = self.generator(pos_node_ids).detach()
            self.optimizer_d.zero_grad()
            pos_loss = F.binary_cross_entropy(self.discriminator(pos_node_ids, pos_node_neighbor_ids), torch.ones(len(pos_node_ids)).to(self.config.device))
            neg_loss = F.binary_cross_entropy(self.discriminator(pos_node_ids, neg_node_neighbor_ids), torch.zeros(len(pos_node_ids)).to(self.config.device))
            fake_loss = F.binary_cross_entropy(self.discriminator(pos_node_ids, node_fake_embedding), torch.zeros(len(pos_node_ids)).to(self.config.device))
            batch_loss = pos_loss + neg_loss + fake_loss
            batch_loss.backward()
            self.optimizer_d.step()
            self.lr_scheduler_d.step()

            info["D_Loss"] += batch_loss.data.item() / batch_count
            info["pos_loss"] += pos_loss.data.item() / batch_count
            info["neg_loss"] += neg_loss.data.item() / batch_count
            info["fake_loss"] += fake_loss.data.item() / batch_count

        info["Training time"] = time.time() - start
        return info


    def train(self):
        n_epochs = self.config.max_epochs
        # Saving and logging
        logging.getLogger().setLevel(logging.INFO)
        if self.config.save:
            if not self.config.save_dir:
                dt = datetime.datetime.now()
                date = f"{dt.year}_{dt.month}_{dt.day}"
                models_dir = os.path.join("./save", self.config.dataset + '_' + self.config.task, date)
                save_dir = get_dir_name(models_dir)
            else:
                save_dir = self.config.save_dir
            logging.basicConfig(level=logging.INFO,
                                handlers=[
                                    logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                    logging.StreamHandler()
                                ])
        logging.info("Start training...")
        best_metric_g, best_metric_d = (0, 0), (0, 0)
        best_g, best_d = None, None
        cnt_g, cnt_d = 0, 0

        for epoch in range(n_epochs):
            start = time.time()
            logging.info("Epoch {}/{}".format(epoch+1, n_epochs))

            # Training Discriminator
            for d_epoch in range(self.config.d_epochs):
                train_info = self.train_dis()
                cnt_d += 1
                if self.config.task == 'lp':
                    roc, ap = self.eval_link_prediction(self.discriminator.embedding)
                    train_info['auc'], train_info['ap'] = roc, ap
                    if roc > best_metric_d[0]:         # Update best info
                        best_d = self.discriminator.state_dict()
                        best_metric_d = (roc, ap)
                        cnt_d = 0
                if self.config.task == 'nc':
                    micro_f1, macro_f1 = self.eval_node_classification(self.discriminator.embedding)
                    train_info['micro'], train_info['macro'] = micro_f1, macro_f1
                    if micro_f1 > best_metric_d[0]:         # Update best info
                        best_d = self.discriminator.state_dict()
                        best_metric_d = (micro_f1, macro_f1)
                        cnt_d = 0
                if self.config.task == 'gr':
                    result = self.eval_graph_reconstruction(self.discriminator.embedding)
                    logging.info(result)
                    # if roc > best_metric_d[0]:         # Update best info
                    #     best_d = self.discriminator.state_dict()
                    #     best_metric_d = (roc, ap)
                    cnt_d = 0
                train_info = dict(zip(train_info, map(lambda x: round(x, 6), train_info.values())))
                logging.info(train_info)

            for g_epoch in range(self.config.g_epochs):
                train_info = self.train_gen()
                cnt_g += 1
                if self.config.task == 'lp':
                    roc, ap = self.eval_link_prediction(self.generator.embedding)
                    train_info['auc'], train_info['ap'] = roc, ap
                    if roc > best_metric_g[0]:
                        best_g = self.generator.state_dict()
                        best_metric_g = (roc, ap)
                        cnt_g = 0
                if self.config.task == 'nc':
                    micro_f1, macro_f1 = self.eval_node_classification(self.generator.embedding)
                    train_info['micro'], train_info['macro'] = micro_f1, macro_f1
                    if micro_f1 > best_metric_g[0]:
                        best_g = self.generator.state_dict()
                        best_metric_g = (micro_f1, macro_f1)
                        cnt_g = 0
                if self.config.task == 'gr':
                    result = self.eval_graph_reconstruction(self.generator.embedding)
                    logging.info(result)
                    # if roc > best_metric_d[0]:         # Update best info
                    #     best_d = self.discriminator.state_dict()
                    #     best_metric_d = (roc, ap)
                    cnt_g = 0
                train_info = dict(zip(train_info, map(lambda x: round(x, 6), train_info.values()))) # 6 digits
                logging.info(train_info)

            # Record best metric and save the best logs
            logging.info("time {:.4f}s; ".format(time.time()-start))
            if cnt_g > self.config.patience and cnt_d > self.config.patience:
                logging.info("Early stop because test metric not improve.")
                break
        # Save model
        if self.config.save:
            torch.save(best_g, os.path.join(save_dir, 'gen.pth'))
            torch.save(best_d, os.path.join(save_dir, 'dis.pth'))
            logging.info("Saved model at {}".format(save_dir))
        logging.info("Training ends. Generator's best: {:.4f} & {:.4f}; Discriminator's best: {:.4f} & {:.4f}".format(
            best_metric_g[0], best_metric_g[1], best_metric_d[0], best_metric_d[1]
        )) 
        return best_metric_g, best_metric_d
                

    def eval_link_prediction(self, embs):
        '''
        ROC-AUC Score using F-D Decoder as logits.
        '''
        self.generator.eval()
        self.discriminator.eval()
        emb_in, emb_out = self.data['test_edges_pos'][0], self.data['test_edges_pos'][1]
        pos_scores = self.discriminator(
            embs.index_select(0, torch.tensor(emb_in).to(self.config.device)), 
            embs.index_select(0, torch.tensor(emb_out).to(self.config.device))
        )
        emb_in, emb_out = self.data['test_edges_neg'][0], self.data['test_edges_neg'][1]
        neg_scores = self.discriminator(
            embs.index_select(0, torch.tensor(emb_in).to(self.config.device)), 
            embs.index_select(0, torch.tensor(emb_out).to(self.config.device))
        )
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.cpu().data.numpy()) + list(neg_scores.cpu().data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        return roc, ap


    def eval_node_classification(self, embs):
        '''
        Micro- and Macro- F1 Score using LinearRegression.
        '''
        embs = self.discriminator.manifold.logmap0(embs, self.discriminator.c)
        embeddings = embs.cpu().data.numpy().tolist()
        X = []
        Y = []
        for idx, key in enumerate(self.data['labels']):
            X.append(embeddings[idx] + embeddings[idx])
            Y.append(key)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

        lr = LinearClassifier(self.config)
        lr.train(torch.tensor(X_train), torch.LongTensor(Y_train))
        Y_pred = lr.test(torch.tensor(X_test))

        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1, macro_f1



if __name__ == "__main__":
    # Parse argument and import config
    args = parser.parse_args()
    yml_path = './configs/{}'.format(args.config)
    with open(yml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = Config(config)             # convert yaml dict to python object for more convinent reference
    config.test_prop = args.testprop
    config.seed = args.seed
    config.dims = args.dim
    print("Using config: {}".format(args.config))
    
    # random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    print("Using seed: {}".format(config.seed))
    # load dataset
    data = load_data(config)
    # create a distance of model
    config.device = 'cuda:' + str(config.cuda) if int(config.cuda) >= 0 else 'cpu'
    model = Model(config, data)
    # CUDA setting
    print("Using device: {}".format(config.device))
    if config.cuda is not None and int(config.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda)
        model.generator = model.generator.to(config.device)
        model.discriminator = model.discriminator.to(config.device)

    # Training
    best_metric_g, best_metric_d = model.train()
           