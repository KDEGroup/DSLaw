import torch
import os
import numpy as np
from torch.optim import Adam
from utils import build_save_dir

class GATTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 epochs,
                 eval_step,
                 save_step,
                 device,
                 lr,
                #  train_id_dict,
                 train_pos_dict,
                 train_neg_dict,
                 train_ids,
                #  val_id_dict,
                 val_pos_dict,
                 val_neg_dict,
                 val_ids,
                 args
                 ):

        self.model = model.to(device)
        self.epochs = epochs
        self.eval_step = eval_step
        self.save_step = save_step
        self.device = device

        self.train_graphs = self.model.train_graphs
        # self.train_id_dict = train_id_dict
        self.train_node_num = self.model.train_num
        self.train_pos_dict = train_pos_dict
        self.train_neg_dict = train_neg_dict
        self.train_ids = train_ids

        self.val_graphs = self.model.val_graphs
        # self.val_id_dict = val_id_dict
        self.val_node_num = self.model.val_num
        self.val_pos_dict = val_pos_dict
        self.val_neg_dict = val_neg_dict
        self.val_ids = val_ids

        self.args=args
        self.save_dir = build_save_dir()

        # 优化器
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train_epoch(self):
        self.model.train()

        loss = 0
        emb, outputs = self.model(self.train_graphs, self.train_node_num, 'train_graph')
        loss = loss + self.model.compute_loss(emb, self.train_pos_dict, self.train_neg_dict, self.train_ids, pos_num=50, neg_num=50, mode='train')
        loss = loss + self.model.degree_regular_loss(emb)

        # loss = loss + self.model.compute_loss(outputs, self.train_pos_dict, self.train_neg_dict, self.train_ids, pos_num=100, neg_num=100, mode='train')
        # loss = loss + self.model.degree_regular_loss(outputs)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def validate_epoch(self):
        self.model.eval()

        loss = 0
        emb, outputs = self.model(self.val_graphs, self.val_node_num, 'val_graph')

        loss = loss + self.model.compute_loss(emb, self.val_pos_dict, self.val_neg_dict, self.val_ids, pos_num=50, neg_num=50, mode='val')
        # loss = loss + self.model.compute_loss(outputs, self.val_pos_dict, self.val_neg_dict, self.val_ids, pos_num=100, neg_num=100, mode='val')

        return loss.item()

    def save_checkpoint(self, model, save_dir, epoch, train_loss, val_loss):
        filename = f"epoch{epoch}_train{train_loss:.4f}_val{val_loss:.4f}.pth"
        filepath = os.path.join(save_dir, filename)


        checkpoint = {
            'model_state_dict': model.state_dict(),
            'args': self.args
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            print(f"[Epoch {epoch:4d}/{self.epochs}] Loss: {train_loss:.4f}")

            if epoch % self.eval_step == 0:
                val_loss = self.validate_epoch()
                print(f"[Validation Loss]: {val_loss:.4f}")

            if epoch % self.save_step == 0:
                self.save_checkpoint(self.model, self.save_dir, epoch, train_loss, val_loss)



