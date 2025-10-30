import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from my_data_loader import load_train_dataset, load_stream_dataset
from tqdm import tqdm
import numpy as np
import time
import logging
import joblib
import os


class Encoder_linear(nn.Module):
    def __init__(self, feature_dim=None):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6144, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
    


class DADSModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.encoder = Encoder_linear(feature_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=1)
        return z


class DADSLoss(nn.Module):
    def __init__(self, device=None, temperature=0.07, contrast_mode='all', base_temperature=0.07, l2_reg_weight=0.01):
        super(DADSLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device
        self.l2_reg_weight = l2_reg_weight

    def forward(self, features, labels=None, domain_label=None, model=None):
        batch_size = features.shape[0]
        
        # Build mask matrix, indicating which sample pairs belong to the same class
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device) # If the labels of sample i and j are the same, mask[i,j]=1, else 0

        # Get the number of contrast views
        contrast_count = features.shape[1]
        # Concatenate all views together
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Calculate dot-product similarity and divide by temperature (all combinations)
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)/self.temperature # Shape: [batch_size, batch_size]

        # Numerical stability: subtract the max value in each row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Expand mask to match number of anchors
        mask = mask.repeat(anchor_count, contrast_count) # Repeat along columns
        
        # Mask out self-comparisons (create a mask matrix with diagonal 0, other positions 1)
        logits_mask = 1 - torch.eye(batch_size * anchor_count).to(self.device)
        mask = mask * logits_mask
        
        # Calculate log-probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Calculate the number of positive sample pairs
        mask_pos_pairs = mask.sum(1) # Count the number of 1s in each row (number of positive pairs for each anchor)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs) # Avoid division by zero: replace zeros with 1 (if no positive pair appears)
        
        # Calculate contrastive loss
        contrastive_loss = - (mask * log_prob).sum(1) / mask_pos_pairs
        contrastive_loss = contrastive_loss.view(anchor_count, batch_size).mean()
        
        # Calculate L2 regularization term
        l2_reg = 0
        if model is not None and self.l2_reg_weight > 0:
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2) ** 2
            l2_reg = self.l2_reg_weight * l2_reg
        
        # Total loss = contrastive loss + L2 regularization
        total_loss = contrastive_loss + l2_reg

        return total_loss




class L_memory_dynamic_temperature(nn.Module):

    def __init__(self, temperature=0.07, device='cuda', epsilon=None):

        super(L_memory_dynamic_temperature, self).__init__()
        self.base_temperature = temperature
        self.device = device
        self.epsilon = epsilon

    def _compute_confidence_weights(self, feature_online, feature_offline, Y_online, Y_offline, conf_online=None):

        # Compute feature similarity as a proxy for confidence
        sim_matrix = torch.matmul(feature_online, feature_offline.T)  # [N_online, N_offline]
        
        # Build positive sample mask
        Y_online = Y_online.contiguous().view(-1, 1)
        Y_offline = Y_offline.contiguous().view(-1, 1)
        mask_positive = torch.eq(Y_online, Y_offline.T).float().to(self.device)
        
        # Confidence weights based on feature similarity
        feature_confidence_weights = torch.where(
            mask_positive > 0.5,
            sim_matrix,  # Positive pair: higher similarity -> higher confidence
            1.0 - sim_matrix  # Negative pair: lower similarity -> higher confidence
        )
        
        # If pseudo-label confidence is provided, integrate it
        if conf_online is not None:
            conf_online = conf_online.to(self.device)
            # Broadcast the confidence to [N_online, N_offline]
            conf_matrix = conf_online.unsqueeze(1).expand(-1, feature_offline.shape[0])  # [N_online, N_offline]
            
            # Combine feature similarity confidence and pseudo-label confidence
            # Weighted average: 70% feature similarity + 30% pseudo-label confidence
            combined_confidence = 0.7 * feature_confidence_weights + 0.3 * conf_matrix
        else:
            combined_confidence = feature_confidence_weights
        
        # Map confidence to [0.1, 1.0] to avoid too small weights
        confidence_weights = torch.sigmoid(combined_confidence * 2)  # sigmoid scaling
        confidence_weights = 0.1 + 0.9 * confidence_weights  # map to [0.1, 1.0]
        
        return confidence_weights

    def _compute_dynamic_temperature_matrix(self, feature_online, feature_offline, Y_online, Y_offline, conf_online=None):
 
        # Compute confidence weights (including pseudo-label confidence)
        confidence_weights = self._compute_confidence_weights(
            feature_online, feature_offline, Y_online, Y_offline, conf_online
        )
        
        
        # Dynamic temperature: τ_ij = τ_0 / (w_ij + ε)
        temperature_matrix = self.base_temperature / (confidence_weights + self.epsilon)
        
        return temperature_matrix

    def forward(self, feature_online, Y_online, feature_offline, Y_offline, conf_online=None):

        # Move features and labels to device
        feature_online = feature_online.to(self.device)
        Y_online = Y_online.to(self.device)
        feature_offline = feature_offline.to(self.device)
        Y_offline = Y_offline.to(self.device)
        
        # Move confidence to device if provided
        if conf_online is not None:
            conf_online = conf_online.to(self.device)

        # Compute dynamic temperature matrix (including pseudo-label confidence)
        temperature_matrix = self._compute_dynamic_temperature_matrix(
            feature_online, feature_offline, Y_online, Y_offline, conf_online
        )

        # Compute similarity matrix (apply dynamic temperature element-wise)
        similarity_raw = torch.matmul(feature_online, feature_offline.T)  # [N_online, N_offline]
        sim_matrix = similarity_raw / temperature_matrix  # Divide element-wise, apply dynamic temperature

        # Build label mask matrix
        Y_online = Y_online.contiguous().view(-1, 1)
        Y_offline = Y_offline.contiguous().view(-1, 1)
        mask_positive = torch.eq(Y_online, Y_offline.T).float().to(self.device)  # [N_online, N_offline]

        # Numerical stability: subtract the max value in each row
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Calculate exp
        exp_logits = torch.exp(logits)

        # Log probability of positive pairs
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Number of positive sample pairs for each sample
        n_pos = mask_positive.sum(1)
        n_pos = torch.where(n_pos < 1e-6, torch.ones_like(n_pos), n_pos)

        # Compute contrastive loss
        # If pseudo-label confidence is provided, use confidence-weighted loss
        if conf_online is not None:
            # Use confidence to weight each online sample's loss
            conf_weights = conf_online.unsqueeze(1)  # [N_online, 1]
            weighted_log_prob = conf_weights * mask_positive * log_prob
            loss = -(weighted_log_prob).sum(1) / n_pos
        else:
            loss = -(mask_positive * log_prob).sum(1) / n_pos
            
        loss = loss.mean()

        return loss

    def get_temperature_statistics(self, feature_online, Y_online, feature_offline, Y_offline, conf_online=None):

        with torch.no_grad():
            feature_online = feature_online.to(self.device)
            Y_online = Y_online.to(self.device)
            feature_offline = feature_offline.to(self.device)
            Y_offline = Y_offline.to(self.device)
            if conf_online is not None:
                conf_online = conf_online.to(self.device)
            
            temperature_matrix = self._compute_dynamic_temperature_matrix(
                feature_online, feature_offline, Y_online, Y_offline, conf_online
            )
            
            # Build positive sample mask
            Y_online = Y_online.contiguous().view(-1, 1)
            Y_offline = Y_offline.contiguous().view(-1, 1)
            mask_positive = torch.eq(Y_online, Y_offline.T).float().to(self.device)
            
            # Collect statistics for positive and negative samples, respectively
            pos_temperatures = temperature_matrix[mask_positive > 0.5]
            neg_temperatures = temperature_matrix[mask_positive < 0.5]
            
            return {
                'base_temperature': self.base_temperature,
                'pos_temp_mean': pos_temperatures.mean().item() if len(pos_temperatures) > 0 else 0,
                'pos_temp_std': pos_temperatures.std().item() if len(pos_temperatures) > 0 else 0,
                'neg_temp_mean': neg_temperatures.mean().item() if len(neg_temperatures) > 0 else 0,
                'neg_temp_std': neg_temperatures.std().item() if len(neg_temperatures) > 0 else 0,
                'temp_min': temperature_matrix.min().item(),
                'temp_max': temperature_matrix.max().item(),
                'temp_range_ratio': (temperature_matrix.max() / temperature_matrix.min()).item()
            }





# Trainer
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()

        self.model = DADSModel(feature_dim=args.feature_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = DADSLoss(
            device=self.device, 
            temperature=args.temperature,
            l2_reg_weight=getattr(args, 'l2_reg_weight', 0.0000)
        )

    def _get_device(self):
        if self.args.gpu_device == "1":
            return torch.device("mps")
        elif self.args.gpu_device == "2":
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def train(self):
        train_loader = load_train_dataset()

        for epoch in range(self.args.max_epoch):
            epoch_start_time = time.time()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.args.max_epoch}')

            for x, y, domain_y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                domain_y = domain_y.to(self.device)

                # Generate two views for augmentation
                x1 = x + torch.randn_like(x) * 0.1
                x2 = x + torch.randn_like(x) * 0.1

                # Forward propagation
                z1 = self.model(x1)  # [N, D]
                z2 = self.model(x2)  # [N, D]

                # Stack features to [N, 2, D]
                features = torch.stack([z1, z2], dim=1)  # [N, 2, D]

                loss = self.criterion(features, y, domain_y, model=self.model)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s, LR: {current_lr:.6f}')

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.save_path + '.pth')
        logging.info(f'Model saved to: {self.args.save_path}.pth')

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.load_path))

    def eval_map(self, data_loader=None):
        self.model.eval()
        test_loader = data_loader
        features_list = []
        y_list = []
        domain_y_list = []
        with torch.no_grad():
            for x, y, domain_y in tqdm(test_loader, desc='Extracting Features'):
                x = x.to(self.device)
                features = self.model(x).cpu()
                y = y.cpu()
                domain_y = domain_y.cpu()

                features_list.append(features)
                y_list.append(y)
                domain_y_list.append(domain_y)

        features = torch.cat(features_list, dim=0)
        y = torch.cat(y_list, dim=0)
        domain_y = torch.cat(domain_y_list, dim=0)

        return features, y, domain_y
    
    def get_memory_features(self, memory_bank, batch_size=128):
        with torch.no_grad():
            X = memory_bank['X']
            memory_features = self.model(X)
        memory_feature_dict = {'feature': memory_features, 'Y': memory_bank['Y']}
        
        return memory_feature_dict


    def update_model_memory_dynamic_temperature(self, memory_bank_offline, memory_bank_online, epsilon=1e-5):
        # Use dynamic temperature loss function
        criterion = L_memory_dynamic_temperature(
            temperature=self.args.temperature,
            device=self.device,
            epsilon=epsilon,
        )

        memory_bank_offline['X'] = memory_bank_offline['X'].to(self.device)
        memory_bank_offline['Y'] = memory_bank_offline['Y'].to(self.device)
        memory_bank_online['X'] = memory_bank_online['X'].to(self.device)
        memory_bank_online['Y'] = memory_bank_online['Y'].to(self.device)
        
        # Check whether pseudo-label confidence is provided
        conf_online = None
        if 'confidence' in memory_bank_online:
            conf_online = memory_bank_online['confidence'].to(self.device)
            logging.info(f"Use pseudo-label confidence. Confidence range: [{conf_online.min().item():.4f}, {conf_online.max().item():.4f}]")
            logging.info(f"Mean confidence: {conf_online.mean().item():.4f}")

        # Record temperature statistics (only for the first epoch)
        temperature_stats = None
        
        for epoch in range(self.args.update_epoch):
            memory_feature_offline = self.model(memory_bank_offline['X'])
            memory_feature_online = self.model(memory_bank_online['X'])
            
            # Calculate dynamic temperature memory loss (including confidence information)
            loss = criterion(
                memory_feature_online, memory_bank_online['Y'], 
                memory_feature_offline, memory_bank_offline['Y'],
                conf_online
            )
            
            # Get temperature statistics (only for the first epoch, to avoid redundant computation)
            if epoch == 0:
                temperature_stats = criterion.get_temperature_statistics(
                    memory_feature_online, memory_bank_online['Y'], 
                    memory_feature_offline, memory_bank_offline['Y'],
                    conf_online
                )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log training information
            if hasattr(self.args, 'verbose') and self.args.verbose and epoch % 5 == 0:
                logging.info(f'Dynamic Temperature Update Epoch {epoch+1}/{self.args.update_epoch}, Loss: {loss.item():.6f}')
        
        # Log final temperature statistics
        if temperature_stats is not None:
            logging.info("Dynamic Temperature Statistics:")
            logging.info(f"  Base Temperature tau_0: {temperature_stats['base_temperature']:.4f}")
            logging.info(f"  Positive Sample Temperature - Mean: {temperature_stats['pos_temp_mean']:.4f}, Std: {temperature_stats['pos_temp_std']:.4f}")
            logging.info(f"  Negative Sample Temperature - Mean: {temperature_stats['neg_temp_mean']:.4f}, Std: {temperature_stats['neg_temp_std']:.4f}")
            logging.info(f"  Temperature Range: [{temperature_stats['temp_min']:.4f}, {temperature_stats['temp_max']:.4f}]")
            logging.info(f"  Temperature Range Ratio: {temperature_stats['temp_range_ratio']:.2f}x")
            
            # If using confidence, log confidence statistics
            if conf_online is not None:
                logging.info(f"  Using pseudo-label confidence weighting, confidence statistics:")
                logging.info(f"    Confidence range: [{conf_online.min().item():.4f}, {conf_online.max().item():.4f}]")
                logging.info(f"    Mean confidence: {conf_online.mean().item():.4f} ± {conf_online.std().item():.4f}")
        
        return temperature_stats


