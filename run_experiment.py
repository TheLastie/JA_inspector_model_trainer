#!/usr/bin/env python3
"""
run_experiment.py – единый скрипт запуска обучения любой модели.
Заменяет train_rae_per_json.py и train_exercise_analyzer.py.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

from data_pipeline import get_dataloaders
from model_zoo import RecurrentAutoencoder, TransformerAutoencoder, LSTMVAE, ExerciseAnalyzer
from experiment_manager import manager


def get_model(args, input_dim: int):
    """Создаёт модель в зависимости от model_type."""
    if args.model_type == 'rae':
        return RecurrentAutoencoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            autoregressive=args.autoregressive
        )
    elif args.model_type == 'transformer':
        return TransformerAutoencoder(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            latent_dim=args.latent_dim,
            seq_len=args.seq_len,
            dropout=args.dropout
        )
    elif args.model_type == 'vae':
        return LSTMVAE(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def train_one_epoch(model, loader, optimizer, criterion, device, model_type):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if model_type == 'vae':
            recon, mu, logvar = model(batch)
            mse = criterion(recon, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + 0.001 * kl
        else:
            recon, _ = model(batch)
            loss = criterion(recon, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, model_type):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        if model_type == 'vae':
            recon, mu, logvar = model(batch)
            mse = criterion(recon, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + 0.001 * kl
        else:
            recon, _ = model(batch)
            loss = criterion(recon, batch)
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description='Run single training experiment')
    parser.add_argument('--model_type', required=True, choices=['rae', 'transformer', 'vae', 'analyzer'])
    parser.add_argument('--json_file', required=True)
    parser.add_argument('--output_dir', default='models')
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    # RAE / VAE
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--autoregressive', type=bool, default=True)
    # Transformer
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    # GPU
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    # Устройство
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Даталоадеры
    train_loader, test_loader = get_dataloaders(
        args.json_file,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio
    )
    # Определяем input_dim по первому батчу
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.shape[-1]

    # Модель
    model = get_model(args, input_dim).to(device)

    # Оптимизатор и loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Эксперимент (если не указан, создаём)
    exp_id = args.experiment_id
    if exp_id is None:
        exp_id = manager.create_experiment(
            model_type=args.model_type,
            exercise=os.path.splitext(os.path.basename(args.json_file))[0],
            params=vars(args)
        )
    manager.update_status(exp_id, 'running')

    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # Папка для артефактов (используем старую структуру или ExperimentManager)
    save_dir = os.path.join(args.output_dir, manager.get_experiment_dir(exp_id))
    os.makedirs(save_dir, exist_ok=True)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.model_type)
            val_loss = validate(model, test_loader, criterion, device, args.model_type) if test_loader else float('inf')
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Сохраняем лучшую модель
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
        manager.update_status(exp_id, 'completed', best_test_loss=best_loss, best_epoch=best_epoch)
        print(f"Training finished. Best Val Loss: {best_loss:.6f} at epoch {best_epoch}")

    except Exception as e:
        manager.update_status(exp_id, 'failed', error_message=str(e))
        raise


if __name__ == '__main__':
    main()