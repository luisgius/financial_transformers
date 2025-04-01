import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import json
import sys
import traceback
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from pathlib import Path
from data.data_loader import FinancialDataLoader
from features.techincal import add_technical_features, add_multi_timeframe_features
from sklearn.preprocessing import RobustScaler
from models.transformer import FinancialTransformer
from Evaluation.evaluation import (
    walk_forward_validation,
    analyze_attention_patterns,
    plot_performance,
    identify_market_regimes,
    calculate_classification_metrics,
    calculate_financial_metrics
)
import seaborn as sns
import argparse
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
import time 
import yaml
import json
import pickle
import torch.utils.data
import torch.optim.lr_scheduler
import torch.nn.functional as F
import torch.optim




def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the financial transformer application.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Financial Transformer")
    # Data loading arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--symbols', nargs='+', required=True,
                           help='List of stock symbols to analyze')
    data_group.add_argument('--start-date', type=str, required=True,
                           help='Start date for data (YYYY-MM-DD)')
    data_group.add_argument('--end-date', type=str, required=True,
                           help='End date for data (YYYY-MM-DD)')
    data_group.add_argument('--data-dir', type=str, default='data/raw',
                           help='Directory for storing raw data')
    data_group.add_argument('--use-cache', action='store_true',
                           help='Use cached data if available')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--seq-length', type=int, default=60,
                           help='Sequence length for the transformer')
    model_group.add_argument('--target-horizon', type=int, default=1,
                           help='Prediction horizon in days')
    model_group.add_argument('--model-type', type=str, default='transformer',
                           choices=['transformer', 'lstm'],
                           help='Type of model to use')
    model_group.add_argument('--hidden-size', type=int, default=128,
                           help='Hidden size of the model')
    model_group.add_argument('--num-layers', type=int, default=3,
                           help='Number of layers in the model')
    model_group.add_argument('--dropout', type=float, default=0.1,
                           help='Dropout rate')
    
    # Training arguments
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('--batch-size', type=int, default=16,
                              help='Batch size for training')
    training_group.add_argument('--epochs', type=int, default=50,
                              help='Number of training epochs')
    training_group.add_argument('--learning-rate', type=float, default=0.0001,
                              help='Learning rate')
    training_group.add_argument('--train-ratio', type=float, default=0.7,
                              help='Ratio of data for training')
    training_group.add_argument('--val-ratio', type=float, default=0.15,
                              help='Ratio of data for validation')
    
    # Market regime arguments
    regime_group = parser.add_argument_group('Market Regime Configuration')
    regime_group.add_argument('--regime-window', type=int, default=63,
                            help='Window for market regime calculation')
    regime_group.add_argument('--volatility-threshold', type=float, default=0.15,
                            help='Threshold for high/low volatility regime')
    regime_group.add_argument('--trend-threshold', type=float, default=0.1,
                            help='Threshold for trending/ranging regime')
    
    # Output and logging
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output-dir', type=str, default='output',
                            help='Directory for saving outputs')
    output_group.add_argument('--model-save-path', type=str, default='models/saved',
                            help='Path to save trained models')
    output_group.add_argument('--log-level', type=str, default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help='Logging level')
    
    # Model configuration additions
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--num-heads', type=int, default=4,
                           help='Number of attention heads (transformer only)')
    model_group.add_argument('--use-time2vec', action='store_true',
                           help='Use Time2Vec encoding instead of standard positional encoding')
    
    # Walk-forward validation arguments
    wf_group = parser.add_argument_group('Walk-Forward Validation Configuration')
    wf_group.add_argument('--window-size', type=int, default=252,
                        help='Window size in days for walk-forward validation')
    wf_group.add_argument('--step-size', type=int, default=21,
                        help='Step size in days for walk-forward validation')
    wf_group.add_argument('--retrain', action='store_true',
                        help='Retrain the model at each walk-forward step')
    
    # Attention analysis arguments
    attention_group = parser.add_argument_group('Attention Analysis Configuration')
    attention_group.add_argument('--analyze-attention', action='store_true',
                               help='Perform attention pattern analysis')
    attention_group.add_argument('--num-samples', type=int, default=5,
                               help='Number of samples for attention analysis')

    
    args = parser.parse_args()

    # Validate arguments
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        parser.error("Dates must be in YYYY-MM-DD format")
    
    if args.start_date >= args.end_date:
        parser.error("Start date must be before end date")
    
    if not (0 < args.train_ratio + args.val_ratio < 1):
        parser.error("Train and validation ratios must sum to less than 1")
    
    # Create necessary directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)
    
    return args


def load_config(config_path: str) -> Dict:
    """
    Load experiment configuration from YAML or JSON file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary
    """
    path = Path(config_path)
    
    # Default configuration
    default_config = {
        'data': {
            'data_dir': 'data/raw',
            'use_cache': True,
            'target_col': 'Close',
            'feature_columns': ['Open', 'High', 'Low', 'Close', 'Volume']
        },
        'model': {
            'type': 'transformer',
            'hidden_size': 128,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1,
            'seq_length': 60
        },
        'training': {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'early_stopping_patience': 20
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'regime_window': 63,
            'volatility_threshold': 0.15,
            'trend_threshold': 0.1
        }
    }
    
    # Load configuration file
    try:
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                user_config = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                user_config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        raise
    
    # Recursive dictionary update function
    def deep_update(source: Dict, update: Dict) -> Dict:
        for k, v in update.items():
            if isinstance(v, dict) and k in source:
                source[k] = deep_update(source[k], v)
            else:
                source[k] = v
        return source
    
    # Merge configurations
    config = deep_update(default_config.copy(), user_config)
    
    # Validate required sections
    required_sections = ['data', 'model', 'training', 'evaluation']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    # Convert special values
    try:
        # Convert date strings to datetime objects if present
        if 'start_date' in config['data']:
            config['data']['start_date'] = datetime.strptime(
                config['data']['start_date'], '%Y-%m-%d'
            )
        if 'end_date' in config['data']:
            config['data']['end_date'] = datetime.strptime(
                config['data']['end_date'], '%Y-%m-%d'
            )
        
        # Convert string values to proper types where needed
        if isinstance(config['model']['hidden_size'], str):
            config['model']['hidden_size'] = int(config['model']['hidden_size'])
        if isinstance(config['training']['learning_rate'], str):
            config['training']['learning_rate'] = float(config['training']['learning_rate'])
            
    except ValueError as e:
        logging.error(f"Error converting configuration values: {e}")
        raise
    
    # Create a nested defaultdict-style access pattern
    class ConfigDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for k, v in self.items():
                if isinstance(v, dict):
                    self[k] = ConfigDict(v)
        
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                return ConfigDict()  # Return empty ConfigDict for missing keys
    
    return ConfigDict(config)

def prepare_data(
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Tuple[FinancialDataLoader, pd.DataFrame, Dict[str, Any]]:
    """
    Prepare data for model training and evaluation.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary
    output_dir : Optional[str]
        Directory to save processed data
        
    Returns:
    --------
    Tuple[FinancialDataLoader, pd.DataFrame, Dict[str, Any]]
        - Data loader instance
        - Full processed dataframe with features
        - Additional metadata (feature names, scaling parameters, etc.)
    """
    logging.info("Starting data preparation...")
    

    # Initialize data loader
    data_loader = FinancialDataLoader(
        symbols=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        seq_length=config['model']['seq_length'],
        target_horizon=config['model']['target_horizon'],
        data_dir=config['data'].get('data_dir', 'data/raw'),
        use_cache=config['data'].get('use_cache', True)
    )
    
    # Load raw price data
    raw_data = data_loader.load_data()
    logging.info(f"Loaded raw data for {len(raw_data)} symbols")
    
    # Calculate returns for each symbol
    data_loader.calculate_returns()
    
    # Process each symbol's data with technical indicators
    processed_data = {}
    all_features = []
    
    for symbol, df in data_loader.raw_data.items():
        logging.info(f"Processing technical indicators for {symbol}...")
        
        # Determine indicator categories to include
        indicators = config['data'].get('indicators', ['trend', 'momentum', 'volatility', 'volume'])
        
        # Generate technical indicators
        if 'timeframes' in config['data'] and len(config['data']['timeframes']) > 1:
            # Multi-timeframe features
            enhanced_df = add_multi_timeframe_features(
                df, 
                timeframes=config['data']['timeframes']
            )
        else:
            # Single timeframe features
            enhanced_df = add_technical_features(
                df, 
                include_categories=indicators
            )
        
        logging.info(f"Generated {enhanced_df.shape[1]} features for {symbol}")
        
        # Handle feature selection to reduce multicollinearity
        if config['data'].get('feature_selection', True):
            # Calculate correlation matrix
            corr_matrix = enhanced_df.corr().abs()
            
            # Find highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns 
                      if any(upper_tri[column] > config['data'].get('correlation_threshold', 0.95))]
            
            # Drop highly correlated features
            if to_drop:
                logging.info(f"Dropping {len(to_drop)} highly correlated features")
                enhanced_df = enhanced_df.drop(columns=to_drop)
        
        # Store processed data
        processed_data[symbol] = enhanced_df
        
        # Keep track of feature names (from first symbol)
        if not all_features:
            # Exclude target columns and metadata
            all_features = [col for col in enhanced_df.columns 
                          if not col.startswith('target_') and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Combine processed data for all symbols
    if len(processed_data) == 1:
        # Single symbol case
        combined_df = next(iter(processed_data.values()))
    else:
        # Multiple symbols case - would need more sophisticated combining logic
        # For simplicity, we'll just use the first symbol for now
        logging.warning("Multiple symbols not fully implemented, using first symbol only")
        combined_df = next(iter(processed_data.values()))
    
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    for feature in all_features:
        if feature in combined_df.columns:
            # Scale each feature independently
            combined_df[feature] = scaler.fit_transform(combined_df[feature].values.reshape(-1, 1))

    logging.info(f"Applied robust scaling to {len(all_features)} features")

    # Create metadata dictionary
    metadata = {
        'feature_names': all_features,
        'num_features': len(all_features),
        'symbols': list(processed_data.keys()),
        'sequence_length': config['model']['seq_length'],
        'target_horizon': config['model']['target_horizon']
    }
    
    # Apply scaling to create the final processed data
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    
    # Select only the features for scaling (not target or metadata columns)
    feature_data = combined_df[all_features].values
    
    # Fit scaler on feature data
    scaled_features = scaler.fit_transform(feature_data)
    
    # Replace original features with scaled versions
    for i, feature in enumerate(all_features):
        combined_df[feature] = scaled_features[:, i]
    
    # Store scaler in metadata
    metadata['scaler'] = scaler
    
    # Save processed data if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined dataframe
        combined_df.to_csv(os.path.join(output_dir, 'processed_data.csv'))
        
        # Save metadata (excluding non-serializable objects like the scaler)
        serializable_metadata = {k: v for k, v in metadata.items() if k != 'scaler'}
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        # Save scaler separately using pickle
        import pickle
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        logging.info(f"Saved processed data and metadata to {output_dir}")
    
    logging.info("Data preparation complete")
    return data_loader, combined_df, metadata



def create_model(
    config: Dict[str, Any],
    input_dim: int,
    pretrained_path: Optional[str] = None
) -> torch.nn.Module:
    """
    Create financial transformer model based on configuration.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Model configuration
    input_dim : int
        Input dimension (number of features)
    pretrained_path : Optional[str]
        Path to pretrained model weights
        
    Returns:
    --------
    torch.nn.Module
        Financial transformer model
    """
    logging.info(f"Creating transformer model with input dimension {input_dim}")
    
    # Extract model configuration
    hidden_size = config['model'].get('hidden_size', 128)
    num_layers = config['model'].get('num_layers', 2)
    num_heads = config['model'].get('num_heads', 8)
    dropout = config['model'].get('dropout', 0.1)
    seq_length = config['model'].get('seq_length', 60)
    use_time2vec = config['model'].get('use_time2vec', True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and 
                         not config.get('no_cuda', False) else 'cpu')
    
    # Calculate feed-forward dimension (typically 4x hidden size)
    d_ff = hidden_size * 4
    
    # Create the FinancialTransformer
    logging.info(f"Building transformer model with {num_layers} layers and {num_heads} heads")
    model = FinancialTransformer(
        input_dim=input_dim,
        d_model=len(feature_template),
        n_head=num_heads,
        n_layers=num_layers,
        d_ff=d_ff,
        seq_length=seq_length,
        dropout=dropout
    )
    
    # Initialize transformer weights properly
    def _init_transformer_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Xavier initialization for linear and embedding layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Layer normalization parameters are already initialized appropriately
            pass
    
    model.apply(_init_transformer_weights)
    
    # Load pretrained weights if specified
    if pretrained_path:
        if os.path.exists(pretrained_path):
            logging.info(f"Loading pretrained weights from {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location=device)
                
                # Check if state_dict is wrapped in a checkpoint
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Try to load state dict, allowing for partial matches
                model_dict = model.state_dict()
                
                # Filter out mismatched keys
                pretrained_dict = {k: v for k, v in state_dict.items() 
                                  if k in model_dict and model_dict[k].shape == v.shape}
                
                if len(pretrained_dict) != len(state_dict):
                    logging.warning(f"Some pretrained weights couldn't be loaded due to shape mismatches. "
                                   f"Loaded {len(pretrained_dict)}/{len(state_dict)} layers.")
                
                # Update model weights
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            except Exception as e:
                logging.error(f"Failed to load pretrained weights: {e}")
        else:
            logging.error(f"Pretrained weights file not found: {pretrained_path}")
    
    # Count the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {trainable_params:,} trainable parameters")
    
    # Move model to the appropriate device
    model = model.to(device)
    logging.info(f"Model moved to {device}")
    
    return model



def train_model(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    train_targets: torch.Tensor,
    val_data: Optional[torch.Tensor] = None,
    val_targets: Optional[torch.Tensor] = None,
    config: Dict[str, Any] = None,
    callbacks: List[Callable] = None
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train the financial transformer model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to train
    train_data : torch.Tensor
        Training data tensor of shape [n_samples, seq_length, n_features]
    train_targets : torch.Tensor
        Training targets of shape [n_samples]
    val_data : Optional[torch.Tensor]
        Validation data tensor, same shape as train_data
    val_targets : Optional[torch.Tensor]
        Validation targets, same shape as train_targets
    config : Dict[str, Any]
        Training configuration
    callbacks : List[Callable]
        List of callback functions for logging, early stopping, etc.
        
    Returns:
    --------
    Tuple[torch.nn.Module, Dict[str, List[float]]]
        - Trained model
        - Training history dictionary with metrics
    """
    if config is None:
        config = {}
    
    # Extract training parameters
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 50)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-5)
    patience = config.get('patience', 10)
    scheduler_type = config.get('scheduler', 'plateau')
    gradient_clip = config.get('gradient_clip', 1.0)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure model is on the correct device
    model = model.to(device)
    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    
    if val_data is not None and val_targets is not None:
        val_data = val_data.to(device)
        val_targets = val_targets.to(device)
        do_validation = True
    else:
        do_validation = False
    
    # Initialize optimizer (AdamW typically works better than Adam for transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize learning rate scheduler
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate/10
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=patience//2, gamma=0.5
        )
    else:
        scheduler = None
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    if do_validation:
        val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    
    # Loss function (binary cross entropy for directional prediction)
    criterion = nn.BCELoss()
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    logging.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_data, batch_targets in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_data)
            
            # Handle different model output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extract predictions from (predictions, attention)
            
            # Ensure outputs have correct shape
            if outputs.shape != batch_targets.shape:
                outputs = outputs.view(batch_targets.shape)
            
            # Calculate loss
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * batch_data.size(0)
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == batch_targets).sum().item()
            total_predictions += batch_targets.size(0)
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_predictions / total_predictions
        
        # Validation phase
        if do_validation:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_targets in val_loader:
                    # Forward pass
                    outputs = model(batch_data)
                    
                    # Handle different model output formats
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Ensure outputs have correct shape
                    if outputs.shape != batch_targets.shape:
                        outputs = outputs.view(batch_targets.shape)
                    
                    # Calculate loss
                    loss = criterion(outputs, batch_targets)
                    
                    # Update metrics
                    val_loss += loss.item() * batch_data.size(0)
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == batch_targets).sum().item()
                    val_total += batch_targets.size(0)
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_accuracy = val_correct / val_total
            
            # Update learning rate scheduler
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()
            
            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            # Update scheduler without validation if needed
            if scheduler is not None and scheduler_type != 'plateau':
                scheduler.step()
            
            # No validation, so we can't do early stopping
            epoch_val_loss = None
            epoch_val_accuracy = None
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_accuracy)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        if do_validation:
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(epoch_val_accuracy)
        
        # Log progress
        log_msg = f"Epoch {epoch+1}/{epochs} - train_loss: {epoch_train_loss:.5f}, train_acc: {epoch_train_accuracy:.5f}"
        if do_validation:
            log_msg += f", val_loss: {epoch_val_loss:.5f}, val_acc: {epoch_val_accuracy:.5f}"
        logging.info(log_msg)
        
        # Execute callbacks if provided
        if callbacks:
            for callback in callbacks:
                callback(epoch, history, model)
    
    # Load best model if we did validation
    if do_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Restored best model from epoch with validation loss: {best_val_loss:.5f}")
    
    return model, history


def run_walk_forward_evaluation(
    model: torch.nn.Module,
    data_loader: FinancialDataLoader,
    price_data: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run walk-forward validation of the model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Financial transformer model
    data_loader : FinancialDataLoader
        Data loader instance for creating sequences
    price_data : pd.DataFrame
        Full price and feature data
    config : Dict[str, Any]
        Evaluation configuration
    output_dir : Optional[str]
        Directory to save evaluation results
        
    Returns:
    --------
    Dict[str, Any]
        Evaluation results dictionary
    """
    logging.info("Starting walk-forward evaluation")
    
    # Extract evaluation parameters
    window_size = config.get('window_size', 252)  # Default to 1 year of trading days
    step_size = config.get('step_size', 21)       # Default to 1 month
    seq_length = config.get('seq_length', 60)     # Sequence length for the model
    retrain = config.get('retrain', True)         # Whether to retrain at each step
    retrain_from_scratch = config.get('retrain_from_scratch', False)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make sure the model is on the correct device
    model = model.to(device)
    
    # Create a function to generate sequences from a data window
    def data_loader_fn(df_slice, seq_length):
        """Generate sequences from a dataframe slice"""
        # Extract feature columns, excluding target and metadata
        feature_cols = [col for col in df_slice.columns 
                       if not col.startswith('target_') and 
                       col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Get target column
        target_col = 'direction'  # Binary classification target
        
        # Create sequences
        X, y = [], []
        for i in range(len(df_slice) - seq_length):
            X.append(df_slice[feature_cols].iloc[i:i+seq_length].values)
            y.append(df_slice[target_col].iloc[i+seq_length])
        
        # Convert to tensors
        if X and y:
            X_tensor = torch.FloatTensor(np.array(X))
            y_tensor = torch.FloatTensor(np.array(y))
            indices = df_slice.index[seq_length:seq_length+len(X)]
            return X_tensor, y_tensor, indices
        else:
            # Return empty tensors if no sequences could be created
            return torch.FloatTensor([]), torch.FloatTensor([]), []
    
    # Create a training function for walk-forward validation
    def train_fn(model, X_train, y_train, train_params=None):
        """Train function for walk-forward validation"""
        if train_params is None:
            train_params = {}
        
        # Use our existing train_model function
        trained_model, history = train_model(
            model=model,
            train_data=X_train,
            train_targets=y_train,
            config={
                'batch_size': train_params.get('batch_size', 32),
                'epochs': train_params.get('epochs', 50),
                'learning_rate': train_params.get('learning_rate', 0.001),
                'weight_decay': train_params.get('weight_decay', 1e-5),
                'patience': train_params.get('patience', 10),
                'device': device
            }
        )
        
        return history
    
    # Run walk-forward validation
    evaluation_results = walk_forward_validation(
        model=model,
        data_loader=data_loader_fn,
        price_data=price_data,
        window_size=window_size,
        step_size=step_size,
        seq_length=seq_length,
        retrain=retrain,
        retrain_from_scratch=retrain_from_scratch,
        train_fn=train_fn,
        train_params={
            'batch_size': config.get('batch_size', 32),
            'epochs': config.get('epochs', 50),
            'learning_rate': config.get('learning_rate', 0.001)
        },
        device=device,
        verbose=1
    )
    
    # Calculate financial metrics
    if 'predictions_df' in evaluation_results and not evaluation_results['predictions_df'].empty:
        predictions_df = evaluation_results['predictions_df']
        
        # Ensure we have price data aligned with predictions
        aligned_price_data = price_data.loc[predictions_df.index]
        
        # Calculate financial metrics
        financial_metrics, returns_df = calculate_financial_metrics(
            price_data=aligned_price_data,
            predictions=predictions_df['PredictedBinary'].values,
            transaction_cost=config.get('transaction_cost', 0.001)
        )
        
        # Add financial metrics to results
        evaluation_results['financial_metrics'] = financial_metrics
        evaluation_results['returns_df'] = returns_df
        
        # Log key financial metrics
        logging.info("Financial Performance Metrics:")
        for key, value in financial_metrics.items():
            if isinstance(value, float):
                logging.info(f"  {key}: {value:.5f}")
            else:
                logging.info(f"  {key}: {value}")
    
    # Save evaluation results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions DataFrame
        if 'predictions_df' in evaluation_results and not evaluation_results['predictions_df'].empty:
            evaluation_results['predictions_df'].to_csv(
                os.path.join(output_dir, 'predictions.csv')
            )
        
        # Save returns DataFrame if it exists
        if 'returns_df' in evaluation_results and not evaluation_results['returns_df'].empty:
            evaluation_results['returns_df'].to_csv(
                os.path.join(output_dir, 'returns.csv')
            )
        
        # Save metrics
        metrics_to_save = {}
        
        # Include classification metrics
        if 'aggregate_metrics' in evaluation_results:
            metrics_to_save['classification'] = evaluation_results['aggregate_metrics']
        
        # Include financial metrics
        if 'financial_metrics' in evaluation_results:
            metrics_to_save['financial'] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in evaluation_results['financial_metrics'].items()
                if k != 'returns_df'  # Skip the returns DataFrame
            }
        
        # Save metrics to JSON
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_to_save, f, indent=2, default=str)
        
        # Generate visualizations
        # Create returns plot
        if 'returns_df' in evaluation_results:
            plt.figure(figsize=(12, 6))
            returns_df = evaluation_results['returns_df']
            
            # Plot cumulative returns
            if 'cumulative_market' in returns_df.columns and 'cumulative_net' in returns_df.columns:
                plt.plot(returns_df.index, returns_df['cumulative_market'], 'b-', label='Market')
                plt.plot(returns_df.index, returns_df['cumulative_net'], 'g-', label='Strategy')
                plt.title('Cumulative Returns')
                plt.xlabel('Date')
                plt.ylabel('Return')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'), dpi=300)
                plt.close()
        
        # Create performance visualizations
        plot_performance(
            results=evaluation_results,
            regime_data=None,  # We'll add regime analysis separately
            figsize=(15, 10),
            save_path=os.path.join(output_dir, 'performance.png')
        )
        
        logging.info(f"Saved evaluation results to {output_dir}")
    
    return evaluation_results


def analyze_market_regimes(
    price_data: pd.DataFrame,
    evaluation_results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze model performance across different market regimes.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Full price and feature data
    evaluation_results : Dict[str, Any]
        Results from walk-forward evaluation
    config : Dict[str, Any]
        Regime analysis configuration
    output_dir : Optional[str]
        Directory to save analysis results
        
    Returns:
    --------
    Dict[str, Any]
        Regime analysis results
    """
    logging.info("Starting market regime analysis...")
    
    # Identify market regimes
    regime_data = identify_market_regimes(
        price_data=price_data,
        window=config.get('regime_window', 63),
        volatility_threshold=config.get('volatility_threshold', 0.15),
        trend_threshold=config.get('trend_threshold', 0.1)
    )
    
    # Get predictions from evaluation results
    predictions_df = evaluation_results.get('predictions_df')
    if predictions_df is None or predictions_df.empty:
        raise ValueError("No predictions found in evaluation results")
    
    # Align regime data with predictions
    aligned_regime_data = regime_data.loc[predictions_df.index]
    
    # Initialize results dictionary
    regime_results = {
        'regime_metrics': {},
        'regime_transitions': {},
        'regime_durations': {}
    }
    
    # Calculate metrics for each regime
    for regime in aligned_regime_data['regime'].unique():
        regime_mask = aligned_regime_data['regime'] == regime
        
        if not any(regime_mask):
            continue
            
        # Classification metrics for this regime
        y_true = predictions_df.loc[regime_mask, 'Actual']
        y_pred = predictions_df.loc[regime_mask, 'Predicted']
        
        regime_results['regime_metrics'][regime] = calculate_classification_metrics(
            y_true=y_true.values,
            y_pred=y_pred.values
        )
        
        # Financial metrics for this regime
        regime_price_data = price_data.loc[regime_mask]
        regime_predictions = predictions_df.loc[regime_mask, 'PredictedBinary'].values
        
        financial_metrics, _ = calculate_financial_metrics(
            price_data=regime_price_data,
            predictions=regime_predictions,
            transaction_cost=config.get('transaction_cost', 0.001)
        )
        
        regime_results['regime_metrics'][regime].update(financial_metrics)
        
        # Calculate regime duration statistics
        regime_durations = aligned_regime_data['regime_duration'][regime_mask]
        regime_results['regime_durations'][regime] = {
            'mean': regime_durations.mean(),
            'median': regime_durations.median(),
            'max': regime_durations.max(),
            'min': regime_durations.min()
        }
    
    # Analyze regime transitions
    regime_transitions = pd.crosstab(
        aligned_regime_data['regime'],
        aligned_regime_data['regime'].shift(-1)
    )
    regime_results['regime_transitions'] = regime_transitions.to_dict()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save regime metrics
        regime_metrics_df = pd.DataFrame.from_dict(
            regime_results['regime_metrics'],
            orient='index'
        )
        regime_metrics_df.to_csv(os.path.join(output_dir, 'regime_metrics.csv'))
        
        # Save regime transitions
        pd.DataFrame(regime_results['regime_transitions']).to_csv(
            os.path.join(output_dir, 'regime_transitions.csv')
        )
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Performance by Regime
        plt.subplot(2, 1, 1)
        metrics_to_plot = ['accuracy', 'sharpe_ratio', 'win_rate']
        regime_metrics_df[metrics_to_plot].plot(kind='bar')
        plt.title('Model Performance by Market Regime')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Plot 2: Regime Distribution Over Time
        plt.subplot(2, 1, 2)
        regime_colors = {
            'Volatile Uptrend': 'red',
            'Volatile Downtrend': 'blue',
            'Volatile Range': 'purple',
            'Quiet Uptrend': 'green',
            'Quiet Downtrend': 'orange',
            'Quiet Range': 'gray'
        }
        
        for regime in regime_colors:
            mask = aligned_regime_data['regime'] == regime
            if any(mask):
                plt.fill_between(
                    aligned_regime_data.index,
                    0, 1,
                    where=mask,
                    color=regime_colors[regime],
                    alpha=0.3,
                    label=regime
                )
        
        plt.title('Market Regimes Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regime_analysis.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Saved regime analysis results to {output_dir}")
    
    return regime_results

def analyze_model_attention(
    model: torch.nn.Module,
    data_loader: FinancialDataLoader,
    feature_names: List[str],
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze attention patterns in the transformer model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained financial transformer model
    data_loader : FinancialDataLoader
        Data loader for test samples
    feature_names : List[str]
        Names of input features
    config : Dict[str, Any]
        Attention analysis configuration
    output_dir : Optional[str]
        Directory to save analysis results
        
    Returns:
    --------
    Dict[str, Any]
        Attention analysis results
    """
    logging.info("Starting attention pattern analysis...")
    
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Initialize results dictionary
    attention_results = {
        'average_attention': [],
        'temporal_importance': [],
        'feature_importance': [],
        'attention_clusters': [],
        'layer_wise_patterns': []
    }
    
    # Get test data
    test_data = data_loader.get_test_data()
    if isinstance(test_data, tuple):
        X_test, _ = test_data  # We only need the features for attention analysis
    else:
        X_test = test_data
    
    # Convert to tensor if necessary
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    X_test = X_test.to(device)
    
    # Collect attention weights
    attention_weights = []
    with torch.no_grad():
        for batch_idx in range(0, len(X_test), config.get('batch_size', 32)):
            batch = X_test[batch_idx:batch_idx + config.get('batch_size', 32)]
            
            # Forward pass with attention weights
            _, attn_weights = model(batch, return_attention=True)
            attention_weights.append(attn_weights)
    
    # Concatenate attention weights from all batches
    attention_weights = torch.cat(attention_weights, dim=0)
    
    # 1. Average Attention Analysis
    avg_attention = attention_weights.mean(dim=(0, 1))  # Average across samples and heads
    attention_results['average_attention'] = {
        'weights': avg_attention.cpu().numpy(),
        'feature_importance': dict(zip(feature_names, avg_attention.mean(dim=0).cpu().numpy()))
    }
    
    # 2. Temporal Importance Analysis
    temporal_importance = attention_weights.mean(dim=(0, 1, 3))  # Average across samples, heads, and features
    attention_results['temporal_importance'] = {
        'weights': temporal_importance.cpu().numpy(),
        'most_attended_timesteps': temporal_importance.topk(5).indices.cpu().numpy()
    }
    
    # 3. Feature Importance Analysis
    feature_importance = attention_weights.mean(dim=(0, 1, 2))  # Average across samples, heads, and timesteps
    attention_results['feature_importance'] = {
        'weights': feature_importance.cpu().numpy(),
        'ranked_features': [
            (feature, weight) for feature, weight in 
            sorted(zip(feature_names, feature_importance.cpu().numpy()), 
                  key=lambda x: x[1], reverse=True)
        ]
    }
    
    # 4. Attention Pattern Clustering
    # Reshape attention weights for clustering
    flat_attention = attention_weights.mean(dim=1).cpu().numpy()  # Average across heads
    flat_attention = flat_attention.reshape(flat_attention.shape[0], -1)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(flat_attention, method='ward')
    clusters = fcluster(linkage_matrix, t=config.get('n_clusters', 5), criterion='maxclust')
    
    attention_results['attention_clusters'] = {
        'cluster_assignments': clusters,
        'cluster_centers': [
            flat_attention[clusters == i].mean(axis=0)
            for i in range(1, clusters.max() + 1)
        ]
    }
    
    # 5. Layer-wise Pattern Analysis
    if hasattr(model, 'transformer'):
        n_layers = len(model.transformer.layers)
        layer_patterns = []
        
        for layer_idx in range(n_layers):
            layer_attention = attention_weights[:, :, layer_idx].mean(dim=0)  # Average across samples
            layer_patterns.append({
                'layer_idx': layer_idx,
                'attention_pattern': layer_attention.cpu().numpy(),
                'top_connections': [
                    (i, j, layer_attention[i, j].item())
                    for i, j in zip(*layer_attention.topk(5).indices.cpu().numpy())
                ]
            })
        
        attention_results['layer_wise_patterns'] = layer_patterns
    
    # Save visualizations if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Feature Importance Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            avg_attention.cpu().numpy(),
            xticklabels=feature_names,
            yticklabels=feature_names,
            cmap='viridis'
        )
        plt.title('Average Attention Weights Across Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_attention_heatmap.png'), dpi=300)
        plt.close()
        
        # 2. Temporal Importance Plot
        plt.figure(figsize=(10, 6))
        plt.plot(temporal_importance.cpu().numpy())
        plt.title('Temporal Attention Importance')
        plt.xlabel('Time Step')
        plt.ylabel('Average Attention Weight')
        plt.savefig(os.path.join(output_dir, 'temporal_importance.png'), dpi=300)
        plt.close()
        
        # 3. Feature Importance Bar Plot
        plt.figure(figsize=(12, 6))
        feature_imp_df = pd.DataFrame(
            attention_results['feature_importance']['ranked_features'],
            columns=['Feature', 'Importance']
        )
        sns.barplot(data=feature_imp_df, x='Feature', y='Importance')
        plt.xticks(rotation=45)
        plt.title('Feature Importance Based on Attention Weights')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        
        # 4. Cluster Analysis Visualization
        if attention_results['attention_clusters']['cluster_centers']:
            plt.figure(figsize=(15, 5))
            for i, center in enumerate(attention_results['attention_clusters']['cluster_centers']):
                plt.subplot(1, len(attention_results['attention_clusters']['cluster_centers']), i+1)
                plt.imshow(center.reshape(len(feature_names), -1), cmap='viridis')
                plt.title(f'Cluster {i+1} Pattern')
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'attention_clusters.png'), dpi=300)
            plt.close()
        
        # Save numerical results
        with open(os.path.join(output_dir, 'attention_analysis.json'), 'w') as f:
            json.dump({
                'feature_importance': attention_results['feature_importance']['ranked_features'],
                'temporal_importance': {
                    'most_attended_timesteps': attention_results['temporal_importance']['most_attended_timesteps'].tolist()
                },
                'cluster_info': {
                    'n_clusters': len(attention_results['attention_clusters']['cluster_centers']),
                    'cluster_sizes': np.bincount(clusters)[1:].tolist()
                }
            }, f, indent=2)
        
        logging.info(f"Saved attention analysis results to {output_dir}")
    
    return attention_results

def run_experiment(
    config: Dict[str, Any],
    output_dir: str,
    experiment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a complete experiment from data preparation to evaluation.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Complete experiment configuration
    output_dir : str
        Directory to save experiment results
    experiment_name : Optional[str]
        Name for this experiment run
        
    Returns:
    --------
    Dict[str, Any]
        Complete experiment results
    """
    # Create experiment directory
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # In your run_experiment function, after creating the experiment directory:
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Add this line to create the data directory:
    data_dir = os.path.join(experiment_dir, "data")
    os.makedirs(data_dir, exist_ok=True)  # Create the data directory explicitly
    
    # Set up logging
    log_file = os.path.join(experiment_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log experiment start
    logging.info(f"Starting experiment: {experiment_name}")
    logging.info(f"Configuration: {config}")
    
    # Save configuration
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v 
                 for k, v in config.items()}, f, indent=2)
    
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Step 1: Prepare data
    logging.info("Step 1: Data preparation")
    data_dir = os.path.join(experiment_dir, "data")
    
    # Create data loader
    data_loader = FinancialDataLoader(
        symbols=config['symbols'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        seq_length=config['seq_length'],
        target_horizon=config.get('target_horizon', 1),
        data_dir=config.get('data_dir', 'data/raw'),
        use_cache=config.get('use_cache', True)
    )
    
    # Load data
    raw_data = data_loader.load_data()
    logging.info(f"Loaded data for {len(raw_data)} symbols")
    
    # Calculate returns
    data_loader.calculate_returns()
    
    # For simplicity, use the first symbol if multiple are provided
    if len(config['symbols']) > 1:
        logging.warning(f"Multiple symbols provided. Using {config['symbols'][0]} for analysis.")
    
    symbol = config['symbols'][0]
    df = data_loader.raw_data[symbol]
    
    # Process each symbol individually and create the correct data structure
    processed_data = {}

    # IMPORTANT: Create a feature template that will work across all symbols first
    feature_template = [
        # Price-based features
        'log_return',                   # Current day's return
        
        
        # Trend indicators
        'sma_10',
        "sma_30",
        
        'macd', 
        "macd_signal",
        "macd_diff"                                                                                              # Moving Average Convergence/Divergence
        'adx',                          # Average Directional Index
        
        # Momentum indicators
        'rsi',                          # Relative Strength Index
        'stoch_k',
        "stoch_d",                                            # Stochastic Oscillator K
        'williams_r',                   # Williams %R
        
        # Volatility indicators
        'atr_pct',                      # Average True Range (percentage)
        
        'hist_volatility',              # Historical Volatility
        
        # Volume indicators
        'obv',                          # On-Balance Volume
        'rel_volume',                   # Relative Volume

        "roc_5",
        "roc_20",
        "keltner_hband",
        "keltner_lband",
        "keltner_width_pct"
    ]

    

    # Add log returns and direction columns directly to the DataFrames
    for sym in config['symbols']:
        if sym in data_loader.raw_data:
            # Get the raw data for this symbol
            df = data_loader.raw_data[sym]
            
            # Add technical indicators using your technical.py module
            from features.techincal import add_technical_features
            enhanced_df = add_technical_features(
                df, 
                categories=['trend', 'momentum', 'volatility', 'volume']
            )
            
            # Verify all required features exist
            missing_features = [f for f in feature_template if f not in enhanced_df.columns]
            if missing_features:
                logging.warning(f"Missing features for {sym}: {missing_features}")
            
            # Replace with enhanced dataframe
            data_loader.raw_data[sym] = enhanced_df
            
            # Add direction column with the correct format
            close_col = 'Close'
            direction_col = 'direction'
            log_return_col = 'log_return'
            
            enhanced_df[direction_col] = (enhanced_df[close_col].shift(-1) > enhanced_df[close_col]).astype(int)
            if log_return_col not in enhanced_df.columns:
                enhanced_df[log_return_col] = np.log(enhanced_df[close_col] / enhanced_df[close_col].shift(1))

    # Set up metadata for model creation - use the number of features for input dimension
    metadata = {
        'feature_names': feature_template,
        'num_features': len(feature_template),
        'symbol': config['symbols'][0]
    }

    logging.info(f"Using feature template: {feature_template}")

    # Make sure processed_data is set correctly
    data_loader.processed_data = data_loader.raw_data

    # Save processed data
    # No need to save at this point since we're working directly with data_loader's data

    # Step 2: Create model
    logging.info("Step 2: Model creation")
    input_dim = metadata['num_features']

    from models.transformer import FinancialTransformer
    model = FinancialTransformer(
        input_dim=input_dim,
        d_model=config['hidden_size'],
        n_head=config.get('num_heads', 8),
        n_layers=config['num_layers'],
        d_ff=config['hidden_size'] * 4,
        seq_length=config['seq_length'],
        dropout=config['dropout']
    )

    logging.info(f"Created model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

    # CRITICAL: Verify feature_names before calling prepare_sequences
    logging.info(f"Feature template before prepare_sequences: {metadata['feature_names']}")
    if not metadata['feature_names']:
        raise ValueError("Feature template list became empty before prepare_sequences!")

    # Step 3: Train the model
    logging.info("Step 3: Model training")

    # Prepare sequences - passing the feature template instead of specific column tuples
    X, y = data_loader.prepare_sequences(
        features=feature_template,  # Now passing just the feature names
        target='direction',
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15)
    )
        
    
    # Train the model
    from models.transformer import FinancialTransformer, train_financial_transformer
    
    
    # Create model
    model = FinancialTransformer(
        input_dim=input_dim,
        d_model=config['hidden_size'],
        n_head=config.get('num_heads', 8),
        n_layers=config['num_layers'],
        d_ff=config['hidden_size'] * 4,
        seq_length=config['seq_length'],
        dropout=config['dropout']
    )

    # Add diagnostic logging
    logging.info(f"Type of X['train']: {type(X['train'])}")
    logging.info(f"Type of y['train']: {type(y['train'])}")

    

    # Convert to tensors if needed
    X_train = torch.FloatTensor(X['train']) if isinstance(X['train'], np.ndarray) else X['train']
    y_train = torch.FloatTensor(y['train']) if isinstance(y['train'], np.ndarray) else y['train']

    logging.info(f"Raw X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Fix the batch dimension - CRITICAL!
    if len(X_train.shape) == 3 and len(y_train.shape) == 1:
        # This is the correct format: [batch, seq, features] and [batch]
        logging.info(f"Tensor shapes look correct: X={X_train.shape}, y={y_train.shape}")
    else:
        logging.error(f"Unexpected tensor shapes: X={X_train.shape}, y={y_train.shape}")
        # If shapes are incorrect, try basic fixes
        if len(y_train.shape) == 0:  # Scalar
            y_train = y_train.unsqueeze(0)  # Add batch dimension



    # Check for NaNs and fix them
    if torch.isnan(X_train).any():
        logging.warning(f"Found {torch.isnan(X_train).sum().item()} NaN values in X_train. Replacing with zeros.")
        X_train = torch.nan_to_num(X_train, nan=0.0)

    if torch.isnan(y_train).any():
        logging.warning(f"Found {torch.isnan(y_train).sum().item()} NaN values in y_train. Replacing with zeros.")
        y_train = torch.nan_to_num(y_train, nan=0.0)

    # Now log that we fixed the NaNs
    logging.info(f"NaN check after fixing - X has NaNs: {torch.isnan(X_train).any().item()}, y has NaNs: {torch.isnan(y_train).any().item()}")

    # Add shape validation
    if X_train.shape[0] != y_train.shape[0]:
        logging.error(f"Batch dimension mismatch: X has {X_train.shape[0]} samples but y has {y_train.shape[0]}")
        raise ValueError("Batch dimension mismatch between X and y")

    
   

    # Check shapes again
    logging.info(f"Final X_train shape: {X['train'].shape}")
    
    # Then update the train_dataset creation:
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    logging.info(f"Creating DataLoader with batch_size={config.get('batch_size', 32)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 32),
        shuffle=False
    )
    test_batch = next(iter(train_loader))
    logging.info(f"Test batch shapes: X={test_batch[0].shape}, y={test_batch[1].shape}")

    # Handle validation data
    val_loader = None
    if 'val' in X and 'val' in y:
        # Add diagnostic logging for validation data
        logging.info(f"Type of X['val']: {type(X['val'])}")
        logging.info(f"Type of y['val']: {type(y['val'])}")
        
        # Convert validation data if needed
        if isinstance(X['val'], np.ndarray):
            X['val'] = torch.FloatTensor(X['val'])
            logging.info("Converted X['val'] from numpy array to tensor")
        
        if isinstance(y['val'], np.ndarray):
            y['val'] = torch.FloatTensor(y['val'])
            logging.info("Converted y['val'] from numpy array to tensor")
        elif isinstance(y['val'], int):
            y['val'] = torch.FloatTensor([y['val']])
            logging.info("Converted y['val'] from integer to tensor")
        
        # Create validation DataLoader
        try:
            val_dataset = torch.utils.data.TensorDataset(X['val'], y['val'])
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=config.get('batch_size', 32), shuffle=False
            )
            logging.info("Successfully created validation DataLoader")
        except Exception as e:
            logging.error(f"Error creating validation DataLoader: {e}")
            logging.warning("Proceeding with training without validation")
            # In this case, we'll just continue without validation rather than failing completely
            val_loader = None
    else:
        logging.info("No validation data provided")


    if torch.isnan(X_train).any() or torch.isnan(y_train).any():
        logging.error("NaN values found in training data. Check preprocessing.")
        raise ValueError("Training data contains NaN values")
        
    logging.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Training data ranges: X=[{X_train.min():.4f}, {X_train.max():.4f}], y=[{y_train.min():.4f}, {y_train.max():.4f}]")

    # Train the model using the standalone function
    model, history = train_financial_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('epochs', 50),
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5),
        patience=config.get('patience', 10)
    )
    # Save model
    model_dir = os.path.join(experiment_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) 
                  else v for k, v in config.items()}
    }, os.path.join(model_dir, "model.pth"))
    
    # Step 4: Evaluate with walk-forward validation
    logging.info("Step 4: Walk-forward evaluation")
    evaluation_dir = os.path.join(experiment_dir, "evaluation")
    
    
    # Step 4: Evaluate with walk-forward validation
    logging.info("Step 4: Walk-forward validation")
    evaluation_dir = os.path.join(experiment_dir, "evaluation")

    # Get the price data for the first symbol (since we're focusing on one symbol)
    symbol = config['symbols'][0]
    price_data = data_loader.raw_data[symbol]


    def sequence_generator(train_slice, seq_length, start_index=None):
        """Custom adapter for walk-forward validation that creates sequences consistently."""
         # Check if train_slice is big enough - with BETTER ERROR HANDLING
        min_required_size = seq_length + 1
        if len(train_slice) < min_required_size:
            logging.warning(f"Insufficient data ({len(train_slice)} rows) to create sequences. Need at least {min_required_size} rows.")
            # Return empty tensors with correct dimensions
            empty_x = torch.zeros((0, seq_length, len(feature_template)), dtype=torch.float32)
            empty_y = torch.zeros((0,), dtype=torch.float32)
            return empty_x, empty_y
        
        # Extract ALL features at once - this is the key fix
        feature_df = train_slice[feature_template].copy()
        
        # Fill NaNs in feature data
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Get target column - must be single column
        target_col = 'direction'
        if target_col not in train_slice.columns:
            logging.warning(f"Target column '{target_col}' not found. Using zeros.")
            target_data = np.zeros(len(train_slice))
        else:
            target_data = train_slice[target_col].fillna(0).values
        
        # Create sequences - similar to _create_sequences method
        X, y = [], []
        for i in range(len(feature_df) - seq_length):
            X.append(feature_df.iloc[i:i+seq_length].values)  # Get all features for sequence
            y.append(target_data[i+seq_length])
        
        # Handle empty sequence case
        if not X:
            logging.warning("No sequences created")
            empty_x = torch.zeros((0, seq_length, len(feature_template)), dtype=torch.float32)
            empty_y = torch.zeros((0,), dtype=torch.float32)
            return empty_x, empty_y
        
        # Convert to numpy arrays then tensors
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)
        
        # Log shapes for debugging
        logging.info(f"Sequence generator created data with shapes - X: {X_np.shape}, y: {y_np.shape}")
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        
        return X_tensor, y_tensor
    
    from Evaluation.evaluation import walk_forward_validation
    evaluation_results = walk_forward_validation(
        model=model,
        data_loader=sequence_generator,  # Pass the sequence preparation function
        price_data=price_data,  # Pass the raw price data
        window_size=config.get('window_size', 504),
        step_size=config.get('step_size', 21),
        seq_length=config['seq_length'],
        retrain=config.get('retrain', True)
    )

    # Save evaluation results if we have an output directory
    if evaluation_dir:
        os.makedirs(evaluation_dir, exist_ok=True)
        if isinstance(evaluation_results, dict):
            if 'predictions_df' in evaluation_results and not evaluation_results['predictions_df'].empty:
                evaluation_results['predictions_df'].to_csv(os.path.join(evaluation_dir, 'predictions.csv'))
            if 'aggregate_metrics' in evaluation_results:
                with open(os.path.join(evaluation_dir, 'metrics.json'), 'w') as f:
                    json.dump(evaluation_results['aggregate_metrics'], f, indent=2)

    
    # Step 5: Optional - Market regime analysis
    if config.get('regime_window'):
        logging.info("Step 5: Market regime analysis")
        regime_dir = os.path.join(experiment_dir, "regimes")
        os.makedirs(regime_dir, exist_ok=True)
        
        from Evaluation.evaluation import identify_market_regimes
        regime_data = identify_market_regimes(
            price_data=price_data,
            window=config.get('regime_window', 63),
            volatility_threshold=config.get('volatility_threshold', 0.15),
            trend_threshold=config.get('trend_threshold', 0.1),
            symbols = config.get('symbols', None)
        )
        
        # Save regime data
        regime_data.to_csv(os.path.join(regime_dir, "regime_data.csv"))
        
        logging.info(f"Identified {regime_data['regime'].nunique()} market regimes")
    
    # Step 6: Optional - Attention pattern analysis
    if config.get('analyze_attention'):
        logging.info("Step 6: Attention pattern analysis")
        attention_dir = os.path.join(experiment_dir, "attention")
        os.makedirs(attention_dir, exist_ok=True)
        
        # Create a test dataset
        test_dataset = torch.utils.data.TensorDataset(X.get('test', X['train'][:10]), 
                                                   y.get('test', y['train'][:10]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        from Evaluation.evaluation import analyze_attention_patterns
        attention_results = analyze_attention_patterns(
            model=model,
            data_loader=test_loader,
            feature_names=metadata['feature_names'],
            n_samples=config.get('num_samples', 5)
        )
        
        # Save feature importance
        if hasattr(attention_results, 'feature_importance') and not attention_results.feature_importance.empty:
            attention_results.feature_importance.to_csv(
                os.path.join(attention_dir, "feature_importance.csv")
            )
            
            logging.info("Completed attention pattern analysis")
    
        logging.info(f"Experiment completed successfully. Results saved to {experiment_dir}")
        

    ## After completing the walk-forward validation, add visualization code
    # After completing the walk-forward validation, add visualization code
    # After completing the walk-forward validation, add visualization code
    if evaluation_results and isinstance(evaluation_results, dict):
        # Create directory for plots
        plots_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Create our own simple plots with dynamic column detection
        if 'predictions_df' in evaluation_results and not evaluation_results['predictions_df'].empty:
            try:
                predictions_df = evaluation_results['predictions_df']
                
                # Log the available columns for debugging
                logging.info(f"Available columns in predictions_df: {predictions_df.columns.tolist()}")
                
                # Try to determine the actual and predicted columns
                # Common naming patterns
                actual_candidates = ['Actual', 'actual', 'y_true', 'true', 'target', 'Target']
                predicted_candidates = ['Predicted', 'predicted', 'y_pred', 'pred', 'prediction', 'Prediction']
                
                # Find the first matching column for actual values
                actual_col = None
                for col in actual_candidates:
                    if col in predictions_df.columns:
                        actual_col = col
                        break
                
                # Find the first matching column for predicted values
                predicted_col = None
                for col in predicted_candidates:
                    if col in predictions_df.columns:
                        predicted_col = col
                        break
                
                # Only plot if we found both columns
                if actual_col and predicted_col:
                    plt.figure(figsize=(12, 6))
                    plt.plot(predictions_df.index, predictions_df[actual_col], 'b-', label='Actual')
                    plt.plot(predictions_df.index, predictions_df[predicted_col], 'r-', alpha=0.7, label='Predicted')
                    plt.title('Actual vs. Predicted Values')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'), dpi=300)
                    plt.close()
                    logging.info(f"Actual vs. Predicted plot saved using columns {actual_col} and {predicted_col}")
                else:
                    # If we couldn't find the expected columns, plot all numeric columns
                    plt.figure(figsize=(12, 6))
                    for col in predictions_df.select_dtypes(include=['number']).columns:
                        plt.plot(predictions_df.index, predictions_df[col], label=col)
                    plt.title('All Numeric Columns')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(plots_dir, 'all_columns.png'), dpi=300)
                    plt.close()
                    logging.info(f"Generic plot of all numeric columns saved to {plots_dir}")
                
                # Find binary columns for confusion matrix
                binary_cols = []
                for col in predictions_df.select_dtypes(include=['number']).columns:
                    unique_vals = predictions_df[col].unique()
                    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                        binary_cols.append(col)
                
                # If we have at least two binary columns, try to plot confusion matrix
                if len(binary_cols) >= 2:
                    from sklearn.metrics import confusion_matrix
                    import seaborn as sns
                    
                    # Use the first two binary columns
                    actual_binary = binary_cols[0]
                    predicted_binary = binary_cols[1]
                    
                    cm = confusion_matrix(predictions_df[actual_binary], predictions_df[predicted_binary])
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
                    plt.title(f'Confusion Matrix ({actual_binary} vs {predicted_binary})')
                    plt.xlabel(f'Predicted ({predicted_binary})')
                    plt.ylabel(f'Actual ({actual_binary})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300)
                    plt.close()
                    logging.info(f"Confusion matrix saved using columns {actual_binary} and {predicted_binary}")
                
            except Exception as e:
                logging.error(f"Error generating custom performance plots: {str(e)}")
                traceback.print_exc()  # Print full traceback for debugging



    return {
        'model': model,
        'history': history,
        'evaluation': evaluation_results
    }

    



def main() -> None:
    """
    Main entry point for the financial transformer application.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Convert arguments to configuration dictionary
    config = vars(args)
    
    # Run the experiment
    try:
        run_experiment(
            config=config,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name if hasattr(args, 'experiment_name') else None
        )
        sys.exit(0)
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


