import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Callable, Dict, List, Tuple, Union, Optional
import copy
import gc
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate classification metrics for directional predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True directional labels (0 for down, 1 for up)
    y_pred : np.ndarray
        Predicted probabilities or binary predictions
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of metrics including accuracy, precision, recall, F1
    """
    # Ensure y_pred is binary
    if y_pred.ndim > 1 or np.any((y_pred > 0) & (y_pred < 1)):
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    
    # Handle potential division by zero or undefined metrics
    try:
        metrics['precision'] = precision_score(y_true, y_pred_binary)
    except:
        metrics['precision'] = np.nan
        
    try:
        metrics['recall'] = recall_score(y_true, y_pred_binary)
    except:
        metrics['recall'] = np.nan
        
    try:
        metrics['f1'] = f1_score(y_true, y_pred_binary)
    except:
        metrics['f1'] = np.nan
    
    # Additional metrics for imbalanced data
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred_binary)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred_binary)
    
    # ROC AUC if y_pred contains probabilities
    if np.any((y_pred > 0) & (y_pred < 1)):
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['roc_auc'] = np.nan
    
    return metrics

def calculate_financial_metrics(price_data: pd.DataFrame, predictions: np.ndarray, transaction_cost: float = 0.001) -> dict:
    """
    Calculate financial performance metrics based on directional predictions.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        DataFrame with price data (must have 'Close' column)
    predictions : np.ndarray
        Binary directional predictions (0 for down, 1 for up)
    transaction_cost : float
        Transaction cost as a fraction of trade value
        
    Returns:
    --------
    dict
        Dictionary of financial metrics including returns, Sharpe ratio, drawdown
    """
    # Ensure price_data and predictions are aligned
    if len(price_data) != len(predictions) + 1:
        raise ValueError("predictions should have one fewer element than price_data (can't predict the first day)")
    
    # Calculate market returns
    market_returns = price_data['Close'].pct_change().dropna()
    
    # Convert predictions to positions (1=long, 0=flat or -1=short)
    positions = predictions.copy()  # Assuming 1=long, 0=flat
    
    # Calculate strategy returns (position * next day's return)
    strategy_returns = positions[:-1] * market_returns[1:]  # Align positions with next day's returns
    
    # Calculate transaction costs
    position_changes = np.diff(np.concatenate([[0], positions]))  # Include initial position
    transaction_costs = np.abs(position_changes) * transaction_cost
    
    # Net returns after costs
    net_returns = strategy_returns - transaction_costs[:-1]  # Exclude the last position change
    
    # Convert to DataFrame for easier calculations
    returns_df = pd.DataFrame({
        'market_returns': market_returns,
        'strategy_returns': pd.Series(strategy_returns.tolist() + [0]),  # Add placeholder for alignment
        'net_returns': pd.Series(net_returns.tolist() + [0])  # Add placeholder for alignment
    })
    
    # Cumulative returns
    returns_df['cumulative_market'] = (1 + returns_df['market_returns']).cumprod() - 1
    returns_df['cumulative_strategy'] = (1 + returns_df['strategy_returns']).cumprod() - 1
    returns_df['cumulative_net'] = (1 + returns_df['net_returns']).cumprod() - 1
    
    # Performance metrics
    # Sharpe ratio (annualized)
    sharpe_ratio = np.sqrt(252) * net_returns.mean() / net_returns.std()
    
    # Maximum drawdown
    cumulative_returns = (1 + net_returns).cumprod() - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / (1 + peak)  # Relative drawdown
    max_drawdown = np.max(drawdown)
    
    # Total return
    total_return = ((1 + net_returns).prod() - 1) if len(net_returns) > 0 else 0
    
    # Annualized return
    days = len(net_returns)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.nan
    
    # Win rate
    win_rate = np.mean(net_returns > 0) if len(net_returns) > 0 else 0
    
    # Metrics dictionary
    metrics = {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "win_rate": win_rate,
        # Add comparison to market
        "market_return": returns_df['cumulative_market'].iloc[-1],
        "excess_return": total_return - returns_df['cumulative_market'].iloc[-1]
    }
    
    return metrics, returns_df


def walk_forward_validation(
    model: nn.Module,
    data_loader: Callable[[pd.DataFrame, int, int], Tuple[torch.Tensor, torch.Tensor]],
    price_data: pd.DataFrame,
    window_size: int = 252,
    step_size: int = 21,
    seq_length: int = 60,
    retrain: bool = True
) -> Dict[str, Union[List, pd.DataFrame]]:
    """
    Perform walk-forward validation of the model.
    
    Parameters:
    -----------
    model : nn.Module
        Transformer model to evaluate
    data_loader : Callable
        Function to prepare data sequences from a given window
    price_data : pd.DataFrame
        Full price and feature data
    window_size : int
        Size of the training window in days
    step_size : int
        Number of days to step forward in each iteration
    seq_length : int
        Sequence length for the transformer input
    retrain : bool
        Whether to retrain the model at each step
        
    Returns:
    --------
    Dict[str, Union[List, pd.DataFrame]]
        Results including predictions, metrics for each window, and aggregated performance
    """
    
    
    # Input validation
    if not isinstance(price_data, pd.DataFrame):
        raise TypeError("price_data must be a pandas DataFrame.")
    if window_size <= seq_length:
        raise ValueError(f"window_size ({window_size}) must be greater than seq_length ({seq_length}).")
    if len(price_data) < window_size + step_size:
        raise ValueError(f"Insufficient data. Total length ({len(price_data)}) is less than window_size + step_size.")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # For tracking predictions and metrics
    all_predictions = []
    all_true_values = []
    all_indices = []
    step_metrics = []
    
    # Initial model state (for retraining from scratch if needed)
    initial_state = copy.deepcopy(model.state_dict()) if retrain else None
    
    # Calculate total steps
    total_data_points = len(price_data)
    start_idx = window_size  # First prediction starts after initial window
    
    # Create steps for walk-forward process
    steps = []
    for i in range(start_idx, total_data_points, step_size):
        end_idx = min(i + step_size, total_data_points)
        steps.append((i, end_idx))
    
    print(f"Walk-forward validation: {len(steps)} steps")
    
    # Walk-forward loop
    for step_idx, (pred_start, pred_end) in enumerate(tqdm(steps)):
        # Define training window (ends right before prediction starts)
        train_end = pred_start
        train_start = max(0, train_end - window_size)
        
        # Training data slice
        train_slice = price_data.iloc[train_start:train_end]
        
        # Prediction data slice
        pred_slice = price_data.iloc[pred_start:pred_end]
        
        # Train/retrain model if required
        if retrain or step_idx == 0:  # Always train on first step
            if retrain and step_idx > 0:
                # Reset model to initial state for consistent training
                model.load_state_dict(initial_state)
            
            # Prepare training data
            X_train, y_train = data_loader(train_slice, seq_length, train_start)
            
            # Ensure data is on the correct device
            X_train, y_train = X_train.to(device), y_train.to(device)
            
            # Here we would train the model - but your signature doesn't include a train_fn
            # Placeholder for actual training code (would need to be provided separately)
            # In a real implementation, you'd either include a train_fn parameter or
            # implement the training loop here
            model.train()
            # ... training code would go here ...
            model.eval()
        
        # Skip prediction if slice is empty
        if pred_slice.empty:
            continue
        
        # Prepare prediction data
        X_pred, y_true = data_loader(pred_slice, seq_length, pred_start)
        
        # No predictions possible
        if X_pred.shape[0] == 0:
            continue
        
        # Make predictions
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_pred), 32):  # Process in batches of 32
                batch_X = X_pred[i:i+32].to(device)
                
                # Use model the same way as in training
                outputs = model(batch_X)
                
                # Handle different output formats consistently
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # First element contains predictions
                    
                # Apply threshold consistently with training
                batch_preds = outputs.cpu().numpy()
                
                # Store raw predictions (not thresholded yet)
                predictions.append(batch_preds)
        
        if predictions:
            # Combine batch predictions
            predictions = np.concatenate(predictions)
            
            # Store results
            all_predictions.append(predictions)
            all_true_values.append(y_true.cpu().numpy())
            all_indices.append(pred_slice.index[:len(predictions)])
            
            # Calculate metrics for this step
            step_metric = calculate_classification_metrics(
                y_true=y_true.cpu().numpy(), 
                y_pred=predictions
            )
            step_metrics.append(step_metric)
        
        # Clean up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # Combine results
    if not all_predictions:
        return {
            'predictions_df': pd.DataFrame(),
            'step_metrics': step_metrics,
            'aggregate_metrics': {}
        }
    
    predictions = np.concatenate(all_predictions)
    true_values = np.concatenate(all_true_values)
    indices = np.concatenate(all_indices)
    
    # Before creating the DataFrame
    if predictions.ndim > 1:
        predictions = predictions.flatten()  # or predictions.squeeze()
    if true_values.ndim > 1:
        true_values = true_values.flatten()  # or actuals.squeeze()
    # Create DataFrame with results
        predictions_df = pd.DataFrame({
        'Actual': true_values,
        'Predicted': predictions,
        'PredictedBinary': (predictions > 0.5).astype(int)
    }, index=indices)
    
    # Calculate aggregate metrics
    aggregate_metrics = calculate_classification_metrics(
        y_true=true_values, 
        y_pred=predictions
    )
    
    return {
        'predictions_df': predictions_df,
        'step_metrics': step_metrics,
        'aggregate_metrics': aggregate_metrics
    }   


    
def analyze_attention_patterns(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader,
    feature_names: List[str], 
    n_samples: int = 5
) -> Dict:
    """
    Analyze attention patterns to interpret model behavior.
    
    Parameters:
    -----------
    model : nn.Module
        Trained transformer model
    data_loader : DataLoader
        DataLoader with test data
    feature_names : List[str]
        Names of input features
    n_samples : int
        Number of samples to analyze
        
    Returns:
    --------
    Dict
        Analysis results including attention weights and feature importance
    """
    
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Storage for attention data
    attention_weights = []
    samples = []
    
    # Collect attention patterns from n_samples batches
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            if i >= n_samples:
                break
                
            # Forward pass through model
            X = X.to(device)
            outputs, attentions = model(X)  # Assuming model returns (outputs, attentions)
            
            # Store sample data and attention weights
            samples.append((X.cpu().numpy(), y.numpy()))
            
            # Process attention weights based on their structure
            if isinstance(attentions, list):
                # Multiple layers of attention
                for layer_idx, layer_attention in enumerate(attentions):
                    # layer_attention shape: [batch_size, n_heads, seq_len, seq_len]
                    attention_weights.append({
                        'sample_idx': i,
                        'layer_idx': layer_idx,
                        'weights': layer_attention.cpu().numpy()
                    })
            else:
                # Single attention layer
                attention_weights.append({
                    'sample_idx': i,
                    'layer_idx': 0,
                    'weights': attentions.cpu().numpy()
                })
    
    # If no attention weights were collected, return empty results
    if not attention_weights:
        return {
            'attention_weights': [],
            'feature_importance': pd.DataFrame(),
            'temporal_importance': pd.DataFrame(),
            'visualizations': {}
        }
    
    # Extract information from collected attention weights
    
    # 1. Calculate feature importance based on attention
    feature_importance_scores = []
    
    for att_data in attention_weights:
        weights = att_data['weights']
        layer_idx = att_data['layer_idx']
        
        # For last layer only
        if 'layer_idx' not in locals() or layer_idx == max(att['layer_idx'] for att in attention_weights):
            # Get attention from last position (what the prediction is based on)
            # Shape: [batch_size, n_heads, 1, seq_len]
            last_token_attention = weights[:, :, -1, :]
            
            # Average across batches and heads
            avg_attention = np.mean(last_token_attention, axis=(0, 1))
            
            feature_importance_scores.append(avg_attention)
    
    if feature_importance_scores:
        # Average across samples
        avg_feature_importance = np.mean(feature_importance_scores, axis=0)
        
        # Create DataFrame with feature importance
        if len(avg_feature_importance) == len(feature_names):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': avg_feature_importance
            }).sort_values('Importance', ascending=False)
        else:
            # Handle case where dimensions don't match
            seq_length = avg_feature_importance.shape[0]
            feature_importance = pd.DataFrame({
                'Position': [f't-{seq_length-i}' for i in range(seq_length)],
                'Importance': avg_feature_importance
            }).sort_values('Importance', ascending=False)
    else:
        feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
    
    # 2. Analyze temporal patterns
    # How attention changes across the sequence
    temporal_patterns = []
    
    for att_data in attention_weights:
        weights = att_data['weights']
        
        # Average across batch and heads, then sum across keys
        # This gives importance of each position as a query
        temporal_importance = np.mean(np.sum(weights, axis=3), axis=(0, 1))
        temporal_patterns.append(temporal_importance)
    
    if temporal_patterns:
        avg_temporal = np.mean(temporal_patterns, axis=0)
        temporal_df = pd.DataFrame({
            'Position': [f't-{len(avg_temporal)-i}' for i in range(len(avg_temporal))],
            'Importance': avg_temporal
        })
    else:
        temporal_df = pd.DataFrame(columns=['Position', 'Importance'])
    
    # 3. Create visualizations
    visualizations = {}
    
    try:
        # 3.1. Feature importance visualization
        if not feature_importance.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            ax.set_title('Feature Importance Based on Attention')
            visualizations['feature_importance'] = fig
        
        # 3.2. Temporal importance
        if not temporal_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x='Position', y='Importance', data=temporal_df, ax=ax)
            ax.set_title('Attention Importance by Position')
            visualizations['temporal_importance'] = fig
        
        # 3.3. Attention heatmap for first sample
        if attention_weights:
            # Get the last layer's attention
            last_layer_idx = max(att['layer_idx'] for att in attention_weights)
            last_layer_weights = [att for att in attention_weights if att['layer_idx'] == last_layer_idx][0]['weights']
            
            # Average across heads
            avg_attention = np.mean(last_layer_weights[0], axis=0)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(avg_attention, cmap='viridis', ax=ax)
            ax.set_title('Attention Heatmap (Average Across Heads)')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            visualizations['attention_heatmap'] = fig
    except Exception as e:
        print(f"Warning: Failed to generate visualizations: {e}")
    
    # 4. Create analysis summary
    analysis_results = {
        'attention_weights': attention_weights,
        'feature_importance': feature_importance,
        'temporal_importance': temporal_df,
        'visualizations': visualizations
    }
    
    return analysis_results


def identify_market_regimes(
    price_data: pd.DataFrame,
    window: int = 63,
    volatility_threshold: float = 0.15,
    trend_threshold: float = 0.1,
    symbols: List[str] = None
) -> pd.DataFrame:
    """
    Identify market regimes based on volatility and trend.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        DataFrame with price data
    window : int
        Window for calculating volatility and trend
    volatility_threshold : float
        Annualized volatility threshold for high/low volatility
    trend_threshold : float
        Absolute return threshold for trending/mean-reverting
        
    Returns:
    --------
    pd.DataFrame
        Original data with regime labels
    """
    # Create a copy to avoid modifying original data
    df = price_data.copy()

    if not symbols:
        price_col = 'Close'
        if price_col not in df.columns:
            # Try to find any Close column
            close_cols = [col for col in df.columns if 'Close' in col or 'close' in col]
            if close_cols:
                price_col = close_cols[0]
                print(f"Using '{price_col}' as price column")
            else:
                raise ValueError("Could not find a suitable 'Close' column")
    else:
        # Use the first symbol to determine price column 
        symbol = symbols[0]
        price_col = f'Close_{symbol}'
        
        # Verify column exists
        if price_col not in df.columns:
            print(f"Warning: Column '{price_col}' not found. Available columns: {df.columns.tolist()[:5]}...")
            # Try to find any Close column for this symbol
            close_cols = [col for col in df.columns if ('Close' in col or 'close' in col) and symbol in col]
            if close_cols:
                price_col = close_cols[0]
                print(f"Using '{price_col}' as price column")
            else:
                raise ValueError(f"Could not find a suitable 'Close' column for symbol {symbol}")
    
    # Calculate log returns using the identified price column
    print(f"Using '{price_col}' for market regime calculations")
    df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Continue with the rest of the function as before
    df['volatility'] = df['log_returns'].rolling(window=window).std() * np.sqrt(252)
    df['trend'] = df[price_col].rolling(window=window).apply(
        lambda x: (x[-1]/x[0] - 1)
    )
    # Initialize regime column
    df['regime'] = 'Unknown'
    
    # Identify regimes
    # High volatility, Strong trend up
    df.loc[(df['volatility'] >= volatility_threshold) & 
           (df['trend'] >= trend_threshold), 'regime'] = 'Volatile Uptrend'
    
    # High volatility, Strong trend down
    df.loc[(df['volatility'] >= volatility_threshold) & 
           (df['trend'] <= -trend_threshold), 'regime'] = 'Volatile Downtrend'
    
    # High volatility, Range bound
    df.loc[(df['volatility'] >= volatility_threshold) & 
           (df['trend'].abs() < trend_threshold), 'regime'] = 'Volatile Range'
    
    # Low volatility, Strong trend up
    df.loc[(df['volatility'] < volatility_threshold) & 
           (df['trend'] >= trend_threshold), 'regime'] = 'Quiet Uptrend'
    
    # Low volatility, Strong trend down
    df.loc[(df['volatility'] < volatility_threshold) & 
           (df['trend'] <= -trend_threshold), 'regime'] = 'Quiet Downtrend'
    
    # Low volatility, Range bound
    df.loc[(df['volatility'] < volatility_threshold) & 
           (df['trend'].abs() < trend_threshold), 'regime'] = 'Quiet Range'
    
    # Add regime change indicator
    df['regime_change'] = (df['regime'] != df['regime'].shift(1)).astype(int)
    
    # Calculate regime duration
    df['regime_duration'] = (df['regime'] == df['regime'].shift(1)).astype(int).cumsum()
    
    # Add regime metrics
    df['regime_volatility'] = df.groupby('regime_duration')['volatility'].transform('mean')
    df['regime_return'] = df.groupby('regime_duration')['log_returns'].transform('sum')
    
    
    df = df.dropna()
    # Or alternatively: df = df.fillna(method='bfill')

    return df   
    


def plot_performance(
    results: Dict[str, Union[List, pd.DataFrame]],
    regime_data: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (20, 16),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize model performance and regime-specific analysis.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from walk-forward validation containing:
        - 'predictions_df': DataFrame with predictions and true values
        - 'step_metrics': List of metrics for each validation step
        - 'aggregate_metrics': Overall metrics
    regime_data : pd.DataFrame, optional
        DataFrame with regime labels from identify_market_regimes function
    figsize : Tuple[int, int]
        Figure size for the plots
    save_path : str, optional
        Path to save the figure, if None the figure is displayed only
    """
    
    # Ensure results exist
    if 'predictions_df' not in results or results['predictions_df'].empty:
        print("No prediction results to display.")
        return
    
    predictions_df = results['predictions_df']
    
    # Create figure and gridspec for layout
    plt.figure(figsize=figsize)
    gs = GridSpec(4, 2)
    
    # Ensure predictions_df has datetime index
    if not isinstance(predictions_df.index, pd.DatetimeIndex):
        try:
            predictions_df.index = pd.to_datetime(predictions_df.index)
        except:
            print("Warning: Could not convert index to datetime. Some plots may not display correctly.")
    
    # 1. Prediction Performance - Time Series
    ax1 = plt.subplot(gs[0, :])
    
    # Check if binary predictions exist
    binary_col = 'PredictedBinary' if 'PredictedBinary' in predictions_df.columns else 'Predicted'
    
    # Calculate cumulative returns for correct predictions
    predictions_df['correct'] = predictions_df['True'] == predictions_df[binary_col]
    accuracy_ma = predictions_df['correct'].rolling(window=50).mean()
    
    # Plot rolling accuracy
    ax1.plot(predictions_df.index, accuracy_ma, 'b-', label='50-period Rolling Accuracy')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Random Guess (50%)')
    
    if 'aggregate_metrics' in results and results['aggregate_metrics']:
        overall_acc = results['aggregate_metrics'].get('accuracy', np.nan)
        ax1.axhline(y=overall_acc, color='g', linestyle='-', label=f'Overall Accuracy: {overall_acc:.4f}')
    
    ax1.set_title('Model Prediction Performance Over Time')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Confusion Matrix
    ax2 = plt.subplot(gs[1, 0])
    cm = confusion_matrix(predictions_df['True'], predictions_df[binary_col])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(ax=ax2, cmap='Blues', colorbar=False)
    ax2.set_title('Confusion Matrix')
    
    # 3. Metrics Over Time (if step metrics available)
    ax3 = plt.subplot(gs[1, 1])
    
    if 'step_metrics' in results and results['step_metrics']:
        # Create a DataFrame from step metrics
        metrics_df = pd.DataFrame(results['step_metrics'])
        
        if not metrics_df.empty:
            # Plot key metrics over validation steps
            for col in ['accuracy', 'f1', 'precision', 'recall']:
                if col in metrics_df.columns:
                    ax3.plot(metrics_df.index, metrics_df[col], label=col.capitalize())
            
            ax3.set_title('Metrics Across Validation Steps')
            ax3.set_xlabel('Validation Step')
            ax3.set_ylabel('Score')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No step metrics available", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Regime Analysis (if available)
    if regime_data is not None and not regime_data.empty:
        # 4. Performance by Market Regime
        ax4 = plt.subplot(gs[2, :])
        
        # Merge predictions with regime data
        merged_df = pd.merge(predictions_df, 
                             regime_data[['regime', 'volatility']], 
                             left_index=True, 
                             right_index=True, 
                             how='left')
        
        # Group by regime
        regime_performance = merged_df.groupby('regime')['correct'].agg(
            ['mean', 'count', 'sum']).sort_values('mean', ascending=False)
        
        # Calculate confidence intervals
        regime_performance['stderr'] = np.sqrt(
            (regime_performance['mean'] * (1 - regime_performance['mean'])) / 
            regime_performance['count']
        )
        regime_performance['ci_lower'] = regime_performance['mean'] - 1.96 * regime_performance['stderr']
        regime_performance['ci_upper'] = regime_performance['mean'] + 1.96 * regime_performance['stderr']
        
        # Plot regime performance with error bars
        bars = ax4.bar(regime_performance.index, regime_performance['mean'], 
                       yerr=1.96 * regime_performance['stderr'], capsize=10,
                       color=sns.color_palette('viridis', len(regime_performance)))
        
        ax4.set_title('Model Accuracy by Market Regime')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim([0, min(1.0, regime_performance['mean'].max() + 0.2)])
        ax4.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
        
        # Add sample counts above bars
        for i, (regime, row) in enumerate(regime_performance.iterrows()):
            ax4.text(i, row['mean'] + 0.02, f"n={int(row['count'])}", 
                     ha='center', va='bottom', fontsize=9)
        
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # 5. Regime Transitions and Model Performance
        ax5 = plt.subplot(gs[3, 0])
        
        # Add regime change indicator
        merged_df['regime_change'] = (merged_df['regime'] != merged_df['regime'].shift(1)).astype(int)
        
        # Calculate performance around regime changes
        transition_periods = 10  # Days before/after transition to analyze
        
        # Find all regime change points
        change_points = merged_df[merged_df['regime_change'] == 1].index
        
        # Initialize containers for performance around changes
        before_change = []
        after_change = []
        
        for change_point in change_points:
            # Get indices before and after change point
            try:
                idx = merged_df.index.get_loc(change_point)
                before_idx = max(0, idx - transition_periods)
                after_idx = min(len(merged_df), idx + transition_periods + 1)
                
                # Collect performance data
                before_perf = merged_df.iloc[before_idx:idx]['correct'].values
                after_perf = merged_df.iloc[idx:after_idx]['correct'].values
                
                # Pad if needed to ensure consistent length
                before_perf = np.pad(before_perf, 
                                     (transition_periods - len(before_perf), 0), 
                                     'constant', 
                                     constant_values=np.nan)
                after_perf = np.pad(after_perf, 
                                    (0, transition_periods - len(after_perf)), 
                                    'constant', 
                                    constant_values=np.nan)
                
                before_change.append(before_perf)
                after_change.append(after_perf)
            except:
                continue
        
        if before_change and after_change:
            # Convert to arrays and calculate average
            before_change_avg = np.nanmean(np.vstack(before_change), axis=0)
            after_change_avg = np.nanmean(np.vstack(after_change), axis=0)
            
            # Create x-axis labels
            days = list(range(-transition_periods, 0)) + list(range(0, transition_periods))
            
            # Plot performance around regime changes
            ax5.plot(days, np.concatenate([before_change_avg, after_change_avg]), 'b-o')
            ax5.axvline(x=0, color='r', linestyle='--', label='Regime Change')
            ax5.set_title(f'Accuracy Around Regime Changes (n={len(change_points)})')
            ax5.set_xlabel('Days Relative to Regime Change')
            ax5.set_ylabel('Average Accuracy')
            ax5.legend(loc='upper right')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "Insufficient regime change data", 
                     horizontalalignment='center', verticalalignment='center')
        
        # 6. Volatility vs. Accuracy Scatter
        ax6 = plt.subplot(gs[3, 1])
        
        # Create scatter plot with regression line
        sns.regplot(x='volatility', y='correct', data=merged_df, ax=ax6, 
                    scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        
        ax6.set_title('Volatility vs. Prediction Accuracy')
        ax6.set_xlabel('Volatility')
        ax6.set_ylabel('Correct Prediction (1=Yes, 0=No)')
        ax6.grid(True, alpha=0.3)
    else:
        # If no regime data, use these slots for other plots
        ax4 = plt.subplot(gs[2, :])
        
        # Distribution of prediction probabilities by actual outcome
        true_probs = predictions_df.loc[predictions_df['True'] == 1, 'PredictedProb'] if 'PredictedProb' in predictions_df.columns else []
        false_probs = predictions_df.loc[predictions_df['True'] == 0, 'PredictedProb'] if 'PredictedProb' in predictions_df.columns else []
        
        if len(true_probs) > 0 and len(false_probs) > 0:
            ax4.hist(false_probs, bins=20, alpha=0.5, label='Actual: Down', color='red')
            ax4.hist(true_probs, bins=20, alpha=0.5, label='Actual: Up', color='green')
            ax4.set_title('Distribution of Prediction Probabilities by Actual Outcome')
            ax4.set_xlabel('Predicted Probability of Up')
            ax4.set_ylabel('Count')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No probability distribution data available", 
                     horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    



def analyze_attention_patterns(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    feature_names: List[str],
    n_samples: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Optional[str] = None
) -> Dict:
    """
    Analyze attention patterns to interpret model behavior.
    
    Parameters:
    -----------
    model : nn.Module
        Trained transformer model that returns attention weights
    data_loader : DataLoader
        DataLoader with test data
    feature_names : List[str]
        Names of input features
    n_samples : int
        Number of samples to analyze
    device : str
        Device to run analysis on
    save_dir : Optional[str]
        Directory to save visualizations, if None no saving is done
        
    Returns:
    --------
    Dict
        Analysis results including:
        - attention_weights: Raw attention weights from model
        - feature_importance: Overall feature importance based on attention
        - temporal_importance: Temporal importance (which time steps matter)
        - head_specialization: What different attention heads focus on
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    from matplotlib.ticker import MaxNLocator
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    
    # Ensure model is on correct device and in evaluation mode
    model = model.to(device)
    model.eval()
    
    # Containers for results
    all_attention_weights = []
    feature_importance_scores = []
    temporal_patterns = []
    head_specializations = []
    
    # Get a few batches of data
    sample_batches = []
    sample_indices = []
    
    # Get up to n_samples batches
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= n_samples:
                break
                
            sample_batches.append((inputs.to(device), targets.to(device)))
            
            # Try to get actual indices if available in the loader
            if hasattr(data_loader.dataset, 'indices'):
                indices = data_loader.dataset.indices[i * data_loader.batch_size:
                                               min((i + 1) * data_loader.batch_size, 
                                                  len(data_loader.dataset))]
                sample_indices.append(indices)
    
    # Check if we have any data
    if not sample_batches:
        print("No data available for attention analysis.")
        return {
            "attention_weights": [],
            "feature_importance": pd.DataFrame(),
            "temporal_importance": pd.DataFrame(),
            "head_specialization": pd.DataFrame()
        }
    
    # Function to extract attention weights from model
    def get_attention_weights(model, inputs):
        # Run forward pass
        outputs = model(inputs)
        
        # Check if model returns attention weights
        if isinstance(outputs, tuple) and len(outputs) > 1:
            # Assume second element contains attention weights
            attention = outputs[1]
            return attention
        else:
            raise ValueError("Model does not return attention weights. Make sure your model returns (output, attention_weights).")
    
    # Process each batch
    for batch_idx, (inputs, targets) in enumerate(sample_batches):
        try:
            # Get attention weights
            attention = get_attention_weights(model, inputs)
            
            # Process attention weights
            if isinstance(attention, list):
                # If we have a list of attention matrices (one per layer)
                for layer_idx, layer_attention in enumerate(attention):
                    # layer_attention shape: [batch_size, n_heads, seq_len, seq_len]
                    # Convert to numpy for easier manipulation
                    if isinstance(layer_attention, torch.Tensor):
                        layer_attention = layer_attention.cpu().numpy()
                    
                    all_attention_weights.append({
                        'batch_idx': batch_idx,
                        'layer_idx': layer_idx,
                        'attention': layer_attention
                    })
            elif isinstance(attention, torch.Tensor):
                # Single attention tensor
                all_attention_weights.append({
                    'batch_idx': batch_idx,
                    'layer_idx': 0,
                    'attention': attention.cpu().numpy()
                })
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # If we didn't get any attention weights, return empty results
    if not all_attention_weights:
        print("Failed to extract attention weights from model.")
        return {
            "attention_weights": [],
            "feature_importance": pd.DataFrame(),
            "temporal_importance": pd.DataFrame(),
            "head_specialization": pd.DataFrame()
        }
    
    # 1. Analyze overall feature importance
    # --------------------------------------
    # We'll focus on the last time step's attention to features
    # since that's what directly influences the prediction
    
    # Extract attention to features from final time step
    feature_attention = []
    
    # Process each batch of attention weights
    for att_data in all_attention_weights:
        attention = att_data['attention']
        layer_idx = att_data['layer_idx']
        
        # For last layer only
        if layer_idx == len(all_attention_weights) // len(sample_batches) - 1:
            # Extract attention from last time step to all features
            # attention shape: [batch_size, n_heads, seq_len, seq_len]
            
            # Get attention from last time step (query) to all other time steps (keys)
            # We use -1 to get the last time step's query
            last_timestep_attention = attention[:, :, -1, :]
            
            # Average across heads and batches
            avg_attention = np.mean(last_timestep_attention, axis=(0, 1))
            
            feature_attention.append(avg_attention)
    
    # Average across all processed batches
    if feature_attention:
        avg_feature_attention = np.mean(feature_attention, axis=0)
        
        # Create feature importance DataFrame
        if len(avg_feature_attention) == len(feature_names):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': avg_feature_attention
            }).sort_values('Importance', ascending=False)
        else:
            # If dimensions don't match, create with indices
            feature_importance = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(avg_feature_attention))],
                'Importance': avg_feature_attention
            }).sort_values('Importance', ascending=False)
        
        feature_importance_scores = feature_importance
    else:
        feature_importance_scores = pd.DataFrame(columns=['Feature', 'Importance'])
    
    # 2. Analyze temporal importance patterns
    # ---------------------------------------
    temporal_importance = []
    
    for att_data in all_attention_weights:
        attention = att_data['attention']
        
        # Average across batches, heads, and sum over keys
        # This gives us importance of each time step as a query
        temporal_imp = np.mean(np.sum(attention, axis=3), axis=(0, 1))
        temporal_importance.append(temporal_imp)
    
    if temporal_importance:
        avg_temporal_importance = np.mean(temporal_importance, axis=0)
        temporal_patterns = pd.DataFrame({
            'Time Step': [f't-{len(avg_temporal_importance)-i}' for i in range(len(avg_temporal_importance))],
            'Importance': avg_temporal_importance
        })
    else:
        temporal_patterns = pd.DataFrame(columns=['Time Step', 'Importance'])
    
    # 3. Analyze head specialization
    # ------------------------------
    # This analysis identifies what different attention heads focus on
    
    # We'll examine correlation patterns between heads
    head_correlations = []
    head_activities = []
    
    for att_data in all_attention_weights:
        attention = att_data['attention']
        
        # For each head, flatten its attention pattern into a vector
        n_heads = attention.shape[1]
        
        for batch_idx in range(attention.shape[0]):
            head_vectors = []
            
            for head_idx in range(n_heads):
                head_attention = attention[batch_idx, head_idx].flatten()
                head_vectors.append(head_attention)
                
                # Also track average activation magnitude per head
                head_activities.append({
                    'Head': f'Head_{head_idx}',
                    'Activity': np.mean(head_attention),
                    'Batch': batch_idx
                })
            
            # Calculate correlation between heads
            if len(head_vectors) > 1:
                head_corr = np.corrcoef(head_vectors)
                head_correlations.append(head_corr)
    
    # Average head correlations across batches
    if head_correlations:
        avg_head_correlation = np.mean(head_correlations, axis=0)
        
        # Perform hierarchical clustering on heads
        if avg_head_correlation.shape[0] > 1:
            head_linkage = linkage(pdist(avg_head_correlation), method='ward')
            head_clusters = fcluster(head_linkage, t=3, criterion='maxclust')
            
            # Create head specialization DataFrame
            head_activity_df = pd.DataFrame(head_activities)
            head_activity_summary = head_activity_df.groupby('Head')['Activity'].mean().reset_index()
            
            # Add cluster information
            head_activity_summary['Cluster'] = [f'Cluster_{cluster}' for cluster in head_clusters]
            
            head_specializations = head_activity_summary
        else:
            head_specializations = pd.DataFrame(head_activities).groupby('Head')['Activity'].mean().reset_index()
    else:
        head_specializations = pd.DataFrame(columns=['Head', 'Activity', 'Cluster'])
    
    # 4. Visualize Results
    # -------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Feature Importance Plot
        if not feature_importance_scores.empty:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_scores)
            plt.title('Feature Importance Based on Attention')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300)
            plt.close()
        
        # 2. Temporal Importance Plot
        if not temporal_patterns.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='Time Step', y='Importance', data=temporal_patterns)
            plt.title('Temporal Importance Pattern')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'temporal_importance.png'), dpi=300)
            plt.close()
        
        # 3. Head Correlation Heatmap
        if head_correlations:
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_head_correlation, annot=True, cmap='viridis')
            plt.title('Attention Head Correlation')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'head_correlation.png'), dpi=300)
            plt.close()
        
        # 4. Head Clustering Plot
        if not head_specializations.empty and 'Cluster' in head_specializations.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Head', y='Activity', hue='Cluster', data=head_specializations)
            plt.title('Attention Head Specialization')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'head_specialization.png'), dpi=300)
            plt.close()
        
        # 5. Attention Heatmaps for Sample Sequences
        if all_attention_weights:
            # Plot attention matrix for first sequence in each batch
            for batch_idx in range(min(n_samples, len(sample_batches))):
                for layer_idx in range(len(all_attention_weights) // len(sample_batches)):
                    # Find matching attention data
                    for att_data in all_attention_weights:
                        if att_data['batch_idx'] == batch_idx and att_data['layer_idx'] == layer_idx:
                            attention = att_data['attention']
                            
                            # Plot attention for first sequence and average across heads
                            avg_attention = np.mean(attention[0], axis=0)
                            
                            plt.figure(figsize=(10, 8))
                            sns.heatmap(avg_attention, cmap='viridis')
                            plt.title(f'Average Attention Pattern - Batch {batch_idx}, Layer {layer_idx}')
                            plt.xlabel('Key Position')
                            plt.ylabel('Query Position')
                            plt.tight_layout()
                            plt.savefig(os.path.join(save_dir, f'attention_heatmap_b{batch_idx}_l{layer_idx}.png'), dpi=300)
                            plt.close()
                            
                            break
    
    # Return comprehensive analysis results
    return {
        "attention_weights": all_attention_weights,
        "feature_importance": feature_importance_scores,
        "temporal_importance": temporal_patterns,
        "head_specialization": head_specializations
    }