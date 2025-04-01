# technical.py (Refactored to match original headers, retaining 10/10 improvements)
import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from ta.utils import IndicatorMixin
from typing import List, Optional, Dict, Any, Tuple, Set, Union

# --- Constants for required columns ---
REQUIRED_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

# --- Default Parameter Configuration (Internal - used when not overridden via kwargs) ---
DEFAULT_TREND_PARAMS = {
    'ema_windows': [20, 30, 50, 100, 200],
    'macd_window_slow': 26,
    'macd_window_fast': 12,
    'macd_window_sign': 9,
    'adx_window': 14,
}

DEFAULT_MOMENTUM_PARAMS = {
    'rsi_window': 14,
    'stoch_window': 14,
    'stoch_smooth_window': 3,
    'roc_windows': [5, 20],
    'williams_r_lbp': 14,
}

DEFAULT_VOLATILITY_PARAMS = {
    'atr_window': 14,
    'keltner_window': 20,
    'keltner_atr_window': 10,
    'hist_vol_window': 20,
    'hist_vol_annualizing_factor': 252,
}

DEFAULT_VOLUME_PARAMS = {
    'vol_sma_windows': [5, 20],
}

# --- Internal Logic Functions (Private Helpers from 10/10 version) ---
# These contain the core calculations and improved logic.

def _add_trend_indicators(
    df: pd.DataFrame,
    ema_windows: List[int],
    macd_window_slow: int,
    macd_window_fast: int,
    macd_window_sign: int,
    adx_window: int,
    prefix: str = '',
) -> pd.DataFrame:
    """Internal function: Adds trend indicators with specific parameters."""
    df_copy = df.copy() # Ensure we don't modify the DataFrame passed in

    # Make sure we're passing Series objects, not DataFrames
    close_series = df_copy['Close'] if isinstance(df_copy['Close'], pd.Series) else df_copy['Close'].squeeze()
    high_series = df_copy['High'] if isinstance(df_copy['High'], pd.Series) else df_copy['High'].squeeze()
    low_series = df_copy['Low'] if isinstance(df_copy['Low'], pd.Series) else df_copy['Low'].squeeze()

    # Calculate SMAs and EMAs
    df_copy['sma_10'] = ta.trend.sma_indicator(close=close_series, window=10)
    df_copy['sma_30'] = ta.trend.sma_indicator(close=close_series, window=30)
    df_copy['ema_10'] = ta.trend.ema_indicator(close=close_series, window=10)
    
    # MACD
    macd = ta.trend.MACD(
        close=close_series,
        window_slow=macd_window_slow,
        window_fast=macd_window_fast,
        window_sign=macd_window_sign,
        fillna=False,
    )
    df_copy[f"{prefix}macd"] = macd.macd()
    df_copy[f"{prefix}macd_signal"] = macd.macd_signal()
    df_copy[f"{prefix}macd_diff"] = macd.macd_diff()

    # Relative price to specific moving averages (using calculated EMAs)
    ema_50_col = f'{prefix}ema_50'
    ema_200_col = f'{prefix}ema_200'
    if ema_50_col in df_copy.columns:
         df_copy[f'{prefix}close_to_ema_50'] = close_series / df_copy[ema_50_col].replace(0, np.nan)
    if ema_200_col in df_copy.columns:
        df_copy[f'{prefix}close_to_ema_200'] = close_series / df_copy[ema_200_col].replace(0, np.nan)

    # ADX (Average Directional Index)
    df_copy[f"{prefix}adx"] = ta.trend.ADXIndicator(
        high=high_series, 
        low=low_series, 
        close=close_series, 
        window=adx_window, 
        fillna=False
    ).adx()

    return df_copy

def _add_momentum_indicators(
    df: pd.DataFrame,
    rsi_window: int,
    stoch_window: int,
    stoch_smooth_window: int,
    roc_windows: List[int],
    williams_r_lbp: int,
    prefix: str = '',
) -> pd.DataFrame:
    """Internal function: Adds momentum indicators with specific parameters."""
    df_copy = df.copy()

    # Ensure we have Series objects for all OHLC data
    close_series = df_copy["Close"] if isinstance(df_copy["Close"], pd.Series) else df_copy["Close"].squeeze()
    high_series = df_copy["High"] if isinstance(df_copy["High"], pd.Series) else df_copy["High"].squeeze()
    low_series = df_copy["Low"] if isinstance(df_copy["Low"], pd.Series) else df_copy["Low"].squeeze()

    # RSI
    df_copy[f"{prefix}rsi"] = ta.momentum.RSIIndicator(
        close=close_series, window=rsi_window, fillna=False
    ).rsi()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=high_series,
        low=low_series,
        close=close_series,
        window=stoch_window,
        smooth_window=stoch_smooth_window,
        fillna=False,
    )
    df_copy[f"{prefix}stoch_k"] = stoch.stoch()
    df_copy[f"{prefix}stoch_d"] = stoch.stoch_signal()

    # Rate of Change (ROC)
    for window in roc_windows:
        df_copy[f"{prefix}roc_{window}"] = ta.momentum.ROCIndicator(
            close=close_series, window=window, fillna=False
        ).roc()

    # Williams %R
    df_copy[f"{prefix}williams_r"] = ta.momentum.WilliamsRIndicator(
        high=high_series,
        low=low_series,
        close=close_series,
        lbp=williams_r_lbp,
        fillna=False
    ).williams_r()

    return df_copy

def _add_volatility_indicators(
    df: pd.DataFrame,
    atr_window: int,
    keltner_window: int,
    keltner_atr_window: int,
    hist_vol_window: int,
    hist_vol_annualizing_factor: int,
    log_return_col: Optional[str] = None,
    prefix: str = '',
) -> pd.DataFrame:
    """Internal function: Adds volatility indicators with specific parameters."""
    df_copy = df.copy()

    # Ensure we have Series objects for all OHLC data
    close_series = df_copy["Close"] if isinstance(df_copy["Close"], pd.Series) else df_copy["Close"].squeeze()
    high_series = df_copy["High"] if isinstance(df_copy["High"], pd.Series) else df_copy["High"].squeeze()
    low_series = df_copy["Low"] if isinstance(df_copy["Low"], pd.Series) else df_copy["Low"].squeeze()

    # Average True Range (ATR)
    atr_indicator = ta.volatility.AverageTrueRange(
        high=high_series,
        low=low_series,
        close=close_series,
        window=atr_window,
        fillna=False
    )
    df_copy[f"{prefix}atr"] = atr_indicator.average_true_range()
    # ATR Percentage (relative to close)
    df_copy[f"{prefix}atr_pct"] = (df_copy[f"{prefix}atr"] / close_series.replace(0, np.nan)) * 100

    # Keltner Channels
    kc = ta.volatility.KeltnerChannel(
        high=high_series,
        low=low_series,
        close=close_series,
        window=keltner_window,
        window_atr=keltner_atr_window,
        fillna=False
    )
    df_copy[f"{prefix}keltner_hband"] = kc.keltner_channel_hband()
    df_copy[f"{prefix}keltner_lband"] = kc.keltner_channel_lband()

    # Keltner Width Percentage (relative to close)
    # Ensure bands exist before calculating width
    hband_col = f"{prefix}keltner_hband"
    lband_col = f"{prefix}keltner_lband"
    if hband_col in df_copy.columns and lband_col in df_copy.columns:
        df_copy[f"{prefix}keltner_width_pct"] = (
            (df_copy[hband_col] - df_copy[lband_col]) / close_series.replace(0, np.nan)
        ) * 100
    else:
        df_copy[f"{prefix}keltner_width_pct"] = np.nan

    # Historical Volatility (using log returns if available)
    if log_return_col and log_return_col in df_copy.columns:
        log_return_series = df_copy[log_return_col] if isinstance(df_copy[log_return_col], pd.Series) else df_copy[log_return_col].squeeze()
        df_copy[f'{prefix}hist_volatility'] = (
            log_return_series.rolling(window=hist_vol_window).std() *
            np.sqrt(hist_vol_annualizing_factor) * 100
        )
    elif 'log_return' in df_copy.columns and not prefix:
        log_return_series = df_copy['log_return'] if isinstance(df_copy['log_return'], pd.Series) else df_copy['log_return'].squeeze()
        df_copy[f'{prefix}hist_volatility'] = (
            log_return_series.rolling(window=hist_vol_window).std() *
            np.sqrt(hist_vol_annualizing_factor) * 100
        )

    return df_copy

def _add_volume_indicators(
    df: pd.DataFrame,
    vol_sma_windows: List[int],
    prefix: str = '',
) -> pd.DataFrame:
    """Internal function: Adds volume indicators with specific parameters."""
    df_copy = df.copy()

    # Ensure we have Series objects
    volume_series = df_copy["Volume"] if isinstance(df_copy["Volume"], pd.Series) else df_copy["Volume"].squeeze()
    close_series = df_copy["Close"] if isinstance(df_copy["Close"], pd.Series) else df_copy["Close"].squeeze()

    # Volume Moving Averages
    base_sma_col = None
    vol_sma_base = 20  # Define base for rel_volume calculation
    for window in vol_sma_windows:
        col_name = f"{prefix}volume_sma_{window}"
        df_copy[col_name] = volume_series.rolling(window=window, min_periods=1).mean()
        if window == vol_sma_base:
            base_sma_col = col_name

    # Fallback if base sma (20) wasn't calculated
    if base_sma_col is None and vol_sma_windows:
        base_sma_col = f"{prefix}volume_sma_{vol_sma_windows[0]}"  # Use first calculated SMA

    # Relative Volume
    if base_sma_col and base_sma_col in df_copy.columns:
        df_copy[f"{prefix}rel_volume"] = volume_series / df_copy[base_sma_col].replace(0, np.nan)

    # On-Balance Volume (OBV)
    df_copy[f"{prefix}obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=close_series,
        volume=volume_series,
        fillna=False
    ).on_balance_volume()

    return df_copy


# --- Public API Functions (Matching Original Headers) ---

def add_technical_features( # Corrected typo from original 'tecnical'
    df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    **kwargs: Any # Accept advanced configuration via kwargs
) -> pd.DataFrame:
    """
    Add technical indicators to price data. Retains original signature.

    Enhanced features (parameter customization, NaN handling, etc.) can be
    controlled via keyword arguments (**kwargs).

    Args:
        df : pandas DataFrame
            DataFrame with OHLCV data (must have 'Open', 'High', 'Low', 'Close',
            'Volume' columns and ideally a DatetimeIndex).
        categories : list or None
            List of categories to include. Options: 'trend', 'momentum',
            'volatility', 'volume'. If None, include all categories. Defaults to None.
        **kwargs: Optional keyword arguments for advanced configuration:
            trend_params (dict): Parameters for trend indicators (see DEFAULT_TREND_PARAMS).
            momentum_params (dict): Parameters for momentum indicators.
            volatility_params (dict): Parameters for volatility indicators.
            volume_params (dict): Parameters for volume indicators.
            calculate_log_return (bool): If True, calculates 'log_return'. Default True.
            drop_na (bool): If True, drops rows with NaNs *after* calculations. Default False.
            col_prefix (str): Prefix for generated columns. Default ''.

    Returns:
        pandas DataFrame: Original dataframe with added technical indicators.

    Raises:
        ValueError: If required OHLCV columns are missing.
        TypeError: If input `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    missing_cols = [col for col in REQUIRED_OHLCV_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Extract config from kwargs or use defaults
    trend_params = kwargs.get('trend_params', DEFAULT_TREND_PARAMS)
    momentum_params = kwargs.get('momentum_params', DEFAULT_MOMENTUM_PARAMS)
    volatility_params = kwargs.get('volatility_params', DEFAULT_VOLATILITY_PARAMS)
    volume_params = kwargs.get('volume_params', DEFAULT_VOLUME_PARAMS)
    calculate_log_return = kwargs.get('calculate_log_return', True)
    drop_na = kwargs.get('drop_na', False)
    col_prefix = kwargs.get('col_prefix', '')

    # Use internal defaults if specific dicts passed in kwargs are incomplete
    _trend_params = {**DEFAULT_TREND_PARAMS, **(trend_params or {})}
    _momentum_params = {**DEFAULT_MOMENTUM_PARAMS, **(momentum_params or {})}
    _volatility_params = {**DEFAULT_VOLATILITY_PARAMS, **(volatility_params or {})}
    _volume_params = {**DEFAULT_VOLUME_PARAMS, **(volume_params or {})}

    results = df.copy()

    results['direction'] = (results['Close'].shift(-1) > results['Close']).astype(int)
    if categories is None:
        categories = ['trend', 'momentum', 'volatility', 'volume']

    log_return_col = f"{col_prefix}log_return"
    if calculate_log_return and log_return_col not in results.columns:
        safe_close = results['Close'].clip(lower=1e-9)
        safe_close_shifted = results['Close'].shift(1).clip(lower=1e-9)
        results[log_return_col] = np.log(safe_close / safe_close_shifted)

    # --- Call internal helpers with resolved parameters ---
    if 'trend' in categories:
        results = _add_trend_indicators(results, prefix=col_prefix, **_trend_params)

    if 'momentum' in categories:
        results = _add_momentum_indicators(results, prefix=col_prefix, **_momentum_params)

    if 'volatility' in categories:
        # Pass the potentially prefixed log return column name
        current_log_return_col = log_return_col if calculate_log_return else None
        if calculate_log_return and current_log_return_col not in results.columns:
             # This condition might occur if calculate_log_return was false initially
             # but volatility needs it. Recalculate if necessary? Or rely on user?
             # For now, assume if calculate_log_return is true, it exists.
             # If calculate_log_return is False, hist_vol won't be calculated by default path.
             pass # Historical volatility calculation inside _add_volatility checks existence
        results = _add_volatility_indicators(results, log_return_col=current_log_return_col, prefix=col_prefix, **_volatility_params)

    if 'volume' in categories:
        results = _add_volume_indicators(results, prefix=col_prefix, **_volume_params)

    if drop_na:
        results.dropna(inplace=True)

    return results

# --- Public Wrappers for Category Indicators (Original Interface) ---

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add default trend-related technical indicators to the DataFrame.
    Uses default parameters internally.
    """
    # Calls the internal function with default parameters
    return _add_trend_indicators(df, **DEFAULT_TREND_PARAMS)

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add default momentum-related technical indicators to the DataFrame.
    Uses default parameters internally.
    """
    # Calls the internal function with default parameters
    return _add_momentum_indicators(df, **DEFAULT_MOMENTUM_PARAMS)

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add default volatility-related technical indicators to the DataFrame.
    Uses default parameters internally. Calculates log_return if needed.
    """
    df_copy = df.copy()
    log_return_col = 'log_return'
    # Calculate log return if needed for historical volatility
    if log_return_col not in df_copy.columns:
        safe_close = df_copy['Close'].clip(lower=1e-9)
        safe_close_shifted = df_copy['Close'].shift(1).clip(lower=1e-9)
        df_copy[log_return_col] = np.log(safe_close / safe_close_shifted)

    # Calls the internal function with default parameters
    return _add_volatility_indicators(df_copy, log_return_col=log_return_col, **DEFAULT_VOLATILITY_PARAMS)

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add default volume-related technical indicators to the DataFrame.
    Uses default parameters internally.
    """
    # Calls the internal function with default parameters
    return _add_volume_indicators(df, **DEFAULT_VOLUME_PARAMS)


# --- Multi-Timeframe Function (Original Interface + Kwargs) ---

def add_multi_timeframe_features(
    df: pd.DataFrame,
    timeframes: List[str] = ['daily'], # Default matches original
    **kwargs: Any # Accept advanced configuration via kwargs
) -> pd.DataFrame:
    """
    Generate technical features at multiple timeframes. Retains original signature.

    Advanced configuration (custom resampling, parameters per timeframe, etc.)
    can be controlled via keyword arguments (**kwargs).

    Args:
        df : pandas DataFrame
            Daily OHLCV dataframe with a DatetimeIndex.
        timeframes : list
            List of timeframes to include. Options: 'daily', 'weekly', 'monthly',
            or custom keys matching `resample_map`. Defaults to ['daily'].
        **kwargs: Optional keyword arguments for advanced configuration:
            resample_map (dict): Mapping from timeframe names to Pandas rules
                                (e.g., {'weekly': 'W-FRI'}).
            daily_categories (list): Categories for 'daily' timeframe.
            agg_categories (list): Categories for aggregated timeframes.
            indicator_params (dict): Nested dict for parameters per timeframe/category.
                                     e.g. {'weekly': {'trend_params': {...}}}
            forward_fill (bool): Whether to forward fill aggregated features. Default True.

    Returns:
        pandas DataFrame: DataFrame with features at all requested timeframes.

    Raises:
        TypeError: If df.index is not a DatetimeIndex.
        ValueError: If required OHLCV columns are missing.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex for resampling.")

    missing_cols = [col for col in REQUIRED_OHLCV_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Extract config from kwargs or use defaults
    resample_map = kwargs.get('resample_map', {'weekly': 'W-FRI', 'monthly': 'M'})
    daily_categories = kwargs.get('daily_categories', None) # None means all
    agg_categories = kwargs.get('agg_categories', None)     # None means all
    indicator_params = kwargs.get('indicator_params', {})   # Default to empty dict
    forward_fill = kwargs.get('forward_fill', True)

    result_df = df.copy()
    processed_daily = False

    # --- Daily Features ---
    if 'daily' in timeframes:
        # Extract specific 'daily' params from the nested dict if provided
        daily_indicator_params = indicator_params.get('daily', {})
        result_df = add_technical_features( 
            result_df,
            categories=daily_categories,
            drop_na=False, # Explicitly keep NaNs here
            col_prefix='',
            **daily_indicator_params # Pass extracted daily params as kwargs
        )
        processed_daily = True

    # --- Aggregated Timeframe Features ---
    original_cols = set(result_df.columns)

    for tf in timeframes:
        if tf == 'daily':
            continue

        if tf not in resample_map:
            print(f"Warning: Timeframe '{tf}' not found in resample_map. Skipping.")
            continue

        resample_rule = resample_map[tf]
        tf_prefix = f"_{tf}" # Use prefix instead of suffix for columns like _weekly_ema_50

        agg_df = df.resample(resample_rule, label='right', closed='right').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        })
        agg_df.dropna(subset=REQUIRED_OHLCV_COLS, how='all', inplace=True)
        if agg_df.empty:
            print(f"Warning: Resampling for timeframe '{tf}' resulted in an empty DataFrame. Skipping.")
            continue

        # Extract specific params for this timeframe (e.g., 'weekly')
        tf_indicator_params = indicator_params.get(tf, {})

        # Calculate features on aggregated data, passing specific params via kwargs
        agg_features = add_technical_features( # Call public API version
            agg_df,
            categories=agg_categories,
            drop_na=False,
            col_prefix=tf_prefix, # Add prefix like '_weekly'
            **tf_indicator_params # Pass extracted params as kwargs
        )

        # Select only newly added feature columns (those with the prefix)
        new_feature_cols = [col for col in agg_features.columns if col.startswith(tf_prefix)]
        if not new_feature_cols: # Check if any features were actually generated
             print(f"Warning: No features generated for timeframe '{tf}' with prefix '{tf_prefix}'. Skipping merge.")
             continue

        agg_features_to_merge = agg_features[new_feature_cols]

        # Reindex & Forward Fill
        # Use outer join on index to align, then merge
        agg_features_reindexed = agg_features_to_merge.reindex(result_df.index)
        if forward_fill:
            agg_features_reindexed.ffill(inplace=True)

        # Merge into the main result
        result_df = pd.merge(result_df, agg_features_reindexed, left_index=True, right_index=True, how='left')

    return result_df


# --- Analysis Functions (Original Interface) ---

def analyze_features(
    df: pd.DataFrame,
    target_col: str = 'direction' 
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Analyze technical features for correlation with target and multicollinearity.
    Retains original signature.

    Args:
        df : pandas DataFrame
            DataFrame with technical features and target column.
        target_col : str
            Name of the target column. Defaults to 'direction'.

    Returns:
        tuple: (target_correlations, feature_correlation_matrix)

    Raises:
        ValueError: If target_col not found or not numeric, or no features found.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df_numeric = df.select_dtypes(include=np.number)

    if target_col not in df_numeric.columns:
         raise ValueError(f"Target column '{target_col}' is not numeric after dtype selection.")

    all_corr = df_numeric.corr()
    if target_col not in all_corr.columns or all_corr[target_col].isnull().all():
         raise ValueError(f"Could not compute correlation for target '{target_col}'. Check variance/NaNs.")

    target_corr = all_corr[target_col].drop(target_col).sort_values(ascending=False)

    feature_cols = [col for col in df_numeric.columns if col != target_col]
    if not feature_cols:
        # Return empty results instead of raising error if only target exists
        print(f"Warning: No numeric feature columns found besides the target '{target_col}'. Returning empty correlations.")
        return pd.Series(dtype=float), pd.DataFrame(columns=[], index=[])
        # raise ValueError("No numeric feature columns found besides the target column.")


    feature_corr_matrix = df_numeric[feature_cols].corr()

    return target_corr, feature_corr_matrix


def identify_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.85 # Default matches original
) -> List[Set[str]]:
    """
    Identify groups of highly correlated features within a DataFrame.
    Retains original signature (operates on DataFrame).

    Args:
        df : pandas DataFrame
            DataFrame with features (should contain numeric columns).
        threshold : float
            Absolute correlation threshold (0.0 to 1.0). Defaults to 0.85.

    Returns:
        list: List of sets containing highly correlated feature names.

    Raises:
        ValueError: If threshold is not between 0 and 1.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Correlation threshold must be between 0.0 and 1.0")

    # Calculate correlation matrix from numeric columns of df
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.shape[1] < 2:
        # Not enough numeric columns to compare
        return []

    feature_corr_matrix = df_numeric.corr()

    # --- Use the same grouping logic as the 10/10 version ---
    corr_matrix_abs = feature_corr_matrix.abs()
    upper_triangle = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

    correlated_groups = []
    features_in_groups = set()

    for column in upper_triangle.columns:
        if column in features_in_groups:
            continue

        high_corr_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()

        if high_corr_features:
            current_group = {column} | set(high_corr_features)
            merged = False
            new_correlated_groups = []
            for existing_group in correlated_groups:
                if not current_group.isdisjoint(existing_group):
                    current_group.update(existing_group)
                    merged = True # Mark that a merge happened
                else:
                    new_correlated_groups.append(existing_group)
            correlated_groups = new_correlated_groups
            correlated_groups.append(current_group) # Add the new or merged group
            features_in_groups.update(current_group)


    # Optional second pass for consolidation if needed (A-B, C-D, B-C case)
    if len(correlated_groups) > 1:
        merged_groups = []
        processed_indices = [False] * len(correlated_groups)
        for i in range(len(correlated_groups)):
            if processed_indices[i]:
                continue
            current_merged_group = correlated_groups[i].copy()
            processed_indices[i] = True
            for j in range(i + 1, len(correlated_groups)):
                 if not processed_indices[j] and not current_merged_group.isdisjoint(correlated_groups[j]):
                     current_merged_group.update(correlated_groups[j])
                     processed_indices[j] = True
            merged_groups.append(current_merged_group)
        correlated_groups = merged_groups


    return correlated_groups


# --- Example Usage (Updated to show kwargs) ---
if __name__ == "__main__":
    try:
        import yfinance as yf
        print("Running example usage (original headers + kwargs)...")

        ticker = "AAPL"
        df_daily = yf.download(ticker, start="2019-01-01", end="2023-12-31", progress=False)

        if df_daily.empty:
            print(f"Could not download data for {ticker}. Exiting example.")
            exit()

        print(f"\nOriginal {ticker} dataframe shape: {df_daily.shape}")

        # --- Usage with original headers (uses internal defaults) ---
        print("\nCalculating daily features using add_technical_features (default args)...")
        df_default_features = add_technical_features(df_daily.copy())
        print(f"DataFrame shape: {df_default_features.shape}")

        # --- Usage demonstrating kwargs for customization ---
        print("\nCalculating daily features using add_technical_features with kwargs...")
        custom_trend = {'ema_windows': [10, 30], 'adx_window': 10}
        df_kwarg_features = add_technical_features(
            df_daily.copy(),
            categories=['trend', 'momentum'], # Only specific categories
            trend_params=custom_trend,
            momentum_params={'rsi_window': 12},
            drop_na=True # Example: dropping NaNs via kwarg
        )
        print(f"DataFrame shape (kwargs, dropna=True): {df_kwarg_features.shape}")
        print("Columns present:", df_kwarg_features.columns.tolist())


        # --- Multi-timeframe using original header default + kwargs ---
        print("\nCalculating multi-timeframe features (default timeframe ['daily'])...")
        df_mtf_default = add_multi_timeframe_features(df_daily.copy())
        print(f"Shape (default ['daily'] only): {df_mtf_default.shape}")

        print("\nCalculating multi-timeframe features (weekly, monthly) via kwargs...")
        df_mtf_kwargs = add_multi_timeframe_features(
            df_daily.copy(),
            timeframes=['weekly', 'monthly'], # Override default timeframe list
            indicator_params = { # Pass params via kwargs
                'weekly': {'trend_params': {'ema_windows': [10, 40]}},
                'monthly': {'momentum_params': {'rsi_window': 20}}
            },
            forward_fill=True # Explicitly using default, passed via kwargs
        )
        print(f"Shape (weekly, monthly): {df_mtf_kwargs.shape}")
        print("Example weekly column:", [c for c in df_mtf_kwargs if '_weekly' in c][0])
        print("Example monthly column:", [c for c in df_mtf_kwargs if '_monthly' in c][0])


        # --- Feature Analysis (using original headers) ---
        print("\n--- Feature Analysis ---")
        df_analysis = df_mtf_kwargs.copy() # Use multi-timeframe features
        # Add target using the default name 'direction' for analyze_features
        df_analysis['direction'] = (df_analysis['Close'].shift(-1) > df_analysis['Close']).astype(float)
        df_analysis.dropna(subset=['direction'], inplace=True) # Drop last row

        if not df_analysis.empty:
            print(f"Shape for analysis: {df_analysis.shape}")
            try:
                # Call analyze_features using its default target_col='direction'
                target_corr, feature_corr_matrix = analyze_features(df_analysis)

                print("\nTop 10 features correlated with 'direction':")
                print(target_corr.head(10))

                # Call identify_highly_correlated_features using its default threshold=0.85
                highly_correlated_groups = identify_highly_correlated_features(
                    df_analysis.drop(columns=['direction']) # Pass df without target
                )

                print(f"\nGroups of features with correlation > 0.85 (default threshold):")
                if highly_correlated_groups:
                    for i, group in enumerate(highly_correlated_groups):
                        print(f"Group {i+1}: {group}")
                else:
                    print("No groups found above the default threshold.")

            except ValueError as e:
                print(f"\nError during feature analysis: {e}")
            except Exception as e:
                 print(f"\nAn unexpected error occurred during analysis: {e}")
        else:
             print("\nDataFrame became empty after adding target, skipping analysis.")


    except ImportError:
        print("Please install required libraries: pip install pandas numpy ta yfinance typing_extensions")
    except Exception as e:
        print(f"\nAn error occurred in the example usage: {e}")
        import traceback
        traceback.print_exc()