# data_loader.py
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialDataLoader:
    """
    Handles loading, processing, and preparing financial time series data
    for sequence modeling tasks (e.g., Transformers).

    Manages data fetching (with caching), feature calculation (log returns,
    direction), sequence creation, chronological train/val/test splitting
    across multiple symbols, and robust feature scaling without data leakage.
    """
    def __init__(self, 
                 symbols: List[str],
                 start_date: str,
                 end_date: str,
                 seq_length: int = 60,
                target_horizon: int = 1,
                data_dir: str = "raw",
                use_cache: bool = True,
                price_col: str = "Close",
                target_col: str = "direction"
    ):
        """
        Initialize the financial data loader.
        
        Args:
            symbols: List of ticker symbols to load
            start_date: Start date for data retrieval (YYYY-MM-DD)
            end_date: End date for data retrieval (YYYY-MM-DD)
            seq_length: Length of sequences for transformer input
            target_horizon: Forecast horizon (in days) for directional prediction
            data_dir: Directory to store/load raw data
            use_cache: Whether to use cached data if available
        """
        # --- Input Validation ---
        if not symbols:
            raise ValueError("Symbols list cannot be empty.")
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format.")
        if start_date >= end_date:
             raise ValueError("Start date must be before end date.")
        if not isinstance(seq_length, int) or seq_length <= 0:
            raise ValueError("seq_length must be a positive integer.")
        if not isinstance(target_horizon, int) or target_horizon <= 0:
            raise ValueError("target_horizon must be a positive integer.")
        
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.target_horizon = target_horizon
        self.data_dir = data_dir
        self.use_cache = use_cache
        self.price_col = price_col
        self.target_col = target_col

        #Create data directory
        os.makedirs(data_dir,  exist_ok=True)

        # Initialize containers and state
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {} # Store data after calcs
        self.scaler: Optional[RobustScaler] = None
        self._data_loaded = False
        self._returns_calculated = False

        logging.info(f"FinancialLoader initialized for symbols: {symbols}")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads raw OHLCV data for symbols from cache or yfinance API.

        Returns:
            Dictionary mapping symbols to raw DataFrames. Skips symbols
            if data fetching fails or returns empty.

        Raises:
            FileNotFoundError: If price_col is not found in loaded data.
        """
        self.raw_data = {} # Clear previous raw data if called again
        logging.info("Starting data loading process...")
    
        for symbol in self.symbols:
            cache_filename = f"{symbol}_{self.start_date}_{self.end_date}.csv"
            cache_path = Path(self.data_dir) / cache_filename

            df: Optional[pd.DataFrame] = None # Define df before conditional assignment

            if self.use_cache and cache_path.exists():
                logging.info(f"Loading cached data for {symbol} from {cache_path}")
                try:
                    # Standardized loading approach without skiprows
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    
                    # Verify required columns exist
                    missing_cols = [col for col in REQUIRED_OHLCV_COLS if col not in df.columns]
                    if missing_cols:
                        raise ValueError(f"Missing columns in cached file: {missing_cols}")
                        
                    # Ensure proper DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                        
                except Exception as e:
                    logging.error(f"Error reading cache file {cache_path}: {e}")
                    logging.info(f"Will attempt to download fresh data for {symbol}")
                    df = None  # Force re-download
            else:
                logging.info(f"Fetching data for {symbol} from yfinance...")
                try:
                    # Fetch data
                    df_fetched = yf.download(symbol,
                                             start=self.start_date,
                                             end=self.end_date,
                                             auto_adjust=True,
                                             progress=False) # Disable yfinance progress bar

                    if df_fetched.empty:
                        logging.warning(f"No data fetched for {symbol} for the given date range.")
                        continue # Skip this symbol

                    # Save to cache
                    logging.info(f"Saving fetched data for {symbol} to {cache_path}")
                    df_fetched.to_csv(cache_path)
                    df = df_fetched

                except Exception as e:
                    logging.error(f"Failed to fetch or save data for {symbol}: {e}")
                    continue # Skip this symbol

            # --- Post-Load Validation (for both cached and fetched) ---
            if df is not None:
                 if df.empty:
                     logging.warning(f"Data for {symbol} is empty after loading/fetching. Skipping.")
                     continue
                 if self.price_col not in df.columns:
                     logging.error(f"'{self.price_col}' not found in columns for {symbol}. Available: {df.columns.tolist()}. Skipping.")
                     continue
                 # Ensure index is DatetimeIndex
                 if not isinstance(df.index, pd.DatetimeIndex):
                     logging.warning(f"Index for {symbol} is not DatetimeIndex. Attempting conversion.")
                     try:
                         df.index = pd.to_datetime(df.index)
                     except Exception as e:
                         logging.error(f"Failed to convert index to Datetime for {symbol}: {e}. Skipping.")
                         continue

                 self.raw_data[symbol] = df # Add valid DataFrame
            # ----

        if not self.raw_data:
             logging.warning("No data loaded successfully for any symbol.")
        else:
             logging.info(f"Successfully loaded data for symbols: {list(self.raw_data.keys())}")

        self._data_loaded = bool(self.raw_data) # Set flag based on successful loads
        self._returns_calculated = False # Reset flags if data is reloaded
        self.processed_data = {}
        self.scaler = None
        return self.raw_data
            
    def calculate_returns(self) -> Dict[str,pd.DataFrame]:
        """
        Calculates log returns and the target direction column.

        Operates on the data stored in `self.raw_data`. Results are stored
        in `self.processed_data`. Handles potential NaNs introduced.

        Returns:
            Dictionary mapping symbols to DataFrames with calculated features.

        Raises:
            RuntimeError: If data has not been loaded first via `load_data`.
        """
        if not self._data_loaded or not self.raw_data:
            raise RuntimeError("Data must be loaded using load_data() before calculating features.")
        
        logging.info("Calculating features (log returns, direction)...")
        self.processed_data = {} # Clear previous processed data

        for symbol, df in self.raw_data.items():
            processed_df = df.copy()
            
            # Get the price column (Handle MultiIndex if necessary)
            if isinstance(processed_df.columns, pd.MultiIndex):
                price_col = [(col[0], col[1]) for col in processed_df.columns if col[0] == self.price_col][0]
                direction_col = ('direction', symbol) if ('direction', symbol) in processed_df.columns else ('direction', '')
                log_return_col = ('log_return', symbol)
            else:
                price_col = self.price_col
                direction_col = self.target_col
                log_return_col = 'log_return'

            # CRITICAL: Print exactly what the column names are before calculating
            print(f"Direction column will be: {direction_col}")
            print(f"Available columns: {processed_df.columns[:5].tolist()}")

            # CRITICAL FIX: Calculate direction properly
            # 1. Calculate price changes directly (don't use shift which can create NaNs)
            price_series = processed_df[price_col].values
            direction_values = np.zeros(len(price_series))

            # 2. Set direction for all but the last target_horizon points
            for i in range(len(price_series) - self.target_horizon):
                # Compare current price with future price
                direction_values[i] = 1.0 if price_series[i + self.target_horizon] > price_series[i] else 0.0

            # 3. Set direction column with no NaNs
            processed_df[direction_col] = direction_values
            
            # Verify the direction column has no NaNs
            if np.isnan(processed_df[direction_col]).any():
                logging.error(f"Direction column for {symbol} still has NaNs! Fixing...")
                processed_df[direction_col] = processed_df[direction_col].fillna(0.0)
            
            # Verify we have both 0s and 1s
            unique_directions = processed_df[direction_col].unique()
            logging.info(f"Unique direction values for {symbol}: {unique_directions}")
            
            # Fill NaNs in the target column
            dir_nan_count = processed_df[direction_col].isna().sum()
            if dir_nan_count > 0:
                logging.warning(f"Found {dir_nan_count} NaNs in direction column for {symbol}")
                processed_df[direction_col] = processed_df[direction_col].fillna(0)
                
            # Calculate log returns
            processed_df[log_return_col] = np.log(processed_df[price_col] / processed_df[price_col].shift(1))

            # Drop initial NaN in log_return (careful with subset syntax)
            processed_df = processed_df.dropna(subset=[log_return_col])
            
            self.processed_data[symbol] = processed_df
        
        self._returns_calculated = bool(self.processed_data)
        return self.processed_data
    


    
    def prepare_sequences(self,
                          features: List[str],
                          target: str="direction",
                          train_ratio: float=0.7,
                          val_ratio:float=0.15) -> Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray]]:
        """
        Prepares sequence data (X, y) for modeling, performs train/val/test
        splitting chronologically, aggregates data across symbols, and applies
        scaling (fitting only on the training set).

        Args:
            features: List of column names to use as input features.
            train_ratio: Proportion of data for training (0.0 to 1.0).
            val_ratio: Proportion of data for validation (0.0 to 1.0).
                       Test ratio is inferred as 1.0 - train_ratio - val_ratio.

        Returns:
            Tuple[Dict, Dict]: Dictionaries for X and y, each containing
                               'train', 'val', 'test' NumPy arrays.
                               X arrays have shape [n_samples, seq_length, n_features].
                               y arrays have shape [n_samples].

        Raises:
            RuntimeError: If features have not been calculated via `calculate_features`.
            ValueError: If feature names are invalid or ratios are incorrect.
        """
        if not self._returns_calculated or not self.processed_data:
            raise RuntimeError("Features must be calculated using calculate_features() before preparing sequences.")
        if not features:
            raise ValueError("Features list cannot be empty.")
        if not (0.0 <= train_ratio <= 1.0):
             raise ValueError("train_ratio must be between 0.0 and 1.0")
        if not (0.0 <= val_ratio <= 1.0):
             raise ValueError("val_ratio must be between 0.0 and 1.0")
        if train_ratio + val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio cannot exceed 1.0")
        test_ratio = 1.0 - train_ratio - val_ratio

        logging.info(f"Preparing sequences with split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")

        # --- Stage 1: Create sequences and split per symbol ---
        temp_X = {'train': [], 'val': [], 'test': []}
        temp_y = {'train': [], 'val': [], 'test': []}
        min_required_len = self.seq_length + 1 # Need at least seq_length history + 1 target

        logging.info(f"Preparing sequences with features: {features}")
        
    
        # After validation, before creating sequences
        for symbol, df in self.processed_data.items():
            # Log column structure
            logging.info(f"Column structure for {symbol}: {df.columns[:5].tolist()}")
            
            # Check for feature existence
            available_features = [f for f in features if f in df.columns]
            if len(available_features) < len(features):
                missing = set(features) - set(available_features)
                logging.warning(f"Missing features for {symbol}: {missing}")
            
            logging.info(f"Using {len(available_features)} features for sequence generation")


            if isinstance(df.columns, pd.MultiIndex):
                # IMPORTANT: Extract the direction data BEFORE flattening columns
                direction_data = None
                direction_col_multiindex = None
                
                # Find direction column in MultiIndex
                for col in df.columns:
                    if col[0] == 'direction':
                        direction_col_multiindex = col
                        direction_data = df[col].values  # Store the actual values
                        logging.info(f"Extracted direction data with shape: {direction_data.shape}")
                        break
                
                # Flatten columns as before
                new_cols = []
                for col in df.columns:
                    if col[1] and col[1] != '':
                        new_cols.append(f"{col[0]}_{col[1]}")
                    else:
                        new_cols.append(f"{col[0]}")
                
                # Apply new column names
                df.columns = new_cols
                logging.info(f"Converted MultiIndex columns to flat format for {symbol}")
                
                close_col = f"Close_{symbol}"
                if close_col in df.columns:
                    logging.info(f"Directly calculating direction column from {close_col}")
                    
                    # Create new direction column
                    df['direction'] = (df[close_col].shift(-self.target_horizon) > df[close_col]).astype(float)
                    
                    # Drop NaNs that result from the shift operation
                    df['direction'] = df['direction'].fillna(0)
                    
                    # Verify we have valid values
                    zeros = (df['direction'] == 0).sum()
                    ones = (df['direction'] == 1).sum()
                    logging.info(f"Created direction column with {zeros} zeros and {ones} ones")
                    
                    # Set target column name
                    self.target_col = 'direction'
                    
                    # CRITICAL: Skip the next section that would overwrite good data
                    direction_data = None  # This prevents re-adding the NaN data
                else:
                    logging.error(f"Cannot find Close price column {close_col}")

                # If we found direction data, ensure it's preserved in the flattened DataFrame
                if direction_data is not None:
                    # Re-add the direction data to the flattened DataFrame
                    target_col_name = 'direction'  # Use simple name in flattened format
                    df[target_col_name] = direction_data
                    self.target_col = target_col_name
                    logging.info(f"Re-added direction data to flattened DataFrame with shape: {df[target_col_name].shape}")
                    
                    # Verify the data contains non-NaN values
                    non_nan_count = np.sum(~np.isnan(df[target_col_name].values))
                    logging.info(f"Direction column contains {non_nan_count} non-NaN values out of {len(df)}")
                else:
                    logging.warning(f"Could not find direction column in MultiIndex for {symbol}")
                
        

            logging.info(f"All columns after flattening: {df.columns.tolist()}")
            logging.info(f"Looking for target column: {self.target_col}")
            if self.target_col not in df.columns:
                logging.error(f"Target column '{self.target_col}' not found after processing")
                
                # Look for alternatives
                direction_cols = [col for col in df.columns if 'direction' in col.lower()]
                if direction_cols:
                    self.target_col = direction_cols[0]
                    logging.info(f"Found alternative target column: {self.target_col}")
                else:
                    # Create it from Close price
                    close_cols = [col for col in df.columns if 'close' in col.lower()]
                    if close_cols:
                        close_col = close_cols[0]
                        df["direction"] = (df[close_col].shift(-self.target_horizon) > df[close_col]).astype(float)
                        self.target_col = "direction"
                        logging.info(f"Created direction column from {close_col}")
                    else:
                        logging.error(f"Cannot create direction column. No close price column found.") 
                
            # Verify all features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logging.warning(f"Missing features for {symbol}: {missing_features}")
                # Add missing features as zeros
                for feature in missing_features:
                    df[feature] = 0.0
                    
            # Fill NaNs in all feature columns
            for feature in features:
                if feature in df.columns and df[feature].isna().any():
                    df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
            # Update processed data
            self.processed_data[symbol] = df
            
            # Verify features and target exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logging.error(f"Missing feature(s) {missing_features} in data for {symbol}. Skipping.")
                continue
            if self.target_col not in df.columns:
                 logging.error(f"Target column '{self.target_col}' not found for {symbol}. Skipping.")
                 continue

            feature_data = df[features].values
            target_data = df[self.target_col].values

            if len(feature_data) < min_required_len:
                logging.warning(f"Insufficient data length ({len(feature_data)}) for {symbol} "
                                f"to create sequence of length {self.seq_length}. Skipping.")
                continue

            # Create sequences for this symbol (unscaled data)
            X_sym, y_sym = self._create_sequences(feature_data, target_data)

            if X_sym.size == 0: # Should not happen if length check above passed, but safe check
                logging.warning(f"Sequence creation resulted in empty array for {symbol}. Skipping.")
                continue

            # Calculate split points based on number of sequences
            n_sequences = len(X_sym)
            train_end_idx = int(n_sequences * train_ratio)
            val_end_idx = train_end_idx + int(n_sequences * val_ratio)

            # --- Validate Split Indices ---
            # Ensure each split gets at least one sample if ratio > 0
            # (adjust if strict adherence to ratio is less important than having data in split)
            valid_split = True
            if train_ratio > 0 and train_end_idx == 0: valid_split = False
            if val_ratio > 0 and val_end_idx <= train_end_idx : valid_split = False
            if test_ratio > 0 and n_sequences <= val_end_idx: valid_split = False
            if n_sequences < 3 and (train_ratio > 0 and val_ratio > 0 and test_ratio > 0) : valid_split = False # Need at least 3 seqs for 3 splits

            if not valid_split:
                 logging.warning(f"Insufficient sequences ({n_sequences}) for {symbol} "
                                 f"to perform valid train/val/test split with given ratios. Skipping.")
                 continue
            # ---

            # Append split data to temporary lists
            temp_X["train"].append(X_sym[:train_end_idx])
            temp_y["train"].append(y_sym[:train_end_idx])

            temp_X["val"].append(X_sym[train_end_idx:val_end_idx])
            temp_y["val"].append(y_sym[train_end_idx:val_end_idx])

            temp_X["test"].append(X_sym[val_end_idx:])
            temp_y["test"].append(y_sym[val_end_idx:])

        # --- Stage 2: Aggregate sequences across symbols ---
        X_dict = {}
        y_dict = {}
        for split in ["train", "val", "test"]:
            if temp_X[split]: # Check if list is not empty
                X_dict[split] = np.vstack(temp_X[split])
                y_dict[split] = np.concatenate(temp_y[split])
                logging.info(f"Aggregated {split} data shape: X={X_dict[split].shape}, y={y_dict[split].shape}")
            else:
                # Create empty arrays with correct dimensions if a split is empty
                # Infer number of features from other splits if possible, else use len(features)
                n_features = len(features)
                if split != 'train' and 'train' in X_dict and X_dict['train'].size > 0:
                    n_features = X_dict['train'].shape[2]
                elif split != 'val' and 'val' in X_dict and X_dict['val'].size > 0:
                     n_features = X_dict['val'].shape[2]

                X_dict[split] = np.empty((0, self.seq_length, n_features), dtype=np.float32) # Specify dtype
                y_dict[split] = np.empty((0,), dtype=np.int64) # Assuming integer target
                logging.warning(f"No data aggregated for the '{split}' split.")

        # --- Stage 3: Scale features (Fit ONLY on Training Data) ---
        self.scaler = None # Reset scaler
        if X_dict['train'].size > 0:
            logging.info("Fitting scaler on training data...")
            self.scaler = RobustScaler()

            # Reshape train data for scaler: [samples * seq_len, features]
            n_samples_train, seq_len_train, n_features_train = X_dict['train'].shape
            X_train_reshaped = X_dict['train'].reshape(-1, n_features_train)

            self.scaler.fit(X_train_reshaped)
            logging.info("Scaler fitted.")

            # Transform all splits using the *fitted* scaler
            for split in ["train", "val", "test"]:
                if X_dict[split].size > 0:
                    logging.debug(f"Scaling {split} data...")
                    n_samples, seq_len, n_features = X_dict[split].shape
                    X_reshaped = X_dict[split].reshape(-1, n_features)
                    X_scaled_reshaped = self.scaler.transform(X_reshaped)
                    # Reshape back to 3D: [samples, seq_len, features]
                    X_dict[split] = X_scaled_reshaped.reshape(n_samples, seq_len, n_features)
        else:
            logging.warning("No training data available. Scaler was not fitted.")

        logging.info("Sequence preparation and scaling complete.")
        return X_dict, y_dict

    def _create_sequences(self, feature_data: np.ndarray, target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper method to create overlapping sequences from feature and target data.

        The target `y[i]` corresponds to `target_data` at the time step *after*
        the end of the sequence `X[i]`.

        Args:
            feature_data: NumPy array of features (n_timesteps, n_features).
            target_data: NumPy array of targets (n_timesteps,).

        Returns:
            Tuple[np.ndarray, np.ndarray]: X sequences ([n_sequences, seq_length, n_features]),
                                           y targets ([n_sequences,]).
        """
        if not isinstance(target_data, np.ndarray):
            target_data = np.array(target_data)

        # Verify we have valid values
        if np.all(np.isnan(target_data)):
            logging.error(f"Target data contains only NaNs! Creating artificial targets.")
            # Check if we can calculate the target from feature_data (if feature_data has Close price)
            # This is a fallback but not ideal - better to fix the issue upstream
            target_data = np.zeros(len(target_data))
        # Ensure feature_data is properly shaped
        if isinstance(feature_data, pd.DataFrame):
            # Convert DataFrame to numpy array
            feature_names = feature_data.columns.tolist()
            feature_data = feature_data.values
        
        # Ensure feature_data is 2D [n_samples, n_features]
        if len(feature_data.shape) == 1:
            feature_data = feature_data.reshape(-1, 1)
            
         #Check if target_data contains any non-NaN values
        valid_targets = ~np.isnan(target_data)
        if not np.any(valid_targets):
            logging.error(f"Target data contains only NaNs! Creating artificial targets (all zeros).")
            # Create artificial targets (all zeros) since we need something
            target_data = np.zeros_like(target_data)
        elif np.any(np.isnan(target_data)):
            # Some NaNs, use the most common valid value to fill
            valid_values = target_data[valid_targets]
            most_common = np.bincount(valid_values.astype(int)).argmax()
            logging.warning(f"Filling {np.isnan(target_data).sum()} NaN targets with most common value: {most_common}")
            target_data = np.nan_to_num(target_data, nan=float(most_common))
        
        # Create sequences
        X, y = [], []
        for i in range(len(feature_data) - self.seq_length):
            # This ensures each sequence item contains all features
            # Shape: [seq_length, n_features]
            X.append(feature_data[i:i+self.seq_length])
            # Target for the sequence
            y.append(target_data[i+self.seq_length])
        
        # Ensure we have sequences
        if not X:
            logging.warning("No sequences could be created! Using empty arrays.")
            return np.array([], dtype=np.float32).reshape(0, self.seq_length, feature_data.shape[1]), np.array([], dtype=np.float32)
        
        # Convert to numpy arrays
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)
        
        # Verify shapes are as expected
        logging.info(f"Created sequences with shape X: {X_np.shape}, y: {y_np.shape}")
        

        
        # Make sure we return the arrays!
        return X_np, y_np
    

    def get_scaler(self) -> Optional[RobustScaler]:
        """Returns the fitted scaler instance, or None if not fitted."""
        return self.scaler

    def get_processed_data(self) -> Dict[str, pd.DataFrame]:
        """Returns the dictionary of DataFrames after feature calculation."""
        if not self._returns_calculated:
             logging.warning("Features have not been calculated yet.")
        return self.processed_data
            
            
            
            
    # Example Usage (Optional)
if __name__ == "__main__":
    symbols_to_load = ['AAPL', 'MSFT', 'NVDA']
    start = '2020-01-01'
    end = '2023-12-31'

    loader = FinancialLoader(
        symbols=symbols_to_load,
        start_date=start,
        end_date=end,
        seq_length=60,
        target_horizon=5, # Predict direction 5 days ahead
        use_cache=True
    )

    try:
        # --- Run the full pipeline ---
        loader.load_data()
        loader.calculate_returns()

        # Define features to use (must include 'log_return' if used)
        features_to_use = ['Close', 'Volume', 'log_return']
        # Add more technical indicators here if calculated

        X_data, y_data = loader.prepare_sequences(
            features=features_to_use,
            train_ratio=0.7,
            val_ratio=0.15
        )

        # --- Access data ---
        print("\n--- Data Shapes ---")
        for split in ['train', 'val', 'test']:
            if X_data[split].size > 0:
                print(f"{split.capitalize()} X: {X_data[split].shape}, y: {y_data[split].shape}")
            else:
                print(f"{split.capitalize()}: No data")

        # --- Access Scaler ---
        scaler_instance = loader.get_scaler()
        if scaler_instance:
            print("\nScaler was fitted.")
            # print(f"Scaler center: {scaler_instance.center_}") # Example property
        else:
            print("\nScaler was not fitted (likely no training data).")

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logging.error(f"An error occurred during data loading/processing: {e}")
    except Exception as e:
         logging.exception(f"An unexpected error occurred: {e}") # Log full traceback        
       

                    
             
             



