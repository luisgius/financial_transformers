import torch
import torch.nn as nn
import torch.optim as optim
import logging

class Time2Vec(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(input_dim, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(input_dim, embed_dim-1))
        self.b = nn.Parameter(torch.randn(embed_dim-1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear time encoding
        v0 = torch.matmul(x, self.w0) + self.b0
        
        # Periodic encodings
        v1 = torch.sin(torch.matmul(x, self.w) + self.b)
        
        # Combine linear and periodic components
        return torch.cat([v0, v1], -1)
    

def generate_causal_mask(seq_len: int) -> torch.Tensor:
    # Create mask where True values are positions that should be *masked out*
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


            
def scaled_dot_product_attention(query, key, value, mask=None):
    """Calculate attention weights and apply them to values."""
    # Calculate attention scores
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale attention scores
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / (d_k ** 0.5) 

    # Scale attention scores
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / (d_k ** 0.5)
    
    # Apply mask (if provided)
    if mask is not None:
        # Convert mask for proper broadcasting (mask has True where attention is blocked)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask, float('-inf'))
    
    # Calculate attention weights
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights    
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multiple heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(output)
        
        return output, attention_weights
   
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.2) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_head, dropout)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
       
       norm_x= self.norm1(x)
       attention_output, attention_weights = self.attention(norm_x, norm_x, norm_x, mask)
       x = x + self.dropout(attention_output)

       norm_x2 = self.norm2(x)
       ff_output = self.feed_forward(norm_x2)
       x = x + self.dropout(ff_output)

       return x, attention_weights
        
        
class FinancialTransformer(nn.Module):
    def __init__(  self, 
        input_dim: int, 
        d_model: int, 
        n_head: int, 
        n_layers: int, 
        d_ff: int, 
        seq_length: int, 
        dropout: float = 0.1
    ) -> None:
        
        super().__init__()

        # Input embedding (projection to d_model dimension)
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Time Encoding
        self.time_encoding = Time2Vec(1, d_model)

        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layers)
        ])

        # IMPORTANT: Add the dropout layer that's missing
        self.dropout = nn.Dropout(dropout)

        # Output head for Binary Classification
        self.output_head = nn.Sequential(
           nn.Linear(d_model, d_model//2),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(d_model//2, 1),
           nn.Sigmoid()
        )

        # Register causal mask buffer
        mask = generate_causal_mask(seq_length)
        self.register_buffer("mask", mask)
        
    def forward(self, x: torch.Tensor, return_attention: bool = True) -> tuple:
        """
        Process input through the financial transformer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_length, input_dim]
        return_attention : bool
            Whether to return attention weights
            
        Returns:
        --------
        tuple[torch.Tensor, list]
            - Predictions of shape [batch_size, 1] (or [batch_size] if squeezed)
            - List of attention weights from each layer (if return_attention=True)
        """

            # Reshape protection
        original_shape = x.shape
        #logging.info(f"Input tensor shape: {original_shape}")

        # First issue: We're getting individual samples instead of batches
        if len(original_shape) == 2:
            # This is a single sample with shape [seq_length, features]
            # Add batch dimension
            x = x.unsqueeze(0)  # Makes [1, seq_length, features]
            logging.info(f"Added batch dimension: {x.shape}")
        
        # Still make sure we have 3D input at this point
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D tensor after reshaping, got shape: {x.shape}")
        
        batch_size, dim1, dim2 = x.shape
        
        # Determine which dimension is likely features vs. sequence
        expected_features = self.input_embedding.in_features
        
        # If dimensions appear swapped (sequence length and features)
        if dim2 == expected_features and dim1 != expected_features:
            # Dimensions are already correct [batch, seq, features]
            pass
        elif dim1 == expected_features and dim2 != expected_features:
            # Need to swap dimensions [batch, features, seq] -> [batch, seq, features]
            logging.info(f"Transposing dimensions 1 and 2 to correct sequence and feature layout")
            x = x.transpose(1, 2)
            logging.info(f"Transposed shape: {x.shape}")
    
           
        
        batch_size, seq_length, input_dim = x.shape
        
        # If input dimension doesn't match model's input dimension, log a warning
        if input_dim != self.input_embedding.in_features:
            logging.warning(f"Input dimension mismatch: got {input_dim}, expected {self.input_embedding.in_features}")
            # Try to fix by padding or truncating
            if input_dim < self.input_embedding.in_features:
                # Pad with zeros
                padding = torch.zeros(batch_size, seq_length, self.input_embedding.in_features - input_dim, device=x.device)
                x = torch.cat([x, padding], dim=2)
                logging.info(f"Padded input features to shape: {x.shape}")
            else:
                # Truncate
                x = x[:, :, :self.input_embedding.in_features]
                logging.info(f"Truncated input features to shape: {x.shape}")

        # Create positional input for time2vec
        positions = torch.arange(0, seq_length, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(x.device)
        positions = positions.expand(batch_size, seq_length,1)

        # Apply Time2Vec encoding
        time_encoding = self.time_encoding(positions)

        # Project input to d_model dimension
        embedded = self.input_embedding(x)

        # Add time encoding
        embedded = embedded + time_encoding
        embedded = self.dropout(embedded)

        # Process through transformer layers
        attention_weights = []

        for layer in self.transformer_layers:
            embedded, attention_weights_layer = layer(embedded, self.mask)
            if return_attention:
                attention_weights.append(attention_weights_layer)

        # Use final token for prediction
        final_hidden = embedded[:,-1,:]

        # Generate prediction 
        prediction = self.output_head(final_hidden)

        prediction = prediction.view(batch_size, -1)
        
        # Standardize output shape
        prediction = prediction.squeeze(-1)  # Consistently return [batch_size] shape
        
        return (prediction, attention_weights) if return_attention else prediction

def train_financial_transformer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple[nn.Module, dict]:
    """
    Train the financial transformer model.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay (L2 regularization)
    patience : int
        Early stopping patience
    device : str
        Device to train on
        
    Returns:
    --------
    tuple[nn.Module, dict]
        - Trained model
        - Training history
    """
    model = model.to(device)
    
    # Binary cross entropy loss
    criterion = nn.BCELoss()
    
    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=15,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Update the training loop handling of outputs:
        for batch_idx, (data, target) in enumerate(train_loader):
            # Log batch shapes
            if batch_idx == 0:  # Just log first batch for debugging
                logging.info(f"Batch {batch_idx}: data={data.shape}, target={target.shape}")
            
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                output = outputs[0]  # First element contains predictions
            else:
                output = outputs
            
            # Ensure compatible shapes
            if output.shape != target.shape:
                if output.dim() > target.dim():
                    output = output.squeeze(-1)  # Remove extra dimension
                if target.dim() > output.dim():
                    target = target.squeeze(-1)
            
            # Ensure target is float for BCE loss
            target = target.float()
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Accumulate metrics - WITH SAFEGUARDS
            train_loss += loss.item() * data.size(0)  # Weighted by batch size
            predicted = (output > 0.5).float()
            
            # Only update if shapes match
            if predicted.shape == target.shape:
                train_correct += (predicted == target.float()).sum().item()
                train_total += target.numel()
            
            # Log progress for some batches
            if batch_idx % 50 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss.item():.6f}")
                
         # Calculate epoch metrics WITH SAFETY CHECK
        if train_total > 0:
            avg_train_loss = train_loss / len(train_loader.dataset)
            train_accuracy = train_correct / train_total
        else:
            logging.warning("No samples were processed correctly in this epoch!")
            avg_train_loss = float('inf')
            train_accuracy = 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output, _ = model(data)

                # Reshape target to match output
                if output.shape != target.shape:
                    # If output is [batch_size] and target is [batch_size, 1]
                    if output.dim() == 1 and target.dim() == 2 and target.size(1) == 1:
                        target = target.squeeze(-1)
                    # If output is [batch_size, 1] and target is [batch_size]
                    elif output.dim() == 2 and output.size(1) == 1 and target.dim() == 1:
                        output = output.squeeze(-1)
                    
                loss = criterion(output, target.float())
                
                # Accumulate metrics
                val_loss += loss.item()
                predicted = (output.squeeze(-1) > 0.5).float()
                val_correct += (predicted == target).sum().item()
                val_total += target.size(0)
        
        # Calculate average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.6f}, Val Acc: {val_accuracy:.4f}")
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history       
        
       
        

