import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.serialization
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import questionary
from scipy.optimize import differential_evolution, minimize
from bayes_opt import BayesianOptimization
import warnings
import json
from tqdm import tqdm
import random

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
tf.random.set_seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.serialization.add_safe_globals([MinMaxScaler , StandardScaler, RobustScaler])

# === Configuration ===
class Config:
    SEED = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    LEARNING_RATE_MIN = 0.00001
    EARLY_STOPPING_PATIENCE = 25
    REDUCE_LR_PATIENCE = 10
    REGULARIZATION = l1_l2(l1=1e-6, l2=1e-5)
    FEATURE_COLS = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
    TARGET_COLS = ['S11', 'S21']
    IMAGE_SAVE_DIR = "Graphs"
    MODEL_DIR = "hybrid_models"
    DATA_PATH = "Coupled_Line_Dataset.csv"
    K_FOLDS = 5
    N_FREQ_POINTS = 33  # Default frequency points for curve-based model
    
    # Physical constraints
    DESIGN_BOUNDS = {
        'l_s': (6, 35),
        'l_2': (6, 35),
        'l_1': (6, 25),
        's_2': (0.15, 0.6),
        's_1': (0.15, 0.6),
        'w_s': (0.6, 2.5),
        'w_2': (0.5, 2.5),
        'w_1': (0.2, 2.5)
    }

Path(Config.IMAGE_SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)

# === Utility Functions ===
def standardize_frequency(freq):
    """Convert frequency string to MHz"""
    freq = str(freq).replace('\xa0', ' ').strip().lower().strip('"').strip("'")
    if 'mhz' in freq:
        return float(freq.replace(' mhz', ''))
    elif 'ghz' in freq:
        return float(freq.replace(' ghz', '')) * 1000
    return float(freq)

def create_enhanced_features(design_params, freq):
    """Enhanced feature engineering combining both approaches"""
    ls, l2, l1, s2, s1, ws, w2, w1 = design_params
    
    features = [
        ls, l2, l1, s2, s1, ws, w2, w1, freq,
        # Your ratios
        w1/ws, s1/s2, l1/ls, w2/ws,
        # Your sums
        ws + w1 + w2, s1 + s2, l1 + l2 + ls,
        # Microwave physics features
        (w1/s1) / (w2/s2),  # Impedance ratio proxy
        l1 * freq / 300,  # Electrical length
        np.sqrt(s1 * s2) / (w1 + w2),  # Coupling factor
        l1 / (300 / freq),  # Wavelength ratio
        # Additional features
        (ws + w1 + w2) / (l1 + l2 + ls),  # Aspect ratio
        (s1 + s2) / (w1 + w2),  # Spacing to width ratio
    ]
    
    return np.array(features, dtype=np.float32)

# === PyTorch Models (Friend's Approach Enhanced) ===
class EnhancedCVAE(nn.Module):
    """Enhanced Conditional VAE for inverse design"""
    def __init__(self, s_dim, design_dim, latent_dim=16, hidden_dim=256):
        super().__init__()

        self.s_dim = s_dim
        self.design_dim = design_dim
        self.latent_dim = latent_dim
        
        # Encoder
        # self.encoder_fc1 = nn.Linear(s_dim + 1, hidden_dim)
        # self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(s_dim + 1, hidden_dim),  # +1 for frequency
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),     # Layer 2  
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, design_dim)
        )
        # self.decoder_fc1 = nn.Linear(latent_dim + 1, hidden_dim // 2)
        # self.decoder_fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        # self.decoder_out = nn.Linear(hidden_dim, design_dim)

        self._init_weights()
        
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

        # Constraint layer for physical bounds
        # self.constraint_layer = nn.Sequential(
        #     nn.Linear(design_dim, design_dim),
        #     nn.ReLU()
        # )
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, s_params, freq):
        """Encode S-parameters and frequency to latent space"""

        if s_params.dim() == 1:
            s_params = s_params.unsqueeze(0)

        if freq.dim() == 1:
            freq = freq.unsqueeze(-1)

        
        x = torch.cat([s_params, freq], dim=-1)
        h = self.encoder(x)

        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        # Prevent numerical instability
        logvar = torch.clamp(logvar, -10, 5)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, freq):
        """Decode from latent space to design parameters"""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        if freq.dim() == 1:
            freq = freq.unsqueeze(-1)

        z_cond = torch.cat([z, freq], dim=-1)
        x = self.decoder(z_cond)
        return x
    
    def forward(self, s_params, freq):
        """Full forward pass"""
        mu, logvar = self.encode(s_params, freq)
        z = self.reparameterize(mu, logvar)
        design_pred = self.decode(z, freq)
        return design_pred, mu, logvar

class CurvePredictor(nn.Module):
    """Predicts full S-parameter curves across frequency"""
    def __init__(self, design_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(design_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# === TensorFlow Models (Your Approach Enhanced) ===
def create_hybrid_forward_model(input_shape):
    """Hybrid forward model for single frequency prediction"""
    inputs = Input(shape=(input_shape,))
    
    # Enhanced feature processing
    x = Dense(256, activation='swish', kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, activation='swish', kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Multiple outputs for better learning
    s11_branch = Dense(64, activation='swish')(x)
    s21_branch = Dense(64, activation='swish')(x)
    
    s11_output = Dense(1, name='s11_output')(s11_branch)
    s21_output = Dense(1, name='s21_output')(s21_branch)
    
    outputs = Concatenate()([s11_output, s21_output])
    
    return Model(inputs, outputs, name="hybrid_forward")

def create_frequency_interpolator():
    """Maps between curve predictions and single frequency points"""
    inputs = Input(shape=(66,))  # 33 S11 + 33 S21 points
    freq_input = Input(shape=(1,))
    
    # Concatenate curve and frequency
    x = Concatenate()([inputs, freq_input])
    
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='swish')(x)
    outputs = Dense(2)(x)  # S11, S21 at specific frequency
    
    return Model([inputs, freq_input], outputs, name="frequency_interpolator")

# === Data Loading and Processing ===
class HybridDataProcessor:
    """Processes data for both single-point and curve-based models"""
    
    @staticmethod
    def load_and_process_data(path):
        """Load data and create both single-point and curve datasets"""
        df = pd.read_csv(path, skiprows=1, header=None, 
                        names=Config.FEATURE_COLS + Config.TARGET_COLS)
        
        # Standardize frequency
        df['freq'] = df['freq'].apply(standardize_frequency)
        
        # Create curve dataset (friend's approach)
        curve_data = HybridDataProcessor._create_curve_dataset(df)
        
        # Create single-point dataset (your approach)
        single_data = HybridDataProcessor._create_single_dataset(df)
        
        return curve_data, single_data
    
    @staticmethod
    def _create_curve_dataset(df):
        """Group by design to get frequency sweeps"""
        design_cols = Config.FEATURE_COLS[:-1]  # Exclude freq
        
        grouped = df.groupby(design_cols)
        designs = []
        s_curves = []
        
        for name, group in grouped:
            # Sort by frequency
            group = group.sort_values('freq')
            
            # Check if we have enough frequency points
            if len(group) >= Config.N_FREQ_POINTS:
                # Take N_FREQ_POINTS evenly spaced points
                indices = np.linspace(0, len(group)-1, Config.N_FREQ_POINTS, dtype=int)
                group = group.iloc[indices]
            
            # Extract curves
            s11_curve = group['S11'].values[:Config.N_FREQ_POINTS]
            s21_curve = group['S21'].values[:Config.N_FREQ_POINTS]
            
            # Pad if necessary
            if len(s11_curve) < Config.N_FREQ_POINTS:
                pad_len = Config.N_FREQ_POINTS - len(s11_curve)
                s11_curve = np.pad(s11_curve, (0, pad_len), mode='edge')
                s21_curve = np.pad(s21_curve, (0, pad_len), mode='edge')
            
            designs.append(np.array(name, dtype=np.float32))
            s_curves.append(np.concatenate([s11_curve, s21_curve]))
        
        return {
            'designs': np.stack(designs, axis=0),
            's_curves': np.stack(s_curves, axis=0),
            'freq_grid': np.linspace(df['freq'].min(), df['freq'].max(), Config.N_FREQ_POINTS)
        }
    
    @staticmethod
    def _create_single_dataset(df):
        """Create single frequency point dataset"""
        X = []
        y = []
        
        for _, row in df.iterrows():
            design_params = row[Config.FEATURE_COLS[:-1]].values
            freq = row['freq']
            
            # Enhanced features
            features = create_enhanced_features(design_params, freq)
            
            X.append(features)
            y.append([row['S11'], row['S21']])
        
        return {
            'X': np.array(X, dtype=np.float32),
            'y': np.array(y, dtype=np.float32)
        }

# === Training Functions ===
class HybridTrainer:
    """Trains both curve-based and single-point models"""
    
    def __init__(self, config):
        self.config = config
        
    def train_curve_predictor(self, curve_data):
        """Train curve predictor (PyTorch)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare data
        X = curve_data['designs']
        y = curve_data['s_curves']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=Config.SEED
        )
        
        # Scale
        design_scaler = StandardScaler()
        S_scaler = StandardScaler()
        
        X_train_scaled = design_scaler.fit_transform(X_train)
        y_train_scaled = S_scaler.fit_transform(y_train)
        
        # Create dataset
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        # Initialize model
        model = CurvePredictor(
            design_dim=X.shape[1],
            output_dim=y.shape[1],
            hidden_dim=256
        ).to(device)
        
        # Training
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(Config.EPOCHS // 2):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Curve Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.6f}")
        
        # Save model and scalers
        torch.save({
            'model_state_dict': model.state_dict(),
            'design_scaler': design_scaler,
            'S_scaler': S_scaler
        }, os.path.join(Config.MODEL_DIR, 'curve_predictor.pt'))
        
        return model, design_scaler, S_scaler
    
    def train_single_predictor(self, single_data):
        """Train single frequency predictor (TensorFlow)"""
        X = single_data['X']
        y = single_data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.SEED
        )
        
        # Scale
        X_scaler = RobustScaler()
        y_scaler = RobustScaler()
        
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_val_scaled = X_scaler.transform(X_val)
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        
        # Create model
        model = create_hybrid_forward_model(X_train_scaled.shape[1])
        model.compile(
            optimizer=Adam(Config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=Config.REDUCE_LR_PATIENCE,
                min_lr=Config.LEARNING_RATE_MIN
            ),
            ModelCheckpoint(
                os.path.join(Config.MODEL_DIR, 'best_single_model.h5'),
                save_best_only=True
            )
        ]
        
        # Train
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save
        model.save(os.path.join(Config.MODEL_DIR, 'single_predictor.h5'))
        joblib.dump(X_scaler, os.path.join(Config.MODEL_DIR, 'single_X_scaler.pkl'))
        joblib.dump(y_scaler, os.path.join(Config.MODEL_DIR, 'single_y_scaler.pkl'))
        
        return model, X_scaler, y_scaler
    

    def train_inverse_cvae(self, single_data):
        """Train inverse CVAE (PyTorch) - STABILIZED VERSION"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training CVAE on device: {device}")
        
        X = single_data['X']
        y = single_data['y']
        
        print(f"Training CVAE with {len(X)} samples")
        
        # Prepare data for CVAE: S-parameters -> Design
        original_designs = X[:, :8]  # First 8 are original design params
        s_params = y
        
        # Add frequency to s_params for conditioning
        freqs = X[:, 8]  # Frequency is 9th feature
        
        # Check for NaN or Inf values
        print(f"Checking for NaN/Inf values...")
        print(f"Original designs: {np.any(np.isnan(original_designs))} NaN, {np.any(np.isinf(original_designs))} Inf")
        print(f"S-params: {np.any(np.isnan(s_params))} NaN, {np.any(np.isinf(s_params))} Inf")
        print(f"Freqs: {np.any(np.isnan(freqs))} NaN, {np.any(np.isinf(freqs))} Inf")
        
        # Clean data
        original_designs = np.nan_to_num(original_designs)
        s_params = np.nan_to_num(s_params)
        freqs = np.nan_to_num(freqs)
        
        # Split
        s_train, s_test, d_train, d_test, f_train, f_test = train_test_split(
            s_params, original_designs, freqs, test_size=0.2, random_state=Config.SEED
        )
        
        # Scale using MinMaxScaler for better stability
        from sklearn.preprocessing import MinMaxScaler
        s_scaler = MinMaxScaler(feature_range=(-1, 1))
        d_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        s_train_scaled = s_scaler.fit_transform(s_train)
        d_train_scaled = d_scaler.fit_transform(d_train)
        
        # Create dataset
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(s_train_scaled, dtype=torch.float32),
            torch.tensor(d_train_scaled, dtype=torch.float32),
            torch.tensor(f_train, dtype=torch.float32)
        )
        
        batch_size = min(32, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # SIMPLIFIED AND STABLE CVAE ARCHITECTURE
        class StableCVAE(nn.Module):
            def __init__(self, s_dim, design_dim, latent_dim=8, hidden_dim=128):
                super().__init__()
                
                # Encoder - simpler architecture
                self.s_dim = s_dim
                self.design_dim = design_dim
                self.latent_dim = latent_dim
                self.encoder = nn.Sequential(
                    nn.Linear(s_dim + 1, hidden_dim),  # +1 for frequency
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU()
                )
                
                self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
                self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)
                
                # Decoder - simpler architecture
                self.decoder = nn.Sequential(
                nn.Linear(latent_dim + 1, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, design_dim)
                )
                # self.decoder_fc1 = nn.Linear(latent_dim + 1, hidden_dim // 2)
                # self.decoder_fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
                # self.decoder_out = nn.Linear(hidden_dim, design_dim)
                self._init_weights()

                # # Activation functions
                # self.relu = nn.ReLU()
                # self.tanh = nn.Tanh()  # Tanh keeps outputs bounded
                
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
                
            def encode(self, s_params, freq):
                # Fix dimension issues
                if s_params.dim() == 1:
                    s_params = s_params.unsqueeze(0)
                if freq.dim() == 1:
                    freq = freq.unsqueeze(-1)
                
                x = torch.cat([s_params, freq], dim=-1)
                h = self.encoder(x)
                mu = self.mu_layer(h)
                logvar = torch.clamp(self.logvar_layer(h), -10, 5)  # CLAMP logvar to prevent NaN
                return mu, logvar
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z, freq):
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                if freq.dim() == 1:
                    freq = freq.unsqueeze(-1)
                
                z_cond = torch.cat([z, freq], dim=-1)
                x = self.decoder(z_cond)                
                return x  

            def forward(self, s_params, freq):
                mu, logvar = self.encode(s_params, freq)
                z = self.reparameterize(mu, logvar)
                design_pred = self.decode(z, freq)
                return design_pred, mu, logvar
        
        # Initialize CVAE with smaller dimensions for stability
        cvae = StableCVAE(
            s_dim=2,
            design_dim=8,
            latent_dim=4,  # Smaller latent dimension
            hidden_dim=64  # Smaller hidden dimension
        ).to(device)
        
        # OPTIMIZER with lower learning rate
        optimizer = optim.Adam(cvae.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # STABLE LOSS FUNCTION
        def cvae_loss(pred, target, mu, logvar, beta=1.0):
            """
            CVAE loss = reconstruction + β * KL
            """
            recon_loss = F.mse_loss(pred, target, reduction='mean')

            kld = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            return recon_loss + beta * kld, recon_loss, kld
        def kl_annealing(epoch, max_beta=1.0, warmup_epochs=50):
            return min(max_beta, epoch / warmup_epochs)

        # def stable_cvae_loss(recon_x, x, mu, logvar, beta=0.001):
        #     # Reconstruction loss with reduction='mean' for stability
        #     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
            
        #     # KL divergence with numerical stability
        #     # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #     # More stable formulation:
        #     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            
        #     # Return total loss
        #     total_loss = recon_loss + beta * kld
            
        #     # Check for NaN
        #     if torch.isnan(total_loss):
        #         print(f"Warning: NaN loss detected!")
        #         return torch.tensor(0.0, requires_grad=True), recon_loss, kld
            
        #     return total_loss, recon_loss, kld
        
        cvae.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\nStarting CVAE training...")
        
        for epoch in range(100):  # Reduced epochs
            epoch_loss = 0
            epoch_recon = 0
            epoch_kld = 0
            num_batches = 0
            
            for s_batch, d_batch, f_batch in train_loader:
                s_batch = s_batch.to(device)
                d_batch = d_batch.to(device)
                f_batch = f_batch.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                d_pred, mu, logvar = cvae(s_batch, f_batch)
                
                # Calculate loss
                beta = kl_annealing(epoch)
                loss, recon_loss, kld_loss = cvae_loss(d_pred, d_batch, mu, logvar, beta)
                
                # Skip if loss is NaN
                if torch.isnan(loss):
                    print(f"Skipping batch due to NaN loss")
                    continue
                
                loss.backward()
                
                # GRADIENT CLIPPING - crucial for stability
                torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kld += kld_loss.item()
                num_batches += 1
            
            if num_batches == 0:
                print(f"Epoch {epoch+1}: No valid batches")
                continue
            
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon / num_batches
            avg_kld = epoch_kld / num_batches
            
            # Validation
            cvae.eval()
            with torch.no_grad():
                s_test_scaled = torch.tensor(s_scaler.transform(s_test), dtype=torch.float32).to(device)
                d_test_scaled = torch.tensor(d_scaler.transform(d_test), dtype=torch.float32).to(device)
                f_test_tensor = torch.tensor(f_test, dtype=torch.float32).to(device)
                beta = kl_annealing(epoch)
                d_pred_test, mu_test, logvar_test = cvae(s_test_scaled, f_test_tensor)
                val_loss, val_recon, val_kld = cvae_loss(
                    d_pred_test, d_test_scaled, mu_test, logvar_test, beta
                )
                val_loss = val_loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"CVAE Epoch {epoch+1}: "
                      f"Loss = {avg_loss:.6f} (Recon: {avg_recon:.6f}, KLD: {avg_kld:.6e}), "
                      f"Val Loss = {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': cvae.state_dict(),
                    's_scaler': s_scaler,
                    'd_scaler': d_scaler,
                    'val_loss': val_loss
                }, os.path.join(Config.MODEL_DIR, 'best_inverse_cvae.pt'))
                print(f"Saved improved model with val loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if os.path.exists(os.path.join(Config.MODEL_DIR, 'best_inverse_cvae.pt')):
            checkpoint = torch.load(os.path.join(Config.MODEL_DIR, 'best_inverse_cvae.pt'), 
                                   map_location='cpu', weights_only=False)
            cvae.load_state_dict(checkpoint['model_state_dict'])
            s_scaler = checkpoint['s_scaler']
            d_scaler = checkpoint['d_scaler']
            print(f"Loaded best model with validation loss: {checkpoint['val_loss']:.6f}")
        else:
            print("Warning: Could not load best model, using current model")
        
        return cvae, s_scaler, d_scaler    
    


# === Main Predictor Class ===
class MicrowavePredictor:
    def __init__(self):
        self.models_loaded = False
        self.curve_predictor = None
        self.single_predictor = None
        self.inverse_cvae = None
        self.freq_interpolator = None
        
        # Scalers
        self.curve_design_scaler = None
        self.curve_S_scaler = None
        self.single_X_scaler = None
        self.single_y_scaler = None
        self.cvae_s_scaler = None
        self.cvae_d_scaler = None
        
        # Frequency grid
        self.freq_grid = None
        
        # Load or train models
        self._initialize_models()
    
    def _initialize_models(self):
        """Load existing models or train new ones"""
        try:
            # Load curve predictor
            curve_path = os.path.join(Config.MODEL_DIR, 'curve_predictor.pt')
            if os.path.exists(curve_path):
                checkpoint = torch.load(curve_path, map_location='cpu', weights_only=False)
                self.curve_design_scaler = checkpoint['design_scaler']
                self.curve_S_scaler = checkpoint['S_scaler']
                
                # Recreate model
                self.curve_predictor = CurvePredictor(
                    design_dim=len(Config.FEATURE_COLS[:-1]),
                    output_dim=Config.N_FREQ_POINTS * 2
                )
                self.curve_predictor.load_state_dict(checkpoint['model_state_dict'])
                self.curve_predictor.eval()
                print("Loaded curve predictor")
            else:
                print("Curve predictor not found, will train when needed")
            
            # Load single predictor
            single_model_path = os.path.join(Config.MODEL_DIR, 'single_predictor.h5')
            if os.path.exists(single_model_path):
                self.single_predictor = load_model(single_model_path, compile=False)
                self.single_X_scaler = joblib.load(os.path.join(Config.MODEL_DIR, 'single_X_scaler.pkl'))
                self.single_y_scaler = joblib.load(os.path.join(Config.MODEL_DIR, 'single_y_scaler.pkl'))
                print("Loaded single predictor")
            
            # Load inverse CVAE
            cvae_path = os.path.join(Config.MODEL_DIR, 'best_inverse_cvae.pt')
            if os.path.exists(cvae_path):
                checkpoint = torch.load(cvae_path, map_location='cpu', weights_only=False)
                self.cvae_s_scaler = checkpoint['s_scaler']
                self.cvae_d_scaler = checkpoint['d_scaler']

                # Create a model that matches the saved architecture
                class LoadedCVAE(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.encoder = nn.Sequential(
                            nn.Linear(3, 64),      # s_dim + 1 = 2 + 1 = 3
                            nn.GELU(),
                            nn.Linear(64, 64),
                            nn.GELU(),
                            nn.Linear(64, 32),
                            nn.GELU()
                        )
                        self.mu_layer = nn.Linear(32, 4)
                        self.logvar_layer = nn.Linear(32, 4)
                        self.decoder = nn.Sequential(
                            nn.Linear(5, 32),      # latent_dim + 1 = 4 + 1 = 5
                            nn.GELU(),
                            nn.Linear(32, 64),
                            nn.GELU(),
                            nn.Linear(64, 8)
                        )

                    def forward(self, s_params, freq):
                        if s_params.dim() == 1:
                            s_params = s_params.unsqueeze(0)
                        if freq.dim() == 1:
                            freq = freq.unsqueeze(-1)
                        
                        x = torch.cat([s_params, freq], dim=-1)
                        h = self.encoder(x)
                        mu = self.mu_layer(h)
                        logvar = self.logvar_layer(h)
                        
                        std = torch.exp(0.5 * logvar)
                        eps = torch.randn_like(std)
                        z = mu + eps * std
                        
                        z_cond = torch.cat([z, freq], dim=-1)
                        design = self.decoder(z_cond)
                        return design, mu, logvar
                
                self.inverse_cvae = LoadedCVAE()
                self.inverse_cvae.load_state_dict(checkpoint['model_state_dict'])
                self.inverse_cvae.eval()
                print("Loaded inverse CVAE")
                
                # self.inverse_cvae = EnhancedCVAE(
                #     s_dim=2,  # S11, S21
                #     design_dim=8,
                #     latent_dim=4,
                #     hidden_dim=64
                # )
                # self.inverse_cvae.load_state_dict(checkpoint['model_state_dict'])
                # self.inverse_cvae.eval()
                # print("Loaded inverse CVAE")
            
            self.models_loaded = True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    def forward_predict(self, design_params, freq):
        """Predict S-parameters at specific frequency"""
        if not self.models_loaded:
            raise ValueError("Models not loaded. Please train models first.")
        
        # Method 1: Use single predictor directly
        features = create_enhanced_features(design_params, freq).reshape(1, -1)
        features_scaled = self.single_X_scaler.transform(features)
        
        s_pred_scaled = self.single_predictor.predict(features_scaled, verbose=0)
        s_pred = self.single_y_scaler.inverse_transform(s_pred_scaled)[0]
        
        return s_pred[0], s_pred[1]
    
    def forward_predict_curve(self, design_params):
        """Predict full S-parameter curves"""
        if not self.models_loaded or self.curve_predictor is None:
            raise ValueError("Curve predictor not available")
        
        design_array = np.array(design_params).reshape(1, -1)
        design_scaled = self.curve_design_scaler.transform(design_array)
        
        with torch.no_grad():
            s_curve_scaled = self.curve_predictor(
                torch.tensor(design_scaled, dtype=torch.float32)
            ).numpy()
        
        s_curve = self.curve_S_scaler.inverse_transform(s_curve_scaled)[0]
        
        # Split into S11 and S21 curves
        n_points = Config.N_FREQ_POINTS
        s11_curve = s_curve[:n_points]
        s21_curve = s_curve[n_points:]
        
        return s11_curve, s21_curve
    
    def inverse_predict(self, s11, s21, freq):
        """Predict design parameters from S-parameters"""
        if not self.models_loaded or self.inverse_cvae is None:
            raise ValueError("Inverse model not available")
        
        # Prepare input
        s_array = np.array([[s11, s21]], dtype=np.float32)
        s_scaled = self.cvae_s_scaler.transform(s_array)
        
        # Predict with CVAE
        with torch.no_grad():
            s_tensor = torch.tensor(s_scaled, dtype=torch.float32)
            freq_tensor = torch.tensor([[freq]], dtype=torch.float32)
            
            design_pred_scaled, _, _ = self.inverse_cvae(s_tensor, freq_tensor)
            design_pred = self.cvae_d_scaler.inverse_transform(design_pred_scaled.numpy())[0]
        
        # Apply physical constraints
        for i, (key, (min_val, max_val)) in enumerate(Config.DESIGN_BOUNDS.items()):
            design_pred[i] = np.clip(design_pred[i], min_val, max_val)
        
        return design_pred
    
    def Dual_Frequency_prediction(self, mode='forward', **kwargs):
        """Enhanced dual-band optimization"""
        if mode == 'forward':
            return self._dual_band_forward(**kwargs)
        elif mode == 'inverse':
            return self._dual_band_inverse(**kwargs)
        else:
            raise ValueError("Mode must be 'forward' or 'inverse'")
    
    def _dual_band_forward(self, freq1, freq2, design_params1):
        """Forward analysis for dual-band"""
        # Get initial predictions
        s11_1_init, s21_1_init = self.forward_predict(design_params1, freq1)
        s11_2_init, s21_2_init = self.forward_predict(design_params1, freq2)
        
        # Optimization function
        def objective(x):
            try:
                s11_1, s21_1 = self.forward_predict(x, freq1)
                s11_2, s21_2 = self.forward_predict(x, freq2)
                
                # Dual-band optimization metric
                error = (2 * (s11_1**2 + s11_2**2) + 
                        (abs(s21_1 + 1) + abs(s21_2 + 1)))
                
                # Add penalty for physical constraints
                penalty = 0
                for i, (key, (min_val, max_val)) in enumerate(Config.DESIGN_BOUNDS.items()):
                    if x[i] < min_val:
                        penalty += (min_val - x[i]) ** 2
                    elif x[i] > max_val:
                        penalty += (x[i] - max_val) ** 2
                
                return error + 10 * penalty
            except:
                return float('inf')
        
        # Bounds
        bounds = []
        for i, param in enumerate(design_params1):
            min_val = max(Config.DESIGN_BOUNDS[list(Config.DESIGN_BOUNDS.keys())[i]][0],
                         param * 0.7)
            max_val = min(Config.DESIGN_BOUNDS[list(Config.DESIGN_BOUNDS.keys())[i]][1],
                         param * 1.3)
            bounds.append((min_val, max_val))
        
        # Optimization
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            seed=Config.SEED
        )
        
        # Local refinement
        refined = minimize(
            objective,
            result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30}
        )
        
        optimized_params = refined.x
        
        # Get final predictions
        s11_1_opt, s21_1_opt = self.forward_predict(optimized_params, freq1)
        s11_2_opt, s21_2_opt = self.forward_predict(optimized_params, freq2)
        
        # Calculate metrics
        errors_freq1 = {
            'S11': {'MSE': (s11_1_init - s11_1_opt)**2, 'MAE': abs(s11_1_init - s11_1_opt)},
            'S21': {'MSE': (s21_1_init - s21_1_opt)**2, 'MAE': abs(s21_1_init - s21_1_opt)}
        }
        
        errors_freq2 = {
            'S11': {'MSE': (s11_2_init - s11_2_opt)**2, 'MAE': abs(s11_2_init - s11_2_opt)},
            'S21': {'MSE': (s21_2_init - s21_2_opt)**2, 'MAE': abs(s21_2_init - s21_2_opt)}
        }
        
        # Calculate composite return loss
        s11_1_lin = 10**(s11_1_opt/20)
        s11_2_lin = 10**(s11_2_opt/20)
        composite_s11 = np.sqrt((s11_1_lin**2 + s11_2_lin**2)/2)
        composite_RL = -20 * np.log10(composite_s11)
        
        return {
            # Initial predictions
            'freq1_s11': s11_1_init,
            'freq1_s21': s21_1_init,
            'freq2_s11': s11_2_init,
            'freq2_s21': s21_2_init,
            
            # Optimized results
            'optimized_params': optimized_params,
            'optimized_freq1_s11': s11_1_opt,
            'optimized_freq1_s21': s21_1_opt,
            'optimized_freq2_s11': s11_2_opt,
            'optimized_freq2_s21': s21_2_opt,
            
            # Metrics
            'composite_RL': composite_RL,
            'bandwidth_MHz': abs(freq2 - freq1),
            'errors_freq1': errors_freq1,
            'errors_freq2': errors_freq2
        }
    
    def _dual_band_inverse(self, freq1, freq2, s11_1, s21_1, s11_2, s21_2):
        """Inverse design for dual-band"""
        # Get individual designs
        design_freq1 = self.inverse_predict(s11_1, s21_1, freq1)
        design_freq2 = self.inverse_predict(s11_2, s21_2, freq2)
        
        # Verify individual designs
        s11_1_pred, s21_1_pred = self.forward_predict(design_freq1, freq1)
        s11_2_pred, s21_2_pred = self.forward_predict(design_freq2, freq2)
        
        # Optimization for dual-band
        def objective(x):
            s11_1_pred, s21_1_pred = self.forward_predict(x, freq1)
            s11_2_pred, s21_2_pred = self.forward_predict(x, freq2)
            
            error = (2 * ((s11_1_pred - s11_1)**2 + (s11_2_pred - s11_2)**2) +
                    ((s21_1_pred - s21_1)**2 + (s21_2_pred - s21_2)**2))
            
            # Center point penalty
            center = (design_freq1 + design_freq2) / 2
            penalty = 0.1 * np.sum((x - center)**2)
            
            return error + penalty
        
        # Bounds based on individual designs
        bounds = []
        for i in range(8):
            min_val = min(design_freq1[i], design_freq2[i]) * 0.8
            max_val = max(design_freq1[i], design_freq2[i]) * 1.2
            
            # Apply physical bounds
            param_name = list(Config.DESIGN_BOUNDS.keys())[i]
            phys_min, phys_max = Config.DESIGN_BOUNDS[param_name]
            min_val = max(min_val, phys_min)
            max_val = min(max_val, phys_max)
            
            bounds.append((min_val, max_val))
        
        # Global optimization
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=60,
            popsize=20,
            seed=Config.SEED
        )
        
        optimized_params = result.x
        
        # Get final performance
        final_s11_1, final_s21_1 = self.forward_predict(optimized_params, freq1)
        final_s11_2, final_s21_2 = self.forward_predict(optimized_params, freq2)
        
        # Calculate composite RL
        s11_1_lin = 10**(final_s11_1/20)
        s11_2_lin = 10**(final_s11_2/20)
        composite_s11 = np.sqrt((s11_1_lin**2 + s11_2_lin**2)/2)
        composite_RL = -20 * np.log10(composite_s11)
        
        return {
            # Individual designs
            'freq1_params': design_freq1,
            'freq1_performance': [s11_1_pred, s21_1_pred],
            'freq2_params': design_freq2,
            'freq2_performance': [s11_2_pred, s21_2_pred],
            
            # Optimized design
            'optimized_params': optimized_params,
            'optimized_freq1_s11': final_s11_1,
            'optimized_freq1_s21': final_s21_1,
            'optimized_freq2_s11': final_s11_2,
            'optimized_freq2_s21': final_s21_2,
            
            # Metrics
            'composite_RL': composite_RL,
            'bandwidth_MHz': abs(freq2 - freq1)
        }

# === Training Function ===
def train_models():
    """Train all models in the hybrid system"""
    print("Loading data...")
    processor = HybridDataProcessor()
    
    try:
        curve_data, single_data = processor.load_and_process_data(Config.DATA_PATH)
        print(f"Curve data: {len(curve_data['designs'])} designs")
        print(f"Single data: {len(single_data['X'])} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    print("\nTraining hybrid model system...")
    
    # Initialize trainer
    trainer = HybridTrainer(Config)
    
    # Train curve predictor
    print("\n1. Training curve predictor...")
    curve_predictor, curve_design_scaler, curve_S_scaler = trainer.train_curve_predictor(curve_data)
    
    # Train single predictor
    print("\n2. Training single frequency predictor...")
    single_predictor, single_X_scaler, single_y_scaler = trainer.train_single_predictor(single_data)
    
    # Train inverse CVAE
    print("\n3. Training inverse CVAE...")
    inverse_cvae, cvae_s_scaler, cvae_d_scaler = trainer.train_inverse_cvae(single_data)
    
    print("\nAll models trained successfully!")
    print(f"Models saved in: {Config.MODEL_DIR}")
    return True

# === User Interface (Your exact interface) ===
def main():
    try:
        predictor = MicrowavePredictor()
    except Exception as e:
        print(f"Initialization error: {e}")
        return
    
    print("\nEnhanced Microwave Structure AI Assistant\n")
    mode = questionary.select(
        "Choose a mode:",
        choices=[
            "Forward Prediction", 
            "Inverse Prediction", 
            "Dual-Frequency Prediction",
            "Train New Models"
        ]
    ).ask()
    
    if mode == "Forward Prediction":
        print("\nEnter the 8 design parameters and frequency:")
        try:
            params = []
            for i, col in enumerate(Config.FEATURE_COLS[:-1]):
                param = questionary.text(f"Enter {col}:").ask()
                params.append(float(param))
            
            freq = questionary.text("Enter frequency (MHz or GHz):").ask()
            freq = standardize_frequency(freq)
            
            s11, s21 = predictor.forward_predict(params, freq)
            print(f"\nPredicted S-parameters at {freq} MHz:")
            print(f"S11: {s11:.2f} dB")
            print(f"S21: {s21:.2f} dB")
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "Inverse Prediction":
        print("\nEnter target S-parameters and frequency:")
        try:
            s11 = float(questionary.text("Target S11 (dB):").ask())
            s21 = float(questionary.text("Target S21 (dB):").ask())
            freq = questionary.text("Frequency (MHz or GHz):").ask()
            freq = standardize_frequency(freq)
            
            design_params = predictor.inverse_predict(s11, s21, freq)
            
            print("\nPredicted Design Parameters:")
            for i, col in enumerate(Config.FEATURE_COLS[:-1]):
                print(f"{col}: {design_params[i]:.4f}")
            
            # Validate with forward prediction
            print("\nValidation with Forward Prediction:")
            s11_pred, s21_pred = predictor.forward_predict(design_params, freq)
            print(f"S11: {s11_pred:.2f} dB (Target: {s11:.2f})")
            print(f"S21: {s21_pred:.2f} dB (Target: {s21:.2f})")
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "Dual-Frequency Prediction":
        print("\nEnhanced Dual-Band Transformer Design System")
        try:
            pred_mode = questionary.select(
                "Choose operation mode:",
                choices=["Forward Analysis", "Inverse Design"]
            ).ask().lower().replace(" ", "_")
            
            freq1 = standardize_frequency(questionary.text("First frequency (MHz/GHz):").ask())
            freq2 = standardize_frequency(questionary.text("Second frequency (MHz/GHz):").ask())
            
            if pred_mode == "forward_analysis":
                print(f"\nEnter microstrip design parameters for {freq1} MHz (mm):")
                params1 = []
                for col in Config.FEATURE_COLS[:-1]:
                    param = float(questionary.text(f"Enter {col} for freq1:").ask())
                    params1.append(param)

                print(f"\nEnter microstrip design parameters for {freq2} MHz (mm):")
                params2 = []
                for col in Config.FEATURE_COLS[:-1]:
                    param = float(questionary.text(f"Enter {col} for freq2:").ask())
                    params2.append(param)

                print("\nPerforming dual-frequency forward prediction and optimization...") 

                
                result = predictor.Dual_Frequency_prediction(
                    mode='forward',
                    freq1=freq1,
                    freq2=freq2,
                    design_params1=params1,
                    design_params2=params2
                )

                
                # Display results
                print("\n=== Forward Prediction Results ===")
                print(f"\nResults for {freq1} MHz:")
                print(f"Design Parameters: {dict(zip(Config.FEATURE_COLS[:-1], params1))}")
                print(f"Predicted S11: {result['freq1_s11']:.2f} dB")
                print(f"Predicted S21: {result['freq1_s21']:.2f} dB")
                
                print(f"\nResults for {freq2} MHz:")
                print(f"Predicted S11: {result['freq2_s11']:.2f} dB")
                print(f"Predicted S21: {result['freq2_s21']:.2f} dB")
                
                print("\n=== Optimized Dual-Band Design ===")
                print("\nOptimized Design Parameters:")
                for name, value in zip(Config.FEATURE_COLS[:-1], result['optimized_params']):
                    print(f"{name}: {value:.4f} mm")
                
                print(f"\nPerformance at {freq1} MHz:")
                print(f"S11: {result['optimized_freq1_s11']:.2f} dB")
                print(f"S21: {result['optimized_freq1_s21']:.2f} dB")
                
                print(f"\nPerformance at {freq2} MHz:")
                print(f"S11: {result['optimized_freq2_s11']:.2f} dB")
                print(f"S21: {result['optimized_freq2_s21']:.2f} dB")
                
                print(f"\nComposite Return Loss: {result['composite_RL']:.2f} dB")
                print(f"Bandwidth: {result['bandwidth_MHz']:.0f} MHz")
                
            elif pred_mode == "inverse_design":
                print(f"\nEnter target signal strengths for {freq1} MHz:")
                s11_1 = float(questionary.text("Target S11 (dB):").ask())
                s21_1 = float(questionary.text("Target S21 (dB):").ask())
                
                print(f"\nEnter target signal strengths for {freq2} MHz:")
                s11_2 = float(questionary.text("Target S11 (dB):").ask())
                s21_2 = float(questionary.text("Target S21 (dB):").ask())
                
                print("\nGenerating optimized dual-band transformer design...")
                result = predictor.Dual_Frequency_prediction(
                    mode='inverse',
                    freq1=freq1,
                    freq2=freq2,
                    s11_1=s11_1,
                    s21_1=s21_1,
                    s11_2=s11_2,
                    s21_2=s21_2
                )
                
                # Display results
                print("\n=== Inverse Design Results ===")
                print(f"\nDesign for {freq1} MHz targets:")
                print(f"S11: {s11_1} dB, S21: {s21_1} dB")
                print("\nSuggested Parameters:")
                for name, value in zip(Config.FEATURE_COLS[:-1], result['freq1_params']):
                    print(f"{name}: {value:.4f} mm")
                print(f"\nPerformance: S11={result['freq1_performance'][0]:.2f} dB, S21={result['freq1_performance'][1]:.2f} dB")
                
                print(f"\nDesign for {freq2} MHz targets:")
                print(f"S11: {s11_2} dB, S21: {s21_2} dB")
                print("\nSuggested Parameters:")
                for name, value in zip(Config.FEATURE_COLS[:-1], result['freq2_params']):
                    print(f"{name}: {value:.4f} mm")
                print(f"\nPerformance: S11={result['freq2_performance'][0]:.2f} dB, S21={result['freq2_performance'][1]:.2f} dB")
                
                print("\n=== Optimized Dual-Band Design ===")
                print("\nOptimized Parameters:")
                for name, value in zip(Config.FEATURE_COLS[:-1], result['optimized_params']):
                    print(f"{name}: {value:.4f} mm")
                
                print(f"\nPerformance at {freq1} MHz:")
                print(f"S11: {result['optimized_freq1_s11']:.2f} dB (Target: {s11_1:.2f})")
                print(f"S21: {result['optimized_freq1_s21']:.2f} dB (Target: {s21_1:.2f})")
                
                print(f"\nPerformance at {freq2} MHz:")
                print(f"S11: {result['optimized_freq2_s11']:.2f} dB (Target: {s11_2:.2f})")
                print(f"S21: {result['optimized_freq2_s21']:.2f} dB (Target: {s21_2:.2f})")
                
                print(f"\nComposite Return Loss: {result['composite_RL']:.2f} dB")
                print(f"Bandwidth: {result['bandwidth_MHz']:.0f} MHz")
                
        except Exception as e:
            print(f"\nError in dual-band design: {str(e)}")
    
    elif mode == "Train New Models":
        print("\nTraining new models...")
        success = train_models()
        if success:
            print("\nModel training completed successfully!")
        else:
            print("\nModel training failed!")

if __name__ == "__main__":
    main()
