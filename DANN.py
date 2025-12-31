import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import questionary
from scipy.optimize import differential_evolution, minimize
from bayes_opt import BayesianOptimization
from scipy.optimize import minimize
import warnings
from sklearn.utils import shuffle

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
tf.random.set_seed(42)
np.random.seed(42)

# === Configuration ===
class Config:
    SEED = 42
    TEST_SIZE = 0.15  # Reduced for more training data
    VAL_SIZE = 0.15
    EPOCHS = 300  # Increased for better convergence
    BATCH_SIZE = 64  # Reduced batch size
    LEARNING_RATE = 0.0005  # Reduced learning rate
    LEARNING_RATE_MIN = 0.00001  # Minimum learning rate
    EARLY_STOPPING_PATIENCE = 30
    REDUCE_LR_PATIENCE = 15
    REGULARIZATION = l1_l2(l1=1e-6, l2=1e-5)  # Adjusted regularization
    FEATURE_COLS = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
    TARGET_COLS = ['S11', 'S21']
    IMAGE_SAVE_DIR = "Graphs"
    FORWARD_MODEL_PATH = "enhanced_forward_model.h5"
    INVERSE_MODEL_PATH = "enhanced_inverse_model.h5"
    FORWARD_CHECKPOINT_PATH = "best_forward_model.h5"
    INVERSE_CHECKPOINT_PATH = "best_inverse_model.h5"
    BAYESIAN_OPT_ITER = 30
    DATA_PATH = "Coupled_Line_Dataset.csv"
    K_FOLDS = 5  # For cross-validation

Path(Config.IMAGE_SAVE_DIR).mkdir(parents=True, exist_ok=True)

# === Enhanced Utility Functions ===
def standardize_frequency(freq):
    freq = str(freq).replace('\xa0', ' ').strip().lower().strip('"').strip("'")
    if 'mhz' in freq:
        return float(freq.replace(' mhz', ''))
    elif 'ghz' in freq:
        return float(freq.replace(' ghz', '')) * 1000
    return float(freq)

def create_enhanced_model(input_shape, output_units, name):
    """Enhanced forward model with residual connections"""
    inputs = Input(shape=(input_shape,), name=f"{name}_input")
    
    # First block with residual connection
    x = Dense(512, activation='swish', kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second block with residual connection
    residual = Dense(256, activation='linear')(x)
    x = Dense(256, activation='swish', kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Add()([x, residual])
    
    # Output block
    x = Dense(128, activation='swish', kernel_regularizer=Config.REGULARIZATION)(x)
    outputs = Dense(output_units, name=f"{name}_output")(x)
    
    return Model(inputs, outputs, name=name)

def create_inverse_model(input_shape, output_units, name):
    inputs = Input(shape=(input_shape,), name=f"{name}_input")
    
    # Input projection
    x = Dense(1024, activation='swish', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Residual block 1
    x1 = Dense(768, activation='swish', kernel_initializer='he_normal')(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.35)(x1)
    
    # Project residual to match dimensions
    if x.shape[-1] != x1.shape[-1]:
        residual = Dense(x1.shape[-1])(x)
    else:
        residual = x
    x = Add()([residual, x1])
    
    # Residual block 2
    x2 = Dense(512, activation='swish', kernel_initializer='he_normal')(x)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Project residual to match dimensions
    if x.shape[-1] != x2.shape[-1]:
        residual = Dense(x2.shape[-1])(x)
    else:
        residual = x
    x = Add()([residual, x2])
    
    # Output block with non-negative constraints
    x = Dense(256, activation='swish', kernel_initializer='he_normal')(x)
    outputs = Dense(output_units, 
                   kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5),
                   kernel_constraint=NonNeg(),  # Ensure positive parameters
                   name=f"{name}_output")(x)
    
    return Model(inputs, outputs, name=name)

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, skiprows=1, header=None, names=Config.FEATURE_COLS + Config.TARGET_COLS)
    df['freq'] = df['freq'].apply(standardize_frequency)
    
    # Feature engineering
    df['w_ratio'] = df['w_1'] / df['w_s']
    df['s_ratio'] = df['s_1'] / df['s_2']
    df['l_ratio'] = df['l_1'] / df['l_s']
    df['w2_ratio'] = df['w_2'] / df['w_s']
    df['w_sum'] = df['w_1'] + df['w_2'] + df['w_s']
    df['s_sum'] = df['s_1'] + df['s_2']
    df['l_sum'] = df['l_1'] + df['l_2'] + df['l_s']
    
    # Update feature columns
    Config.FEATURE_COLS += ['w_ratio', 's_ratio', 'l_ratio', 'w2_ratio', 'w_sum', 's_sum', 'l_sum']
    
    return df

def get_callbacks(model_type):
    checkpoint_path = Config.FORWARD_CHECKPOINT_PATH if model_type == 'forward' else Config.INVERSE_CHECKPOINT_PATH
    
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=Config.LEARNING_RATE_MIN,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        TensorBoard(log_dir=f"logs/{model_type}"),
        tf.keras.callbacks.TerminateOnNaN()
    ]

def run_bayesian_optimization(X, y, inverse_X, inverse_y):
    print("\n=== Running Bayesian Optimization for Inverse Model ===")
    
    def inverse_model_hyperopt(hidden_units1, hidden_units2, dropout1, dropout2, l1, l2):
        """Function to optimize for inverse model"""
        hidden_units1 = int(hidden_units1)
        hidden_units2 = int(hidden_units2)
        
        # Create model with current hyperparameters
        inputs = Input(shape=(inverse_X.shape[1],))
        x = Dense(hidden_units1, activation='swish', 
                 kernel_regularizer=l1_l2(l1=l1, l2=l2))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout1)(x)
        x = Dense(hidden_units2, activation='swish', 
                 kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout2)(x)
        outputs = Dense(inverse_y.shape[1], activation='linear',
                       kernel_constraint=NonNeg())(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(Config.LEARNING_RATE), loss="mse")
        
        # Train with early stopping
        history = model.fit(
            inverse_X, inverse_y,
            validation_split=0.2,
            epochs=50,
            batch_size=Config.BATCH_SIZE,
            verbose=0
        )
        
        return -history.history['val_loss'][-1]  # Return negative validation loss
    
    # Define parameter bounds
    pbounds = {
        'hidden_units1': (256, 1024),
        'hidden_units2': (128, 512),
        'dropout1': (0.1, 0.5),
        'dropout2': (0.1, 0.5),
        'l1': (1e-8, 1e-4),
        'l2': (1e-8, 1e-4)
    }
    
    # Run optimization
    optimizer = BayesianOptimization(
        f=inverse_model_hyperopt,
        pbounds=pbounds,
        random_state=Config.SEED,
        verbose=2
    )
    
    optimizer.maximize(init_points=5, n_iter=Config.BAYESIAN_OPT_ITER)
    
    return optimizer.max['params']

def train_models():
    try:
        df = load_data(Config.DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None
    
    # 1. Enhanced Data Preparation
    X = df[Config.FEATURE_COLS].values.astype('float32')
    y = df[Config.TARGET_COLS].values.astype('float32')
    
    # Create frequency bins for stratification
    freq_bins = pd.cut(df['freq'], bins=10, labels=False)
    
    # 2. K-Fold Cross Validation
    print("\n=== Running K-Fold Cross Validation ===")
    kfold = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.SEED)
    fold_metrics = {'forward': [], 'inverse': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nProcessing Fold {fold + 1}/{Config.K_FOLDS}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scaling
        forward_scaler = RobustScaler()
        X_train_scaled = forward_scaler.fit_transform(X_train)
        X_val_scaled = forward_scaler.transform(X_val)
        
        # Forward model
        forward_model = create_enhanced_model(X_train_scaled.shape[1], y.shape[1], f"forward_fold_{fold}")
        forward_model.compile(optimizer=Adam(Config.LEARNING_RATE), loss="mse", metrics=['mae'])
        
        forward_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=Config.EPOCHS//2,
            batch_size=Config.BATCH_SIZE,
            verbose=0
        )
        
        # Evaluate forward model
        forward_pred = forward_model.predict(X_val_scaled)
        forward_mae = mean_absolute_error(y_val, forward_pred)
        forward_mse = mean_squared_error(y_val, forward_pred)
        fold_metrics['forward'].append((forward_mae, forward_mse))
        
        # Inverse model data
        inverse_X_train = np.concatenate([y_train, X_train[:, Config.FEATURE_COLS.index('freq')].reshape(-1,1)], axis=1)
        inverse_y_train = X_train[:, :8]  # Original 8 features
        
        inverse_X_val = np.concatenate([y_val, X_val[:, Config.FEATURE_COLS.index('freq')].reshape(-1,1)], axis=1)
        inverse_y_val = X_val[:, :8]
        
        # Scaling for inverse model
        inverse_scaler_X = RobustScaler()
        inverse_scaler_y = RobustScaler()
        inverse_X_train_scaled = inverse_scaler_X.fit_transform(inverse_X_train)
        inverse_y_train_scaled = inverse_scaler_y.fit_transform(inverse_y_train)
        inverse_X_val_scaled = inverse_scaler_X.transform(inverse_X_val)
        
        # Inverse model
        inverse_model = create_inverse_model(inverse_X_train_scaled.shape[1], inverse_y_train.shape[1], f"inverse_fold_{fold}")
        inverse_model.compile(optimizer=Adam(Config.LEARNING_RATE), loss="mse", metrics=['mae'])
        
        inverse_model.fit(
            inverse_X_train_scaled, inverse_y_train_scaled,
            validation_data=(inverse_X_val_scaled, inverse_scaler_y.transform(inverse_y_val)),
            epochs=Config.EPOCHS//2,
            batch_size=Config.BATCH_SIZE,
            verbose=0
        )
        
        # Evaluate inverse model
        inverse_pred_scaled = inverse_model.predict(inverse_X_val_scaled)
        inverse_pred = inverse_scaler_y.inverse_transform(inverse_pred_scaled)
        inverse_mae = mean_absolute_error(inverse_y_val, inverse_pred)
        inverse_mse = mean_squared_error(inverse_y_val, inverse_pred)
        fold_metrics['inverse'].append((inverse_mae, inverse_mse))
    
    # Print cross-validation results
    print("\n=== Cross-Validation Results ===")
    print("Forward Model:")
    print(f"Average MAE: {np.mean([m[0] for m in fold_metrics['forward']]):.4f}")
    print(f"Average MSE: {np.mean([m[1] for m in fold_metrics['forward']]):.4f}")
    
    print("\nInverse Model:")
    print(f"Average MAE: {np.mean([m[0] for m in fold_metrics['inverse']]):.4f}")
    print(f"Average MSE: {np.mean([m[1] for m in fold_metrics['inverse']]):.4f}")
    
    # 3. Final Training with Best Hyperparameters
    print("\n=== Final Model Training ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED, stratify=freq_bins)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.SEED)
    
    # Forward model scaling and training
    forward_scaler = RobustScaler()
    X_train_scaled = forward_scaler.fit_transform(X_train)
    X_val_scaled = forward_scaler.transform(X_val)
    X_test_scaled = forward_scaler.transform(X_test)
    joblib.dump(forward_scaler, "forward_scaler.pkl")
    
    # Create and train final forward model
    forward_model = create_enhanced_model(X_train_scaled.shape[1], y.shape[1], "forward_model")
    forward_model.compile(
        optimizer=Adam(Config.LEARNING_RATE),
        loss="mse",
        metrics=['mae']
    )
    
    forward_history = forward_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=get_callbacks('forward'),
        verbose=1
    )
    
    # Prepare inverse data
    inverse_X = np.concatenate([y_train, X_train[:, Config.FEATURE_COLS.index('freq')].reshape(-1,1)], axis=1)
    inverse_y = X_train[:, :8]  # Original 8 features
    
    # Scaling for inverse model
    inverse_scaler_X = RobustScaler()
    inverse_scaler_y = RobustScaler()
    inverse_X_scaled = inverse_scaler_X.fit_transform(inverse_X)
    inverse_y_scaled = inverse_scaler_y.fit_transform(inverse_y)
    joblib.dump(inverse_scaler_X, "inverse_scaler_x.pkl")
    joblib.dump(inverse_scaler_y, "inverse_scaler_y.pkl")
    
    # Run Bayesian optimization for inverse model
    best_params = run_bayesian_optimization(X_train, y_train, inverse_X_scaled, inverse_y_scaled)
    
    # Create final inverse model with best hyperparameters
    inputs = Input(shape=(inverse_X_scaled.shape[1],))
    x = Dense(int(best_params['hidden_units1']), activation='swish', 
             kernel_regularizer=l1_l2(l1=best_params['l1'], l2=best_params['l2']))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(best_params['dropout1'])(x)
    x = Dense(int(best_params['hidden_units2']), activation='swish', 
             kernel_regularizer=l1_l2(l1=best_params['l1'], l2=best_params['l2']))(x)
    x = BatchNormalization()(x)
    x = Dropout(best_params['dropout2'])(x)
    outputs = Dense(inverse_y.shape[1], activation='linear',
                   kernel_constraint=NonNeg())(x)
    
    inverse_model = Model(inputs, outputs, name="inverse_model")
    inverse_model.compile(
        optimizer=Adam(Config.LEARNING_RATE),
        loss="mse",
        metrics=['mae']
    )
    
    # Train final inverse model
    inverse_history = inverse_model.fit(
        inverse_X_scaled, inverse_y_scaled,
        validation_data=(
            inverse_scaler_X.transform(np.concatenate([y_val, X_val[:, Config.FEATURE_COLS.index('freq')].reshape(-1,1)], axis=1)),
            inverse_scaler_y.transform(X_val[:, :8])
        ),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=get_callbacks('inverse'),
        verbose=1
    )
    
    # Save models
    forward_model.save(Config.FORWARD_MODEL_PATH)
    inverse_model.save(Config.INVERSE_MODEL_PATH)
    
    # Final Evaluation
    print("\n=== Final Model Evaluation ===")
    
    # Forward model evaluation
    forward_pred = forward_model.predict(X_test_scaled)
    print("\nForward Model Metrics:")
    for i, col in enumerate(Config.TARGET_COLS):
        print(f"\n{col}:")
        print(f"MAE: {mean_absolute_error(y_test[:, i], forward_pred[:, i]):.4f}")
        print(f"MSE: {mean_squared_error(y_test[:, i], forward_pred[:, i]):.4f}")
        print(f"R²: {r2_score(y_test[:, i], forward_pred[:, i]):.4f}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test[:, i], forward_pred[:, i], alpha=0.3)
        plt.plot([y_test[:, i].min(), y_test[:, i].max()], 
                 [y_test[:, i].min(), y_test[:, i].max()], 'r--')
        plt.xlabel(f'Actual {col}')
        plt.ylabel(f'Predicted {col}')
        plt.title(f'{col} Prediction Accuracy')
        plt.savefig(f"{Config.IMAGE_SAVE_DIR}/forward_{col}_validation.png")
        plt.close()
    
    # Inverse model evaluation
    inverse_pred_scaled = inverse_model.predict(
        inverse_scaler_X.transform(np.concatenate([y_test, X_test[:, Config.FEATURE_COLS.index('freq')].reshape(-1,1)], axis=1))
    )
    inverse_pred = inverse_scaler_y.inverse_transform(inverse_pred_scaled)
    
    print("\nInverse Model Metrics:")
    original_features = Config.FEATURE_COLS[:8]
    for i, col in enumerate(original_features):
        print(f"\n{col}:")
        print(f"MAE: {mean_absolute_error(X_test[:, i], inverse_pred[:, i]):.4f}")
        print(f"MSE: {mean_squared_error(X_test[:, i], inverse_pred[:, i]):.4f}")
        print(f"R²: {r2_score(X_test[:, i], inverse_pred[:, i]):.4f}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[:, i], inverse_pred[:, i], alpha=0.3)
        plt.plot([X_test[:, i].min(), X_test[:, i].max()], 
                 [X_test[:, i].min(), X_test[:, i].max()], 'r--')
        plt.xlabel(f'Actual {col}')
        plt.ylabel(f'Predicted {col}')
        plt.title(f'{col} Prediction Accuracy')
        plt.savefig(f"{Config.IMAGE_SAVE_DIR}/inverse_{col}_validation.png")
        plt.close()
    
    return forward_model, inverse_model, forward_scaler, inverse_scaler_X, inverse_scaler_y

# === Prediction Functions ===
class MicrowavePredictor:
    def __init__(self):
        self.models_trained = all(os.path.exists(f) for f in [
            Config.FORWARD_MODEL_PATH, 
            Config.INVERSE_MODEL_PATH,
            "forward_scaler.pkl",
            "inverse_scaler_x.pkl",
            "inverse_scaler_y.pkl"
        ])
        
        if not self.models_trained:
            print("Models not found. Training new models...")
            self.forward_model, self.inverse_model, self.forward_scaler, \
            self.inverse_scaler_x, self.inverse_scaler_y = train_models()
            
            if not all([self.forward_model, self.inverse_model, self.forward_scaler, 
                       self.inverse_scaler_x, self.inverse_scaler_y]):
                raise RuntimeError("Failed to train models. Please check your data file.")
        else:
            self.forward_model = load_model(Config.FORWARD_MODEL_PATH, compile=False)
            self.inverse_model = load_model(Config.INVERSE_MODEL_PATH, compile=False)
            self.forward_scaler = joblib.load("forward_scaler.pkl")
            self.inverse_scaler_x = joblib.load("inverse_scaler_x.pkl")
            self.inverse_scaler_y = joblib.load("inverse_scaler_y.pkl")

    def expanded_feature_cols(self, params,freq):
        ls,l1,l2, s1, s2, ws, w1, w2 = params
        base = [ls, l2, l1, s2, s1, ws, w2, w1, freq]
        derived_features = [
            ws / w1,  # w_ratio
            s1 / s2,  # s_ratio
            l1 / ls,  # l_ratio
            w2 / ws,  # w2_ratio
            ws + w1 + w2,  # w_sum
            s1 + s2,  # s_sum
            l1 + l2 + ls  # l_sum
        ]
        return np.array(base + derived_features).reshape(1, -1)
    
    def forward_predict(self, design_params, freq):
        """S-parameters prediction from design parameters and frequency"""
        if len(design_params) != len(Config.FEATURE_COLS) - 1:
            raise ValueError(f"Expected {len(Config.FEATURE_COLS)-1} design parameters, got {len(design_params)}")
        
        # Ensure frequency is included in the input array
        input_array = self.expanded_feature_cols(design_params, freq)
        # Scale the input
        input_scaled = self.forward_scaler.transform(input_array)
        return self.forward_model.predict(input_scaled, verbose=0)[0]
        
    
    def inverse_predict(self, s11, s21, freq):
        """design parameters prediction from S-parameters and frequency"""

        input_array = np.array([[s11, s21, freq]], dtype=np.float32)
        input_scaled = self.inverse_scaler_x.transform(input_array)
        predicted_scaled = self.inverse_model.predict(input_scaled, verbose=0)
        predicted_params = self.inverse_scaler_y.inverse_transform(predicted_scaled)[0]
        predicted_params = np.clip(predicted_params, [
        6, 6, 6, 0.15, 0.15, 0.6, 0.5, 0.2], 
        [13, 25, 25, 0.6, 0.6, 1.8, 2, 2])
        return predicted_params
    
    def format_parameters(self, params):
        """parameters format for display"""
        return {
            'l_s': params[0],
            'l_2': params[1],
            'l_1': params[2],
            's_2': params[3],
            's_1': params[4],
            'w_s': params[5],
            'w_2': params[6],
            'w_1': params[7]
        }
    
    def Dual_Frequency_prediction(self, mode='forward', freq1=None, freq2=None, 
                                design_params1=None, design_params2=None, 
                                s11_1=None, s21_1=None, s11_2=None, s21_2=None):
       
        def calculate_errors(target, predicted):
                return {
                    'MSE': np.mean((target - predicted)**2),
                    'MAE': np.mean(np.abs(target - predicted))
                }

        def calculate_dual_band_metrics(s11_1, s21_1, s11_2, s21_2):
                s11_1_lin = 10**(s11_1/20)
                s11_2_lin = 10**(s11_2/20)
                composite_s11 = np.sqrt((s11_1_lin**2 + s11_2_lin**2)/2)
                return {
                    'composite_RL': -20*np.log10(composite_s11),
                    'bandwidth_MHz': abs(freq2 - freq1)
                }

        if mode == 'forward':
            if design_params1 is None or len(design_params1) != len(Config.FEATURE_COLS)-1:
                 raise ValueError(f"Expected {len(Config.FEATURE_COLS)-1} design parameters for freq1")
                
            # Predict performance at both frequencies with their respective parameters

            
            input_array1 = self.expanded_feature_cols(design_params1 ,freq1).reshape(1, -1)
            input_array2 = self.expanded_feature_cols(design_params1 ,freq2).reshape(1, -1)
                
            input_scaled1 = self.forward_scaler.transform(input_array1)
            input_scaled2 = self.forward_scaler.transform(input_array2)

            s11_1_init, s21_1_init = self.forward_model.predict(input_scaled1, verbose=0)[0]
            s11_2_init, s21_2_init = self.forward_model.predict(input_scaled2, verbose=0)[0]
                
            # Optimize for dual-band performance
            def objective(x):
                input_array1 = self.expanded_feature_cols(x, freq1).reshape(1, -1)
                input_array2 = self.expanded_feature_cols(x, freq2).reshape(1, -1)
                input_scaled1 = self.forward_scaler.transform(input_array1)
                input_scaled2 = self.forward_scaler.transform(input_array2)
                s11_1, s21_1 = self.forward_model.predict(input_scaled1, verbose=0)[0]
                s11_2, s21_2 = self.forward_model.predict(input_scaled2, verbose=0)[0]
                    
                
                return (3*s11_1**2 + abs(s21_1 + 1) + 3*s11_2**2 + abs(s21_2 + 1))
                # return (s11_1**2 + s11_2**2) + (abs(s21_1 + 1) + abs(s21_2 + 1))
                
            bounds = [
                (max(6, 0.8*design_params1[0]), min(35, 1.2*design_params1[0])),  # l_s
                (max(6, 0.8*design_params1[1]), min(35, 1.2*design_params1[1])),  # l_2
                (max(6, 0.8*design_params1[2]), min(25, 1.2*design_params1[2])),  # l_1
                (max(0.15, 0.8*design_params1[3]), min(0.6, 1.2*design_params1[3])),  # s_2
                (max(0.15, 0.8*design_params1[4]), min(0.6, 1.2*design_params1[4])),  # s_1
                (max(0.6, 0.8*design_params1[5]), min(2.5, 1.2*design_params1[5])),  # w_s
                (max(0.5, 0.8*design_params1[6]), min(2.5, 1.2*design_params1[6])),  # w_2
                (max(0.2, 0.8*design_params1[7]), min(2.5, 1.2*design_params1[7]))   # w_1
            ]
            best_error = float('inf')
            best_design = None

            result_de = differential_evolution(
                objective,
                bounds=bounds,
                strategy='best1bin',
                maxiter=30,
                popsize=10,
                init=np.vstack([
                    design_params1,
                    np.random.uniform([b[0] for b in bounds], 
                                    [b[1] for b in bounds], 
                                    size=(9, len(bounds)))
                ])
            )
    
            # Phase 2: Local refinement
            result = minimize(
                objective,
                result_de.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )

            

            if result.fun < best_error:
                # Update best design if new result is better
                best_error = result.fun
                best_design = result.x
                
            
                
            # Get optimized performance
            input_array1 = self.expanded_feature_cols(best_design, freq1).reshape(1, -1)
            input_array2 = self.expanded_feature_cols(best_design, freq2).reshape(1, -1)
            input_scaled1 = self.forward_scaler.transform(input_array1)
            input_scaled2 = self.forward_scaler.transform(input_array2)
            s11_1_opt, s21_1_opt = self.forward_model.predict(input_scaled1, verbose=0)[0]
            s11_2_opt, s21_2_opt = self.forward_model.predict(input_scaled2, verbose=0)[0]
                
            metrics = calculate_dual_band_metrics(s11_1_opt, s21_1_opt, s11_2_opt, s21_2_opt)
                
            # Calculate errors for initial predictions
            errors_freq1 = {
                'S11': calculate_errors(s11_1_init, s11_1_opt),
                'S21': calculate_errors(s21_1_init, s21_1_opt)
                }
            errors_freq2 = {
                'S11': calculate_errors(s11_2_init, s11_2_opt),
                'S21': calculate_errors(s21_2_init, s21_2_opt)
                }
                
            return {
                # Initial predictions
                'freq1_params': design_params1,
                'freq1_s11': s11_1_init,
                'freq1_s21': s21_1_init,
                # 'freq2_params': design_params2,
                'freq2_s11': s11_2_init,
                'freq2_s21': s21_2_init,
                    
                # Optimized design
                'optimized_params': best_design,
                'optimized_freq1_s11': s11_1_opt,
                'optimized_freq1_s21': s21_1_opt,
                'optimized_freq2_s11': s11_2_opt,
                'optimized_freq2_s21': s21_2_opt,
                    
                # Metrics
                'composite_RL': metrics['composite_RL'],
                'bandwidth_MHz': metrics['bandwidth_MHz'],
                'optimization_error': best_error,
                'errors_freq1': errors_freq1,
                'errors_freq2': errors_freq2,
                'mse': (errors_freq1['S11']['MSE'] + errors_freq1['S21']['MSE'] + 
                        errors_freq2['S11']['MSE'] + errors_freq2['S21']['MSE']) / 4,
                'mae': (errors_freq1['S11']['MAE'] + errors_freq1['S21']['MAE'] + 
                        errors_freq2['S11']['MAE'] + errors_freq2['S21']['MAE']) / 4
            }
        
        elif mode == 'inverse':
            if None in [s11_1, s21_1, s11_2, s21_2]:
                raise ValueError("Target S11 and S21 values required for both frequencies")
            
            # Get individual frequency designs
            design_freq1 = self.inverse_predict(s11_1, s21_1, freq1)
            design_freq2 = self.inverse_predict(s11_2, s21_2, freq2)
            

            # Verify individual designs
            forward_input1 = self.expanded_feature_cols(design_freq1, freq1).reshape(1, -1)
            forward_input2 = self.expanded_feature_cols(design_freq2, freq2).reshape(1, -1)
            input_scaled1 = self.forward_scaler.transform(forward_input1)
            input_scaled2 = self.forward_scaler.transform(forward_input2)
            perf_freq1 = self.forward_model.predict(input_scaled1, verbose=0)[0]
            perf_freq2 = self.forward_model.predict(input_scaled2, verbose=0)[0]
            
            bounds = [
                (min(6, 0.8*min(design_freq1[0], design_freq2[0])), 
                max(35, 1.2*max(design_freq1[0], design_freq2[0]))),  # l_s
                (min(6, 0.8*min(design_freq1[1], design_freq2[1])), 
                max(35, 1.2*max(design_freq1[1], design_freq2[1]))),  # l_2
                (min(6, 0.8*min(design_freq1[2], design_freq2[2])), 
                max(25, 1.2*max(design_freq1[2], design_freq2[2]))),  # l_1
                (0.15, 0.6),  # s_2
                (0.15, 0.6),  # s_1
                (min(0.6, 0.8*min(design_freq1[5], design_freq2[5])), 
                max(2.5, 1.2*max(design_freq1[5], design_freq2[5]))),  # w_s
                (min(0.5, 0.8*min(design_freq1[6], design_freq2[6])), 
                max(2.5, 1.2*max(design_freq1[6], design_freq2[6]))),  # w_2
                (min(0.2, 0.8*min(design_freq1[7], design_freq2[7])), 
                max(2.5, 1.2*max(design_freq1[7], design_freq2[7])))   # w_1
            ]
            # Optimize for dual-band performance
            def objective(x):
                # Create proper input arrays with frequency included
                input_array1 = self.expanded_feature_cols(x, freq1).reshape(1, -1)
                input_array2 = self.expanded_feature_cols(x, freq2).reshape(1, -1)
                
                # Scale and predict
                input_scaled1 = self.forward_scaler.transform(input_array1)
                input_scaled2 = self.forward_scaler.transform(input_array2)
                s11_1_pred, s21_1_pred = self.forward_model.predict(input_scaled1, verbose=0)[0]
                s11_2_pred, s21_2_pred = self.forward_model.predict(input_scaled2, verbose=0)[0]
                # Balanced error metric
                error = (3*(s11_1_pred - s11_1)**2 + 
                        1*(s21_1_pred - s21_1)**2 +
                        3*(s11_2_pred - s11_2)**2 + 
                        1*(s21_2_pred - s21_2)**2)
                
                penalty = 0
                if x[5] <= x[7] + x[6]:  # w_s <= w1 + w2
                    penalty += 10 * (x[7] + x[6] - x[5])**2
                if x[4] > 3*x[7]:  # s1 > 3*w1
                    penalty += (x[4] - 3*x[7])**2

                # Parameter drift penalty
                center_point = (np.array(design_freq1) + np.array(design_freq2))/2
                penalty += 0.1 * np.sum((x - center_point)**2)

                return error + penalty
            
            # bounds = [
            #     (6, 13), (6, 25), (6, 25), (0.15, 0.6), 
            #     (0.15, 0.6), (0.6, 1.8), (0.5, 2), (0.2, 2)
            # ]
            result_de = differential_evolution(
                objective,
                bounds=bounds,
                strategy='best1bin',
                maxiter=50,
                popsize=15,
                tol=0.01,
                recombination=0.9,
                init=np.vstack([
                    design_freq1,
                    design_freq2,
                    (np.array(design_freq1) + np.array(design_freq2))/2,
                    np.random.uniform([b[0] for b in bounds], 
                                    [b[1] for b in bounds], 
                                    size=(12, len(bounds)))
                ])
            )
            result = minimize(
                objective,
                result_de.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            # 5. Get final performance
            final_design = result.x
            input1 = self.expanded_feature_cols(final_design, freq1).reshape(1,-1)
            input2 = self.expanded_feature_cols(final_design, freq2).reshape(1,-1)
            
            final_s11_1, final_s21_1 = self.forward_model.predict(
                self.forward_scaler.transform(input1), verbose=0)[0]
            final_s11_2, final_s21_2 = self.forward_model.predict(
                self.forward_scaler.transform(input2), verbose=0)[0]
          
            
            metrics = calculate_dual_band_metrics(final_s11_1, final_s21_1, final_s11_2 , final_s21_2)
            
            # Calculate errors
            errors_freq1 = {
                'S11': calculate_errors(s11_1, final_s11_1),
                'S21': calculate_errors(s21_1, final_s21_1)
            }
            errors_freq2 = {
                'S11': calculate_errors(s11_2, final_s11_2),
                'S21': calculate_errors(s21_2,final_s21_2)
            }
            
            return {
                # Individual designs
                'freq1_params': design_freq1,
                'freq1_performance': perf_freq1,
                'freq2_params': design_freq2,
                'freq2_performance': perf_freq2,
                
                # Optimized design
                'optimized_params': final_design,
                'optimized_freq1_s11': final_s11_1,
                'optimized_freq1_s21': final_s21_1,
                'optimized_freq2_s11': final_s11_2,
                'optimized_freq2_s21': final_s21_2,

                # Metrics
                'composite_RL': metrics['composite_RL'],
                'bandwidth_MHz': metrics['bandwidth_MHz'],
                'optimization_error': best_error,
                'errors_freq1': errors_freq1,
                'errors_freq2': errors_freq2,
                'mse': (errors_freq1['S11']['MSE'] + errors_freq1['S21']['MSE'] + 
                        errors_freq2['S11']['MSE'] + errors_freq2['S21']['MSE']) / 4,
                'mae': (errors_freq1['S11']['MAE'] + errors_freq1['S21']['MAE'] + 
                        errors_freq2['S11']['MAE'] + errors_freq2['S21']['MAE']) / 4
            }
        
        else:
            raise ValueError("Mode must be 'forward' or 'inverse'")
        
        

# === User Interface ===
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
            forward_input = predictor.expanded_feature_cols([[*design_params, freq]]).reshape(1, -1)
            forward_input_scaled = predictor.forward_scaler.transform(forward_input)
            pred_s11, pred_s21 = predictor.forward_model.predict(forward_input_scaled, verbose=0)[0]
            print(f"S11: {pred_s11:.2f} dB (Target: {s11:.2f})")
            print(f"S21: {pred_s21:.2f} dB (Target: {s21:.2f})")
            
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
                
                
                result = predictor.Dual_Frequency_prediction(
                    mode='forward',
                    freq1=freq1,
                    freq2=freq2,
                    design_params1=params1,
                    
                )
                
                # Display results
                print("\n=== Forward Prediction Results ===")
                print(f"\nResults for {freq1} MHz:")
                print(f"Design Parameters: {dict(zip(Config.FEATURE_COLS[:-1], params1))}")
                print(f"Predicted S11: {result['freq1_s11']:.2f} dB")
                print(f"Predicted S21: {result['freq1_s21']:.2f} dB")
                
                print(f"\nResults for {freq2} MHz:")
                # print(f"Design Parameters: {dict(zip(Config.FEATURE_COLS[:-1], params1))}")
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
                
                
        except Exception as e:
            print(f"\nError in dual-band design: {str(e)}")

   
    
    elif mode == "Train New Models":
        print("\nTraining new models...")
        train_models()
        print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()