import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # Features and target
    X = df[['pressure_MPa', 'mass_flux_kg_m2_s', 'x_e_out__', 
            'D_e_mm', 'D_h_mm', 'length_mm', 'geometry_encoded']].values
    y = df['chf_exp_MW_m2'].values.reshape(-1, 1)
    
    # Feature scaling
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    return X_scaled, y_scaled, X_scaler, y_scaler

# Physics-informed loss function
def physics_loss(y_true, y_pred, X_original, pressure_idx=0, mass_flux_idx=1, quality_idx=2):
    # Ensure we're using the original unscaled features
    pressure = X_original[:, pressure_idx]
    mass_flux = X_original[:, mass_flux_idx]
    quality = X_original[:, quality_idx]
    
    # Gradient calculation
    with tf.GradientTape() as tape:
        tape.watch(pressure)
        tape.watch(mass_flux)
        tape.watch(quality)
        
        # Create input tensors for gradient calculation
        inputs_p = tf.concat([
            pressure[:, tf.newaxis],
            mass_flux[:, tf.newaxis],
            quality[:, tf.newaxis],
            tf.zeros_like(pressure)[:, tf.newaxis],
            tf.zeros_like(pressure)[:, tf.newaxis],
            tf.zeros_like(pressure)[:, tf.newaxis],
            tf.zeros_like(pressure)[:, tf.newaxis]
        ], axis=1)
        
        # Predict CHF
        chf_pred = model(inputs_p)
    
    # Compute gradients
    dchf_dp = tape.gradient(chf_pred, pressure)
    dchf_dg = tape.gradient(chf_pred, mass_flux)
    dchf_dx = tape.gradient(chf_pred, quality)
    
    # Physics-based constraints
    # 1. CHF should decrease with increasing quality (x_e_out)
    loss_x = tf.reduce_mean(tf.square(tf.nn.relu(dchf_dx)))
    
    # 2. CHF should generally increase with mass flux
    loss_g = tf.reduce_mean(tf.square(tf.nn.relu(-dchf_dg)))
    
    # 3. Pressure effect: CHF first increases then decreases with pressure
    # Penalize when dCHF/dP is positive at high pressures (>15 MPa)
    high_p_mask = tf.cast(pressure > 15, tf.float32)
    loss_p_high = tf.reduce_mean(high_p_mask * tf.square(tf.nn.relu(dchf_dp)))
    
    # Penalize when dCHF/dP is negative at low pressures (<5 MPa)
    low_p_mask = tf.cast(pressure < 5, tf.float32)
    loss_p_low = tf.reduce_mean(low_p_mask * tf.square(tf.nn.relu(-dchf_dp)))
    
    # Combine physics losses
    physics_loss = 0.1 * loss_x + 0.1 * loss_g + 0.05 * loss_p_high + 0.05 * loss_p_low
    
    return physics_loss

# Combined loss function
def combined_loss(X_original):
    def loss(y_true, y_pred):
        # Data loss (MSE)
        data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Physics loss
        p_loss = physics_loss(y_true, y_pred, X_original)
        
        # Weighted combination
        return data_loss + p_loss
    return loss

# Create neural network model
def create_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X, y, X_scaler, y_scaler = load_data('train.csv')
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    model = create_model(X_train.shape[1])
    
    # Compile with physics-informed loss
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss=combined_loss(X_scaler.inverse_transform(X_train)))
    
    # Train model
    history = model.fit(X_train, y_train, 
                       validation_data=(X_val, y_val),
                       epochs=500, 
                       batch_size=64,
                       verbose=1)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training History')
    plt.savefig('training_history.png', dpi=300)
    plt.close()
    
    # Generate validation predictions
    y_pred = model.predict(X_val)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_val_orig = y_scaler.inverse_transform(y_val)
    
    # Create parity plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val_orig, y_pred, alpha=0.6)
    plt.plot([y_val_orig.min(), y_val_orig.max()], 
             [y_val_orig.min(), y_val_orig.max()], 'r--')
    plt.xlabel('Experimental CHF (MW/m²)')
    plt.ylabel('Predicted CHF (MW/m²)')
    plt.title('Parity Plot: Experimental vs Predicted CHF')
    plt.grid(True)
    plt.savefig('parity_plot.png', dpi=300)
    plt.close()
    
    # Create contour plots for key parameter relationships
    def create_contour_plot(param1, param2, param1_name, param2_name):
        # Create grid
        x = np.linspace(np.min(param1), np.max(param1), 100)
        y = np.linspace(np.min(param2), np.max(param2), 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        
        # Create constant values for other parameters (mean values)
        other_params = np.array([
            np.mean(X_scaler.mean_[2]),  # x_e_out__
            np.mean(X_scaler.mean_[3]),  # D_e_mm
            np.mean(X_scaler.mean_[4]),  # D_h_mm
            np.mean(X_scaler.mean_[5]),  # length_mm
            np.mean(X_scaler.mean_[6])   # geometry_encoded
        ])
        
        # Create input grid
        grid_points = np.array([
            X_grid.ravel(),
            Y_grid.ravel(),
            np.full(X_grid.ravel().shape, other_params[0]),
            np.full(X_grid.ravel().shape, other_params[1]),
            np.full(X_grid.ravel().shape, other_params[2]),
            np.full(X_grid.ravel().shape, other_params[3]),
            np.full(X_grid.ravel().shape, other_params[4])
        ]).T
        
        # Scale grid points
        grid_points_scaled = X_scaler.transform(grid_points)
        
        # Predict CHF
        Z = model.predict(grid_points_scaled)
        Z = y_scaler.inverse_transform(Z)
        Z_grid = Z.reshape(X_grid.shape)
        
        # Create contour plot
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X_grid, Y_grid, Z_grid, levels=50, cmap='viridis')
        plt.colorbar(contour, label='CHF (MW/m²)')
        plt.scatter(param1, param2, c='red', s=10, alpha=0.5)
        plt.xlabel(param1_name)
        plt.ylabel(param2_name)
        plt.title(f'CHF Contour: {param1_name} vs {param2_name}')
        plt.savefig(f'contour_{param1_name}_vs_{param2_name}.png', dpi=300)
        plt.close()
    
    # Generate contour plots for key relationships
    X_orig = X_scaler.inverse_transform(X)
    create_contour_plot(X_orig[:, 1], X_orig[:, 2], 'Mass Flux (kg/m²s)', 'Quality (x_e_out)')
    create_contour_plot(X_orig[:, 0], X_orig[:, 1], 'Pressure (MPa)', 'Mass Flux (kg/m²s)')
    create_contour_plot(X_orig[:, 0], X_orig[:, 2], 'Pressure (MPa)', 'Quality (x_e_out)')
    
    # Save model
    model.save('chf_model.h5')
    
    print("Model training and evaluation completed. Results saved.")