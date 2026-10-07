import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\bridge_monitor")

SEQUENCE_DIR = BASE_DIR / "data" / "sequences"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 200
N_FEATURES = 24

BATCH_SIZE = 32
EPOCHS = 30

LATENT_DIM = 64
LEARNING_RATE = 0.001

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("=" * 70)
print("LSTM AUTOENCODER TRAINING")
print("=" * 70)

print(f"TensorFlow version : {tf.__version__}")
print(f"Window size        : {WINDOW_SIZE}")
print(f"Features            : {N_FEATURES}")
print(f"Latent dimension    : {LATENT_DIM}")
print(f"Batch size          : {BATCH_SIZE}")
print(f"Maximum epochs      : {EPOCHS}")


# ============================================================
# LOAD SEQUENCE SUMMARY
# ============================================================

summary_file = SEQUENCE_DIR / "sequence_summary.csv"

summary_df = pd.read_csv(summary_file)

train_files = summary_df[
    summary_df["split"] == "train"
]["output_file"].tolist()

validation_files = summary_df[
    summary_df["split"] == "validation"
]["output_file"].tolist()

print("\nTraining recordings :", len(train_files))
print("Validation recordings:", len(validation_files))


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_sequences(file_list):

    arrays = []

    for filename in file_list:

        filepath = SEQUENCE_DIR / filename

        if not filepath.exists():
            print(f"WARNING: Missing {filename}")
            continue

        data = np.load(filepath)

        X = data["X"]

        if len(X) > 0:
            arrays.append(X)

    if not arrays:
        raise RuntimeError("No sequences were loaded.")

    return np.concatenate(arrays, axis=0).astype(np.float32)


print("\nLoading training sequences...")

X_train = load_sequences(train_files)

print(f"X_train shape: {X_train.shape}")

print("\nLoading validation sequences...")

X_val = load_sequences(validation_files)

print(f"X_val shape:   {X_val.shape}")


# ============================================================
# VERIFY DATA
# ============================================================

print("\nData verification:")

print(f"Training NaN values     : {np.isnan(X_train).sum():,}")
print(f"Training infinite values: {np.isinf(X_train).sum():,}")

print(f"Validation NaN values     : {np.isnan(X_val).sum():,}")
print(f"Validation infinite values: {np.isinf(X_val).sum():,}")

if np.isnan(X_train).any() or np.isinf(X_train).any():
    raise ValueError("Invalid values detected in training data.")

if np.isnan(X_val).any() or np.isinf(X_val).any():
    raise ValueError("Invalid values detected in validation data.")


# ============================================================
# BUILD LSTM AUTOENCODER
# ============================================================

print("\nBuilding LSTM Autoencoder...")

inputs = Input(
    shape=(WINDOW_SIZE, N_FEATURES),
    name="sensor_input"
)

# ------------------------------------------------------------
# Encoder
# ------------------------------------------------------------

encoded = LSTM(
    LATENT_DIM,
    activation="tanh",
    return_sequences=False,
    name="encoder_lstm"
)(inputs)

# ------------------------------------------------------------
# Convert latent vector back to sequence
# ------------------------------------------------------------

repeated = RepeatVector(
    WINDOW_SIZE,
    name="repeat_latent"
)(encoded)

# ------------------------------------------------------------
# Decoder
# ------------------------------------------------------------

decoded = LSTM(
    LATENT_DIM,
    activation="tanh",
    return_sequences=True,
    name="decoder_lstm"
)(repeated)

outputs = TimeDistributed(
    Dense(N_FEATURES),
    name="reconstruction"
)(decoded)


# ============================================================
# MODEL
# ============================================================

autoencoder = Model(
    inputs,
    outputs,
    name="bridge_lstm_autoencoder"
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

autoencoder.compile(
    optimizer=optimizer,
    loss="mse"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel architecture:")

autoencoder.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = MODEL_DIR / "lstm_autoencoder_best.keras"

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = autoencoder.fit(
    X_train,
    X_train,

    validation_data=(
        X_val,
        X_val
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "lstm_autoencoder_final.keras"

autoencoder.save(final_model_path)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)

history_file = MODEL_DIR / "training_history.csv"

history_df.to_csv(
    history_file,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Best model saved to:")
print(checkpoint_path)

print(f"\nFinal model saved to:")
print(final_model_path)

print(f"\nTraining history saved to:")
print(history_file)

print("\nBest validation loss:")

best_val_loss = min(history.history["val_loss"])

print(f"{best_val_loss:.8f}")

print("=" * 70)