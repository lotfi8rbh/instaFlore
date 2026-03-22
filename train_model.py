"""
InstaFlore - Script d'entraînement du modèle de classification de fleurs.

Architecture : MobileNetV2 (transfer learning) en 2 phases :
  Phase 1 — base gelée, apprentissage de la tête de classification
  Phase 2 — fine-tuning des dernières couches de la base

Anti-overfitting :
  - Augmentation de données (flip, rotation, zoom, luminosité, contraste)
    intégrée comme layer dans le modèle (appliquée uniquement au training)
  - Dropout(0.4) + Dropout(0.2)
  - L2 regularization sur la couche Dense
  - EarlyStopping (patience=8)
  - ReduceLROnPlateau
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# ─── Reproductibilité ────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ─── Hyperparamètres ─────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS_P1     = 20
EPOCHS_P2     = 30
LR_P1         = 1e-3
LR_P2         = 5e-5
DROPOUT       = 0.4
L2            = 1e-4
FINE_TUNE_AT  = 100
PATIENCE      = 8

OUT_DIR = pathlib.Path("model_output")
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
NUM_CLASSES  = len(CLASS_NAMES)

# ─── Dataset ─────────────────────────────────────────────────────────────────
print("Chargement du dataset tf_flowers...")

(ds_train_raw, ds_val_raw, ds_test_raw), info = tfds.load(
    "tf_flowers",
    split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
    as_supervised=True,
    with_info=True,
    data_dir="D:/tfds_data",
)

tfds_labels = info.features["label"].names
print(f"Labels TFDS    : {tfds_labels}")
print(f"Labels attendus: {CLASS_NAMES}")

label_map = {tfds_labels.index(c): CLASS_NAMES.index(c) for c in CLASS_NAMES}
label_map_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=list(label_map.keys()),
        values=list(label_map.values()),
        key_dtype=tf.int64,
        value_dtype=tf.int64,
    ),
    default_value=-1,
)


def preprocess(image, label):
    """Preprocessing standard (val / test)."""
    label = label_map_table.lookup(tf.cast(label, tf.int64))
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def preprocess_train(image, label):
    """Preprocessing avec augmentation sur [0,255] AVANT normalisation."""
    label = label_map_table.lookup(tf.cast(label, tf.int64))
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)

    # Augmentations appliquées sur [0, 255]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=40.0)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.clip_by_value(image, 0.0, 255.0)

    # Normalisation MobileNetV2 [-1, 1]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


AUTOTUNE = tf.data.AUTOTUNE

ds_train = (
    ds_train_raw
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
ds_val = (
    ds_val_raw
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
ds_test = (
    ds_test_raw
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

n_train = sum(1 for _ in ds_train_raw)
n_val   = sum(1 for _ in ds_val_raw)
n_test  = sum(1 for _ in ds_test_raw)
print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

# ─── Modèle ───────────────────────────────────────────────────────────────────
print("\nConstruction du modèle...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
x       = base_model(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.Dropout(DROPOUT)(x)
x       = tf.keras.layers.Dense(
    128, activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(L2),
)(x)
x       = tf.keras.layers.Dropout(DROPOUT / 2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

model = tf.keras.Model(inputs, outputs, name="instaflore_mobilenetv2")
model.summary(line_length=80)


def make_callbacks(patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT_DIR / "model_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5, patience=4, min_lr=1e-7,
            verbose=1,
        ),
    ]


# ─── Phase 1 ─────────────────────────────────────────────────────────────────
print("\n=== Phase 1 : tête de classification ===")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_P1),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

h1 = model.fit(
    ds_train, epochs=EPOCHS_P1,
    validation_data=ds_val,
    callbacks=make_callbacks(PATIENCE),
)

# ─── Phase 2 : fine-tuning ────────────────────────────────────────────────────
print(f"\n=== Phase 2 : fine-tuning à partir du layer {FINE_TUNE_AT} ===")

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_P2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

h2 = model.fit(
    ds_train, epochs=EPOCHS_P2,
    validation_data=ds_val,
    callbacks=make_callbacks(PATIENCE),
)

# ─── Évaluation ───────────────────────────────────────────────────────────────
print("\n=== Évaluation ===")
best = tf.keras.models.load_model(str(OUT_DIR / "model_best.keras"))
test_loss, test_acc = best.evaluate(ds_test, verbose=1)
print(f"Test accuracy : {test_acc*100:.2f}%   Test loss: {test_loss:.4f}")

# ─── Courbes ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
p2 = len(h1.history["accuracy"])

for key, label, ax_idx in [("accuracy", "Accuracy", 0), ("loss", "Loss", 1)]:
    all_train = h1.history[key] + h2.history[key]
    all_val   = h1.history[f"val_{key}"] + h2.history[f"val_{key}"]
    axes[ax_idx].plot(all_train, label=f"Train {label}")
    axes[ax_idx].plot(all_val,   label=f"Val {label}")
    axes[ax_idx].axvline(p2, color="gray", linestyle="--", label="Fine-tuning")
    axes[ax_idx].set_title(label)
    axes[ax_idx].set_xlabel("Epoch")
    axes[ax_idx].legend()
    axes[ax_idx].grid(True)

plt.suptitle(f"InstaFlore — Test acc: {test_acc*100:.2f}%", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(str(OUT_DIR / "training_curves.png"), dpi=120)
print(f"Courbes: {OUT_DIR / 'training_curves.png'}")

# ─── Conversion TFLite ────────────────────────────────────────────────────────
print("\n=== Conversion TFLite float32 ===")

converter = tf.lite.TFLiteConverter.from_keras_model(best)
tflite_bytes = converter.convert()

out_tflite = OUT_DIR / "model_instaflore.tflite"
out_tflite.write_bytes(tflite_bytes)
size_kb = len(tflite_bytes) / 1024
print(f"TFLite: {out_tflite}  ({size_kb:.0f} Ko)")

# ─── Vérification rapide ──────────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"Input : {inp_d['shape']}  {inp_d['dtype']}")
print(f"Output: {out_d['shape']}  {out_d['dtype']}")

correct = total = 0
for imgs, lbls in ds_test.take(5):
    for i in range(len(imgs)):
        x_in = np.expand_dims(imgs[i].numpy(), 0).astype(np.float32)
        interp.set_tensor(inp_d['index'], x_in)
        interp.invoke()
        pred = np.argmax(interp.get_tensor(out_d['index'])[0])
        correct += int(pred == lbls[i].numpy())
        total   += 1

tflite_acc = correct / total if total else 0
print(f"TFLite accuracy (sample): {tflite_acc*100:.1f}% sur {total} images")

# ─── Labels ───────────────────────────────────────────────────────────────────
(OUT_DIR / "labels.txt").write_text("\n".join(CLASS_NAMES))

# ─── Résumé ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RÉSUMÉ FINAL")
print("="*60)
print(f"  Test accuracy  : {test_acc*100:.2f}%")
print(f"  TFLite (sample): {tflite_acc*100:.1f}%")
print(f"  Taille modèle  : {size_kb:.0f} Ko")
print(f"  Fichiers       : {OUT_DIR}/")
print("="*60)
