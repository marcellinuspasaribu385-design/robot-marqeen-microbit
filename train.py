"""Transfer learning (MobileNetV2) untuk klasifikasi gestur tangan Robot Marqeen.

Dataset default : TFDS `rock_paper_scissors` (gambar tangan batu/gunting/kertas).
Dataset sendiri : folder data/<nama_kelas>/*.jpg  ->  python train.py --data_dir data
"""
import argparse, json
import tensorflow as tf

IMG = 160
AUTOTUNE = tf.data.AUTOTUNE

def load_tfds(batch):
    import tensorflow_datasets as tfds
    (tr, va), info = tfds.load("rock_paper_scissors", split=["train[:85%]", "train[85%:]"],
                               as_supervised=True, with_info=True)
    prep = lambda x, y: (tf.image.resize(x, (IMG, IMG)), y)
    tr = tr.map(prep, AUTOTUNE).shuffle(1000).batch(batch).prefetch(AUTOTUNE)
    va = va.map(prep, AUTOTUNE).batch(batch).prefetch(AUTOTUNE)
    return tr, va, info.features["label"].names

def load_dir(path, batch):
    tr = tf.keras.utils.image_dataset_from_directory(path, validation_split=0.2, subset="training",
            seed=42, image_size=(IMG, IMG), batch_size=batch)
    va = tf.keras.utils.image_dataset_from_directory(path, validation_split=0.2, subset="validation",
            seed=42, image_size=(IMG, IMG), batch_size=batch)
    names = tr.class_names
    return tr.prefetch(AUTOTUNE), va.prefetch(AUTOTUNE), names

def build(n_classes):
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                               tf.keras.layers.RandomRotation(0.1),
                               tf.keras.layers.RandomZoom(0.1)])
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = tf.keras.Input((IMG, IMG, 3))
    x = aug(inp)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out), base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--finetune_epochs", type=int, default=5)
    a = ap.parse_args()

    tr, va, names = load_dir(a.data_dir, a.batch) if a.data_dir else load_tfds(a.batch)
    print("Kelas:", names)
    model, base = build(len(names))
    loss = "sparse_categorical_crossentropy"
    model.compile(tf.keras.optimizers.Adam(1e-3), loss, metrics=["accuracy"])
    model.fit(tr, validation_data=va, epochs=a.epochs)

    # Fine-tuning: buka 30 layer terakhir dengan learning rate kecil
    base.trainable = True
    for l in base.layers[:-30]:
        l.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-5), loss, metrics=["accuracy"])
    hist = model.fit(tr, validation_data=va, epochs=a.finetune_epochs)

    model.save("marqeen_model.keras")
    json.dump(names, open("labels.json", "w"))
    print("Val accuracy akhir:", hist.history["val_accuracy"][-1])

if __name__ == "__main__":
    main()
