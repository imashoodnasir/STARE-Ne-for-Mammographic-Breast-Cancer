# train.py - STARE-Net module implementation

import tensorflow as tf
from preprocess import preprocess_image
from backbone import build_backbone
from hmt import build_hmt
from elt import build_elt
from tan import fuse_hmt_elt_outputs
from acr import acr_refinement_layer
from ensemble import MetaEnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def build_stare_model(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    backbone = build_backbone(input_shape)
    F1, F2, F3, F4 = backbone(inputs)

    Z = build_hmt(F1, F2, F3)
    FELT = build_elt(F3, tf.nn.softmax(F4, axis=-1))
    TAN = fuse_hmt_elt_outputs(Z, FELT)

    final_map = tf.reshape(TAN, (tf.shape(TAN)[0], 28, 28, -1))  # reshape if needed
    refined = acr_refinement_layer(final_map)

    pooled = tf.keras.layers.GlobalAveragePooling2D()(refined)
    output = tf.keras.layers.Dense(2, activation='softmax')(pooled)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def compile_and_train_model(model, train_ds, val_ds, epochs=100):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

def run_meta_ensemble(X_train, y_train, X_test, y_test):
    ensemble = MetaEnsembleClassifier(weights=(0.6, 0.4))
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    print(classification_report(y_test, predictions))
