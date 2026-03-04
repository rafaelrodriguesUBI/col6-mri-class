
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, 
    Conv2D, 
    MaxPooling2D, 
    Flatten, 
    Dense, 
    Dropout, 
    GlobalAveragePooling2D, 
    Concatenate,
    Normalization,
    RandomFlip,
    RandomTranslation,
    RandomRotation,
    RandomContrast
)

# Preprocessing and Augmentation Layers
data_normalization = Normalization(axis=-1)

data_augmentation = tf.keras.Sequential(
    [
        RandomFlip("horizontal", seed=123),
        RandomTranslation(0, 0.1, fill_mode="constant", seed=123),
        RandomRotation(0.2, fill_mode="constant", seed=123),
        RandomContrast(0.2, seed=123),
    ]
)

# CNN encoder only
def build_cnn_model(sz=96, num_nodes=512, nclasses=3, dropout=True):
    
    inputs = Input(shape=(sz, sz, 1), name="input_img")
    u1 = data_augmentation(inputs)
    u1 = data_normalization(u1)
    c = Conv2D(
        64,
        (7, 7),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv1_1",
    )(u1)
    c2 = MaxPooling2D((2, 2), name="pool1")(c)
    c = Conv2D(
        128,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv2_1",
    )(c2)
    c3 = MaxPooling2D((2, 2), name="pool2")(c)
    c = Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv3_1",
    )(c3)
    c4 = MaxPooling2D((2, 2), name="pool3")(c)
    c5 = Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv4",
    )(c4)
    g = Conv2D(64, (1, 1), name="conv5")(c5)
    h = Flatten(name="flatten1")(g)
    z = Dense(num_nodes, activation='relu', name="dense1")(h)
    if dropout == True:
        z = Dropout(0.2, name="dropout1")(z)
    z = Dense(num_nodes, activation='relu', name="dense2")(z)
    if dropout == True:
        z = Dropout(0.2, name="dropout2")(z)
    outputs = Dense(nclasses, activation="softmax", name="output1")(z)
        
    return tf.keras.Model(inputs, outputs, name="CNN")

# CNN+GAP
def build_cnn_gap_model(sz=96, nclasses=3):
    
    inputs = Input(shape=(sz, sz, 1), name="input_img")
    u1 = data_augmentation(inputs)
    u1 = data_normalization(u1)
    c = Conv2D(
        64,
        (7, 7),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv1_1",
    )(u1)
    c2 = MaxPooling2D((2, 2), name="pool1")(c)
    c = Conv2D(
        128,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv2_1",
    )(c2)
    c3 = MaxPooling2D((2, 2), name="pool2")(c)
    c = Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv3_1",
    )(c3)
    c4 = MaxPooling2D((2, 2), name="pool3")(c)
    c5 = Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv4",
    )(c4)
    g = Conv2D(64, (1, 1), name="conv5")(c5)
    z = GlobalAveragePooling2D()(g)
    outputs = Dense(nclasses, activation="softmax", name="output1")(z)

    return tf.keras.Model(inputs, outputs, name="CNN_GAP")

#CNN + Texture
def build_cnn_texture_model(sz=96, sz_t=355, num_nodes=512, nclasses=3, dropout=True):
    
    inputs = Input(shape=(sz, sz, 1), name="input_img")
    inputs_tex = Input(shape=(sz_t,), name="input_tex")
    u1 = data_augmentation(inputs)
    u1 = data_normalization(u1)
    c = Conv2D(
        64,
        (7, 7),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv1_1",
    )(u1)
    c2 = MaxPooling2D((2, 2), name="pool1")(c)
    c = Conv2D(
        128,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv2_1",
    )(c2)
    c3 = MaxPooling2D((2, 2), name="pool2")(c)
    c = Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv3_1",
    )(c3)
    c4 = MaxPooling2D((2, 2), name="pool3")(c)
    c5 = Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv4",
    )(c4)
    g = Conv2D(64, (1, 1), name="conv5")(c5)
    h = Flatten(name="flatten1")(g)
    t = Concatenate(axis=-1, name="conc")([h, inputs_tex])
    z = Dense(num_nodes, activation='relu', name="dense1")(t)
    if dropout == True:
        z = Dropout(0.2, name="dropout1")(z)
    z = Dense(num_nodes, activation='relu', name="dense2")(z)
    if dropout == True:
        z = Dropout(0.2, name="dropout2")(z)
    outputs = Dense(nclasses, activation="softmax", name="output1")(z)

    return tf.keras.Model(inputs=[inputs, inputs_tex], outputs=outputs, name="CNN_Texture")

# CNN+GAP + Texture
def build_cnn_gap_texture_model(sz=96, sz_t=355, nclasses=3):
    
    inputs = Input(shape=(sz, sz, 1), name="input_img")
    inputs_tex = Input(shape=(sz_t,), name="input_tex")
    u1 = data_augmentation(inputs)
    u1 = data_normalization(u1)
    c = Conv2D(
        64,
        (7, 7),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv1_1",
    )(u1)
    c2 = MaxPooling2D((2, 2), name="pool1")(c)
    c = Conv2D(
        128,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv2_1",
    )(c2)
    c3 = MaxPooling2D((2, 2), name="pool2")(c)
    c = Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv3_1",
    )(c3)
    c4 = MaxPooling2D((2, 2), name="pool3")(c)
    c5 = Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv4",
    )(c4)
    g = Conv2D(64, (1, 1), name="conv5")(c5)
    h = GlobalAveragePooling2D()(g)
    t = Concatenate(axis=-1, name="conc")([h, inputs_tex])
    outputs = Dense(nclasses, activation="softmax", name="output1")(t)

    return tf.keras.Model(inputs=[inputs, inputs_tex], outputs=outputs, name="CNN_GAP_Texture")
    
# MLP    
def build_mlp_model(sz_t=355, num_nodes=256, nclasses=3, dropout=True):
    
    inputs_tex = Input(shape=(sz_t,), name="input_tex")
    z = Dense(num_nodes, activation="relu", name="dense3")(inputs_tex)
    if dropout == True:
        z = Dropout(0.2, name="dropout1")(z)
    z = Dense(num_nodes, activation="relu", name="dense4")(z)
    if dropout == True:
        z = Dropout(0.2, name="dropout2")(z)
    outputs = Dense(nclasses, activation="softmax", name="output2")(z)
    
    return tf.keras.Model(inputs=inputs_tex, outputs=outputs, name="MLP")
    
