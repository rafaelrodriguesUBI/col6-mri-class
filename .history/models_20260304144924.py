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
    RandomContrast,
    Lambda # Added Lambda for cleaner syntax
)
from kymatio.keras import Scattering2D # Added Kymatio import

# ... [Preprocessing and CNN models remain exactly the same] ...

# Scattering-based CNN
def build_scattering_cnn_model(sz=96, Jv=3, Lv=8, num_nodes=64, nclasses=3, dropout=True):
    
    image_input = Input(shape=(sz, sz))
    x = Scattering2D(J=Jv, L=Lv)(image_input)
    x = tf.transpose(x, [0, 2, 3, 1])
    x = Lambda(lambda x: x[:,:,:,1:])(x)
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)  
    x = Conv2D(16, (1,1), strides=(1,1), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(num_nodes, activation='relu')(x)
    if dropout == True:
        x = Dropout(0.2)(x)
    x = Dense(num_nodes, activation='relu')(x)
    if dropout == True:
        x = Dropout(0.2)(x)
    outputs = Dense(nclasses, activation='softmax')(x)

    return tf.keras.Model(inputs=image_input, outputs=outputs, name="Scat_CNN")

# Scattering-based CNN + Texture
def build_scattering_cnn_texture_model(sz=96, sz_t=355, Jv=3, Lv=8, num_nodes=64, nclasses=3, dropout=True):
    
    image_input = Input(shape=(sz, sz))
    inputs_tex = Input(shape=(sz_t,), name="input_tex")
    
    x = Scattering2D(J=Jv, L=Lv)(image_input)
    x = tf.transpose(x, [0, 2, 3, 1])
    x = Lambda(lambda x: x[:,:,:,1:])(x)
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(2,2)(x)  
    x = Conv2D(16, (1,1), strides=(1,1), padding='same', activation='relu')(x)
    h = Flatten()(x)
    t = Concatenate(axis=-1, name="conc")([h, inputs_tex])
    
    x = Dense(num_nodes, activation='relu')(t)
    if dropout == True:
        x = Dropout(0.2)(x)
    x = Dense(num_nodes, activation='relu')(x)
    if dropout == True:
        x = Dropout(0.2)(x)
    outputs = Dense(nclasses, activation='softmax')(x)

    return tf.keras.Model(inputs=[image_input, inputs_tex], outputs=outputs, name="Scat_CNN_Texture")