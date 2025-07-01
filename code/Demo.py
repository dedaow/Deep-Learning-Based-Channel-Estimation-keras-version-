import numpy as np
import math
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input, BatchNormalization, Activation, Lambda, Subtract, Conv2DTranspose, PReLU
from keras.regularizers import l2
from keras.layers import Reshape, Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
import keras.backend as K
from scipy import interpolate
import matplotlib.pyplot as plt


def psnr(target, ref):
    """Calculate Peak Signal-to-Noise Ratio"""
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def interpolation(noisy, SNR, Number_of_pilot, interp):
    """Perform interpolation on pilot symbols"""
    # Handle case where noisy is a dictionary (from loadmat)
    if isinstance(noisy, dict):
        # Filter out MATLAB metadata keys (those starting with '__')
        data_keys = [k for k in noisy.keys() if not k.startswith('__')]
        print(f"Available data keys: {data_keys}")
        
        if len(data_keys) == 1:
            key = data_keys[0]
        elif len(data_keys) > 1:
            # Try to find the most likely key
            possible_keys = [k for k in data_keys if 'noisy' in k.lower() or 'h' in k.lower()]
            if possible_keys:
                key = possible_keys[0]
            else:
                key = data_keys[0]  # Use first available key
            print(f"Multiple keys found, using: {key}")
        else:
            raise ValueError("No data keys found in the dictionary")
        
        noisy = noisy[key]
        print(f"Extracted data shape: {noisy.shape}")
    
    # Get the actual dimensions from the data
    data_shape = noisy.shape
    num_samples = data_shape[0] if len(data_shape) > 0 else 40000
    
    noisy_image = np.zeros((num_samples, 72, 14, 2))

    noisy_image[:, :, :, 0] = np.real(noisy)
    noisy_image[:, :, :, 1] = np.imag(noisy)

    # Define pilot positions based on number of pilots
    if (Number_of_pilot == 48):
        idx = [14*i for i in range(1, 72, 6)] + [4+14*(i) for i in range(4, 72, 6)] + \
              [7+14*(i) for i in range(1, 72, 6)] + [11+14*(i) for i in range(4, 72, 6)]
    elif (Number_of_pilot == 16):
        idx = [4+14*(i) for i in range(1, 72, 9)] + [9+14*(i) for i in range(4, 72, 9)]
    elif (Number_of_pilot == 24):
        idx = [14*i for i in range(1, 72, 9)] + [6+14*i for i in range(4, 72, 9)] + \
              [11+14*i for i in range(1, 72, 9)]
    elif (Number_of_pilot == 8):
        idx = [4+14*(i) for i in range(5, 72, 18)] + [9+14*(i) for i in range(8, 72, 18)]
    elif (Number_of_pilot == 36):
        idx = [14*(i) for i in range(1, 72, 6)] + [6+14*(i) for i in range(4, 72, 6)] + \
              [11+14*i for i in range(1, 72, 6)]

    r = [x//14 for x in idx]
    c = [x % 14 for x in idx]

    interp_noisy = np.zeros((num_samples, 72, 14, 2))

    for i in range(len(noisy)):
        # Interpolate real part
        z = [noisy_image[i, j, k, 0] for j, k in zip(r, c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z, function='gaussian')
            X, Y = np.meshgrid(range(72), range(14))
            z_intp = f(X, Y)
            interp_noisy[i, :, :, 0] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72), range(14), tck)
            interp_noisy[i, :, :, 0] = z_intp

        # Interpolate imaginary part
        z = [noisy_image[i, j, k, 1] for j, k in zip(r, c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z, function='gaussian')
            X, Y = np.meshgrid(range(72), range(14))
            z_intp = f(X, Y)
            interp_noisy[i, :, :, 1] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72), range(14), tck)
            interp_noisy[i, :, :, 1] = z_intp

    interp_noisy = np.concatenate((interp_noisy[:, :, :, 0], interp_noisy[:, :, :, 1]), axis=0).reshape(2*num_samples, 72, 14, 1)
    
    return interp_noisy


def SRCNN_model():
    """Create SRCNN model architecture"""
    input_shape = (72, 14, 1)
    x = Input(shape=input_shape)
    
   
    c1 = Conv2D(64, (9, 9), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    c2 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c3 = Conv2D(1, (5, 5), kernel_initializer='he_normal', padding='same')(c2)
    
    model = Model(inputs=x, outputs=c3)
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def SRCNN_train(train_data, train_label, val_data, val_label, channel_model, num_pilots, SNR):
    """Train SRCNN model"""
    srcnn_model = SRCNN_model()
    print(srcnn_model.summary())
    
    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(train_data, train_label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs=300, verbose=0)
    
    srcnn_model.save_weights("SRCNN_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".weights.h5")


def SRCNN_predict(input_data, channel_model, num_pilots, SNR):
    """Make predictions using trained SRCNN model"""
    srcnn_model = SRCNN_model()
    srcnn_model.load_weights("SRCNN_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".weights.h5")
    predicted = srcnn_model.predict(input_data)
    return predicted


def DNCNN_model():
    """Create DNCNN model architecture"""
    inpt = Input(shape=(None, None, 1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(18):
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def DNCNN_train(train_data, train_label, val_data, val_label, channel_model, num_pilots, SNR):
    """Train DNCNN model"""
    dncnn_model = DNCNN_model()
    print(dncnn_model.summary())

    checkpoint = ModelCheckpoint("DNCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    dncnn_model.fit(train_data, train_label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs=200, verbose=0)
    dncnn_model.save_weights("DNCNN_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".weights.h5")


def DNCNN_predict(input_data, channel_model, num_pilots, SNR):
    """Make predictions using trained DNCNN model"""
    dncnn_model = DNCNN_model()
    dncnn_model.load_weights("DNCNN_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".weights.h5")
    predicted = dncnn_model.predict(input_data)
    return predicted


if __name__ == "__main__":
    # Load datasets
    channel_model = "VehA"
    SNR = 22
    Number_of_pilots = 48
    
    # Load data files
    print("Loading data files...")
    perfect_data = loadmat("Perfect_H_40000.mat")
    noisy_data = loadmat("My_noisy_H_12.mat")
    
    # Debug: print available keys (excluding MATLAB metadata)
    perfect_keys = [k for k in perfect_data.keys() if not k.startswith('__')]
    noisy_keys = [k for k in noisy_data.keys() if not k.startswith('__')]
    
    print("Keys in perfect_data:", perfect_keys)
    print("Keys in noisy_data:", noisy_keys)
    
    # Extract the actual data arrays - use the first non-metadata key
    perfect_key = perfect_keys[0] if perfect_keys else 'My_perfect_H'
    noisy_key = noisy_keys[0] if noisy_keys else 'My_noisy_H'
    
    perfect = perfect_data[perfect_key]
    noisy_input = noisy_data[noisy_key]
    
    print(f"Perfect data shape: {perfect.shape}")
    print(f"Noisy input shape: {noisy_input.shape}")
    
    # Perform interpolation
    interp_noisy = interpolation(noisy_input, SNR, Number_of_pilots, 'rbf')
    
    # Prepare perfect channel data
    perfect_image = np.zeros((len(perfect), 72, 14, 2))
    perfect_image[:, :, :, 0] = np.real(perfect)
    perfect_image[:, :, :, 1] = np.imag(perfect)
    perfect_image = np.concatenate((perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]), axis=0).reshape(2*len(perfect), 72, 14, 1)
    
    # Split data for training and validation
    idx_random = np.random.rand(len(perfect_image)) < (8/9)  # uses 8/9 for training and 1/9 for validation
    train_data, train_label = interp_noisy[idx_random, :, :, :], perfect_image[idx_random, :, :, :]
    val_data, val_label = interp_noisy[~idx_random, :, :, :], perfect_image[~idx_random, :, :, :]
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
    # Training SRCNN
    print("Training SRCNN...")
    SRCNN_train(train_data, train_label, val_data, val_label, channel_model, Number_of_pilots, SNR)
    
    # Prediction using SRCNN
    print("Making SRCNN predictions...")
    srcnn_pred_train = SRCNN_predict(train_data, channel_model, Number_of_pilots, SNR)
    srcnn_pred_validation = SRCNN_predict(val_data, channel_model, Number_of_pilots, SNR)
    
    # Training DNCNN
    print("Training DNCNN...")
    DNCNN_train(train_data, train_label, val_data, val_label, channel_model, Number_of_pilots, SNR)
    
    # Prediction using DNCNN
    print("Making DNCNN predictions...")
    dncnn_pred_train = DNCNN_predict(train_data, channel_model, Number_of_pilots, SNR)
    dncnn_pred_validation = DNCNN_predict(val_data, channel_model, Number_of_pilots, SNR)
    
    print("Training and prediction completed!")
