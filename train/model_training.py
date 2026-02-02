from src.model import create_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def train_model(noisy_train_images, noisy_test_images, image_generator_train, image_generator_test):
    model = create_model()
    steps_per_epoch_train = len(noisy_train_images)
    steps_per_epoch_validation = len(noisy_test_images)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=0.00003))
    best_models_path = "Best Models/PRIDNet model/"
    callbacks_lst = [
                    tf.keras.callbacks.ModelCheckpoint(filepath=best_models_path+"best_PRIDNet_blindnoise_256x256.h5", save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=0.0000009, min_delta=0.0001, factor=0.70, patience=3, verbose=1, mode='min'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.0001, patience=10)
    ]
    model.fit(image_generator_train, 
            validation_data=image_generator_test,
                            steps_per_epoch=steps_per_epoch_train,
                            validation_steps=steps_per_epoch_validation,
                            epochs=100,
                            verbose=1,
                        callbacks=callbacks_lst)