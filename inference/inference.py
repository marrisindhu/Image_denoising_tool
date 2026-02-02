import tensorflow as tf
from src.convolution_block import Convolutional_block
from src.channel_attention_module import Channel_attention
from src.unet_module import Avg_pool_Unet_Upsample_msfe
from src.feature_extraction_module import Multi_scale_feature_extraction
from src.kernel_selection_module import Kernel_selecting_module
import numpy as np
import matplotlib.pyplot as plt
import random

def inference_single_image(model, noisy_image):
    input_image = np.expand_dims(noisy_image, axis=0)
    predicted_image = model.predict(input_image)
    
    return predicted_image[0]

def inference_batch_images(model, noisy_images):
    predicted_image = model.predict(noisy_images)
    return predicted_image

def visualize_predictions(model, X_test, y_test, n):
    random_numbers = random.choices(range(X_test.shape[0]), k=n)    # Get n random indices
    for i in random_numbers:
        noisy_image = X_test[i]
        gt_image = y_test[i]
        predicted_image = inference_single_image(model, X_test[i])
        predicted_image/=255

        f, axarr = plt.subplots(1,3, figsize=(21,21))
        axarr[0].imshow(noisy_image)
        axarr[0].set_title("Noisy image")
        axarr[0].set_axis_off()
        axarr[1].imshow(gt_image)
        axarr[1].set_title("Ground truth image")
        axarr[1].set_axis_off()
        axarr[2].imshow(predicted_image)
        axarr[2].set_title("Predicted image")
        axarr[2].set_axis_off()

def model_inference():
    best_models_path = "/Best Models/PRIDNet model/"
    model = tf.keras.models.load_model(best_models_path+'best_PRIDNet_blindnoise_256x256.h5', custom_objects={'Convolutional_block': Convolutional_block,
                                                                                                            'Channel_attention':Channel_attention,
                                                                                                            'Avg_pool_Unet_Upsample_msfe':Avg_pool_Unet_Upsample_msfe,
                                                                                                            'Multi_scale_feature_extraction':Multi_scale_feature_extraction,
                                                                                                            'Kernel_selecting_module':Kernel_selecting_module})
    return model