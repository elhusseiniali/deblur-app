import gradio as gr
import numpy as np
import tensorflow as tf
from huggingface_hub.keras_mixin import from_pretrained_keras
from PIL import Image

from create_model import Model
from model.configs import MAXIM_CONFIGS

CKPT = "google/maxim-s3-deblurring-gopro"
VARIANT = CKPT.split("/")[-1].split("-")[1]
VARIANT = VARIANT[0].upper() + "-" + VARIANT[1]
_MODEL = from_pretrained_keras(CKPT)


def mod_padding_symmetric(image, factor=64):
    """Padding the image to be divided by factor."""
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = ((height + factor) // factor) * factor, (
        (width + factor) // factor
    ) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = tf.pad(
        image,
        [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)],
        mode="REFLECT"
    )
    return image


def make_shape_even(image):
    """Pad the image to have even shapes."""
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = tf.pad(image, [(0, padh), (0, padw), (0, 0)], mode="REFLECT")
    return image


def process_image(image: Image):
    input_img = np.asarray(image) / 255.0
    height, width = input_img.shape[0], input_img.shape[1]

    # Padding images to have even shapes
    input_img = make_shape_even(input_img)
    height_even, width_even = input_img.shape[0], input_img.shape[1]

    # padding images to be multiplies of 64
    input_img = mod_padding_symmetric(input_img, factor=64)
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img, height, width, height_even, width_even


def init_new_model(input_img):
    configs = MAXIM_CONFIGS.get(VARIANT)
    configs.update(
        {
            "variant": VARIANT,
            "dropout_rate": 0.0,
            "num_outputs": 3,
            "use_bias": True,
            "num_supervision_scales": 3,
        }
    )
    configs.update(
        {"input_resolution": (input_img.shape[1], input_img.shape[2])}
    )
    with tf.device("/GPU:0"):
        new_model = Model(**configs)
        new_model.set_weights(_MODEL.get_weights())
    return new_model


def infer(image):
    preprocessed_image, \
        height, width, \
        height_even, width_even = process_image(image)
    new_model = init_new_model(preprocessed_image)

    preds = new_model.predict(preprocessed_image)
    if isinstance(preds, list):
        preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]

    preds = np.array(preds[0], np.float32)

    new_height, new_width = preds.shape[0], preds.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width
    preds = preds[h_start:h_end, w_start:w_end, :]

    return Image.fromarray(
        np.array((np.clip(preds, 0.0, 1.0) * 255.0).astype(np.uint8))
    )


title = "Deblur blurry images."
description = "Demo for deblurring."

interface = gr.Interface(
    infer,
    inputs="image",
    outputs=gr.Image(),
    title=title,
    description=description,
    allow_flagging="never",
    examples=[
        ["1fromGOPR1096.MP4.png"],
        ["1fromGOPR0950.png"],
        ["109fromGOPR1096.MP4.png"],
        ["110fromGOPR1087.MP4.png"],
    ],
)
# interface.launch(debug=True, share=True)
