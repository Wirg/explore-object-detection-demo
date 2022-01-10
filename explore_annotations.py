import pandas as pd
import streamlit as st
from stqdm import stqdm

from src.utils.display import load_and_annotate_image, load_crop

COLOR_MAP = {
    "airplane": (0, 0, 0),
    "apple": (0, 0, 128),
    "backpack": (0, 0, 128),
    "banana": (0, 0, 139),
    "baseball glove": (0, 0, 255),
    "bear": (0, 100, 0),
    "bed": (0, 128, 0),
    "bench": (0, 139, 139),
    "bicycle": (0, 191, 255),
    "bird": (0, 206, 209),
    "boat": (0, 250, 154),
    "book": (0, 255, 0),
    "bottle": (0, 255, 127),
    "bowl": (0, 255, 255),
    "broccoli": (0, 255, 255),
    "bus": (25, 25, 112),
    "cake": (30, 144, 255),
    "car": (32, 178, 170),
    "carrot": (34, 139, 34),
    "cat": (46, 139, 87),
    "cell phone": (47, 79, 79),
    "chair": (47, 79, 79),
    "clock": (50, 205, 50),
    "couch": (60, 179, 113),
    "cow": (64, 224, 208),
    "cup": (65, 105, 225),
    "dining table": (70, 130, 180),
    "dog": (72, 61, 139),
    "donut": (72, 209, 204),
    "elephant": (75, 0, 130),
    "fire hydrant": (85, 107, 47),
    "fork": (95, 158, 160),
    "frisbee": (100, 149, 237),
    "giraffe": (102, 205, 170),
    "hair drier": (105, 105, 105),
    "handbag": (105, 105, 105),
    "horse": (106, 90, 205),
    "hot dog": (107, 142, 35),
    "keyboard": (112, 128, 144),
    "kite": (112, 128, 144),
    "knife": (119, 136, 153),
    "laptop": (119, 136, 153),
    "microwave": (123, 104, 238),
    "motorcycle": (124, 252, 0),
    "mouse": (127, 255, 0),
    "orange": (128, 0, 0),
    "oven": (128, 0, 128),
    "parking meter": (128, 128, 0),
    "person": (128, 128, 128),
    "pizza": (128, 128, 128),
    "potted plant": (132, 112, 255),
    "refrigerator": (138, 43, 226),
    "remote": (139, 0, 0),
    "sandwich": (139, 0, 139),
    "scissors": (139, 69, 19),
    "sheep": (143, 188, 143),
    "sink": (144, 238, 144),
    "skateboard": (147, 112, 219),
    "skis": (148, 0, 211),
    "snowboard": (152, 251, 152),
    "spoon": (153, 50, 204),
    "sports ball": (154, 205, 50),
    "stop sign": (160, 82, 45),
    "suitcase": (165, 42, 42),
    "surfboard": (169, 169, 169),
    "teddy bear": (169, 169, 169),
    "tennis racket": (173, 255, 47),
    "tie": (178, 34, 34),
    "toaster": (184, 134, 11),
    "toilet": (186, 85, 211),
    "toothbrush": (188, 143, 143),
    "traffic light": (189, 183, 107),
    "train": (199, 21, 133),
    "truck": (205, 92, 92),
    "tv": (205, 133, 63),
    "umbrella": (208, 32, 144),
    "vase": (210, 105, 30),
    "wine glass": (210, 180, 140),
    "zebra": (218, 112, 214),
}


@st.experimental_memo
def load_all_annotations():
    return pd.read_csv("data/annotations/coco_val_2020.csv")


annotations = load_all_annotations()

with st.sidebar:
    st.write(annotations.category_name.value_counts())
    display_crops = st.checkbox("Display Crops")
    if not display_crops and st.checkbox("Filter by image"):
        images_to_display = st.multiselect(
            "Images to display", annotations.image_name.unique()
        )
        annotations = annotations.loc[lambda df: df.image_name.isin(images_to_display)]
    else:
        labels_to_display = st.multiselect(
            "Labels to display", annotations.category_name.unique()
        )
        if display_crops:
            annotations = annotations.loc[
                lambda df: df.category_name.isin(labels_to_display)
            ]
        else:
            display_only_selected_labels = st.checkbox(
                "Display only selected labels", value=False
            )
            if display_only_selected_labels:
                annotations = annotations.loc[
                    lambda df: df.category_name.isin(labels_to_display)
                ]
            else:
                annotations = annotations.groupby("image_name").filter(
                    lambda g: g.category_name.isin(labels_to_display).any()
                )


if display_crops:
    for _, crop_annotations in stqdm(annotations.iterrows(), desc="Displaying Images"):
        image_name = crop_annotations["image_name"]
        coco_url = crop_annotations["coco_url"]
        coordinates = crop_annotations[["x1", "y1", "x2", "y2"]]
        st.image(load_crop(coco_url, coordinates, shape=(640, 480)), caption=image_name)
else:
    for (image_name, coco_url), image_annotations in annotations.groupby(
        ["image_name", "coco_url"]
    ):
        st.image(
            load_and_annotate_image(coco_url, image_annotations, color_map=COLOR_MAP),
            caption=image_name,
        )