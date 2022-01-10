from typing import List

import pandas as pd
import streamlit as st
from stqdm import stqdm

from src.constants.colors import COLOR_MAP
from src.utils.display import load_and_annotate_image, load_crop

st.set_page_config(
    "Explore coco dataset val set",
    page_icon=":mag_right:",
)


@st.experimental_memo
def load_all_annotations():
    return pd.read_csv("data/annotations/coco_val_2020.csv")


def get_arguments_from_query(
    key_name: str, available_values: List[str], default_values: List[str]
) -> List[str]:
    query_params = st.experimental_get_query_params()
    if query_params and key_name in query_params:
        default_values = query_params[key_name]
        non_existing_values = set(default_values) - set(available_values)
        if non_existing_values:
            st.error(
                f"The following {key_name}s in your query don't exist in the dataset {non_existing_values}"
            )
            st.stop()
    return default_values


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
        available_labels = annotations.category_name.unique()
        labels_to_display = st.multiselect(
            "Labels to display",
            available_labels,
            get_arguments_from_query("label", available_labels, default_values=[]),
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
    for _, crop_annotations in stqdm(
        annotations.iterrows(), desc="Displaying Images", total=len(annotations)
    ):
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
