from typing import List, Literal, Optional

import pandas as pd
import streamlit as st
from stqdm import stqdm

from src.constants.colors import COLOR_MAP
from src.utils.display import HOUR, load_and_annotate_image, load_crop

Subset = Literal["train", "val"]

st.set_page_config(
    "Explore coco dataset val set",
    page_icon=":mag_right:",
)


@st.experimental_memo(ttl=1 * HOUR)
def load_all_annotations(subset: Subset) -> pd.DataFrame:
    return pd.read_parquet(f"data/annotations/coco_{subset}_2020.parquet.gzip")


@st.experimental_memo()
def get_category_count(subset: Subset) -> pd.DataFrame:
    return load_all_annotations(subset).category_name.value_counts()


@st.experimental_memo(ttl=6 * HOUR)
def cached_isin(series: pd.Series, elements: List[str]) -> pd.Series:
    return series.isin(elements)


@st.experimental_memo(ttl=3 * HOUR)
def get_selected_annotations(
    subset: Subset, selected_images: Optional[List[str]], selected_categories: List[str]
) -> pd.DataFrame:
    all_annotations = load_all_annotations(subset)

    if selected_images is None:
        selected_annotations_for_image = True
    else:
        selected_annotations_for_image = cached_isin(
            all_annotations.image_name, selected_images
        )
    selected_annotations_for_category = cached_isin(
        all_annotations.category_name, selected_categories
    )
    return all_annotations.loc[
        selected_annotations_for_image & selected_annotations_for_category
    ]


@st.experimental_memo(ttl=3 * HOUR)
def get_selected_images(
    subset: Subset, selected_images: Optional[List[str]], selected_categories: List[str]
) -> pd.DataFrame:
    all_annotations = load_all_annotations(subset)
    selected_images = get_selected_annotations(
        subset, selected_images, selected_categories
    ).image_name.drop_duplicates()
    return all_annotations.loc[cached_isin(all_annotations.image_name, selected_images)]


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


with st.sidebar:
    subset = st.selectbox("Subset", ["val", "train"])
    category_count = get_category_count(subset)
    st.write(get_category_count(subset))
    display_crops = st.checkbox("Display Crops")
    available_labels = category_count.index.tolist()
    images_to_display = st.multiselect(
        "Images to display [Empty = All]",
        available_labels,
    )
    if not images_to_display:
        images_to_display = None
    labels_to_display = st.multiselect(
        "Labels to display",
        available_labels,
        get_arguments_from_query("label", available_labels, default_values=[]),
    )
    display_only_selected_labels = st.checkbox(
        "Display only selected labels", value=False
    )
    if display_only_selected_labels:
        annotations = get_selected_annotations(
            subset, images_to_display, labels_to_display
        )
    else:
        annotations = get_selected_images(subset, images_to_display, labels_to_display)

if display_crops:
    for _, crop_annotations in stqdm(
        annotations.iterrows(), desc="Displaying Images", total=len(annotations)
    ):
        image_name = crop_annotations["image_name"]
        coco_url = crop_annotations["coco_url"]
        coordinates = crop_annotations[["x1", "y1", "x2", "y2"]]
        st.image(load_crop(coco_url, coordinates, shape=(640, 480)), caption=image_name)
else:
    for (image_name, coco_url), image_annotations in stqdm(
        annotations.groupby(["image_name", "coco_url"]),
        desc="Displaying Images",
        total=len(annotations[["image_name", "coco_url"]].drop_duplicates()),
    ):
        st.image(
            load_and_annotate_image(coco_url, image_annotations, color_map=COLOR_MAP),
            caption=image_name,
        )
