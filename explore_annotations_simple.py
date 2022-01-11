from typing import List

import pandas as pd
import streamlit as st
from stqdm import stqdm

from src.constants.colors import COLOR_MAP
from src.utils.display import HOUR, load_and_annotate_image

st.set_page_config(
    "Explore coco dataset",
    page_icon=":mag_right:",
)


def load_all_annotations() -> pd.DataFrame:
    return pd.read_parquet("data/annotations/coco_val_2020.parquet.gzip")


@st.experimental_memo(max_entries=2)
def get_category_count() -> pd.DataFrame:
    return load_all_annotations().category_name.value_counts()


@st.experimental_memo(ttl=0.5 * HOUR)
def get_selected_images(selected_categories: List[str]) -> pd.DataFrame:
    all_annotations = load_all_annotations()
    return all_annotations.groupby("image_name").filter(
        lambda g: g.category_name.isin(selected_categories).any()
    )


with st.sidebar:
    category_count = get_category_count()
    st.write(category_count)
    st.write(f"{len(category_count)} Labels available")
    available_labels = category_count.index.tolist()
    labels_to_display = st.multiselect(
        "Labels to display",
        available_labels,
    )
    annotations = get_selected_images(labels_to_display)


n_images = len(annotations[["image_name", "coco_url"]].drop_duplicates())
with st.sidebar:
    st.write(f"{n_images} images selected")
for (image_name, coco_url), image_annotations in stqdm(
    annotations.groupby(["image_name", "coco_url"]),
    desc="Displaying Images",
    total=n_images,
):
    st.image(
        load_and_annotate_image(coco_url, image_annotations, color_map=COLOR_MAP),
        caption=image_name,
    )
