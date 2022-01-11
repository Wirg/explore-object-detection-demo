from typing import List, Literal, Optional, TypeVar

import pandas as pd
import streamlit as st
from stqdm import stqdm

from src.constants.colors import COLOR_MAP
from src.utils.display import HOUR, load_and_annotate_image, load_crop

Subset = Literal["train", "val"]

st.set_page_config(
    "Explore coco dataset",
    page_icon=":mag_right:",
)


def load_all_annotations(subset: Subset) -> pd.DataFrame:
    return pd.read_parquet(f"data/annotations/coco_{subset}_2020.parquet.gzip")


@st.experimental_memo(max_entries=2)
def get_category_count(subset: Subset) -> pd.DataFrame:
    return load_all_annotations(subset).category_name.value_counts()


@st.experimental_memo(max_entries=2)
def get_available_image_names(subset: Subset) -> List[str]:
    return load_all_annotations(subset).image_name.drop_duplicates().tolist()


@st.experimental_memo(max_entries=10)
def cached_isin(series: pd.Series, elements: List[str]) -> pd.Series:
    return series.isin(elements)


@st.experimental_memo(ttl=0.5 * HOUR)
def get_selected_annotations(
    subset: Subset, selected_images: List[str], selected_categories: List[str]
) -> pd.DataFrame:
    all_annotations = load_all_annotations(subset)

    if not selected_images:
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


@st.experimental_memo(ttl=0.5 * HOUR)
def get_selected_images(
    subset: Subset, selected_images: Optional[List[str]], selected_categories: List[str]
) -> pd.DataFrame:
    all_annotations = load_all_annotations(subset)
    selected_images = get_selected_annotations(
        subset, selected_images, selected_categories
    ).image_name.drop_duplicates()
    return all_annotations.loc[cached_isin(all_annotations.image_name, selected_images)]


def get_arguments_list_from_query(
    key_name: str, available_values: List[str], default_values: List[str]
) -> List[str]:
    query_params = st.experimental_get_query_params()
    if query_params and key_name in query_params:
        values = query_params[key_name]
        non_existing_values = set(values) - set(available_values)
        if non_existing_values:
            st.error(
                f"The following {key_name}s in your query don't exist in the dataset {non_existing_values}"
            )
            st.stop()
        return values
    return default_values


ParamType = TypeVar("ParamType")


def get_single_argument_from_query(
    key_name: str, available_values: List[ParamType], default: int = 0
) -> ParamType:
    query_params = st.experimental_get_query_params()
    if query_params and key_name in query_params:
        param_value = query_params[key_name][0]
        try:
            return next(
                real_value
                for real_value in available_values
                if str(real_value) == param_value
            )
        except StopIteration:
            st.error(
                f"The following {key_name} in your query is not valid {param_value}. Authorized : {available_values}"
            )
            st.stop()
    return available_values[default]


with st.sidebar:
    selected_subset = st.selectbox(
        "Subset",
        ["val", "train"],
        ["val", "train"].index(
            get_single_argument_from_query("subset", ["val", "train"])
        ),
    )
    category_count = get_category_count(selected_subset)
    st.write(category_count)
    st.write(f"{len(category_count)} Labels available")
    display_crops = st.checkbox(
        "Display Crops",
        value=get_single_argument_from_query(
            "display_crops", available_values=[False, True]
        ),
    )
    available_labels = category_count.index.tolist()
    available_image_names = get_available_image_names(selected_subset)
    images_to_display = st.multiselect(
        "Images to display [Empty = All]",
        available_image_names,
        get_arguments_list_from_query(
            "image_name", available_image_names, default_values=[]
        ),
    )
    if st.button("Empty selection", key="empty_images"):
        images_to_display = []
    labels_to_display = st.multiselect(
        "Labels to display",
        available_labels,
        get_arguments_list_from_query("label", available_labels, default_values=[]),
    )
    if st.button("Empty selection", key="empty_labels"):
        labels_to_display = []
    display_only_selected_labels = st.checkbox(
        "Display only selected labels",
        value=get_single_argument_from_query(
            "display_only_selected_labels", available_values=[False, True]
        ),
    )
    if display_only_selected_labels:
        annotations = get_selected_annotations(
            selected_subset, images_to_display, labels_to_display
        )
    else:
        annotations = get_selected_images(
            selected_subset, images_to_display, labels_to_display
        )

st.experimental_set_query_params(
    subset=selected_subset,
    display_only_selected_labels=display_only_selected_labels,
    display_crops=display_crops,
    image_name=images_to_display,
    label=labels_to_display,
)

if display_crops:
    n_annotations = len(annotations)
    with st.sidebar:
        st.write(f"{n_annotations} crops selected")
    for _, crop_annotations in stqdm(
        annotations.iterrows(), desc="Displaying Images", total=n_annotations
    ):
        image_name = crop_annotations["image_name"]
        coco_url = crop_annotations["coco_url"]
        coordinates = crop_annotations[["x1", "y1", "x2", "y2"]]
        st.image(load_crop(coco_url, coordinates, shape=(640, 480)), caption=image_name)
else:
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
