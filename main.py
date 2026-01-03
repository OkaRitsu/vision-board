from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.embedder import Embedder
from src.encoders import CLIPStrategy, DinoV2Strategy, ResNetStrategy
from src.reducers import PCAStrategy, TSNEStrategy


def app():
    st.sidebar.title("Configuration")

    st.sidebar.subheader("Data")
    data_dir = st.sidebar.text_input("Dataset directory", "data/images")

    st.sidebar.subheader("Encoder")
    # st.sidebar.write("Select the encoder model for feature extraction:")
    encoder_type = st.sidebar.selectbox(
        "Encoder type",
        ("resnet18", "clip", "dinov2"),
        label_visibility="collapsed",
    )
    encoder_config = {}
    if encoder_type == "clip":
        clip_model = st.sidebar.text_input("CLIP model", "ViT-B-32")
        clip_pretrained = st.sidebar.text_input(
            "CLIP pretrained weights", "laion2b_s34b_b79k"
        )
        encoder_config["clip_model"] = clip_model
        encoder_config["clip_pretrained"] = clip_pretrained
    elif encoder_type == "dinov2":
        dinov2_variant = st.sidebar.text_input("DINOv2 variant", "dinov2_vits14_reg")
        dinov2_image_size = st.sidebar.number_input(
            "DINOv2 image size", min_value=64, max_value=1024, value=224, step=1
        )
        encoder_config["dinov2_variant"] = dinov2_variant
        encoder_config["dinov2_image_size"] = dinov2_image_size

    st.sidebar.subheader("Dimensionality Reduction")
    st.sidebar.write("Choose a method to reduce embedding dimensions:")
    dim_reduction = st.sidebar.selectbox(
        "Reduction method",
        ("pca", "tsne"),
        label_visibility="collapsed",
    )
    dim_reduction_config = {}
    if dim_reduction == "tsne":
        tsne_perplexity = st.sidebar.number_input(
            "t-SNE perplexity", min_value=5.0, max_value=100.0, value=30.0, step=1.0
        )
        tsne_learning_rate = st.sidebar.number_input(
            "t-SNE learning rate",
            min_value=10.0,
            max_value=1000.0,
            value=200.0,
            step=10.0,
        )
        tsne_iterations = st.sidebar.number_input(
            "t-SNE iterations", min_value=250, max_value=5000, value=1000, step=250
        )
        dim_reduction_config["tsne_perplexity"] = tsne_perplexity
        dim_reduction_config["tsne_learning_rate"] = tsne_learning_rate
        dim_reduction_config["tsne_iterations"] = tsne_iterations

    st.title("Vision Board")
    if st.sidebar.button("Run"):
        # Validate data directory
        data_path = Path(data_dir)
        if not data_path.exists():
            st.error(f"Dataset directory not found: {data_dir}")
            return

        # Prepare encoder
        if encoder_type == "resnet18":
            encoder = ResNetStrategy()
        elif encoder_type == "dinov2":
            encoder = DinoV2Strategy()
        elif encoder_type == "clip":
            encoder = CLIPStrategy()

        # Prepare dimensionality reducer
        if dim_reduction == "pca":
            reducer = PCAStrategy()
        elif dim_reduction == "tsne":
            reducer = TSNEStrategy()

        embedder = Embedder(
            encoder_strategy=encoder,
            reducer_strategy=reducer,
        )

        # Embedding images
        with st.spinner("Embedding images..."):
            coords, labels, paths = embedder.embed(
                data_dir=data_dir,
                encoder_config=encoder_config,
                reducer_config=dim_reduction_config,
            )

        # Create visualization
        st.session_state.df = pd.DataFrame(
            {
                "dim1": coords[:, 0],
                "dim2": coords[:, 1],
                "label": labels,
                "filename": [Path(p).name for p in paths],
                "path": paths,
            }
        )

    if "df" in st.session_state:
        df = st.session_state.df.copy()
        range_cols = st.columns(2)
        with range_cols[0]:
            x_range = st.slider(
                "X range",
                float(df["dim1"].min()),
                float(df["dim1"].max()),
                (
                    float(df["dim1"].min()),
                    float(df["dim1"].max()),
                ),
            )
        with range_cols[1]:
            y_range = st.slider(
                "Y range",
                float(df["dim2"].min()),
                float(df["dim2"].max()),
                (
                    float(df["dim2"].min()),
                    float(df["dim2"].max()),
                ),
            )
        df = df[
            (df["dim1"] >= x_range[0])
            & (df["dim1"] <= x_range[1])
            & (df["dim2"] >= y_range[0])
            & (df["dim2"] <= y_range[1])
        ]
        axis_titles = {
            "pca": ("PC1", "PC2"),
            "tsne": ("Dim 1", "Dim 2"),
        }
        xaxis, yaxis = axis_titles.get(dim_reduction, ("Dim 1", "Dim 2"))
        title = (
            f"{dim_reduction.upper()} projection of {encoder_type.upper()} embeddings"
        )

        st.subheader("Scatter Plot")
        fig = px.scatter(
            df,
            x="dim1",
            y="dim2",
            color="label",
            hover_data=["filename", "path"],
            title=title,
        )
        fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, legend_title="Class")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Loaded images")

        # Select columns for caption
        available_cols = [col for col in df.columns.tolist() if col != "path"]
        caption_cols = st.multiselect(
            "Columns for caption",
            options=available_cols,
            default=["filename", "label"],
        )

        sort_option = st.selectbox("Sort by", options=df.columns.tolist())
        img_cols = st.columns(3)
        df_sorted = df.sort_values(by=sort_option).reset_index(drop=True)
        for i, row in df_sorted.iterrows():
            # Build caption from selected columns
            caption_parts = [f"{col}: {row[col]}" for col in caption_cols]
            caption = ", ".join(caption_parts) if caption_parts else "No caption"

            with img_cols[i % 3]:
                st.image(row["path"], caption=caption)


if __name__ == "__main__":
    app()
