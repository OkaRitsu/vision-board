from __future__ import annotations

from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from sklearn.metrics import pairwise_distances

from src.embedder import Embedder
from src.encoders import EncoderStrategyFactory
from src.reducers import ReducerStrategyFactory


def app():
    st.sidebar.title("Configuration")

    st.sidebar.subheader("Data")
    dataset_csv = st.sidebar.file_uploader(
        "Upload dataset CSV",
        type=["csv"],
        help="CSV file with columns: 'filepath' and 'label'",
    )
    if dataset_csv is None:
        st.warning("Please upload a dataset CSV file.")
        st.stop()
    dataset_df = pd.read_csv(dataset_csv)

    st.sidebar.subheader("Encoder")
    # st.sidebar.write("Select the encoder model for feature extraction:")
    encoder_type = st.sidebar.selectbox(
        "Encoder type",
        ("resnet", "clip", "dinov2", "depth_anything_v2"),
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
        # Prepare encoder and reducer
        encoder = EncoderStrategyFactory.get_strategy(encoder_type)
        reducer = ReducerStrategyFactory.get_strategy(dim_reduction)
        embedder = Embedder(
            encoder_strategy=encoder,
            reducer_strategy=reducer,
        )

        # Embedding images
        with st.spinner(f"Embedding {len(dataset_df)} images..."):
            st.session_state.df = embedder.embed(
                dataset_df=dataset_df,
                encoder_config=encoder_config,
                reducer_config=dim_reduction_config,
            )

    if "df" in st.session_state:
        df = st.session_state.df.copy()
        range_cols = st.columns(2)
        with range_cols[0]:
            x_range = st.slider(
                "X range",
                float(df["x"].min()),
                float(df["x"].max()),
                (
                    float(df["x"].min()),
                    float(df["x"].max()),
                ),
            )
        with range_cols[1]:
            y_range = st.slider(
                "Y range",
                float(df["y"].min()),
                float(df["y"].max()),
                (
                    float(df["y"].min()),
                    float(df["y"].max()),
                ),
            )
        df = df[
            (df["x"] >= x_range[0])
            & (df["x"] <= x_range[1])
            & (df["y"] >= y_range[0])
            & (df["y"] <= y_range[1])
        ]

        if "selected" not in df.columns:
            df["selected"] = False

        # FIXME: static ディレクトリに画像を置く必要がある（アプリを起動する前に）
        # TODO: 相対パスを使った方法を検討
        # 例:
        # mkdir static && cd static
        # ln -s ../data/InsPLAD-fault/defect_supervised/glass-insulator/val/*/*.jpg .
        df["url"] = df["filename"].apply(lambda p: f"app/static/{quote(p)}")

        st.subheader("Table View")
        edited_df = st.data_editor(
            df,
            column_config={
                "url": st.column_config.ImageColumn("image", pinned=True),
                "selected": st.column_config.CheckboxColumn("Select", pinned=True),
            },
            hide_index=True,
            key="data_editor",
        )

        # Update scatter plot with selected highlights

        # Create separate dataframes for selected and non-selected points
        df_not_selected = edited_df[~edited_df["selected"]]
        df_selected = edited_df[edited_df["selected"]]

        st.subheader("Scatter")
        color_by = st.selectbox(
            "Color by",
            options=df.columns.tolist(),
            index=0,
            help="Select the column to color the points by.",
        )
        fig = px.scatter(
            df_not_selected,
            x="x",
            y="y",
            color=color_by,
            hover_data=["filename"],
        )

        # Add selected points with border
        if not df_selected.empty:
            fig_selected = px.scatter(
                df_selected,
                x="x",
                y="y",
                color=color_by,
                hover_data=["filename"],
            )

            # Update selected points to have thick border
            for trace in fig_selected.data:
                trace.marker.line.width = 3
                trace.marker.line.color = "red"
                trace.marker.size = 12
                trace.showlegend = False  # Don't duplicate in legend
                fig.add_trace(trace)

        # Update layout for non-selected points (no border)
        fig.update_traces(
            marker=dict(size=8, line=dict(width=0)), selector=dict(showlegend=True)
        )

        fig.update_layout(xaxis_title="X", yaxis_title="Y", legend_title=color_by)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            label="Download Image Embeddings CSV",
            data=edited_df[["filename", "x", "y"]].to_csv(index=False).encode("utf-8"),
            file_name="image_embeddings.csv",
            mime="text/csv",
        )

        st.subheader("Distance Matrix")
        distance_type = st.selectbox(
            "Distance type",
            ("euclidean", "manhattan", "cosine"),
            label_visibility="collapsed",
        )

        coords = edited_df[["x", "y"]].to_numpy()
        distance_matrix = pairwise_distances(coords, metric=distance_type)
        heatmap_fig = go.Figure(
            data=go.Heatmap(
                x=edited_df["filename"],
                y=edited_df["filename"],
                z=distance_matrix,
                colorscale="plasma",
            ),
            layout=go.Layout(
                xaxis={"showticklabels": False},
                yaxis={"showticklabels": False},
            ),
        )
        st.plotly_chart(heatmap_fig)
        st.download_button(
            label="Download Distance Matrix CSV",
            data=pd.DataFrame(
                distance_matrix,
                index=edited_df["filename"],
                columns=edited_df["filename"],
            )
            .to_csv()
            .encode("utf-8"),
            file_name="distance_matrix.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    app()
