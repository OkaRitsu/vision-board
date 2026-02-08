from sklearn.decomposition import PCA

from src.reducers.strategy import ReducerStrategy


class PCAStrategy(ReducerStrategy):
    def build(self, **config):
        pca = PCA(n_components=2, random_state=42, **config)
        return pca

    def reduce(self, features, projector):
        return projector.fit_transform(features)
