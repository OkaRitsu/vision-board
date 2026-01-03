from sklearn.decomposition import PCA

from src.reducers.strategy import ReducerStrategy


class PCAStrategy(ReducerStrategy):
    def build(self, **config):
        n_components = config.get("n_components", 2)
        pca = PCA(n_components=n_components)
        return pca

    def reduce(self, features, projector):
        return projector.fit_transform(features)
