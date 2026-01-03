from sklearn.manifold import TSNE

from src.reducers.strategy import ReducerStrategy


class TSNEStrategy(ReducerStrategy):
    def build(self, **config):
        n_components = config.get("n_components", 2)
        tsne = TSNE(n_components=n_components)
        return tsne

    def reduce(self, features, projector):
        return projector.fit_transform(features)
