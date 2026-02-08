from sklearn.manifold import TSNE

from src.reducers.strategy import ReducerStrategy


class TSNEStrategy(ReducerStrategy):
    def build(self, **config):
        tsne = TSNE(n_components=2, random_state=42, **config)
        return tsne

    def reduce(self, features, projector):
        return projector.fit_transform(features)
