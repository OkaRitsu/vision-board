from src.reducers import PCAStrategy, TSNEStrategy


class ReducerStrategyFactory:
    strategies = {
        "pca": PCAStrategy,
        "tsne": TSNEStrategy,
    }

    @classmethod
    def get_strategy(cls, strategy_name: str):
        strategy_class = cls.strategies.get(strategy_name.lower())
        if not strategy_class:
            raise ValueError(f"Reducer strategy '{strategy_name}' is not supported.")
        return strategy_class()

    @classmethod
    def list_strategies(cls):
        return list(cls.strategies.keys())
