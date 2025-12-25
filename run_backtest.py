from src.strategy import FractalStrategy


if __name__ == "__main__":
    strategy = FractalStrategy(ticker="QQQ")
    metrics = strategy.run(start_date="2019-01-01", end_date="2024-12-01")
