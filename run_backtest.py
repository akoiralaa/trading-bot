from src.strategy import FractalStrategy
from src.csv_loader import CSVLoader


if __name__ == "__main__":
    strategy = FractalStrategy(ticker="QQQ")
    strategy.data_loader = CSVLoader(ticker="QQQ")
    
    metrics = strategy.run(start_date="2019-01-01", end_date="2024-12-01")
