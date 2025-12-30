"""
Monte Carlo Backtester for Stress Testing

Problem with single backtest:
- "I backtested and made 12% annually" = ONE path through history
- What if markets had occurred in different order?
- What if we got unlucky on 3 early trades?
- What about black swan events?

Solution: Monte Carlo
- Shuffle historical trades 10,000 times
- See distribution of possible outcomes
- Calculate survival rate, VaR, Conditional VaR
- Inject artificial crashes to test risk management

This is what separates "lucky backtests" from "robust strategies"
Jane Street will ask: "How do you know your edge is real and not luck?"
Answer: "We ran 10,000 Monte Carlo simulations"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats


@dataclass
class SimulationResult:
    """Container for a single Monte Carlo simulation result"""
    equity_curve: np.ndarray
    final_equity: float
    max_drawdown: float
    total_return_pct: float
    sharpe_ratio: float
    survived: bool


class MonteCarloBacktester:
    """
    Runs Monte Carlo simulations to stress test a strategy
    
    Core insight: Your backtest is just ONE possible path through history.
    This shows the distribution of possible outcomes.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.simulations: List[SimulationResult] = []
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown as percentage of peak
        
        Max Drawdown = (Trough - Peak) / Peak
        
        This is the worst peak-to-trough decline during the period.
        """
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        return abs(np.min(drawdown))
    
    def calculate_sharpe_ratio(self, 
                              equity_curve: np.ndarray,
                              rf_rate: float = 0.02) -> float:
        """
        Sharpe Ratio = (Return - Risk-Free Rate) / Std Dev of Returns
        
        Higher is better (more return per unit of risk)
        Typically:
        - < 1.0 = poor
        - 1.0-2.0 = acceptable
        - 2.0-3.0 = good
        - > 3.0 = excellent
        """
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) - rf_rate / 252  # Daily risk-free rate
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
        
        sharpe = excess_return / volatility * np.sqrt(252)  # Annualize
        return sharpe
    
    def run_simulation_shuffled_trades(self,
                                      trade_returns: List[float]) -> SimulationResult:
        """
        Run ONE Monte Carlo simulation by shuffling historical trade order
        
        Args:
            trade_returns: List of returns from historical trades (e.g., [0.02, -0.01, 0.03, ...])
        
        Returns:
            SimulationResult with equity curve and metrics
        """
        
        # Shuffle the order of trades
        shuffled_returns = np.random.permutation(trade_returns)
        
        # Replay with shuffled order
        equity = self.initial_capital
        equity_curve = [equity]
        
        for ret in shuffled_returns:
            equity = equity * (1 + ret)
            equity_curve.append(equity)
        
        equity_curve = np.array(equity_curve)
        
        # Calculate metrics
        final_equity = equity_curve[-1]
        max_dd = self.calculate_max_drawdown(equity_curve)
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        sharpe = self.calculate_sharpe_ratio(equity_curve)
        
        # Did account survive? (didn't go to zero)
        survived = final_equity > 0
        
        return SimulationResult(
            equity_curve=equity_curve,
            final_equity=final_equity,
            max_drawdown=max_dd,
            total_return_pct=total_return,
            sharpe_ratio=sharpe,
            survived=survived
        )
    
    def run_full_monte_carlo(self,
                            trade_returns: List[float],
                            num_simulations: int = 10000) -> Dict:
        """
        Run NUM_SIMULATIONS Monte Carlo simulations
        
        Returns statistics about the distribution of outcomes
        """
        
        self.simulations = []
        
        for sim in range(num_simulations):
            result = self.run_simulation_shuffled_trades(trade_returns)
            self.simulations.append(result)
            
            if (sim + 1) % 1000 == 0:
                print(f"Completed {sim + 1}/{num_simulations} simulations...")
        
        return self.analyze_simulations()
    
    def analyze_simulations(self) -> Dict:
        """
        Analyze results from Monte Carlo simulations
        
        Returns statistics for interview presentation
        """
        
        if not self.simulations:
            return {}
        
        final_equities = np.array([s.final_equity for s in self.simulations])
        max_drawdowns = np.array([s.max_drawdown for s in self.simulations])
        returns = np.array([s.total_return_pct for s in self.simulations])
        sharpe_ratios = np.array([s.sharpe_ratio for s in self.simulations])
        survived = np.array([s.survived for s in self.simulations])
        
        return {
            # Survival statistics
            'survival_rate': np.sum(survived) / len(survived),
            'blow_up_rate': 1 - (np.sum(survived) / len(survived)),
            'num_survived': np.sum(survived),
            
            # Return distribution
            'mean_return_pct': np.mean(returns),
            'median_return_pct': np.median(returns),
            'std_return_pct': np.std(returns),
            'min_return_pct': np.min(returns),
            'max_return_pct': np.max(returns),
            
            # Percentiles (equity cone)
            'percentile_1': np.percentile(final_equities, 1),
            'percentile_5': np.percentile(final_equities, 5),
            'percentile_25': np.percentile(final_equities, 25),
            'percentile_50': np.percentile(final_equities, 50),  # median
            'percentile_75': np.percentile(final_equities, 75),
            'percentile_95': np.percentile(final_equities, 95),
            'percentile_99': np.percentile(final_equities, 99),
            
            # Risk metrics
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'worst_max_drawdown': np.max(max_drawdowns),
            'percentile_95_max_drawdown': np.percentile(max_drawdowns, 95),
            'percentile_99_max_drawdown': np.percentile(max_drawdowns, 99),
            
            # Sharpe
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            
            # Value at Risk
            'var_95': np.percentile(returns, 5),  # 5th percentile = 95% VaR
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
            
            # Simulations
            'num_simulations': len(self.simulations),
        }
    
    def run_crash_injection_test(self,
                                trade_returns: List[float],
                                crash_magnitude: float = -0.05,
                                num_tests: int = 100) -> Dict:
        """
        Stress test: Inject a sudden crash at random points
        
        Simulate: "What if we had a -5% gap-down overnight?"
        
        Args:
            trade_returns: Historical trade returns
            crash_magnitude: e.g., -0.05 = -5% crash
            num_tests: How many different crash points to test
        
        Returns:
            Statistics on how often the account survives crashes
        """
        
        survival_by_position = {}
        
        for test in range(num_tests):
            # Pick random point to inject crash
            crash_position = np.random.randint(0, len(trade_returns))
            
            # Create modified returns with crash
            modified_returns = trade_returns.copy()
            modified_returns[crash_position] = crash_magnitude
            
            # Run simulation
            equity = self.initial_capital
            for ret in modified_returns:
                equity = equity * (1 + ret)
            
            # Track survival
            if crash_position not in survival_by_position:
                survival_by_position[crash_position] = []
            
            survived = equity > self.initial_capital * 0.5  # Survive if didn't lose > 50%
            survival_by_position[crash_position].append(survived)
        
        # Analyze
        survival_rates = {
            pos: np.mean(results)
            for pos, results in survival_by_position.items()
        }
        
        return {
            'overall_survival_rate': np.mean(list(survival_rates.values())),
            'worst_position': min(survival_rates, key=survival_rates.get),
            'worst_survival_rate': min(survival_rates.values()),
            'crash_magnitude': crash_magnitude,
            'num_crash_tests': num_tests,
            'message': f"Account survived {np.mean(list(survival_rates.values())):.1%} of crash scenarios"
        }
    
    def plot_equity_cone(self, output_path: str = None):
        """
        Visualize Monte Carlo results as equity cone
        
        Shows: What are possible equity paths?
        """
        
        if not self.simulations:
            print("No simulations to plot")
            return
        
        # Align all paths to same length
        min_length = min(len(s.equity_curve) for s in self.simulations)
        aligned_curves = np.array([s.equity_curve[:min_length] for s in self.simulations])
        
        # Calculate percentiles
        perc_1 = np.percentile(aligned_curves, 1, axis=0)
        perc_5 = np.percentile(aligned_curves, 5, axis=0)
        perc_50 = np.percentile(aligned_curves, 50, axis=0)
        perc_95 = np.percentile(aligned_curves, 95, axis=0)
        perc_99 = np.percentile(aligned_curves, 99, axis=0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        bars = np.arange(min_length)
        
        # Plot percentile bands
        ax.fill_between(bars, perc_1, perc_99, alpha=0.1, label='1-99 percentile')
        ax.fill_between(bars, perc_5, perc_95, alpha=0.2, label='5-95 percentile')
        ax.plot(bars, perc_50, 'b-', linewidth=2, label='Median')
        ax.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        
        # Plot a few random simulations
        for i in np.random.choice(len(self.simulations), min(100, len(self.simulations)), replace=False):
            ax.plot(bars, aligned_curves[i], alpha=0.05, color='gray')
        
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Equity ($)')
        ax.set_title(f'Monte Carlo Equity Cone ({len(self.simulations):,} simulations)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved equity cone to {output_path}")
        else:
            plt.show()
    
    def plot_return_distribution(self, output_path: str = None):
        """Plot distribution of returns across simulations"""
        
        returns = np.array([s.total_return_pct * 100 for s in self.simulations])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(returns), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.1f}%')
        ax.axvline(np.median(returns), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.1f}%')
        ax.axvline(np.percentile(returns, 5), color='orange', linestyle='--', linewidth=2, label=f'5th %ile: {np.percentile(returns, 5):.1f}%')
        
        ax.set_xlabel('Total Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Returns ({len(self.simulations):,} simulations)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved return distribution to {output_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    
    # Simulate historical trades (what your backtester produces)
    np.random.seed(42)
    
    # Strategy with 55% win rate, average of +2% wins and -1.5% losses
    num_trades = 100
    trades = []
    
    for i in range(num_trades):
        if np.random.rand() < 0.55:  # 55% win rate
            ret = np.random.normal(0.02, 0.01)  # +2% average win
        else:
            ret = np.random.normal(-0.015, 0.01)  # -1.5% average loss
        
        trades.append(ret)
    
    print(f"Historical trades: {len(trades)}")
    print(f"Average trade return: {np.mean(trades):.2%}")
    print(f"Win rate: {len([t for t in trades if t > 0]) / len(trades):.1%}")
    
    # Run Monte Carlo
    mc = MonteCarloBacktester(initial_capital=100000)
    results = mc.run_full_monte_carlo(trades, num_simulations=10000)
    
    print("\n=== Monte Carlo Results ===")
    print(f"Survival rate: {results['survival_rate']:.1%}")
    print(f"Median return: {results['median_return_pct']:.2%}")
    print(f"5th percentile (VaR): {results['percentile_5']:.0f}")
    print(f"95th percentile: {results['percentile_95']:.0f}")
    print(f"Mean max drawdown: {results['mean_max_drawdown']:.1%}")
    print(f"Worst case drawdown (1%ile): {np.percentile([s.max_drawdown for s in mc.simulations], 99):.1%}")
    
    # Crash injection test
    crash_results = mc.run_crash_injection_test(trades, crash_magnitude=-0.05, num_tests=100)
    print("\n=== Crash Injection Test (-5%) ===")
    print(f"Survival rate with crashes: {crash_results['overall_survival_rate']:.1%}")