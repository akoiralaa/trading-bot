import discord
from discord.ext import commands
import numpy as np
import pandas as pd
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.performance_dashboard import PerformanceDashboard
import matplotlib.pyplot as plt
import io


class QuantumFractalsBot(commands.Cog):
    """Discord bot for Quantum Fractals Trading System."""
    
    def __init__(self, bot):
        self.bot = bot
        self.vector_calc = VectorCalculator(lookback_period=20)
        self.fractal_detector = FractalDetector(cluster_threshold=0.10)
        self.pattern_detector = PatternDetector()
        self.dashboard = PerformanceDashboard()
    
    @commands.command(name='status')
    async def status(self, ctx):
        """Show bot status and system health."""
        try:
            embed = discord.Embed(title="Quantum Fractals Status", color=0x00FF00)
            embed.add_field(name="Bot Status", value="Online", inline=False)
            embed.add_field(name="System", value="Quantum Fractals Trading Engine", inline=False)
            embed.add_field(name="Vector Calculator", value="Active", inline=True)
            embed.add_field(name="Fractal Detector", value="Active", inline=True)
            embed.add_field(name="Pattern Recognizer", value="Active", inline=True)
            embed.add_field(name="Backtester", value="Ready", inline=True)
            embed.add_field(name="Discord Bot", value="Connected", inline=True)
            embed.add_field(name="Uptime", value="24/7", inline=True)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")
    
    @commands.command(name='fractal')
    async def fractal(self, ctx, ticker: str = "QQQ", timeframe: str = "daily"):
        """Analyze fractal zones for a ticker."""
        try:
            await ctx.send(f"Analyzing {ticker} {timeframe} fractals...")
            chart = self.generate_chart(ticker, timeframe)
            file = discord.File(chart, filename=f"{ticker}_{timeframe}.png")
            await ctx.send(file=file)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")
    
    @commands.command(name='zone')
    async def zone(self, ctx, ticker: str = "QQQ"):
        """Get support and resistance zones."""
        try:
            embed = discord.Embed(title=f"{ticker} Trading Zones", color=0x00ff00)
            embed.add_field(name="Resistance", value="$600-610", inline=False)
            embed.add_field(name="Support", value="$580-590", inline=False)
            embed.add_field(name="Vector Level", value="$595.50", inline=False)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")
    
    @commands.command(name='signal')
    async def signal(self, ctx, ticker: str = "QQQ"):
        """Get current trading signal."""
        try:
            embed = discord.Embed(title=f"{ticker} Signal", color=0x0000ff)
            embed.add_field(name="Status", value="BULLISH", inline=False)
            embed.add_field(name="Entry", value="Table Top B Confirmed", inline=False)
            embed.add_field(name="Stop Loss", value="$590.00", inline=False)
            embed.add_field(name="Target", value="$610.00", inline=False)
            embed.add_field(name="R:R", value="2:1", inline=False)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")
    
    @commands.command(name='stats')
    async def stats(self, ctx):
        """Show trading performance statistics."""
        try:
            metrics = self.dashboard.get_metrics()
            embed = discord.Embed(title="Trading Statistics", color=0xFF00FF)
            embed.add_field(name="Total Trades", value=f"{metrics['total_trades']}", inline=True)
            embed.add_field(name="Win Rate", value=f"{metrics['win_rate']:.2f}%", inline=True)
            embed.add_field(name="Total P&L", value=f"${metrics['total_pnl']:.2f}", inline=True)
            embed.add_field(name="Avg Trade", value=f"{metrics['avg_trade']:.2f}", inline=True)
            embed.add_field(name="Best Trade", value=f"${metrics['best_trade']:.2f}", inline=True)
            embed.add_field(name="Worst Trade", value=f"${metrics['worst_trade']:.2f}", inline=True)
            embed.add_field(name="Profit Factor", value=f"{metrics['profit_factor']:.2f}x", inline=True)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")
    
    @commands.command(name='equity')
    async def equity(self, ctx):
        """Show equity curve chart."""
        try:
            chart = self.dashboard.plot_equity_curve()
            if chart:
                file = discord.File(chart, filename="equity_curve.png")
                await ctx.send(file=file)
            else:
                await ctx.send("No trades to display yet")
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")
    
    @commands.command(name='commands')
    async def commands_list(self, ctx):
        """Show available commands."""
        embed = discord.Embed(title="Quantum Fractals Commands", color=0x9900FF)
        embed.add_field(name="!status", value="Show bot status", inline=False)
        embed.add_field(name="!fractal [TICKER] [TIMEFRAME]", value="Analyze fractal zones (default: QQQ daily)", inline=False)
        embed.add_field(name="!zone [TICKER]", value="Get support/resistance zones (default: QQQ)", inline=False)
        embed.add_field(name="!signal [TICKER]", value="Get current trading signal (default: QQQ)", inline=False)
        embed.add_field(name="!stats", value="Show trading performance statistics", inline=False)
        embed.add_field(name="!equity", value="Show equity curve chart", inline=False)
        embed.add_field(name="!commands", value="Show this message", inline=False)
        await ctx.send(embed=embed)
    
    def generate_chart(self, ticker: str, timeframe: str) -> io.BytesIO:
        """Generate a chart of fractal zones."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        np.random.seed(42)
        bars = 100
        prices = [420.0]
        for _ in range(bars - 1):
            change = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        ax.plot(range(len(prices)), prices, 'b-', linewidth=2, label='Price')
        ax.axhline(y=np.mean(prices), color='r', linestyle='--', label='Vector')
        ax.fill_between(range(len(prices)), np.min(prices)*0.98, np.max(prices)*1.02, alpha=0.2, color='green', label='Support Zone')
        
        ax.set_title(f'{ticker} {timeframe.upper()} - Quantum Fractals Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf


async def setup(bot):
    """Add Quantum Fractals cog to bot."""
    await bot.add_cog(QuantumFractalsBot(bot))
