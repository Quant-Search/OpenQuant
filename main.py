#!/usr/bin/env python3
"""
OpenQuant MVP Robot - Entry Point

Simple, clean entry point that delegates to the modular robot package.
"""
import argparse
from robot import Robot, Config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OpenQuant MVP Trading Robot")
    parser.add_argument(
        "--mode", 
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode: paper (simulated), live (real MT5), backtest (historical)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (e.g., EURUSD,GBPUSD)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe: 1h, 4h, 1d"
    )
    
    args = parser.parse_args()
    
    # Override config from args
    if args.symbols:
        Config.SYMBOLS = [s.strip() for s in args.symbols.split(",")]
    if args.timeframe:
        Config.TIMEFRAME = args.timeframe
    
    # Create and run robot
    robot = Robot(mode=args.mode)
    robot.run()


if __name__ == "__main__":
    main()


