# Order Book Depth Integration

Complete order book integration for OpenQuant, providing real-time market depth analysis, liquidity-aware sizing, and optimal execution strategies.

## Features

### 1. **Order Book Fetching**
- Real-time order book data via CCXT
- Support for any exchange with order book API
- Built-in caching (configurable duration)
- Automatic retry and fallback mechanisms

### 2. **Market Metrics**
- **Spread Analysis**: Bid-ask spread in absolute and basis points
- **Liquidity Depth**: Total available liquidity at multiple price levels
- **Order Book Imbalance**: Buy/sell pressure indicator (-1 to +1)
- **Mid Price**: Real-time fair value estimate

### 3. **Market Impact Modeling**
- **Slippage Estimation**: Walk through order book to calculate expected fill price
- **Impact Calculation**: Square-root model (Almgren-Chriss) for permanent impact
- **Level Analysis**: Determine how many price levels an order will consume

### 4. **Execution Optimization**
- **Optimal Pricing**: Calculate ideal limit price based on urgency
- **Strategy Selection**: Auto-select market/limit/TWAP based on conditions
- **Price Improvement**: Balance execution probability vs. better price

### 5. **Liquidity-Aware Sizing**
- **Participation Rate**: Limit order size relative to visible liquidity
- **Impact Constraints**: Cap position size to keep impact below threshold
- **Depth Requirements**: Ensure minimum price levels available
- **Dynamic Adjustment**: Automatically reduce size if liquidity insufficient

## Architecture

### Core Classes

```
OrderBookSnapshot
├── OrderBookLevel (price/size pairs)
├── Metrics (spread, imbalance, liquidity)
└── Analytics (depth_at_price, to_dataframe)

OrderBookFetcher
├── Exchange connection via CCXT
├── Caching mechanism
└── Error handling & fallback

MarketImpactModel
├── estimate_slippage()
├── estimate_market_impact()
└── optimal_execution_price()

LiquidityAwareSizer
├── calculate_max_size()
└── adjust_position_size()

OrderBookAnalyzer (High-level API)
├── analyze_execution()
└── get_execution_strategy()
```

## Usage Examples

### Basic Order Book Fetching

```python
from openquant.data.orderbook import OrderBookFetcher

fetcher = OrderBookFetcher("binance")
orderbook = fetcher.fetch_order_book("BTC/USDT", limit=20)

print(f"Mid Price: ${orderbook.mid_price:,.2f}")
print(f"Spread: {orderbook.spread_bps:.2f} bps")
print(f"Best Bid: ${orderbook.best_bid.price:,.2f}")
print(f"Best Ask: ${orderbook.best_ask.price:,.2f}")
print(f"Imbalance: {orderbook.order_book_imbalance():.3f}")
```

### Slippage and Impact Estimation

```python
from openquant.data.orderbook import OrderBookFetcher, MarketImpactModel

fetcher = OrderBookFetcher("binance")
impact_model = MarketImpactModel(impact_coefficient=0.1)

orderbook = fetcher.fetch_order_book("ETH/USDT")

# Estimate slippage for buying 5 ETH
slippage = impact_model.estimate_slippage(orderbook, 5.0, "buy")
print(f"Expected price: ${slippage['avg_price']:,.2f}")
print(f"Slippage: {slippage['slippage_bps']:.2f} bps")
print(f"Levels consumed: {slippage['levels_consumed']}")

# Estimate market impact
impact = impact_model.estimate_market_impact(orderbook, 5.0, "buy")
print(f"Market impact: {impact:.2f} bps")
```

### Liquidity-Aware Position Sizing

```python
from openquant.data.orderbook import OrderBookFetcher, LiquidityAwareSizer

fetcher = OrderBookFetcher("binance")
sizer = LiquidityAwareSizer(
    max_participation_rate=0.05,  # Max 5% of visible liquidity
    max_impact_bps=10.0,           # Max 10 bps impact
    min_depth_levels=3             # Need at least 3 price levels
)

orderbook = fetcher.fetch_order_book("BTC/USDT")

# Check if desired size is feasible
desired_size = 1.0
sizing = sizer.calculate_max_size(orderbook, "buy", desired_size)

if sizing["feasible"]:
    print(f"Order is feasible at {desired_size} BTC")
else:
    print(f"Order too large: {sizing['reason']}")
    print(f"Recommended: {sizing['recommended_quantity']:.4f} BTC")
```

### Complete Execution Analysis

```python
from openquant.data.orderbook import OrderBookAnalyzer

analyzer = OrderBookAnalyzer("binance")

analysis = analyzer.analyze_execution(
    symbol="BTC/USDT",
    quantity=0.5,
    side="buy",
    urgency=0.5  # 0.0=patient, 1.0=aggressive
)

print(f"Mid Price: ${analysis['mid_price']:,.2f}")
print(f"Optimal Price: ${analysis['optimal_price']:,.2f}")
print(f"Expected Slippage: {analysis['slippage_estimate']['slippage_bps']:.2f} bps")
print(f"Market Impact: {analysis['market_impact_bps']:.2f} bps")
print(f"Recommended Qty: {analysis['recommended_quantity']:.4f}")

# Get execution strategy recommendation
strategy = analyzer.get_execution_strategy("BTC/USDT", 0.5, "buy")
print(f"Strategy: {strategy['strategy']}")  # market, limit, or twap
print(f"Reason: {strategy['reason']}")
```

### Integration with SmartExecutor

```python
from openquant.trading.smart_executor import SmartExecutor, ExecutionConfig

# Configure executor with order book integration
config = ExecutionConfig(
    use_orderbook=True,
    orderbook_exchange="binance",
    max_impact_bps=10.0
)

executor = SmartExecutor(broker=my_broker, config=config)

# Execute with automatic optimization
result = executor.execute(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.5,
    current_price=50000.0,
    urgency=0.5
)

# The executor automatically:
# - Fetches order book
# - Adjusts quantity based on liquidity
# - Calculates optimal limit price
# - Selects execution strategy (market/limit/TWAP)
# - Minimizes slippage and impact
```

## Configuration Options

### OrderBookFetcher
- `exchange`: Exchange name (any CCXT-supported exchange)
- `cache_seconds`: Cache duration (default: 1.0s)

### MarketImpactModel
- `impact_coefficient`: Impact sensitivity (default: 0.1, higher=more conservative)

### LiquidityAwareSizer
- `max_participation_rate`: Max % of visible liquidity (default: 0.05 = 5%)
- `max_impact_bps`: Max acceptable impact in bps (default: 10.0)
- `min_depth_levels`: Min required price levels (default: 3)

### ExecutionConfig (SmartExecutor)
- `use_orderbook`: Enable order book integration (default: False)
- `orderbook_exchange`: Exchange for order book data
- `limit_offset_bps`: Limit price offset (default: 2.0)
- `max_retries`: Retry attempts for partial fills (default: 3)

## Market Impact Model

Uses the **Almgren-Chriss square-root model**:

```
Impact (bps) = k × √(Q / L) × 10,000
```

Where:
- `k` = impact coefficient (configurable)
- `Q` = order quantity
- `L` = available liquidity (approximated from order book depth)

This model captures:
1. **Temporary impact**: Walking through the order book
2. **Permanent impact**: Market's response to information in the order

## Optimal Execution Pricing

Calculates limit price based on **urgency parameter** (0.0 to 1.0):

- **urgency = 0.0 (Patient)**: Offer below best price, maximize price improvement
- **urgency = 0.5 (Balanced)**: Near best price, balanced execution probability
- **urgency = 1.0 (Aggressive)**: Pay through multiple levels, ensure fill

Algorithm considers:
- Best bid/ask
- Expected slippage from walking the book
- Mid price as reference
- Urgency-weighted interpolation

## Execution Strategy Selection

Automatic strategy recommendation based on:

1. **Market Order**
   - Low impact (< 10 bps)
   - Tight spread (< 5 bps)
   - Small order relative to liquidity

2. **Limit Order**
   - Moderate impact (10-20 bps)
   - Wide spread (> 5 bps)
   - Acceptable order size

3. **TWAP (Time-Weighted Average Price)**
   - High impact (> 20 bps)
   - Large order relative to liquidity
   - Size > 1% of visible depth

## Performance Considerations

### Caching
- Default 1-second cache reduces API calls
- Critical for high-frequency analysis
- Configurable per use case

### Rate Limiting
- Integrated with OpenQuant's rate limiter
- Respects exchange limits (10 req/sec default)
- Automatic backoff on errors

### Error Handling
- Graceful degradation on fetch failures
- Falls back to cached data when available
- Returns safe defaults on total failure

## Integration Points

### With Existing Modules

1. **SmartExecutor** (`openquant/trading/smart_executor.py`)
   - Automatic order book analysis before execution
   - Liquidity-adjusted position sizing
   - Optimal limit price calculation

2. **AdaptiveSizer** (`openquant/risk/adaptive_sizing.py`)
   - Can incorporate order book liquidity metrics
   - Adjust Kelly/volatility sizing based on depth

3. **TCA Monitor** (`openquant/analysis/tca.py`)
   - Compare actual vs. expected slippage
   - Track execution quality vs. order book predictions

4. **Brokers** (`openquant/broker/*.py`)
   - Pass order book data to execution layer
   - Validate orders against liquidity constraints

## Exchange Support

Works with **any CCXT-supported exchange** that provides order book data:

- ✅ Binance
- ✅ Coinbase Pro
- ✅ Kraken
- ✅ Bitfinex
- ✅ Bitstamp
- ✅ OKX
- ✅ Bybit
- And 100+ more...

## Limitations & Considerations

1. **Latency**: Order book data may be stale by execution time
2. **Hidden Liquidity**: Doesn't account for iceberg orders or dark pools
3. **Slippage Model**: Assumes instantaneous execution (no time component)
4. **Market Impact**: Square-root model is approximate, varies by asset/exchange
5. **Exchange Differences**: Order book depth varies significantly across exchanges

## Best Practices

1. **Use Short Cache Duration** for high-volatility assets (0.5-1.0s)
2. **Validate Feasibility** before submitting large orders
3. **Monitor Actual vs. Expected** slippage for model calibration
4. **Adjust Impact Coefficient** based on historical execution data
5. **Consider Multiple Exchanges** for better liquidity analysis
6. **Use TWAP** for orders > 5% of visible liquidity
7. **Set Conservative Limits** on participation rate (3-5%)

## Testing

Run the example script to test all functionality:

```bash
python scripts/example_orderbook_usage.py
```

This demonstrates:
- Order book fetching
- Market impact estimation
- Liquidity-aware sizing
- Optimal execution pricing
- Complete execution analysis
- SmartExecutor integration

## Future Enhancements

Potential improvements:
- [ ] Multi-exchange aggregation for better liquidity view
- [ ] Time-series analysis of order book dynamics
- [ ] Machine learning for impact prediction
- [ ] Order book imbalance signals for alpha generation
- [ ] Websocket streaming for real-time updates
- [ ] Historical order book replay for backtesting

## References

- Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions"
- Hasbrouck, J. (2007). "Empirical Market Microstructure"
- Gould, M. et al. (2013). "Limit order books"

## Support

For issues or questions:
1. Check example script: `scripts/example_orderbook_usage.py`
2. Review logs in `logs/openquant.log`
3. Verify CCXT installation: `pip install ccxt`
4. Test exchange connectivity independently
