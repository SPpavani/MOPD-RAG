"""Token cost in US dollars.

PRICES ARE A SNAPSHOT and change over time: verify at https://www.anthropic.com/pricing, or override with the
environment variables MOPD_PRICE_IN and MOPD_PRICE_OUT (USD per million tokens). A model that is not in the
table has no price: cost is reported as None rather than guessed.
"""
import os
from typing import Optional, Tuple

# (input, output) USD per million tokens, matched by model-name prefix
PRICES_USD_PER_MTOK = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
}


def price_for(model: str) -> Optional[Tuple[float, float]]:
    env_in, env_out = os.environ.get("MOPD_PRICE_IN"), os.environ.get("MOPD_PRICE_OUT")
    if env_in and env_out:
        return float(env_in), float(env_out)
    for prefix, price in PRICES_USD_PER_MTOK.items():
        if model.startswith(prefix):
            return price
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    price = price_for(model)
    if price is None:
        return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 6)
