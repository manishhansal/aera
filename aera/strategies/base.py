"""Strategy contract: each strategy ingests market state and emits Signals."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional

from aera.markets import Market

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution -> strategies)
    from aera.core import Portfolio


@dataclass
class Leg:
    """One side of a trade.

    ``leverage`` is the multiplier the executor and risk manager should apply
    when translating ``size_usd`` (a *notional* USD figure) into the actual
    margin required. ``1.0`` means "1 USD of notional costs 1 USD of margin"
    (spot, cash equities). On leveraged perps (Delta, futures) the strategy
    sets this to whatever leverage Delta will actually apply to the order so
    the sizer treats ``bankroll`` as *buying power = bankroll × leverage*.
    The Leg still carries notional in ``size_usd``; ``size_usd / leverage``
    is the implied margin commitment.
    """
    market_id: str
    outcome_id: str
    side: str               # "BUY" or "SELL"
    limit_price: float
    size_usd: float
    reason: str = ""
    leverage: float = 1.0
    # If True, this leg only ever *reduces* an existing position and must
    # never open a new opposite-direction position. Set by strategies on
    # their take-profit / stop-loss / unwind paths so the executor:
    #   * skips the (additive) per-market exposure cap in RiskManager.vet,
    #   * clamps size_usd to the existing position notional so a "close"
    #     can never accidentally flip into a same-size opposite position,
    #   * on live Delta, submits the order with reduce_only=true so the
    #     venue rejects it instead of opening a new leg if our size math is
    #     ever off.
    reduce_only: bool = False
    # ---- per-leg execution-mode overrides ----------------------------
    # When set, override the live exchange's defaults for this single
    # leg only. Used by maker-only strategies (e.g. bid_ask_spread_fade)
    # to post resting limit orders even though the rest of the bot
    # operates with IOC taker semantics. Both default to ``None`` so
    # existing strategies are unaffected and continue to use whatever
    # the live exchange was constructed with.
    #
    # ``time_in_force``: typically ``"ioc"`` (default, immediate-or-
    # cancel = taker fill) or ``"gtc"`` (rest on book = maker eligible).
    #
    # ``post_only``: when True, Delta refuses the order if it would
    # cross the spread (i.e. take liquidity), so the leg is guaranteed
    # to either rest as a maker or be rejected outright. Without
    # ``time_in_force="gtc"`` this rarely makes sense — IOC + post_only
    # together is functionally a pure rejection rule.
    time_in_force: Optional[str] = None
    post_only: Optional[bool] = None


@dataclass
class Signal:
    """A trade idea produced by a strategy."""
    strategy: str
    confidence: float        # 0..1; arb is 1.0
    edge: float              # decimal (0.01 = 1%)
    legs: List[Leg] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def total_notional(self) -> float:
        return sum(l.size_usd for l in self.legs)


class Strategy(abc.ABC):
    name: str = "base"

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    @abc.abstractmethod
    def scan(self, markets: Iterable[Market]) -> List[Signal]:
        """Inspect markets and emit zero or more Signals."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # phantom-position guard (shared across all strategies)
    # ------------------------------------------------------------------
    #
    # Strategies stamp internal "I'm LONG / SHORT" state at signal-
    # emission time, BEFORE the brain / risk vet / executor see the
    # signal. When any of those layers veto the entry, the strategy
    # is left out of sync with the portfolio: it thinks it has an
    # open position, and on subsequent ticks fires TP/SL/timeout close
    # signals on a phantom position. The executor drops those closes
    # (``no open position to close``) and the dashboard fills with
    # noise rejections.
    #
    # ``sync_position_state`` is the universal escape hatch: every
    # strategy that maintains a ``_SymbolState`` with a
    # ``position_side`` attribute calls this at the top of each
    # per-symbol scan. If the live portfolio shows no shares for the
    # (market, outcome) key, the strategy's internal state is reset
    # to flat. The strategy can then re-emit a fresh entry on the
    # next tick instead of bleeding phantom closes forever.

    @staticmethod
    def sync_position_state(
        state,
        portfolio: Optional["Portfolio"],
        market_id: str,
        outcome_id: str,
    ) -> bool:
        """Reset ``state.position_side`` (and entry_* fields) when the
        live portfolio shows no open position for the (market,
        outcome) pair.

        Returns True when state was reset (= caller should treat the
        strategy as flat on this market for this tick). Returns False
        when no reset was needed (state already flat, or live position
        confirmed open).

        Defensive — if no portfolio is wired in we leave state alone
        so unit tests with hand-built fixtures continue to work.
        """
        if portfolio is None or state is None:
            return False
        if getattr(state, "position_side", None) is None:
            return False
        from aera.core.portfolio import Portfolio as _PF
        key = _PF._key(market_id, outcome_id)
        pos = portfolio.positions.get(key)
        if pos is not None and pos.shares != 0:
            return False
        # Phantom — reset every entry_* field the strategy may carry.
        state.position_side = None
        for attr in (
            "entry_mid", "entry_price", "entry_size_usd",
            "entry_shares", "entry_time", "entry_vwap",
            "entry_wick_low", "entry_wick_high", "best_pnl_usd",
            "tp1_done", "scaled_out_qty",
        ):
            if hasattr(state, attr):
                cur = getattr(state, attr)
                if isinstance(cur, bool):
                    setattr(state, attr, False)
                elif isinstance(cur, (int, float)):
                    setattr(state, attr, type(cur)(0))
        return True
