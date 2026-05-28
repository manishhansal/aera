"""Portfolio state: bankroll, open positions, realised + unrealised P&L.

This is the *single source of truth* for the bot's money. Every strategy reads
the live bankroll before sizing a bet, so growth compounds automatically —
that's the mechanical reason $1 can ever become $1M.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aera.logging import get_logger


log = get_logger(__name__)


@dataclass
class Fill:
    """A single execution event.

    ``leverage`` is the venue leverage at fill time. ``1.0`` (the default)
    means "1 USD of notional = 1 USD of cash committed" — correct for spot
    and any non-leveraged venue. ``50.0`` means a leveraged perpetual at
    50× — opening the leg commits ``notional / leverage`` of cash as
    *margin*, not the full notional. The Portfolio uses this to decide
    whether a fill drains cash or just reallocates it from free to locked.

    Without this, paper-trading a leveraged perp blows the bankroll deeply
    negative on the first fill (a $25k notional open on a $1k bankroll
    looks like spending $25k of cash). The accounting fix lives in
    ``Portfolio.apply_fill``.
    """
    timestamp: float
    market_id: str
    outcome_id: str
    side: str           # "BUY" or "SELL"
    price: float        # USD per share / contract
    size: float         # number of shares / contracts
    fee: float = 0.0
    leverage: float = 1.0

    @property
    def notional(self) -> float:
        return self.price * self.size

    @property
    def margin_required(self) -> float:
        """USD of cash needed to open ``size`` at ``price``."""
        return self.notional / max(self.leverage, 1.0)


@dataclass
class Position:
    """Aggregated holding in one (market, outcome)."""
    market_id: str
    outcome_id: str
    shares: float = 0.0          # signed; positive = long YES, negative = short
    avg_cost: float = 0.0        # average entry price per share
    realised_pnl: float = 0.0
    fills: List[Fill] = field(default_factory=list)

    def apply_fill(self, fill: Fill) -> None:
        signed_size = fill.size if fill.side == "BUY" else -fill.size
        new_shares = self.shares + signed_size

        # If we are adding to the same direction (or opening from flat),
        # recompute weighted-average cost.
        if self.shares == 0 or (self.shares > 0) == (signed_size > 0):
            total_cost = self.avg_cost * abs(self.shares) + fill.price * abs(signed_size)
            self.avg_cost = total_cost / abs(new_shares) if new_shares != 0 else 0.0
        else:
            # Closing or reducing: realise PnL on the closed portion.
            closing_size = min(abs(self.shares), abs(signed_size))
            direction = 1 if self.shares > 0 else -1
            pnl_per_share = (fill.price - self.avg_cost) * direction
            if fill.side == "SELL":  # closing a long
                pnl_per_share = fill.price - self.avg_cost
            else:                    # closing a short
                pnl_per_share = self.avg_cost - fill.price
            self.realised_pnl += pnl_per_share * closing_size - fill.fee

            if abs(signed_size) > abs(self.shares):
                # We flipped direction; remaining size opens a new position
                self.avg_cost = fill.price
            # else: partial close, avg_cost unchanged

        self.shares = new_shares
        self.fills.append(fill)

    def unrealised_pnl(self, mark_price: float) -> float:
        if self.shares == 0:
            return 0.0
        direction = 1 if self.shares > 0 else -1
        return (mark_price - self.avg_cost) * self.shares * direction / max(direction, 1)

    def notional_exposure(self, mark_price: float) -> float:
        return abs(self.shares) * mark_price


@dataclass
class Portfolio:
    """Bankroll + book of positions.

    Money is tracked in two buckets:

    * ``bankroll``        — *free cash* available for new margin commitments
                            and fees. This is what the risk manager checks.
    * ``locked_margin``   — cash currently posted as margin for open positions.
                            Reallocated, not spent. Returns to ``bankroll``
                            when the position closes.

    ``settled_wealth = bankroll + locked_margin`` is the realised wealth (i.e.
    excludes unrealised mark-to-market PnL). Drawdown is computed against
    ``settled_wealth`` so simply *posting margin* on a leveraged trade does
    not look like a loss — only realised P&L (and fees) move it.

    For non-leveraged fills (``Fill.leverage == 1.0``, the default), the
    accounting collapses to the classic cash-flow model:
    ``opening_size × price / 1 = notional`` leaves the bankroll on BUY and
    ``closing_size × prior_avg_cost + realised_pnl = closing_notional`` re-
    enters on SELL.
    """
    bankroll: float
    starting_bankroll: float = field(init=False)
    peak_bankroll: float = field(init=False)
    locked_margin: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    consecutive_losses: int = 0

    def __post_init__(self) -> None:
        self.starting_bankroll = self.bankroll
        # Track the peak of *settled wealth* (bankroll + locked margin) so
        # drawdown is invariant to the open/close cycle of leveraged trades.
        # On a fresh portfolio with no positions, this collapses to bankroll.
        self.peak_bankroll = self.bankroll

    @staticmethod
    def _key(market_id: str, outcome_id: str) -> str:
        return f"{market_id}:{outcome_id}"

    def position(self, market_id: str, outcome_id: str) -> Position:
        key = self._key(market_id, outcome_id)
        if key not in self.positions:
            self.positions[key] = Position(market_id=market_id, outcome_id=outcome_id)
        return self.positions[key]

    @property
    def settled_wealth(self) -> float:
        """Free cash + locked margin. Excludes unrealised PnL."""
        return self.bankroll + self.locked_margin

    def apply_fill(self, fill: Fill) -> None:
        """Apply a Fill to bankroll, locked margin, and the position book.

        We decompose every fill into a *closing* portion (the part that
        offsets the existing position, if same-side opposite) and an
        *opening* portion (everything else). For each:

        * opening_margin   = ``opening_size × fill.price / leverage`` — cash
                             moves from ``bankroll`` to ``locked_margin``.
        * closing_returned = ``closing_size × prior_avg_cost / leverage`` —
                             the margin originally posted for the closed
                             portion returns to ``bankroll``.
        * realised_pnl     — already computed inside ``Position.apply_fill``,
                             flows back to ``bankroll``. ``Position`` also
                             subtracts ``fee`` from ``realised_pnl`` on the
                             closing branch, so we must NOT charge it again
                             when a fill has any closing component.

        For ``leverage == 1.0`` the two formulas above sum to exactly
        ``±notional``, matching a plain cash-flow ledger.
        """
        pos = self.position(fill.market_id, fill.outcome_id)
        prior_shares = pos.shares
        prior_avg_cost = pos.avg_cost
        prior_realised = pos.realised_pnl

        leverage = max(getattr(fill, "leverage", 1.0) or 1.0, 1.0)
        signed_size = fill.size if fill.side == "BUY" else -fill.size

        if prior_shares == 0 or (prior_shares > 0) == (signed_size > 0):
            # Same direction or opening from flat: pure open.
            closing_size = 0.0
            opening_size = abs(signed_size)
        else:
            # Opposite direction: the overlap with the existing position
            # closes; anything beyond that opens an opposite leg (a flip).
            closing_size = min(abs(prior_shares), abs(signed_size))
            opening_size = abs(signed_size) - closing_size

        opening_margin = opening_size * fill.price / leverage
        closing_returned = closing_size * prior_avg_cost / leverage

        pos.apply_fill(fill)
        realised_delta = pos.realised_pnl - prior_realised

        # Bankroll math
        # ─────────────
        # Opening only:  bankroll -= margin posted + fee
        # Closing only:  bankroll += margin returned + realised PnL
        #                (fee already inside realised_delta from Position.apply_fill)
        # Flip (both):   close first (fee already inside realised_delta), open second
        if closing_size > 0:
            self.bankroll += closing_returned + realised_delta - opening_margin
            self.locked_margin += opening_margin - closing_returned
        else:
            self.bankroll += -opening_margin - fill.fee
            self.locked_margin += opening_margin

        # Tiny floating-point negative values can sneak in when a close
        # exactly empties locked margin; clamp to zero so summaries stay clean.
        if -1e-9 < self.locked_margin < 0:
            self.locked_margin = 0.0

        self.fills.append(fill)

        # update peak + loss streak
        if realised_delta < 0:
            self.consecutive_losses += 1
        elif realised_delta > 0:
            self.consecutive_losses = 0

        # Track peak of *settled* wealth (bankroll + locked margin). Marking
        # to fill.price would briefly inflate peak with unrealised PnL and
        # make every subsequent unrealised dip look like drawdown.
        self.peak_bankroll = max(self.peak_bankroll, self.settled_wealth)
        log.debug(
            "fill %s %s %.4f x %.2f lev=%g -> bankroll=%.4f locked=%.4f",
            fill.side, fill.market_id[:8], fill.price, fill.size, leverage,
            self.bankroll, self.locked_margin,
        )

    def equity(self, marks: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """Settled wealth + mark-to-market unrealised PnL on open positions.

        Settled wealth (= bankroll + locked margin) is what we'd have if we
        closed every position at its entry. Adding unrealised PnL on top
        gives the live mark-to-market equity that the dashboard shows.
        """
        marks = marks or {}
        unrl = 0.0
        for pos in self.positions.values():
            mark = marks.get(pos.market_id, {}).get(pos.outcome_id, pos.avg_cost)
            unrl += pos.notional_exposure(mark) * (1 if pos.shares > 0 else -1) - (
                pos.avg_cost * pos.shares
            )
        return self.settled_wealth + unrl

    def total_realised_pnl(self) -> float:
        return sum(p.realised_pnl for p in self.positions.values())

    def drawdown(self) -> float:
        """Realised drawdown from peak settled wealth.

        Uses ``settled_wealth`` (bankroll + locked margin), not ``bankroll``
        alone, so opening a leveraged position — which just moves cash from
        free to locked — is not treated as a drawdown event. Only realised
        P&L (and fees) move settled wealth, so the halt only trips on
        actual losses.
        """
        if self.peak_bankroll <= 0:
            return 0.0
        return 1.0 - (self.settled_wealth / self.peak_bankroll)

    def growth_multiple(self) -> float:
        if self.starting_bankroll <= 0:
            return 0.0
        return self.settled_wealth / self.starting_bankroll

    def summary(self) -> dict:
        return {
            "starting_bankroll": self.starting_bankroll,
            "bankroll": self.bankroll,
            "locked_margin": self.locked_margin,
            "settled_wealth": self.settled_wealth,
            "peak_bankroll": self.peak_bankroll,
            "growth_multiple": self.growth_multiple(),
            "drawdown": self.drawdown(),
            "open_positions": sum(1 for p in self.positions.values() if p.shares != 0),
            "total_fills": len(self.fills),
            "consecutive_losses": self.consecutive_losses,
            "realised_pnl": self.total_realised_pnl(),
        }
