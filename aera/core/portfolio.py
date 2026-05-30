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
    # When a partial-close fill leaves a position with notional below this
    # USD floor, the residual is force-flattened: ``(mark - avg_cost) ×
    # shares`` is booked to realised PnL and shares are zeroed. Without
    # this, sub-min residuals (e.g. 0.0001 BTC after a 0.003 BTC partial)
    # are unclosable on the venue — min contract size is enforced — so they
    # sit on the book forever locking margin and surfacing as ghost
    # positions in the dashboard. Set to 0 to disable.
    dust_threshold_usd: float = 1.0

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

        Strategy:

        1. Compute ``realised_delta`` from the Position (handles closing
           PnL + fee accounting on the close branch). This is the only
           value that changes *settled_wealth* (bankroll + locked_margin),
           and on opens we additionally deduct the fill fee directly.
        2. Update settled_wealth by ``realised_delta - opening_fee``.
        3. Recompute ``locked_margin`` deterministically from the
           post-fill open positions (each position's margin is its
           notional / leverage at last-fill on the prevailing side).
           This eliminates a class of accounting drift the previous
           incremental scheme exhibited when per-fill leverage differed
           between open and close (e.g. the dashboard reporting
           ``locked_margin = -$78`` after a long paper run — physically
           impossible, the result of a leverage mismatch silently
           shifting cash from locked to free over hundreds of fills).
        4. Derive ``bankroll = settled_wealth - locked_margin``.

        For ``leverage == 1.0`` the math collapses to the classic
        cash-flow ledger: opens deduct full notional + fee from bankroll,
        closes add (close_size × close_price) + realised_pnl back.
        """
        pos = self.position(fill.market_id, fill.outcome_id)
        prior_shares = pos.shares
        prior_realised = pos.realised_pnl

        signed_size = fill.size if fill.side == "BUY" else -fill.size

        # Determine whether this fill has any opening portion — used to
        # decide if the fill fee needs to be deducted separately. (Closing
        # fills route the fee through ``Position.apply_fill`` →
        # ``realised_pnl``, so we'd double-count if we deducted it again.)
        is_pure_open = prior_shares == 0 or (prior_shares > 0) == (signed_size > 0)

        pos.apply_fill(fill)
        realised_delta = pos.realised_pnl - prior_realised

        # Dust sweep: if the fill REDUCED the position and the remainder
        # has notional below ``dust_threshold_usd``, force-flatten it.
        # Books ``(fill.price - avg_cost) × shares`` as additional
        # realised PnL and zeros out shares. Without this the residual
        # sits on the book unclosable (sub-min contract size on the
        # venue) and silently locks margin / shows up as a ghost
        # position. The mark we use is ``fill.price`` — the most recent
        # traded price for this market, which is the best mark
        # available without piping the live book through to portfolio.
        if (
            self.dust_threshold_usd > 0
            and not is_pure_open
            and pos.shares != 0
            and abs(pos.shares) * pos.avg_cost < self.dust_threshold_usd
        ):
            dust_pnl = (fill.price - pos.avg_cost) * pos.shares
            pos.realised_pnl += dust_pnl
            pos.shares = 0.0
            pos.avg_cost = 0.0
            realised_delta += dust_pnl
            log.info(
                "portfolio: swept dust position %s — booked $%+.4f to "
                "realised PnL (notional was $%.4f, below dust threshold "
                "$%.2f)",
                fill.market_id,
                dust_pnl,
                abs(prior_shares + signed_size) * pos.avg_cost
                if pos.avg_cost > 0 else 0.0,
                self.dust_threshold_usd,
            )

        # Recompute settled_wealth as the source of truth. For pure opens
        # the fee is paid out of cash and Position never sees it; for
        # closes / flips the fee is already inside realised_delta.
        prior_settled = self.settled_wealth
        new_settled = prior_settled + realised_delta
        if is_pure_open:
            new_settled -= float(fill.fee or 0.0)

        # Locked margin = sum of margin posted across open positions.
        # Recomputed from scratch so per-fill leverage drift cannot
        # accumulate. (Incremental updates broke when greedy stamped
        # one leverage on entries and the closing path used a slightly
        # different leverage — the difference snuck into locked_margin
        # over hundreds of fills.)
        self.locked_margin = self._recompute_locked_margin()
        # Bankroll is the free-cash slot of the new settled wealth.
        self.bankroll = new_settled - self.locked_margin

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
        leverage = max(getattr(fill, "leverage", 1.0) or 1.0, 1.0)
        log.debug(
            "fill %s %s %.4f x %.2f lev=%g -> bankroll=%.4f locked=%.4f",
            fill.side, fill.market_id[:8], fill.price, fill.size, leverage,
            self.bankroll, self.locked_margin,
        )

    def sweep_dust(self, market_mids: Dict[str, float]) -> List[str]:
        """Flatten every open position whose notional is below
        ``dust_threshold_usd``, booking the mark-to-market as realised
        PnL and zeroing shares.

        Useful for cleaning up positions that became dust *before* the
        per-fill sweep ran (e.g. on bot restart with stale residuals on
        the book). Per-fill cleanup in ``apply_fill`` handles the
        normal case; this method is the bulk-cleanup escape hatch.

        Returns the list of swept market ids. ``locked_margin`` and
        ``bankroll`` are re-derived from the post-sweep state so the
        caller never sees an inconsistent snapshot.
        """
        if self.dust_threshold_usd <= 0 or not self.positions:
            return []
        swept: List[str] = []
        cumulative_pnl_delta = 0.0
        for pos in self.positions.values():
            if pos.shares == 0 or pos.avg_cost == 0:
                continue
            notional = abs(pos.shares) * pos.avg_cost
            if notional >= self.dust_threshold_usd:
                continue
            mark = market_mids.get(pos.market_id)
            if mark is None or mark <= 0:
                mark = pos.avg_cost
            dust_pnl = (mark - pos.avg_cost) * pos.shares
            pos.realised_pnl += dust_pnl
            pos.shares = 0.0
            pos.avg_cost = 0.0
            cumulative_pnl_delta += dust_pnl
            swept.append(pos.market_id)
            log.info(
                "portfolio: swept pre-existing dust %s — booked $%+.4f "
                "(notional was $%.4f)",
                pos.market_id, dust_pnl, notional,
            )
        if swept:
            # Realised PnL changed; settled_wealth shifts by the same
            # total. Locked margin gets recomputed from the (now
            # smaller) set of open positions; bankroll absorbs the
            # rest so bankroll + locked = previous_settled + delta.
            prior_settled = self.bankroll + self.locked_margin
            new_settled = prior_settled + cumulative_pnl_delta
            self.locked_margin = self._recompute_locked_margin()
            self.bankroll = new_settled - self.locked_margin
        return swept

    def _recompute_locked_margin(self) -> float:
        """Sum of margin posted across all open positions.

        For each open position the margin is
        ``|shares| × avg_cost / leverage`` where ``leverage`` is the
        most-recent fill's leverage on the side that opened the
        prevailing position direction. With constant leverage across
        all opens (the typical case) this collapses to the original
        margin posted; with scale-ins at varying leverages it tracks
        the best-effort margin given the avg_cost reduction the
        Position has performed.
        """
        total = 0.0
        for pos in self.positions.values():
            if pos.shares == 0 or pos.avg_cost == 0:
                continue
            # Find the most recent fill on the side that is the current
            # net direction — that's the leverage we should bill margin at.
            target_side = "BUY" if pos.shares > 0 else "SELL"
            lev = 1.0
            for f in reversed(pos.fills):
                if f.side == target_side:
                    lev = max(1.0, float(getattr(f, "leverage", 1.0) or 1.0))
                    break
            total += abs(pos.shares) * pos.avg_cost / lev
        return total

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
