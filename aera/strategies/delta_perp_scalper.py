"""Delta Exchange perpetual scalper.

A directional mean-reversion strategy designed for Delta's USD-quoted
perpetuals (BTCUSD, ETHUSD, etc.). Delta products are continuous contracts
that trade at a true mark price tied to spot, so a z-score reversion has
real mean-reversion gravity behind it (the funding-rate mechanism enforces
that the perp tracks the index). Short-horizon mean-reversion in liquid
perps is a long-known and well-documented edge.

How it fires
------------

For each Delta market (one ``LONG`` outcome representing the perp itself):

1. Update a rolling z-score of mid-price over the last ``window`` ticks.
2. Update an order-flow-imbalance (OFI) EMA over top-of-book sizes.
3. If ``z <= -zscore_entry``  AND ``ofi >= +ofi_threshold``  → emit a BUY.
   If ``z >= +zscore_entry``  AND ``ofi <= -ofi_threshold``  → emit a SELL.
   (Two-condition gating cuts false positives where price is just trending.)
4. ``edge`` is reported as a normalised signal strength
   ``min(|z| / zscore_entry, 1.0) * 0.01`` so it slots into the risk
   manager's edge-based sizing.

The strategy is intentionally conservative on size and depends on the
existing `RiskManager` to cap exposure per market / per trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market
from aera.signals.microstructure import OrderFlowImbalance, RollingZScore

from .base import Leg, Signal, Strategy

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution -> strategies)
    from aera.core import Portfolio


log = get_logger(__name__)


@dataclass
class _SymbolState:
    z: RollingZScore
    ofi: OrderFlowImbalance
    last_signal_mid: float = 0.0
    # Open-position tracking for take-profit / stop-loss exits.
    # ``position_side`` is "LONG", "SHORT", or None (flat). ``entry_mid`` and
    # ``entry_size_usd`` are recorded at the tick we emit the entry signal.
    # We track from emitted intent rather than executor fills because the
    # strategy doesn't see fills directly; if a fill is rejected, the next
    # opposite signal still closes cleanly (Portfolio.apply_fill is signed).
    position_side: str | None = None
    entry_mid: float = 0.0
    entry_size_usd: float = 0.0


class DeltaPerpetualScalper(Strategy):
    """Mean-reversion + OFI scalper for Delta Exchange perpetuals.

    Parameters
    ----------
    zscore_window : int
        Rolling window for the mid-price z-score (in scan ticks).
    zscore_entry : float
        Trigger threshold; ``|z|`` must exceed this to fire.
    ofi_threshold : float
        Required |OFI| in the *reversion* direction. Range: [0, 1].
    min_edge : float
        Minimum normalised edge to emit a signal (used for sorting).
    notional_usd : float
        Reference USD notional for a single leg. Will be scaled by the
        executor against bankroll + risk caps; this is just the "target"
        size that determines proportions when multiple symbols fire.
    min_depth_contracts : float
        Require at least this many contracts on top-of-book on the side
        we'd hit, otherwise skip — protects against thin venues.
    rearm_distance_bps : float
        Don't re-fire on the same symbol until mid has moved by this many
        basis points from the last firing mid. Cheap debouncer.
    take_profit_pct : float
        Close an open position when mid moves in our favour by this fraction
        of the entry mid (e.g. ``0.01`` = +1% from entry for a long). Set to
        ``0.0`` to disable — the position then only flattens when the
        opposite-direction reversion signal fires.
    stop_loss_pct : float
        Close an open position when mid moves against us by this fraction of
        the entry mid (e.g. ``0.005`` = −0.5% from entry for a long). Set to
        ``0.0`` to disable. Stop-loss takes precedence over take-profit when
        both could fire in the same tick (defensive: realise losses first).
    take_profit_usd : float
        Close an open position when its **unrealised P&L in USD** reaches this
        amount (e.g. ``5.0`` = close at +$5 of profit). Computed from the
        real ``Portfolio`` position (signed shares × (mark − avg_cost)) so it
        reflects the actual filled size after executor scaling, not the
        strategy's emitted intent. Requires ``portfolio`` to be passed at
        construction. ``0.0`` disables. When enabled, takes precedence over
        ``take_profit_pct`` for the same symbol.
    stop_loss_usd : float
        Close an open position when its unrealised P&L drops to ``-stop_loss_usd``
        (e.g. ``3.0`` = close at −$3 of loss). Same accounting as
        ``take_profit_usd``. Stop-loss still takes precedence over take-profit
        on a simultaneous breach.
    portfolio : Portfolio, optional
        Live portfolio reference used by the USD thresholds to read the
        actual open position. The strategy never mutates it. Without this,
        the USD thresholds are silently inactive (the percentage thresholds
        still work, since they read entry state stamped at signal emission).
    """

    name = "delta_perp_scalper"

    def __init__(
        self,
        *,
        zscore_window: int = 60,
        zscore_entry: float = 2.0,
        ofi_threshold: float = 0.2,
        min_edge: float = 0.002,
        notional_usd: float = 5.0,
        min_depth_contracts: float = 1.0,
        rearm_distance_bps: float = 5.0,
        take_profit_pct: float = 0.0,
        stop_loss_pct: float = 0.0,
        take_profit_usd: float = 0.0,
        stop_loss_usd: float = 0.0,
        portfolio: Optional["Portfolio"] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.zscore_window = zscore_window
        self.zscore_entry = zscore_entry
        self.ofi_threshold = ofi_threshold
        self.min_edge = min_edge
        self.notional_usd = notional_usd
        self.min_depth_contracts = min_depth_contracts
        self.rearm_distance_bps = rearm_distance_bps
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.take_profit_usd = max(0.0, float(take_profit_usd))
        self.stop_loss_usd = max(0.0, float(stop_loss_usd))
        self.portfolio = portfolio
        self._state: Dict[str, _SymbolState] = {}

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState(
                z=RollingZScore(window=self.zscore_window),
                ofi=OrderFlowImbalance(),
            )
            self._state[symbol] = st
        return st

    def scan(self, markets: Iterable[Market]) -> List[Signal]:
        signals: List[Signal] = []
        for m in markets:
            if m.venue != "delta":
                continue
            outcome = next(iter(m.outcomes.values()), None)
            if outcome is None or outcome.label != DELTA_OUTCOME_LABEL:
                continue

            bid = outcome.best_bid
            ask = outcome.best_ask
            if bid is None or ask is None or ask <= 0:
                continue
            mid = 0.5 * (bid + ask)
            if mid <= 0:
                continue

            st = self._state_for(m.id)

            # Reconcile internal "I'm LONG / SHORT" state with the live
            # portfolio. If a previous entry signal was vetoed (by the
            # brain, risk vet, or executor), we'd otherwise emit a
            # phantom close on a position that never opened.
            self.sync_position_state(st, self.portfolio, m.id, outcome.id)

            # Exit logic runs before entry: a tripped stop-loss or take-profit
            # always wins over opening a fresh position on the same tick.
            # Skips the depth gate intentionally — when we need to flatten,
            # we do so even on a thin tick (the executor will still respect
            # min_trade_notional / overshoot caps).
            close = self._maybe_emit_close(st, m, outcome, bid, ask, mid)
            if close is not None:
                signals.append(close)
                continue

            best_bid_lvl = outcome.book.best_bid()
            best_ask_lvl = outcome.book.best_ask()
            if best_bid_lvl is None or best_ask_lvl is None:
                continue
            if (best_bid_lvl.size < self.min_depth_contracts
                    or best_ask_lvl.size < self.min_depth_contracts):
                continue

            z = st.z.update(mid)
            ofi = st.ofi.update(best_bid_lvl.size, best_ask_lvl.size)
            if z is None:
                continue

            # debounce — must move at least N bps since the last firing mid
            if st.last_signal_mid > 0:
                move_bps = abs(mid - st.last_signal_mid) / st.last_signal_mid * 1e4
                if move_bps < self.rearm_distance_bps:
                    continue

            side: str | None = None
            if z <= -self.zscore_entry and ofi >= self.ofi_threshold:
                side = "BUY"
            elif z >= self.zscore_entry and ofi <= -self.ofi_threshold:
                side = "SELL"
            if side is None:
                continue

            # Don't double up on an open position in the same direction;
            # otherwise we'd average our entry mid and confuse TP/SL bands.
            # An *opposite*-direction signal still flows through — that's
            # the existing reversion-exit / flip behaviour and is desirable.
            if (side == "BUY" and st.position_side == "LONG") or \
               (side == "SELL" and st.position_side == "SHORT"):
                continue

            strength = min(abs(z) / self.zscore_entry, 2.0) / 2.0  # 0..1
            edge = max(self.min_edge, strength * 0.01)
            limit_price = ask if side == "BUY" else bid
            # Propagate the market's leverage so the executor sizes against
            # leveraged buying power rather than raw cash bankroll. Falls back
            # to 1.0 if Delta metadata is missing (e.g. hand-built fixtures).
            try:
                leverage = float(m.metadata.get("leverage", 1.0) or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0
            leg = Leg(
                market_id=m.id,
                outcome_id=outcome.id,
                side=side,
                limit_price=float(limit_price),
                size_usd=self.notional_usd,
                reason=f"z={z:.2f} ofi={ofi:.2f}",
                leverage=leverage,
            )
            signals.append(
                Signal(
                    strategy=self.name,
                    confidence=strength,
                    edge=edge,
                    legs=[leg],
                    metadata={
                        "symbol": m.id,
                        "z": float(z),
                        "ofi": float(ofi),
                        "mid": float(mid),
                    },
                )
            )
            st.last_signal_mid = mid
            # Record (or refresh) the position so TP/SL can fire later.
            # If the previous position was on the opposite side, the executor
            # nets it out via signed Position.shares — so overwriting entry
            # state to the new direction matches the realised book state.
            st.position_side = "LONG" if side == "BUY" else "SHORT"
            st.entry_mid = mid
            st.entry_size_usd = self.notional_usd
            log.debug(
                "delta-scalper FIRE %s side=%s z=%.2f ofi=%.2f mid=%.2f",
                m.id, side, z, ofi, mid,
            )
        return signals

    def _maybe_emit_close(
        self,
        st: _SymbolState,
        market: Market,
        outcome,
        bid: float,
        ask: float,
        mid: float,
    ) -> Optional[Signal]:
        """Return a flattening Signal if TP or SL has been hit, else None.

        Two exit modes:

        1. **USD P&L thresholds** (``take_profit_usd`` / ``stop_loss_usd``):
           when a ``portfolio`` is attached and either threshold is > 0, the
           strategy queries the live ``Position`` for actual signed shares
           and ``avg_cost``, then computes unrealised P&L in USD as
           ``(close_price − avg_cost) × shares`` where ``close_price`` is the
           bid (selling into for a long) or ask (buying from for a short).
           Using the close-side price (not mid) means the threshold reflects
           what would actually be realised on close, not an over-optimistic
           mid mark. Stop-loss precedence still applies.

        2. **Percentage thresholds** (``take_profit_pct`` / ``stop_loss_pct``,
           legacy): fall back when no portfolio is attached or USD thresholds
           are disabled. Compares mid against the entry mid stamped at signal
           emission.

        The close limit price crosses the spread (sell into the bid, buy
        from the ask) so a paper-fill or live-IOC order is very likely to
        execute on the same tick the threshold is breached.
        """
        if st.position_side is None or st.entry_mid <= 0:
            return None
        if (self.take_profit_pct <= 0 and self.stop_loss_pct <= 0
                and self.take_profit_usd <= 0 and self.stop_loss_usd <= 0):
            return None

        side = st.position_side
        entry = st.entry_mid
        close_side = "SELL" if side == "LONG" else "BUY"
        limit = bid if side == "LONG" else ask

        hit_sl = False
        hit_tp = False
        pnl_usd: Optional[float] = None

        # USD-based exit (preferred when configured + portfolio attached).
        # We query the *real* portfolio position so the threshold tracks
        # actual filled size, which can be smaller than the emitted intent
        # after the executor's trade-size scaling.
        usd_active = (
            (self.take_profit_usd > 0 or self.stop_loss_usd > 0)
            and self.portfolio is not None
        )
        if usd_active:
            from aera.core import Portfolio  # local import; avoids cycle at module load
            key = Portfolio._key(market.id, outcome.id)
            pos = self.portfolio.positions.get(key) if self.portfolio else None
            if pos is None or pos.shares == 0:
                # No live position to close: flatten our internal book so we
                # stop trying, and let the next entry signal restamp state.
                st.position_side = None
                st.entry_mid = 0.0
                st.entry_size_usd = 0.0
                return None
            mark = limit  # bid for long, ask for short — the price we'd actually close at
            pnl_usd = (mark - pos.avg_cost) * pos.shares
            if self.stop_loss_usd > 0 and pnl_usd <= -self.stop_loss_usd:
                hit_sl = True
            if self.take_profit_usd > 0 and pnl_usd >= self.take_profit_usd:
                hit_tp = True
        else:
            # Percentage-based exit (legacy path)
            tp = self.take_profit_pct
            sl = self.stop_loss_pct
            if side == "LONG":
                if sl > 0 and mid <= entry * (1.0 - sl):
                    hit_sl = True
                if tp > 0 and mid >= entry * (1.0 + tp):
                    hit_tp = True
            else:  # SHORT
                if sl > 0 and mid >= entry * (1.0 + sl):
                    hit_sl = True
                if tp > 0 and mid <= entry * (1.0 - tp):
                    hit_tp = True

        if not (hit_sl or hit_tp):
            return None

        kind = "stop-loss" if hit_sl else "take-profit"
        move_bps = (mid - entry) / entry * 1e4
        try:
            leverage = float(market.metadata.get("leverage", 1.0) or 1.0)
        except (TypeError, ValueError):
            leverage = 1.0
        pnl_label = f" pnl=${pnl_usd:+.2f}" if pnl_usd is not None else ""
        leg = Leg(
            market_id=market.id,
            outcome_id=outcome.id,
            side=close_side,
            limit_price=float(limit),
            size_usd=st.entry_size_usd,
            reason=f"{kind}: entry={entry:.4f} mid={mid:.4f} ({move_bps:+.1f}bps){pnl_label}",
            leverage=leverage,
            reduce_only=True,
        )
        meta = {
            "symbol": market.id,
            "exit": kind,
            "position_side": side,
            "entry_mid": float(entry),
            "mid": float(mid),
        }
        if pnl_usd is not None:
            meta["pnl_usd"] = float(pnl_usd)
        sig = Signal(
            strategy=self.name,
            confidence=1.0,
            edge=max(self.min_edge, 0.01),
            legs=[leg],
            metadata=meta,
        )
        # Flatten our internal book; rearm baseline stays put so the standard
        # debounce still applies if a fresh entry triggers right after.
        st.position_side = None
        st.entry_mid = 0.0
        st.entry_size_usd = 0.0
        log.info(
            "delta-scalper EXIT %s %s side=%s entry=%.4f mid=%.4f (%.1f bps)%s",
            market.id, kind.upper(), side, entry, mid, move_bps, pnl_label,
        )
        return sig
