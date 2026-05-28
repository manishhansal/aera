/* ─────────────────────────────────────────────────────────────────────
   aera · live dashboard
   websocket-driven render, Chart.js charts, glassy bento UI
   ───────────────────────────────────────────────────────────────────── */
(() => {
    "use strict";

    // ─────────────────────────────────────────────────────────────────
    // helpers
    // ─────────────────────────────────────────────────────────────────
    const $ = (id) => document.getElementById(id);

    const fmtUsd = (v, dp = 4) => {
        if (v === null || v === undefined || Number.isNaN(v)) return "$—";
        const abs = Math.abs(v);
        const sign = v < 0 ? "-" : "";
        if (abs >= 1e9) return sign + "$" + (abs / 1e9).toFixed(2) + "B";
        if (abs >= 1e6) return sign + "$" + (abs / 1e6).toFixed(2) + "M";
        if (abs >= 1e3) return sign + "$" + (abs / 1e3).toFixed(2) + "k";
        return sign + "$" + abs.toFixed(dp);
    };
    const fmtSignedUsd = (v, dp = 2) => {
        if (v === null || v === undefined || Number.isNaN(v)) return "$—";
        const sign = v > 0 ? "+" : v < 0 ? "−" : "";
        return sign + fmtUsd(Math.abs(v), dp);
    };
    const fmtPct = (v, dp = 2) => {
        if (v === null || v === undefined || Number.isNaN(v)) return "—%";
        return (v * 100).toFixed(dp) + "%";
    };
    const fmtNum = (v, dp = 2) => {
        if (v === null || v === undefined || Number.isNaN(v)) return "—";
        return Number(v).toFixed(dp);
    };
    const fmtTime = (ts) => {
        if (!ts) return "—";
        const d = new Date(ts * 1000);
        return d.toLocaleTimeString(undefined, { hour12: false });
    };
    const fmtAgo = (ts) => {
        if (!ts) return "—";
        const d = (Date.now() / 1000) - ts;
        if (d < 60)   return Math.max(0, d).toFixed(0) + "s ago";
        if (d < 3600) return (d / 60).toFixed(0) + "m ago";
        return (d / 3600).toFixed(1) + "h ago";
    };
    const fmtUptime = (s) => {
        if (s == null) return "0s";
        s = Math.max(0, Math.floor(s));
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = s % 60;
        if (h) return `${h}h ${m}m`;
        if (m) return `${m}m ${sec}s`;
        return `${sec}s`;
    };
    const fmtDuration = (s) => {
        if (s == null) return "—";
        s = Math.max(0, Math.floor(s));
        if (s < 60)   return s + "s";
        if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        return h + "h " + m + "m";
    };
    const shortId = (id, n = 8) => (id ? String(id).slice(0, n) : "—");

    // ─────────────────────────────────────────────────────────────────
    // chart theme tokens
    // ─────────────────────────────────────────────────────────────────
    const C = {
        grid:    "rgba(255, 255, 255, 0.04)",
        tick:    "#5a6677",
        text:    "#e7eef5",
        accent:  "#00e5a8",
        accent2: "#00a8ff",
        accent3: "#7c5cff",
        good:    "#00e5a8",
        bad:     "#ff5e7c",
        warn:    "#fbbf24",
        tooltipBg: "#11181f",
        tooltipBorder: "#243140",
    };

    function lineChartOpts({ ySuffix = "$", yPct = false, yMin } = {}) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { intersect: false, mode: "index" },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: C.tooltipBg,
                    borderColor: C.tooltipBorder,
                    borderWidth: 1,
                    titleColor: C.text,
                    bodyColor: C.text,
                    titleFont: { family: "ui-monospace" },
                    bodyFont:  { family: "ui-monospace" },
                    callbacks: {
                        label: (ctx) => {
                            const v = ctx.parsed.y;
                            if (yPct) return ctx.dataset.label + ": " + (v * 100).toFixed(2) + "%";
                            return ctx.dataset.label + ": " + ySuffix + Number(v).toFixed(4);
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid:  { color: C.grid, drawTicks: false },
                    border: { display: false },
                    ticks: { color: C.tick, maxTicksLimit: 6, font: { size: 10 } },
                },
                y: {
                    grid:  { color: C.grid, drawTicks: false },
                    border: { display: false },
                    ticks: {
                        color: C.tick,
                        font: { size: 10 },
                        callback: (v) =>
                            yPct ? (v * 100).toFixed(1) + "%" : ySuffix + Number(v).toFixed(4),
                    },
                    min: yMin,
                },
            },
        };
    }

    function barChartOpts() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: C.tooltipBg,
                    borderColor: C.tooltipBorder,
                    borderWidth: 1,
                    titleColor: C.text,
                    bodyColor: C.text,
                },
            },
            scales: {
                x: {
                    grid:  { color: C.grid, drawTicks: false },
                    border: { display: false },
                    ticks: { color: C.tick, font: { size: 10 } },
                },
                y: {
                    grid:  { color: C.grid, drawTicks: false },
                    border: { display: false },
                    ticks: {
                        color: C.tick,
                        font: { size: 10 },
                        callback: (v) => fmtUsd(v, 2),
                    },
                },
            },
        };
    }

    function donutOpts(labelFmt = (v) => v) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            cutout: "62%",
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: C.tooltipBg,
                    borderColor: C.tooltipBorder,
                    borderWidth: 1,
                    titleColor: C.text,
                    bodyColor: C.text,
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${labelFmt(ctx.parsed)}`,
                    },
                },
            },
        };
    }

    function gradientFill(ctx, color, alpha0 = 0.18, alpha1 = 0.0) {
        const g = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
        g.addColorStop(0, hexA(color, alpha0));
        g.addColorStop(1, hexA(color, alpha1));
        return g;
    }
    function hexA(hex, a) {
        const h = hex.replace("#", "");
        const r = parseInt(h.slice(0, 2), 16);
        const g = parseInt(h.slice(2, 4), 16);
        const b = parseInt(h.slice(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${a})`;
    }

    // ─────────────────────────────────────────────────────────────────
    // chart instances
    // ─────────────────────────────────────────────────────────────────
    const heroSparkCtx = $("heroSpark").getContext("2d");
    const heroSpark = new Chart(heroSparkCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: C.accent,
                backgroundColor: gradientFill(heroSparkCtx, C.accent, 0.35),
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: { display: false },
            },
            elements: { line: { borderJoinStyle: "round" } },
        },
    });

    const equityCtx = $("equityChart").getContext("2d");
    const equityChart = new Chart(equityCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "equity",
                    data: [],
                    borderColor: C.accent2,
                    backgroundColor: gradientFill(equityCtx, C.accent2, 0.18),
                    borderWidth: 2,
                    tension: 0.25,
                    fill: true,
                    pointRadius: 0,
                },
                {
                    label: "settled wealth",
                    data: [],
                    borderColor: C.accent3,
                    backgroundColor: "transparent",
                    borderWidth: 1.5,
                    tension: 0.25,
                    fill: false,
                    pointRadius: 0,
                },
                {
                    label: "bankroll",
                    data: [],
                    borderColor: C.accent,
                    backgroundColor: "transparent",
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    tension: 0.25,
                    fill: false,
                    pointRadius: 0,
                },
            ],
        },
        options: lineChartOpts({ ySuffix: "$" }),
    });

    const ddCtx = $("ddChart").getContext("2d");
    const ddChart = new Chart(ddCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "drawdown",
                data: [],
                borderColor: C.bad,
                backgroundColor: gradientFill(ddCtx, C.bad, 0.22),
                borderWidth: 2,
                tension: 0.25,
                fill: true,
                pointRadius: 0,
            }],
        },
        options: lineChartOpts({ yPct: true, yMin: 0 }),
    });

    const cumPnlCtx = $("cumPnlChart").getContext("2d");
    const cumPnlChart = new Chart(cumPnlCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "cumulative p&l",
                data: [],
                borderColor: C.accent,
                backgroundColor: gradientFill(cumPnlCtx, C.accent, 0.18),
                borderWidth: 2,
                tension: 0.15,
                fill: true,
                pointRadius: 0,
            }],
        },
        options: lineChartOpts({ ySuffix: "$" }),
    });

    const winLossCtx = $("winLossChart").getContext("2d");
    const winLossChart = new Chart(winLossCtx, {
        type: "doughnut",
        data: {
            labels: ["wins", "losses"],
            datasets: [{
                data: [0, 0],
                backgroundColor: [C.good, C.bad],
                borderColor: "#0b1118",
                borderWidth: 3,
                hoverOffset: 6,
            }],
        },
        options: donutOpts((v) => v + " trades"),
    });

    const pnlDistCtx = $("pnlDistChart").getContext("2d");
    const pnlDistChart = new Chart(pnlDistCtx, {
        type: "bar",
        data: {
            labels: [],
            datasets: [{
                label: "trades",
                data: [],
                backgroundColor: [],
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: C.tooltipBg,
                    borderColor: C.tooltipBorder,
                    borderWidth: 1,
                    titleColor: C.text,
                    bodyColor: C.text,
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.y} trade(s)`,
                    },
                },
            },
            scales: {
                x: {
                    grid:   { display: false },
                    border: { display: false },
                    ticks:  { color: C.tick, font: { size: 9 }, maxTicksLimit: 7 },
                },
                y: {
                    grid:   { color: C.grid, drawTicks: false },
                    border: { display: false },
                    ticks:  { color: C.tick, font: { size: 10 }, precision: 0, stepSize: 1 },
                },
            },
        },
    });

    const strategyCtx = $("strategyChart").getContext("2d");
    const strategyChart = new Chart(strategyCtx, {
        type: "bar",
        data: { labels: [], datasets: [{ label: "p&l", data: [], backgroundColor: [], borderWidth: 0 }] },
        options: barChartOpts(),
    });

    const exposureCtx = $("exposureChart").getContext("2d");
    const exposureChart = new Chart(exposureCtx, {
        type: "doughnut",
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [],
                borderColor: "#0b1118",
                borderWidth: 3,
                hoverOffset: 6,
            }],
        },
        options: donutOpts((v) => fmtUsd(v, 2)),
    });

    // palette for the strategy/exposure donut
    const PALETTE = [
        C.accent, C.accent2, C.accent3, "#fbbf24", "#ff5e7c",
        "#22d3ee", "#a78bfa", "#34d399", "#f97316", "#60a5fa",
    ];

    // ─────────────────────────────────────────────────────────────────
    // rendering — KPIs / hero / strip
    // ─────────────────────────────────────────────────────────────────
    function renderHero(snap) {
        const p = snap.portfolio || {};
        const an = snap.analytics || {};
        const eng = snap.engine || {};

        const start = p.starting_bankroll ?? 0;
        const br = p.bankroll ?? 0;
        const eq = p.equity ?? br;
        const settled = p.settled_wealth ?? br;
        const locked = p.locked_margin ?? 0;
        const peak = p.peak_bankroll ?? start;
        const gm = p.growth_multiple ?? 1;
        const dd = p.drawdown ?? 0;
        const realised = p.realised_pnl ?? 0;

        const heroEq = $("hero-equity");
        heroEq.textContent = fmtUsd(eq, 4);
        heroEq.classList.toggle("up",   eq > start);
        heroEq.classList.toggle("down", eq < start);

        const delta = eq - start;
        const pct = start ? (eq / start - 1) : 0;
        const dEl = $("hero-delta");
        dEl.textContent = `${fmtSignedUsd(delta, 4)}  (${(pct * 100).toFixed(2)}%)`;
        dEl.classList.toggle("up",   delta > 0);
        dEl.classList.toggle("down", delta < 0);

        $("hero-start").textContent = "start " + fmtUsd(start, 4);
        $("hero-peak").textContent  = "peak "  + fmtUsd(peak, 4);

        $("strip-bankroll").textContent = fmtUsd(br, 4);
        $("strip-margin").textContent   = fmtUsd(locked, 4);
        $("strip-settled").textContent  = fmtUsd(settled, 4);
        const bp = an.buying_power ?? 0;
        $("strip-bp").textContent = fmtUsd(bp, 0);
        $("strip-bp-sub").textContent = `@ ${(an.approx_leverage || 1).toFixed(0)}x lev`;

        const grEl = $("strip-growth");
        grEl.textContent = (gm).toFixed(4) + "x";
        grEl.classList.toggle("up",   gm > 1);
        grEl.classList.toggle("down", gm < 1);

        const ddEl = $("strip-dd");
        ddEl.textContent = fmtPct(dd);
        ddEl.classList.toggle("down", dd > 0.05);
        $("strip-dd-sub").textContent = "streak " + (p.consecutive_losses ?? 0);
        $("dd-current").textContent = fmtPct(dd);
    }

    function renderKpiRibbon(snap) {
        const p = snap.portfolio || {};
        const an = snap.analytics || {};
        const eng = snap.engine || {};

        // Use the closed-trade ledger (after fees) so the KPI agrees
        // with the per-strategy and per-trade tables. portfolio.realised_pnl
        // is a different bucket (price-only realised on the position
        // object) and diverges from the fee-adjusted trade ledger by
        // exactly 2x round-trip fees per closed round trip.
        const realised = (typeof snap.closed_trade_pnl === "number")
            ? snap.closed_trade_pnl
            : (p.realised_pnl ?? 0);
        const rEl = $("kpi-realised");
        rEl.textContent = fmtSignedUsd(realised, 2);
        rEl.classList.toggle("up",   realised > 0);
        rEl.classList.toggle("down", realised < 0);

        const wr = an.win_rate ?? 0;
        $("kpi-winrate").textContent = fmtPct(wr);
        $("kpi-winrate-sub").textContent =
            `${snap.closed_trade_wins || 0} W / ${snap.closed_trade_losses || 0} L`;

        const pf = an.profit_factor;
        $("kpi-pf").textContent = (pf == null) ? "∞" : pf.toFixed(2);
        $("kpi-pf-sub").textContent =
            `+${fmtUsd(an.gross_profit ?? 0, 2)} / −${fmtUsd(an.gross_loss ?? 0, 2)}`;

        const exp = an.expectancy ?? 0;
        const eEl = $("kpi-exp");
        eEl.textContent = fmtSignedUsd(exp, 4);
        eEl.classList.toggle("up",   exp > 0);
        eEl.classList.toggle("down", exp < 0);

        $("kpi-awl").textContent =
            `${fmtSignedUsd(an.avg_win ?? 0, 2)} / ${fmtSignedUsd(an.avg_loss ?? 0, 2)}`;

        $("kpi-hold").textContent = fmtDuration(an.avg_hold_seconds ?? 0);

        $("kpi-trades").textContent     = (snap.trades_closed ?? 0).toString();
        $("kpi-trades-rej").textContent = (eng.trades_rejected ?? 0).toString();
        $("kpi-signals").textContent    = (eng.signals_emitted ?? 0).toString();
        $("kpi-iters").textContent      = (eng.iterations ?? 0).toString();
        $("kpi-positions").textContent  = (p.open_positions ?? 0).toString();
        $("kpi-markets").textContent    = (snap.markets_count ?? 0).toString();

        $("uptime").textContent = fmtUptime(snap.uptime_seconds ?? 0);
        $("footer-info").textContent =
            `loop ${eng.loop_interval_ms ?? 0}ms · ${eng.use_websocket ? "ws" : "rest"} feed`;
        $("ws-badge").textContent = eng.use_websocket ? "WS" : "REST";

        // mode pill
        const modeEl = $("mode-pill");
        if (snap.paper_mode === false) {
            modeEl.textContent = "LIVE";
            modeEl.classList.add("live");
        } else {
            modeEl.textContent = "PAPER";
            modeEl.classList.remove("live");
        }

        // status pill
        const pill = $("status-pill");
        pill.classList.remove("ok", "paused", "error");
        if (eng.running) {
            if (eng.paused) {
                pill.classList.add("paused");
                $("status-label").textContent = "paused";
            } else {
                pill.classList.add("ok");
                $("status-label").textContent = "running";
            }
        } else {
            $("status-label").textContent = "idle";
        }
        $("btn-pause").classList.toggle("hidden", !!eng.paused);
        $("btn-resume").classList.toggle("hidden", !eng.paused);

        // exposure badges
        $("exp-gross").textContent = "gross " + fmtUsd(an.gross_exposure ?? 0, 2);
        const netEl = $("exp-net");
        const ne = an.net_exposure ?? 0;
        netEl.textContent = "net " + fmtSignedUsd(ne, 2);
        netEl.classList.toggle("num-pos", ne > 0);
        netEl.classList.toggle("num-neg", ne < 0);
    }

    // ─────────────────────────────────────────────────────────────────
    // rendering — charts
    // ─────────────────────────────────────────────────────────────────
    function renderEquity(points) {
        if (!points || !points.length) return;
        const labels = points.map((p) => fmtTime(p.timestamp));
        equityChart.data.labels = labels;
        equityChart.data.datasets[0].data = points.map((p) => p.equity);
        equityChart.data.datasets[1].data = points.map((p) => p.settled_wealth ?? p.bankroll);
        equityChart.data.datasets[2].data = points.map((p) => p.bankroll);
        equityChart.update("none");

        ddChart.data.labels = labels;
        ddChart.data.datasets[0].data = points.map((p) => p.drawdown);
        ddChart.update("none");

        // sparkline (last ~80 points)
        const tail = points.slice(-80);
        heroSpark.data.labels = tail.map((_, i) => i);
        heroSpark.data.datasets[0].data = tail.map((p) => p.equity);
        heroSpark.update("none");
    }

    function renderCumPnl(trades) {
        // trades arrive newest-first; reverse to chronological for cumulative
        const chrono = [...trades].reverse();
        let cum = 0;
        const labels = [];
        const data = [];
        for (const t of chrono) {
            cum += t.pnl || 0;
            labels.push(fmtTime(t.closed_at));
            data.push(cum);
        }
        cumPnlChart.data.labels = labels;
        cumPnlChart.data.datasets[0].data = data;
        // recolour gradient based on final sign
        const finalSign = data.length ? data[data.length - 1] >= 0 : true;
        const colour = finalSign ? C.accent : C.bad;
        cumPnlChart.data.datasets[0].borderColor = colour;
        cumPnlChart.data.datasets[0].backgroundColor = gradientFill(cumPnlCtx, colour, 0.18);
        cumPnlChart.update("none");

        const cumNet = data.length ? data[data.length - 1] : 0;
        const cumEl = $("cum-net");
        cumEl.textContent = "net " + fmtSignedUsd(cumNet, 2);
        cumEl.classList.toggle("num-pos", cumNet >= 0);
        cumEl.classList.toggle("num-neg", cumNet < 0);
    }

    function renderWinLoss(snap) {
        const w = snap.closed_trade_wins || 0;
        const l = snap.closed_trade_losses || 0;
        winLossChart.data.datasets[0].data = (w + l === 0) ? [1, 0] : [w, l];
        winLossChart.data.datasets[0].backgroundColor =
            (w + l === 0) ? ["rgba(255,255,255,0.05)", "rgba(255,255,255,0.0)"] : [C.good, C.bad];
        winLossChart.update("none");
        $("wl-wins").textContent = w;
        $("wl-losses").textContent = l;
    }

    function renderPnlDistribution(dist) {
        if (!dist || !dist.bins || !dist.bins.length) {
            pnlDistChart.data.labels = [];
            pnlDistChart.data.datasets[0].data = [];
            pnlDistChart.update("none");
            return;
        }
        const bins = dist.bins;
        const counts = dist.counts;
        const labels = [];
        const colours = [];
        for (let i = 0; i < counts.length; i++) {
            const lo = bins[i];
            const hi = bins[i + 1];
            const mid = (lo + hi) / 2;
            labels.push(fmtUsd(mid, 2));
            colours.push(mid >= 0 ? C.good : C.bad);
        }
        pnlDistChart.data.labels = labels;
        pnlDistChart.data.datasets[0].data = counts;
        pnlDistChart.data.datasets[0].backgroundColor = colours;
        pnlDistChart.update("none");
    }

    function renderStrategyBars(strategy_pnl) {
        const rows = (strategy_pnl || []).slice().sort((a, b) => b.pnl - a.pnl);
        const labels = rows.map((r) => r.name);
        const data   = rows.map((r) => r.pnl);
        const cols   = rows.map((r) => (r.pnl >= 0 ? C.good : C.bad));
        strategyChart.data.labels = labels;
        strategyChart.data.datasets[0].data = data;
        strategyChart.data.datasets[0].backgroundColor = cols;
        strategyChart.update("none");
    }

    function renderExposure(exposure) {
        const rows = (exposure || []).filter((e) => e.notional > 0);
        if (!rows.length) {
            exposureChart.data.labels = ["flat"];
            exposureChart.data.datasets[0].data = [1];
            exposureChart.data.datasets[0].backgroundColor = ["rgba(255,255,255,0.05)"];
            exposureChart.update("none");
            return;
        }
        exposureChart.data.labels = rows.map((r) => `${r.side} ${shortId(r.market_id, 6)}`);
        exposureChart.data.datasets[0].data = rows.map((r) => r.notional);
        exposureChart.data.datasets[0].backgroundColor =
            rows.map((_, i) => PALETTE[i % PALETTE.length]);
        exposureChart.update("none");
    }

    // ─────────────────────────────────────────────────────────────────
    // rendering — tables
    // ─────────────────────────────────────────────────────────────────
    let prevFillKey = null;
    let prevSignalKey = null;
    let prevTradeKey = null;

    const fillKey = (f) =>
        `${f.timestamp}|${f.market_id}|${f.outcome_id}|${f.side}|${f.price}|${f.size}`;
    const signalKey = (s) =>
        `${s.timestamp}|${s.strategy}|${s.edge}|${s.legs}|${s.notional}|${s.status}`;
    const tradeKey = (t) =>
        `${t.closed_at}|${t.market_id}|${t.outcome_id}|${t.side}|${t.open_price}|${t.close_price}|${t.size}`;

    function renderFills(fills) {
        const body = $("fills-body");
        $("fills-count").textContent = fills.length;
        if (!fills.length) {
            body.innerHTML = '<tr class="empty"><td colspan="7">no fills yet — bot is scanning…</td></tr>';
            $("last-fill-ago").textContent = "—";
            return;
        }
        $("last-fill-ago").textContent = fmtAgo(fills[0].timestamp);

        const newestKey = fillKey(fills[0]);
        const isNew = newestKey !== prevFillKey;
        prevFillKey = newestKey;

        body.innerHTML = fills.map((f, idx) => {
            const sideCls = f.side === "BUY" ? "buy" : "sell";
            const rowCls = isNew && idx === 0 ? "flash-in" : "";
            return `<tr class="${rowCls}">
                <td>${fmtTime(f.timestamp)}</td>
                <td>${f.strategy}</td>
                <td><span class="tag ${sideCls}">${f.side}</span></td>
                <td>$${Number(f.price).toFixed(4)}</td>
                <td>${Number(f.size).toFixed(2)}</td>
                <td>$${Number(f.notional).toFixed(4)}</td>
                <td>${fmtUsd(f.bankroll_after, 4)}</td>
            </tr>`;
        }).join("");
    }

    function renderTrades(trades, snap) {
        const body = $("trades-body");
        $("trades-count").textContent = trades.length;

        const netPnl = (snap && typeof snap.closed_trade_pnl === "number")
            ? snap.closed_trade_pnl
            : trades.reduce((s, t) => s + (t.pnl || 0), 0);
        const wins = (snap && snap.closed_trade_wins) || 0;
        const losses = (snap && snap.closed_trade_losses) || 0;
        const badge = $("trades-pnl");
        if (badge) {
            badge.textContent = `net ${fmtSignedUsd(netPnl, 2)} · ${wins}W / ${losses}L`;
            badge.classList.toggle("num-pos", netPnl >= 0);
            badge.classList.toggle("num-neg", netPnl < 0);
        }

        if (!trades.length) {
            body.innerHTML = '<tr class="empty"><td colspan="9">no completed trades yet — open a position to start the round trip…</td></tr>';
            return;
        }

        const newestKey = tradeKey(trades[0]);
        const isNew = newestKey !== prevTradeKey;
        prevTradeKey = newestKey;

        body.innerHTML = trades.map((t, idx) => {
            const sideCls = t.side === "LONG" ? "long" : "short";
            const pnlCls  = t.pnl >= 0 ? "num-pos" : "num-neg";
            const rowCls  = isNew && idx === 0 ? "flash-in" : "";
            return `<tr class="${rowCls}">
                <td>${fmtTime(t.closed_at)}</td>
                <td>${t.strategy}</td>
                <td title="${t.market_id}">${shortId(t.market_id)}</td>
                <td><span class="tag ${sideCls}">${t.side}</span></td>
                <td>$${Number(t.open_price).toFixed(4)}</td>
                <td>$${Number(t.close_price).toFixed(4)}</td>
                <td>${Number(t.size).toFixed(4)}</td>
                <td class="${pnlCls}">${fmtSignedUsd(t.pnl, 2)}</td>
                <td>${fmtDuration(t.duration_seconds)}</td>
            </tr>`;
        }).join("");
    }

    function renderSignals(signals) {
        const body = $("signals-body");
        $("signals-count").textContent = signals.length;
        if (!signals.length) {
            body.innerHTML = '<tr class="empty"><td colspan="7">no signals yet — waiting on market data…</td></tr>';
            return;
        }
        const newestKey = signalKey(signals[0]);
        const isNew = newestKey !== prevSignalKey;
        prevSignalKey = newestKey;

        body.innerHTML = signals.map((s, idx) => {
            const rowCls = isNew && idx === 0 ? "flash-in" : "";
            return `<tr class="${rowCls}">
                <td>${fmtTime(s.timestamp)}</td>
                <td>${s.strategy}</td>
                <td>${(s.edge * 100).toFixed(3)}%</td>
                <td>${s.legs}</td>
                <td>$${Number(s.notional).toFixed(4)}</td>
                <td><span class="tag ${s.status}">${s.status}</span></td>
                <td title="${s.reason || ''}">${s.reason || "—"}</td>
            </tr>`;
        }).join("");
    }

    function renderPositions(positions) {
        const body = $("positions-body");
        $("positions-count").textContent = positions.length;
        if (!positions.length) {
            body.innerHTML = '<tr class="empty"><td colspan="6">flat — no open positions</td></tr>';
            return;
        }
        body.innerHTML = positions.map((p) => {
            const side = p.shares >= 0 ? "LONG" : "SHORT";
            const sideCls = side === "LONG" ? "long" : "short";
            return `<tr>
                <td title="${p.market_id}">${shortId(p.market_id)}</td>
                <td><span class="tag ${sideCls}">${side}</span></td>
                <td class="${p.shares >= 0 ? "num-pos" : "num-neg"}">${Number(p.shares).toFixed(2)}</td>
                <td>$${Number(p.avg_cost).toFixed(4)}</td>
                <td class="${p.realised_pnl >= 0 ? "num-pos" : "num-neg"}">${fmtSignedUsd(p.realised_pnl, 4)}</td>
                <td>${fmtUsd(p.notional, 2)}</td>
            </tr>`;
        }).join("");
    }

    function renderStrategies(strats, strategy_pnl) {
        const body = $("strategies-body");
        if (!strats || !strats.length) {
            body.innerHTML = '<tr class="empty"><td colspan="7">no strategies bound</td></tr>';
            return;
        }
        const pnlByName = {};
        for (const r of (strategy_pnl || [])) pnlByName[r.name] = r.pnl;
        body.innerHTML = strats.map((s) => {
            const pnl = pnlByName[s.name] ?? 0;
            return `<tr>
                <td>${s.name}</td>
                <td>${s.signals_emitted}</td>
                <td class="num-pos">${s.trades_executed}</td>
                <td class="num-neg">${s.trades_rejected}</td>
                <td>${(s.avg_edge * 100).toFixed(3)}%</td>
                <td class="${pnl >= 0 ? "num-pos" : "num-neg"}">${fmtSignedUsd(pnl, 2)}</td>
                <td>${s.last_signal_ts ? fmtAgo(s.last_signal_ts) : "—"}</td>
            </tr>`;
        }).join("");
    }

    function renderMarkets(markets) {
        const body = $("markets-body");
        $("markets-count").textContent = markets.length;
        if (!markets.length) {
            body.innerHTML = '<tr class="empty"><td colspan="3">discovering markets…</td></tr>';
            return;
        }
        body.innerHTML = markets.map((m) => {
            const oc = m.outcomes.map((o) => {
                const bid = o.best_bid != null ? `$${o.best_bid.toFixed(3)}` : "—";
                const ask = o.best_ask != null ? `$${o.best_ask.toFixed(3)}` : "—";
                return `<span class="muted">${o.label}</span> ${bid} / ${ask}`;
            }).join(" &nbsp;·&nbsp; ");
            return `<tr>
                <td title="${m.question}">${m.question}</td>
                <td>${m.category || "—"}</td>
                <td>${oc}</td>
            </tr>`;
        }).join("");
    }

    // ─────────────────────────────────────────────────────────────────
    // websocket connection
    // ─────────────────────────────────────────────────────────────────
    let ws = null;
    let reconnectTimer = null;
    let backoff = 1000;

    function connect() {
        const proto = location.protocol === "https:" ? "wss" : "ws";
        const url = `${proto}://${location.host}/ws`;
        try {
            ws = new WebSocket(url);
        } catch (e) {
            return scheduleReconnect();
        }

        ws.addEventListener("open", () => {
            backoff = 1000;
        });

        ws.addEventListener("message", (evt) => {
            let payload;
            try { payload = JSON.parse(evt.data); } catch { return; }
            if (payload.type !== "tick") return;
            const snap = payload.state || {};
            renderHero(snap);
            renderKpiRibbon(snap);
            renderEquity(payload.equity || []);
            renderCumPnl(payload.trades || []);
            renderWinLoss(snap);
            renderPnlDistribution(snap.pnl_distribution || {});
            renderStrategyBars(snap.strategy_pnl || []);
            renderExposure(snap.exposure || []);
            renderFills(payload.fills || []);
            renderTrades(payload.trades || [], snap);
            renderSignals(payload.signals || []);
            renderPositions(payload.positions || []);
            renderStrategies(snap.strategies || [], snap.strategy_pnl || []);
        });

        ws.addEventListener("close", scheduleReconnect);
        ws.addEventListener("error", () => ws && ws.close());
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        const pill = $("status-pill");
        pill.classList.remove("ok", "paused");
        pill.classList.add("error");
        $("status-label").textContent = "reconnecting…";
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            backoff = Math.min(backoff * 1.5, 10000);
            connect();
        }, backoff);
    }

    async function pollMarkets() {
        try {
            const r = await fetch("/api/markets");
            const j = await r.json();
            renderMarkets(j.markets || []);
        } catch { /* swallow */ }
    }

    // ─────────────────────────────────────────────────────────────────
    // controls
    // ─────────────────────────────────────────────────────────────────
    async function postJson(path) {
        try {
            const r = await fetch(path, { method: "POST" });
            return await r.json();
        } catch { return null; }
    }
    $("btn-pause").addEventListener("click",  () => postJson("/api/control/pause"));
    $("btn-resume").addEventListener("click", () => postJson("/api/control/resume"));

    // ─────────────────────────────────────────────────────────────────
    // boot
    // ─────────────────────────────────────────────────────────────────
    connect();
    pollMarkets();
    setInterval(pollMarkets, 5000);
})();
