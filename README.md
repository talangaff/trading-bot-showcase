> **Showcase copy.** Credentials, hostnames and deployment details have been removed. It trades on Alpaca *paper* accounts only unless you explicitly configure otherwise; supply your own keys via environment variables or the Accounts page. Educational and research use only. Not investment advice.

# Earnings Volatility Screener

Screens stocks for the earnings-volatility / long-calendar setup (based on the
"Volatility Vibes" criteria): 30-day average volume ≥ 1.5M, IV30/RV30 ≥ 1.25,
and term-structure slope (0–45 DTE) ≤ −0.00406.

Two front-ends over the **same calculation engine** ([`engine.py`](engine.py)):

| | |
|---|---|
| `calculator.py` | Original desktop app (FreeSimpleGUI). `pip install -r requirements.txt` |
| `webapp/`       | FastAPI web app. `pip install -r requirements-web.txt` |

**The calculations in `engine.py` are copied verbatim from `calculator.py` and must not be changed.**

## Web app

Pages, one per section of the desktop app:

- **Bot Fleet** — your own fleet of evolving 0DTE/swing options bots, on your
  own Alpaca paper keys.
- **Incubator** (on the Bot Fleet page) — a second population of *simulated*
  bots (`pool='sim'`, `webapp/incubator.py`) that run the same strategy code
  as the fleet on the live market with modeled fills and **no orders**, so
  hundreds more genomes are tested at once. A sim bot that trades enough,
  stays in profit and inside the drawdown guardrail **graduates** into the
  fleet as a real-fill bot (first choice for any free slot; a clearly losing,
  flat, established fleet bot can be swapped for one — a bot that is doing
  well never is). The best performers anywhere (sim, promoted, fleet) are the
  **parents** of every new sim bot and of fleet refills, by mutation and
  crossover, so each generation starts from the last one's winners; the page
  shows fitness by generation. Runs from `trade-calculator-incubator.timer`
  (every 5 min) and after the daily pass. Tunables: `BOT_SIM_PER_FLEET`,
  `BOT_SIM_GRAD_TRADES/DAYS/FITNESS`, `BOT_SIM_TICK_BUDGET_S`.
- **Shared Pool** — the genome population every fleet is collectively
  searching, each genome's pooled record, and who is feeding the shared
  market history.
- **Single Symbol** — analyze one ticker.
- **Bulk Import** — paste a comma-separated list; screened in a background job.
- **Pull from Nasdaq** — screen the top *N* earnings by market cap for a date.
- **This Week** — 7 day tabs. Results are cached **on the server** so every
  client loads instantly. A systemd timer refreshes the cache **hourly**; the
  **Update All** button refreshes every day on demand.
- **Trade Journals** — hand-logged trades, paper and live. Each journal holds
  **one strategy** (`journals.strategy`): a calendar spread, a credit spread, a
  single option or stock. The trade form on a marked journal offers that type
  and nothing else, and the rule is enforced in `webapp/journal.py` — create,
  update and patch all retype whatever they are handed, so a `?type=` on the
  URL or a hand-rolled POST cannot smuggle another shape in. `strategy=any`
  keeps the full picker and is what every journal that pre-dates the rule was
  migrated to; a journal can be re-marked later only to a type its existing
  trades already match. A **credit spread** is scored on what it can lose —
  `(width − credit) × contracts × 100` — not on the credit it took in, so
  position sizing and P/L % read as return-on-risk; it is entered per leg and
  closed through the Exits section at the net debit paid (0 = expired
  worthless). Calls and puts both live in one credit-spread journal.
  **Quick add** takes one typed line — `AAL 10/16 $11/12 put credit spread` —
  and opens a draft with the symbol, expiration, type and both strikes filled
  in, leaving quantity and the leg prices to the form. Which leg is which
  follows from the strategy, not from the typing order: a put credit spread
  sells the higher strike and buys the lower, a call one the reverse. A bare
  `10/16` rolls forward to the next such date. Credit spreads only, and
  refused on a journal marked for a different strategy, since that journal
  would retype the trade on the way in. See `webapp/quick_add.py`.

### Accounts

Username + password sign-in. Every account gets the same thing — the screener,
its own journals, its own **bot fleet** and its own custom bots — and an
**admin** additionally gets **Users** (create accounts, reset passwords,
disable / delete, grant or revoke admin), the house guardrail defaults, the
installation-wide engine switch, and the ability to view any account's fleet
for support.

A fresh install has one admin, `admin` / `admin`, and forces a password change
on first login. New accounts are created on the Users page; each starts with a
temporary password (shown once) that it must replace at first sign-in.

Journals are private to their owner, admins included. So is every bot: a
fleet bot, its ledger and its P&L belong to one account exactly as a custom
bot does. Each account sets its own Alpaca paper keys, guardrails and engine
switch on its Bot Fleet page, and its custom bots use a second set of keys on
its Custom Bots page.

Upgrading from the single shared fleet: the primary admin inherits the fleet
that was already running, along with its Alpaca keys, data feed, execution
mode, guardrails and custom-bot capital pool, plus every existing journal and
custom bot. It keeps trading the same account and nothing needs re-entering.

```bash
./venv/bin/python -m webapp.auth list                    # accounts
./venv/bin/python -m webapp.auth reset-admin             # locked out: admin back to the default
./venv/bin/python -m webapp.auth create NAME [--admin]   # prints a temporary password
```

### Run locally

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements-web.txt
TRADE_INSECURE_COOKIE=1 ./venv/bin/uvicorn webapp.main:app --reload --port 5005
# open http://127.0.0.1:5005  (first-run default login for local development only; change it immediately)
```

Runtime state (accounts, session key, week cache, per-account Alpaca keys) is
written to `./data/` (override with `TRADE_DATA_DIR`). `data/` is git-ignored.

Build the "This Week" cache manually:

```bash
./venv/bin/python -m webapp.refresh_week          # all 7 days
./venv/bin/python -m webapp.refresh_week 2026-08-27
```

### Tests

```bash
./venv/bin/pip install pytest
./venv/bin/pytest tests/
```

Covers the 0DTE bot fleet's fill pricing (spread/commission), live-quote
blending and its fallback/sanity checks, evolution's eligibility gating, and
the genome mutation rules — the parts most at risk of a silent regression —
plus live execution (`tests/test_live_execution.py`: who trades on which
keys, real fills booked as reported, per-bot sell caps on a shared account,
the buying-power gate, reconcile, order-journal recovery, multi-leg order
bodies; Alpaca itself is stubbed) and the account store and route-level access (who can see the
fleet, whose journals and custom bots are whose).

## One search, many fleets

Every account runs **its own fleet** on **its own Alpaca paper account**. What
the accounts share is the *search*, never the money:

- **Own fleet, own keys.** A user's fleet and baseline bots trade on the
  Alpaca paper keys that user entered on their Bot Fleet page, under their own
  guardrails and their own engine switch. There is no shared or fallback key
  anywhere in the code (`webapp/broker.keys_for_bot`): a fleet with no keys of
  its own simply does not trade. That is the guarantee that one account's bots
  can never reach another's money, however much they share otherwise.
- **One gene pool** (`webapp/gene_pool.py`, the `/pool` page). Every genome
  that has ever been given a live trial is registered by a fingerprint of its
  behaviour genes, and every bot row carries that fingerprint. Generations and
  lineage belong to the population, not to whichever fleet ran a genome first.
- **Split search.** Breeding never hands one fleet a candidate another fleet
  is already running (`gene_pool.distinct_child`). *N* fleets therefore explore
  *N* times as many distinct genomes per market day — which is where the
  acceleration actually comes from, since this search is short of calendar
  days, not CPU. (*N* fleets running the *same* genome on the *same* minutes
  would only produce *N* correlated copies of one observation, so the breeder
  is deliberately built not to do that.) A genome registered but not currently
  running is left *pending*; the next bot any fleet spawns may adopt it and
  finish the trial someone else started.
- **Pooled evidence.** A genome's track record is the union of the closed
  trades of every user's bot that has run it (`gene_pool.pooled_stats`), using
  the same day-adjusted shrunk t-statistic as a single bot — only the sample is
  bigger. So a genome two fleets each gave thirty trades is *proven* (or
  culled) on sixty, in half the calendar days either would have needed. The
  multiple-comparison bar counts **distinct genomes**, not bot rows, so a new
  user joining widens the search without silently raising the bar on everyone.
  Culling reads the pooled record too: a bad genome only has to be discovered
  once, by anybody, and dies everywhere.
- **Pooled market data.** Bars, the VIX snapshot and option chains are fetched
  **once per cycle**, on the best Alpaca entitlement anyone has connected
  (`bot_settings.data_source`), and recorded into the one shared history
  (`webapp/recorder.py`) that every fleet's replay and evolution runs on. One
  user with a SIP subscription upgrades the data quality — and so the fill
  realism — of every other fleet. `/pool` shows who has contributed what.
- **The house prior is shared too.** `fleet_prior_return` and
  `baseline_day_returns` are computed across every fleet: what a no-edge genome
  earns on these symbols, and what simply being long returned on a given
  session, are facts about the market rather than about one account. Every new
  fleet makes them better estimates, and they are what makes trades from
  different accounts comparable at all.
- **Your numbers stay yours.** A fleet page shows your bots' own ledgers, fills
  and P&L. Only evolution's *decisions* read the pooled numbers.

Each user's own P&L still depends on their own account: different paper
balances, entitlements and fill quality mean two fleets running the same genome
will not book identical trades. Pooling those fills buys a bigger sample at the
cost of some execution noise; the sample-size gain is why it is worth it, and
it is the reason day-adjustment against each account's own baseline bots
matters.

## How the fleet trades

Every bot places **real orders on an Alpaca paper account** and books only
the fill Alpaca reports — there is no modeled price anywhere in a live
ledger, and no setting that can introduce one, so evolution breeds and
culls on live paper fills. A fleet with no keys does not trade; it never
trades modeled ones instead. To evaluate a genome without spending market
days, use the offline replay harness.

- **Fleet and baseline bots** trade on their **owner's** paper keys (that
  account's Bot Fleet settings panel), all of that user's fleet together on
  that one account. Each bot keeps its own ledger starting at `$10,000`
  (`BOT_START_BALANCE`); the account's P&L is the sum of its bots', each
  bot's P&L is its own. Baseline bots are per account too, so a fleet's
  null-hypothesis floor paid the same costs on the same account it did.
- **Custom bots** trade on their owner's *second* set of keys (Custom Bots
  page) — a different account again, so one genome can be tested by itself.
  They never fall back to the fleet keys; the page warns if you enter the same
  key ID your own fleet trades on.
  A clone keeps its source's style (0DTE / swing) and structure (single /
  vertical) and runs the exact same execution path.
- **Shared contracts.** Several bots on one account can hold the same
  contract. Each ledger tracks only its own contracts; a sell is capped at
  that bot's own claim *and* at what the account actually holds
  (`webapp/account_book.py`), so a bot never sells another bot's contracts.
  Once per cycle the sum of every bot's claims is reconciled against the
  account's positions; mismatches are shown on the Bot Fleet / Custom Bots
  pages and logged, never "fixed" automatically — except a contract the
  account no longer holds at all, which is closed in the ledger at its last
  mark (flagged) so the bot isn't left managing a phantom.
- **Order journal.** Every real order is written to `pending_orders` before
  it goes out and cleared once booked; a cycle that dies in between is
  recovered on the next one by `client_order_id`, so a fill on the account
  can't go unbooked.
- **Alpaca's options rules.** Only market and limit orders exist for
  options: a genome's stop-limit goes out as a limit at its limit price and
  a trailing stop as a market order once the engine's own rule fires. A
  vertical debit spread is one multi-leg (`mleg`) order — long the chosen
  strike, short the wider one — with one net fill price, so the account is
  never left holding a naked short leg. The paper account needs options
  level 3 for spreads; a rejection is logged under History and the entry
  simply doesn't happen.
- **API budget.** Alpaca rate-limits per key and meters its trading API
  (orders, fills, positions) separately from its market-data API. Every
  outbound call in the app goes through `webapp/api_meter.py`, which counts
  them per account, per API, per minute in SQLite — the engine cycle is a
  different process from the web app, so an in-memory counter would miss
  almost everything. The Bot Fleet and Custom Bots pages show that account's
  **peak calls/min**, the fleet size at that minute, the resulting
  calls-per-bot-per-minute, and the bot count that rate would take to the
  limit — which is the number to look at before raising the fleet-size
  guardrail.
  An entry is one POST plus a poll per second until it fills, so the trading
  figure scales with bot count; the data figure does not (one fetch per
  cycle serves every fleet). `ALPACA_RATE_LIMIT_PER_MIN` (default 200, the
  documented basic-plan figure) only renders the headroom — nothing throttles
  against it.
- **Fleet size.** **`max_fleet_bots`** (default `BOT_MAX_FLEET_BOTS`, 100)
  is the whole story: both the number every maintenance pass refills the
  fleet to and the number it may never exceed. There is no separate, lower
  target — after the daily pass retires the idle and the weakest, it spawns
  fresh candidates into every empty slot, so each account starts the day
  with a full fleet. The cap is a guardrail
  like the risk limits — a house default an admin sets, each account's own
  override on its Guardrails page, both clamped to
  `BOT_ABS_MIN_FLEET_BOTS`/`BOT_ABS_MAX_FLEET_BOTS`. It is enforced in
  `bots.spawn_bot()`, which every fleet bot is born through, so no route
  gets past it: not breeding, not the daily top-up, not a fleet reset, not
  the *Spawn* button (which spawns into the room available and says so).
  Where the cap is tighter than the target it becomes the target
  (`bots.fleet_target`), so a capped fleet settles at the cap instead of
  trying to top up to a number it may not reach; the daily pass retires its
  weakest before breeding, rather than after, so children still have
  somewhere to be born. Lowering the cap under a running fleet retires
  nothing — growth stops and the fleet drifts down as evolution retires
  bots normally. Custom and baseline bots sit outside it: they are not part
  of the evolving fleet, so they neither count against it nor are blocked
  by it.
- **Capital.** Three different numbers, easy to confuse. The bots' *virtual
  capital* (60 × `$10,000` = `$600k`) is pure bookkeeping and never needs to
  exist as cash. What the account actually has to fund is *peak exposure*:
  each bot's total open premium is capped at `position_pct_limit` (10%) of
  its own balance, so ~64 bots can hold at most ~`$64k` at once — about 6%
  of a `$1M` paper account. The Bot Fleet page shows all three plus live
  utilization.
- **Running short.** There is **no Alpaca API to add money to a paper
  account** — it is a long-standing feature request, and the dashboard's
  *Reset* is the only way to change a paper balance. So the engine refuses
  an entry it can't fund (`account_book.can_afford`, logged once under
  History) instead of firing an order Alpaca would reject. That matters for
  more than tidiness: rejections would otherwise land on whichever bots
  happen to tick last, handing the earlier ones a survivorship edge that has
  nothing to do with their genomes. Cached buying power is debited and
  credited as fills happen, so bots later in a cycle see what is really left.

  To give the fleet more room, prefer lowering the **Maximum bots in my
  fleet** guardrail (below) or `BOT_START_BALANCE`. If you do want a bigger paper balance, in Alpaca's
  dashboard: **Reset** the paper account with the new starting amount — but
  note it **wipes the account's positions and history and issues new API
  keys**. Do it only while the fleet is flat, then paste the new keys into
  your Bot Fleet settings panel and reset your fleet, since the old ledgers no
  longer correspond to anything on the account. This is per account — other
  users' fleets, and everything the shared pool has learned, are untouched.
- **No simulated mode.** Every fleet trades on its owner's Alpaca account,
  always. There is deliberately no modeled-fill setting: a genome must
  never be able to look proven on the strength of a spread model that
  flatters it, and a fitness number is only worth acting on if the fills
  behind it really happened. Modeled fills exist in exactly one place — the
  offline replay harness, on its own in-memory ledger, which is the right
  tool for testing a genome without spending market days.
  *Upgrading:* an account that was on the old "simulated" mode now places
  real orders (or, with no keys saved, stops trading). Its existing ledger
  mixes modeled and real fills, so reset that fleet to judge genomes under
  one set of rules — the change is logged under History.
- **Forced exits** (drawdown trip, retirement, fleet reset, expiry
  settlement) sell for real too and book the actual fill; only if the
  broker call fails is the close booked synthetically, flagged in the note.

## How the fleet is judged

- **Fitness** is a t-statistic of per-trade return on premium, day-adjusted
  against the buy-and-hold baseline bot for the same symbol (so a trend day
  doesn't bless every long-call bot), shrunk toward the fleet mean with a
  20-trade pseudo-count and divided by its standard error. Win rate, trade
  count and raw P&L% no longer score on their own. Open positions are
  marked to market; `robust_fitness` is the worst of lifetime, recent-window
  and marked fitness, and culling ranks on the marked number (flattening at
  the mark before retiring).
- **Proven** (may breed) means ≥ 60 closed trades over ≥ 15 sessions in ≥ 2
  volatility regimes (low / mid / high VIX or a macro event day), fitness
  above a Bonferroni bar `Φ⁻¹(1 − 0.05/N)` for the N genomes tried since the
  last pool reset, and a non-negative recent window. **All three sample
  gates are counted over the pooled record** — every fleet's trades on that
  genome — so several fleets reach them in a fraction of the calendar days
  one would need, and N counts distinct genomes rather than bot rows so more
  fleets never raise the bar by themselves. Breeding is balanced across
  strategy families; half of bred children are crossovers; mutation step
  size is self-adaptive per lineage. Fleet size is `max_fleet_bots` (100 by default) **per account**, refilled in full every day.
- **Cost gate** (not a gene): an entry needs expected directional move over
  its planned hold ≥ spread + theta + commission (`BOT_COST_GATE_MULT`).
- **Stops / targets / trails** are measured mid-to-mid from the entry mid,
  never from the ask paid to the current mid.
- **Quotes** older than `BOT_MAX_QUOTE_AGE_S` (60 s) or from the delayed
  yfinance chain are never filled against; they calibrate implied vol and
  spread width and are repriced at the live spot.
- **Structure**: a genome trades a naked call/put (`single`) or a vertical
  debit spread (`vertical`, `spread_width` strikes wide) booked as one
  defined-risk package — the buy-to-open-only rule keeps its meaning while
  theta and spread drag drop.

These are rule changes: every bot alive before them was selected under the
old rules. Reset once after deploying (`python -m webapp.bot_engine reset`,
which resets every fleet and restarts the shared multiple-comparison count,
or the per-account button on the Bot Fleet page) so the new rules get a clean
cohort. `python -m webapp.bot_engine reset USER_ID` resets one account's
fleet only.

## Replay harness (offline evolution)

The live fleet can only test a genome at the cost of a real market day, on
a sample far too small to tell skill from luck. The engine therefore
**records** every 1-minute bar, VIX snapshot, prior-day level and real
option quote it sees (`webapp/recorder.py` → `data/market_history.db`), and
`webapp/replay.py` runs the *same* entry / exit / position-management code
over that history in memory. That history is one file every account shares:
each cycle fetches it once on the best entitlement anyone has connected, so
more connected accounts means a deeper record per user, not a divided one,
and a genome tested offline is tested against everything all of them have
ever seen.

```bash
./venv/bin/python -m webapp.recorder backfill --days 30      # bars from Alpaca (years) or yfinance (~30d)
./venv/bin/python -m webapp.recorder status
./venv/bin/python -m webapp.replay backtest --bot 12 --days 20
./venv/bin/python -m webapp.replay evolve --pop 60 --generations 20 --workers 4 --promote 5
```

`evolve` runs a GA on the first 75% of recorded days (or `--train A:B`),
then validates the best distinct genomes on the remaining days they never
saw (`--validate C:D`). Only a genome whose **validation** fitness is
positive with real out-of-sample trades is a survivor; `--promote N` seeds
the top N into a live fleet as fresh bots (origin `replay`) that still have
to prove themselves live — the primary admin's fleet by default,
`--promote-to USER_ID` for another. Promoted survivors are registered in the
shared pool like any other candidate, so whoever runs one, its live trades
build the same record and any account can clone it from `/pool`. Runs are saved under `data/replay_runs/`
and logged on the engine history page.

Fidelity: backfilled days have bars only, so every fill on them is priced
off the modeled chain; days the recorder itself captured replay against the
real quotes it stored. The replay ticks once per minute (live: four times).
Budget roughly 0.25 s per genome-day single-threaded.

## Deployment

Designed to run as a FastAPI app behind a reverse proxy with systemd services and timers for the scheduled jobs (daily pass, incubator ticks, refreshes). Deployment scripts and server configuration are not included in this showcase copy.

## Disclaimer

Educational and research use only. Not investment advice.
