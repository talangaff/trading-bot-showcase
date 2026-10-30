"""Quick add — one typed line in, a credit spread out.

"AAL 10/16 $11/12 Put Credit Spread" is how a spread gets talked about, so
it should be how one gets logged. This turns that line into the fields a
credit-spread trade needs. What it cannot know — quantity, the two leg
prices, the account value — it deliberately leaves alone: the trade is
created as a draft and the form asks for the rest.

Credit spreads ONLY, for now. Every other type has its own shape and its
own shorthand, and guessing which one a line meant would make the feature
unreliable at exactly the moment it is supposed to save time.

Which leg is which is not a detail the typist should have to spell out,
because it follows from the strategy: a PUT credit spread sells the higher
strike and buys the lower one (you are paid to promise a floor, and buy a
cheaper floor below it), and a CALL credit spread is the mirror — sell the
lower, buy the higher. So "$11/12" means short 12 / long 11 as a put, and
short 11 / long 12 as a call, whichever order it was typed in.
"""

import re
from datetime import date

# a number pair — "10/16", "$11/12", "10/16/26" — with whatever marks it
# carries: a $ says strikes, a trailing year says expiration
_PAIR = re.compile(
    r"(?P<d1>\$)?\s*(?P<a>\d+(?:\.\d+)?)\s*[/\-]\s*(?P<d2>\$)?\s*(?P<b>\d+(?:\.\d+)?)"
    r"(?:\s*[/\-]\s*(?P<year>\d{2}|\d{4})\b)?"
)

# words that appear in these lines but are never the ticker
_NOT_A_TICKER = {
    "put", "puts", "call", "calls", "credit", "debit", "spread", "spreads",
    "bull", "bear", "vertical", "pcs", "ccs", "sell", "sold", "buy", "bought",
    "short", "long", "exp", "expiry", "expires", "expiration", "x", "at", "for",
}
_WORD = re.compile(r"[A-Za-z]+")

# the one trade type this understands — callers check against it rather
# than spelling the string again
CREDIT_SPREAD = "credit_spread"

EXAMPLE = "AAL 10/16 $11/12 put credit spread"


def _is_datey(a, b, year) -> bool:
    """Could this pair be a month/day? A year makes it certain."""
    if year is not None:
        return True
    return (float(a).is_integer() and float(b).is_integer()
            and 1 <= int(a) <= 12 and 1 <= int(b) <= 31)


def _resolve_expiration(month: int, day: int, year, today: date):
    """The date that "10/16" means: this year, or next if it has passed.

    A typed expiration is always the NEXT one — nobody quick-adds a trade
    into the past — so a bare month/day rolls forward rather than landing
    in a month that is already over."""
    if year is not None:
        y = int(year)
        y = 2000 + y if y < 100 else y
        try:
            return date(y, month, day), None
        except ValueError:
            return None, "{}/{}/{} isn't a real date.".format(month, day, year)
    for y in (today.year, today.year + 1):
        try:
            d = date(y, month, day)
        except ValueError:
            return None, "{}/{} isn't a real date.".format(month, day)
        if d >= today:
            return d, None
    return None, "{}/{} isn't a real date.".format(month, day)


def _pair_text(pair) -> str:
    def n(v):
        return str(int(v)) if float(v).is_integer() else str(v)
    return "{}/{}".format(n(pair["a"]), n(pair["b"]))


def _option_type(low: str):
    if re.search(r"\bputs?\b|\bpcs\b", low):
        return "put"
    if re.search(r"\bcalls?\b|\bccs\b", low):
        return "call"
    return None


def _ticker(text: str, skip_spans):
    """The first word that isn't part of the strategy's own vocabulary."""
    for m in _WORD.finditer(text):
        if any(s <= m.start() < e for s, e in skip_spans):
            continue
        w = m.group(0)
        if w.lower() in _NOT_A_TICKER or len(w) > 6:
            continue
        return w.upper()
    return None


def parse_credit_spread(text: str, today: date | None = None):
    """(values, None) ready for journal.create_trade, or (None, message).

    The message is written to be shown to whoever typed the line, so it
    says what was missing rather than what the regex did."""
    raw = (text or "").strip()
    today = today or date.today()
    if not raw:
        return None, "Type the spread first — like: {}".format(EXAMPLE)

    low = raw.lower()
    ot = _option_type(low)
    if not ot:
        return None, ("Say whether it's a put or a call credit spread — like: {}"
                      .format(EXAMPLE))

    pairs = []
    for m in _PAIR.finditer(raw):
        pairs.append({
            "a": float(m.group("a")), "b": float(m.group("b")),
            "year": m.group("year"),
            "dollar": bool(m.group("d1") or m.group("d2")),
            "span": m.span(),
        })
    if not pairs:
        return None, ("Needs two strikes and an expiration — like: {}".format(EXAMPLE))

    # A $ marks strikes and a year marks a date; when a line carries
    # neither mark, the convention in these lines is expiration first.
    strikes = next((p for p in pairs if p["dollar"]), None)
    dated = next((p for p in pairs if p["year"] is not None), None)
    if strikes is None and dated is not None:
        strikes = next((p for p in pairs if p is not dated), None)
    if dated is None:
        rest = [p for p in pairs if p is not strikes]
        dated = next((p for p in rest if _is_datey(p["a"], p["b"], p["year"])), None)
    if strikes is None:
        rest = [p for p in pairs if p is not dated]
        strikes = rest[0] if rest else (pairs[0] if dated is None else None)
        if strikes is None and len(pairs) == 1:
            # one bare pair and nothing else: strikes are the part a trade
            # cannot do without, so read it as strikes and ask for the date
            strikes, dated = pairs[0], None

    if strikes is None:
        return None, "Needs two strikes — like the $11/12 in: {}".format(EXAMPLE)
    if strikes["a"] == strikes["b"]:
        return None, "The two strikes have to be different."
    if dated is None:
        spare = next((p for p in pairs if p is not strikes), None)
        if spare is not None:
            return None, ("Couldn't read {} as an expiration — needs a month/day, "
                          "like the 10/16 in: {}".format(_pair_text(spare), EXAMPLE))
        return None, "Needs an expiration — like the 10/16 in: {}".format(EXAMPLE)

    exp, err = _resolve_expiration(int(dated["a"]), int(dated["b"]), dated["year"], today)
    if err:
        return None, err

    ticker = _ticker(raw, [p["span"] for p in pairs])
    if not ticker:
        return None, "Couldn't find a ticker — like the AAL in: {}".format(EXAMPLE)

    hi, lo = max(strikes["a"], strikes["b"]), min(strikes["a"], strikes["b"])
    # a put credit spread sells the higher strike, a call one the lower
    short_strike, long_strike = (hi, lo) if ot == "put" else (lo, hi)

    return {
        "trade_type": CREDIT_SPREAD,
        "ticker": ticker,
        "option_type": ot,
        "expiration": exp.isoformat(),
        "short_strike": short_strike,
        "long_strike": long_strike,
    }, None


def describe(values: dict) -> str:
    """What the parse decided, in the words the ticket uses — so a
    misread line is obvious before any money is typed in."""
    def n(v):
        return str(int(v)) if float(v).is_integer() else str(v)
    return ("{}: sell the {} {}, buy the {} {}, expiring {}".format(
        values["ticker"], n(values["short_strike"]), values["option_type"],
        n(values["long_strike"]), values["option_type"], values["expiration"]))
