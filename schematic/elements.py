"""Generic addressable-element machinery: registry, override grammar, group
anchoring, per-element RNG streams.

Extracted as its own module rather than re-typed per node. Nothing here knows what
an element looks like; it only resolves WHICH are drawn and WHERE they are nudged.

Two rules from the Micrographics build carry over and are load-bearing:

  * Offsets are normalised canvas fractions applied AFTER anchoring, so a
    detection-anchored element is nudged relative to what the detection chose
    rather than ripped off it.
  * Every element draws from its OWN RNG stream keyed on (seed, element_id). With
    one shared sequential stream, disabling any element re-rolls the content of
    every element drawn after it - found by the blind teeth, with no visual tell.
"""

from . import engine

OFFSET_LIMIT = 1.0


class ElementSyntaxError(ValueError):
    """Malformed or unknown directive. Never silently ignored: a silent no-op on a
    typo is the bug class that costs hours."""


def element_seed(seed, element_id):
    """Stable per-element seed. FNV-1a rather than hash(), which is salted per
    process and would break determinism across runs.

    The FNV hash is then run through a MurmurHash3 fmix32 finaliser. Without it,
    XORing a small user seed with a constant id-hash changes only the LOW BITS,
    so seeds 1, 7 and 42 produce LCG states within ~100 of each other - and a
    Lehmer generator started from nearby states produces nearly identical first
    outputs. Measured before the fix: with zone sizes over [2, 150], the first
    zone came out 109.8px for every seed tried.
    """
    h = 2166136261
    for ch in str(element_id):
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    x = (int(seed) & 0xFFFFFFFF) ^ h
    x ^= (x >> 16)
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= (x >> 13)
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= (x >> 16)
    return (x % (engine.MinstdLCG.MOD - 1)) + 1


# Lehmer generators correlate their first outputs with the seed regardless of how
# well the seed is mixed, so the stream is also warmed up before use.
_WARMUP = 3


def rng_for(seed, element_id):
    r = engine.MinstdLCG(element_seed(seed, element_id))
    for _ in range(_WARMUP):
        r.draw()
    return r


class Registry:
    """An ordered set of (element_id, group_id, default_enabled)."""

    def __init__(self, entries):
        self.entries = tuple(entries)
        self._group = {e: g for e, g, _ in self.entries}
        self._default = {e: d for e, _, d in self.entries}

    def ids(self):
        return tuple(e for e, _, _ in self.entries)

    def groups(self):
        seen = []
        for _, g, _ in self.entries:
            if g not in seen:
                seen.append(g)
        return tuple(seen)

    def members(self, group):
        return tuple(e for e, g, _ in self.entries if g == group)

    def group_of(self, eid):
        return self._group.get(eid)

    def default(self, eid):
        return self._default.get(eid, False)


def _parse_offset(token, line_no):
    parts = token.split(",")
    if len(parts) != 2:
        raise ElementSyntaxError(
            f"[Construction] elements line {line_no}: expected an offset as "
            f"'dx,dy' in normalised canvas fractions, got {token!r}.")
    try:
        dx, dy = float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        raise ElementSyntaxError(
            f"[Construction] elements line {line_no}: offset components must be "
            f"numbers, got {token!r}.")
    dx = max(-OFFSET_LIMIT, min(OFFSET_LIMIT, dx))
    dy = max(-OFFSET_LIMIT, min(OFFSET_LIMIT, dy))
    return dx, dy


def parse_elements(text, registry):
    """Returns (toggles, offsets, group_offsets). Empty text yields three empty
    dicts, which is what makes an empty override identical to the defaults."""
    toggles, offsets, group_offsets = {}, {}, {}
    if not text:
        return toggles, offsets, group_offsets

    valid_e = set(registry.ids())
    valid_g = set(registry.groups())

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ElementSyntaxError(
                f"[Construction] elements line {line_no}: expected "
                f"'<name>: <directive>', got {line!r}.")
        name, rest = line.split(":", 1)
        name, rest = name.strip(), rest.strip()

        is_group = name in valid_g
        if not is_group and name not in valid_e:
            raise ElementSyntaxError(
                f"[Construction] elements line {line_no}: unknown name {name!r}.\n"
                f"  valid elements: {', '.join(sorted(valid_e))}\n"
                f"  valid groups:   {', '.join(sorted(valid_g))}")
        if not rest:
            raise ElementSyntaxError(
                f"[Construction] elements line {line_no}: {name!r} has no "
                f"directive. Use 'off', 'on', 'dx,dy', or 'on dx,dy'.")

        state = None
        offset = None
        for tok in rest.split():
            low = tok.lower()
            if low in ("on", "off"):
                state = (low == "on")
            elif "," in tok:
                offset = _parse_offset(tok, line_no)
            else:
                raise ElementSyntaxError(
                    f"[Construction] elements line {line_no}: unrecognised token "
                    f"{tok!r}. Use 'off', 'on', 'dx,dy', or 'on dx,dy'.")

        if state is not None:
            if is_group:
                for m in registry.members(name):
                    toggles[m] = state
            else:
                toggles[name] = state
        if offset is not None:
            if is_group:
                group_offsets[name] = offset
            else:
                offsets[name] = offset

    return toggles, offsets, group_offsets


class ElementState:
    """Resolved enable/offset state for one render."""

    def __init__(self, registry, text=""):
        self.reg = registry
        self._t, self._o, self._g = parse_elements(text or "", registry)

    def on(self, eid):
        return bool(self._t.get(eid, self.reg.default(eid)))

    def offset(self, eid):
        """Group offset plus the element's own. The SUM is clamped, not each term,
        so a group offset equals the same offset applied to every member."""
        gdx, gdy = self._g.get(self.reg.group_of(eid), (0.0, 0.0))
        odx, ody = self._o.get(eid, (0.0, 0.0))
        return (max(-OFFSET_LIMIT, min(OFFSET_LIMIT, gdx + odx)),
                max(-OFFSET_LIMIT, min(OFFSET_LIMIT, gdy + ody)))

    def px(self, eid, w, h):
        """Normalised units resolved to pixels exactly once, here, which is what
        keeps a render at 1024 geometrically similar to one at 2048."""
        dx, dy = self.offset(eid)
        return dx * float(w), dy * float(h)

    def table(self, title):
        lines = [f"# {title} - resolved element table",
                 "# Paste any line into `elements` and edit it. Offsets are",
                 "# normalised canvas fractions, applied after anchoring."]
        for eid in self.reg.ids():
            dx, dy = self.offset(eid)
            lines.append(f"{eid}: {'on' if self.on(eid) else 'off'} {dx:+.3f},{dy:+.3f}")
        lines.append("#")
        for g in self.reg.groups():
            lines.append(f"# group {g}: {', '.join(self.reg.members(g))}")
        return "\n".join(lines)
