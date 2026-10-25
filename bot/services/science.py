"""Scientific notation → Unicode, for formulas that Telegram can render.

Telegram has no math typesetting: it supports neither LaTeX nor subscript /
superscript *formatting* (MarkdownV2 and HTML have no ``<sub>``/``<sup>``).
Models nevertheless answer with ``H_2SO_4``, ``$E = mc^2$``, ``\\frac{s}{t}``,
``\\alpha``, ``6.022 \\times 10^{23}`` — which reaches the user as raw source.

:func:`normalize_formulas` rewrites that notation with Unicode characters that
display identically everywhere:

=================================  =====================================
model output                       shown in Telegram
=================================  =====================================
``H_2SO_4``, ``\\ce{H2SO4}``       H₂SO₄
``Ca(OH)_2``, ``Na_2CO_3``         Ca(OH)₂, Na₂CO₃
``$E = mc^2$``, ``m/s^2``          E = mc², m/s²
``6.022 \\times 10^{23}``          6.022 × 10²³
``\\frac{s}{t}``, ``\\sqrt{2}``    s/t, √2
``\\alpha \\approx \\Omega``       α ≈ Ω
``\\Delta G = \\Delta H - T\\Delta S``   ΔG = ΔH − TΔS
``25 ^\\circ C``                   25 °C
``\\vec{F}``, ``x_{max}``          F⃗, xₘₐₓ
``A -> B``, ``a <= b``             A → B, a ≤ b
=================================  =====================================

Text inside fenced code blocks and inline code is **never** touched. The
subscript rules are element-aware (a periodic-table parse), so ``snake_case``
identifiers, ``MAX_VALUE`` constants, ``item_1`` and ``x86_64`` survive
unchanged, and bare ``H2O`` in prose is not rewritten either.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# --- Character tables ------------------------------------------------------

_SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼",
    "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ", "g": "ᵍ",
    "h": "ʰ", "j": "ʲ", "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "o": "ᵒ", "p": "ᵖ",
    "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ",
    "y": "ʸ", "z": "ᶻ",
    "A": "ᴬ", "B": "ᴮ", "D": "ᴰ", "E": "ᴱ", "G": "ᴳ", "H": "ᴴ", "I": "ᴵ",
    "J": "ᴶ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ", "N": "ᴺ", "O": "ᴼ", "P": "ᴾ",
    "R": "ᴿ", "T": "ᵀ", "U": "ᵁ", "V": "ⱽ", "W": "ᵂ",
}

_SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌",
    "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "l": "ₗ",
    "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ",
    "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
}

# Periodic table, used to tell a chemical formula from an identifier.
ELEMENT_SYMBOLS = frozenset(
    ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "D", "T"]
)

# LaTeX command → Unicode. Longest names first so ``\neq`` wins over ``\ne``.
_SYMBOLS: dict[str, str] = {
    # Greek (lowercase)
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ",
    "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    # Greek (uppercase)
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
    # Arithmetic / relations
    "times": "×", "cdot": "·", "ast": "∗", "star": "⋆", "div": "÷", "pm": "±",
    "mp": "∓", "approx": "≈", "cong": "≅", "simeq": "≃", "sim": "∼",
    "equiv": "≡", "neq": "≠", "ne": "≠", "leq": "≤", "le": "≤", "geq": "≥",
    "ge": "≥", "ll": "≪", "gg": "≫", "propto": "∝", "infty": "∞",
    "partial": "∂", "nabla": "∇", "sum": "Σ", "prod": "∏", "int": "∫",
    "oint": "∮", "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂",
    "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇", "cup": "∪", "cap": "∩",
    "emptyset": "∅", "varnothing": "∅", "forall": "∀", "exists": "∃",
    "nexists": "∄", "neg": "¬", "lnot": "¬", "land": "∧", "lor": "∨",
    "oplus": "⊕", "ominus": "⊖", "otimes": "⊗", "odot": "⊙", "perp": "⊥",
    "parallel": "∥", "angle": "∠", "triangle": "△", "square": "□",
    "therefore": "∴", "there4": "∴", "because": "∵", "prime": "′",
    "bullet": "•", "dagger": "†", "checkmark": "✓",
    # Arrows
    "rightarrow": "→", "to": "→", "Rightarrow": "⇒", "leftarrow": "←",
    "Leftarrow": "⇐", "leftrightarrow": "↔", "Leftrightarrow": "⇔",
    "uparrow": "↑", "downarrow": "↓", "mapsto": "↦", "longrightarrow": "⟶",
    "rightleftharpoons": "⇌", "hookrightarrow": "↪",
    # Units / science
    "degree": "°", "celsius": "°C", "angstrom": "Å", "ohm": "Ω", "mho": "℧",
    "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ", "Im": "ℑ", "aleph": "ℵ", "imath": "ı",
    "jmath": "ȷ",
    # Delimiters and dots
    "langle": "⟨", "rangle": "⟩", "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊",
    "rfloor": "⌋", "vert": "|", "Vert": "‖", "ldots": "…", "dots": "…",
    "cdots": "⋯", "vdots": "⋮", "ddots": "⋱", "colon": ":",
    # Spacing / line breaks
    "quad": "  ", "qquad": "    ", "nbsp": " ", "space": " ",
    ",": " ", ";": " ", ":": " ", "!": "", "newline": "\n", "linebreak": "\n",
}

# Symbols that read as letters: the LaTeX delimiter space is dropped before a
# letter or digit, so ``\Delta G`` becomes ``ΔG`` (not ``Δ G``) and ``2\pi r``
# becomes ``2πr``.
_TIGHT_SYMBOLS = frozenset(
    ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega", "partial", "nabla", "infty", "hbar", "ell", "aleph", "Re", "Im", "degree", "celsius", "angstrom", "ohm", "mho", "imath", "jmath", "angle", "sum", "prod", "int", "oint"]
)

_SYMBOL_NAMES = sorted(
    (name for name in _SYMBOLS if name[0].isalpha()), key=len, reverse=True
)
# Group 2 captures the single delimiter space LaTeX swallows after a command.
_SYMBOL_RE = re.compile(r"\\(" + "|".join(_SYMBOL_NAMES) + r")(?![A-Za-z])( ?)")
# Spacing escapes such as ``\,`` ``\;`` ``\:`` ``\!`` and ``\ ``.
_SPACE_ESCAPE_RE = re.compile(r"\\([,;:!])(?![A-Za-z])")

# Combining decorations: ``\hat{x}`` → x̂
_DECORATIONS = {
    "vec": "\u20d7", "overrightarrow": "\u20d7", "hat": "\u0302",
    "widehat": "\u0302", "bar": "\u0304", "overline": "\u0305",
    "tilde": "\u0303", "widetilde": "\u0303", "dot": "\u0307",
    "ddot": "\u0308", "underline": "\u0332", "check": "\u030c",
}
_DECORATION_RE = re.compile(
    r"\\(" + "|".join(sorted(_DECORATIONS, key=len, reverse=True)) + r")\s*"
    r"(?:\{([^{}]*)\}|([A-Za-z0-9]))"
)

# ``\text{…}`` and friends: keep the content, drop the wrapper.
_TEXT_WRAPPER_RE = re.compile(
    r"\\(?:text|textrm|textbf|textit|emph|mathrm|mathbf|mathit|mathsf|mathtt"
    r"|mbox|hbox|operatorname|rm|bf|it)\s*\{([^{}]*)\}"
)
# ``\left( … \right)`` — sizing commands mean nothing in plain text. The space
# they consume in LaTeX (after an opener, before a closer) is dropped too.
_LEFT_RE = re.compile(r"\\(?:left|big|Big|bigg|Bigg)[lrm]?\s*([(\[{|.])\s*")
_RIGHT_RE = re.compile(r"\s*\\(?:right|big|Big|bigg|Bigg)[lrm]?\s*([)\]}|.])")

# Degree written as a superscript: ``^\circ``, ``^{\circ}``, ``^\degree``.
_DEGREE_RE = re.compile(r"\^\s*\{?\s*\\(?:circ|degree)\s*\}?")

# ``^{…}``, ``^23``, ``^-1``, ``^n``, ``^-`` (ion charges).
_SUPER_RE = re.compile(r"\^\s*(\{[^{}]*\}|[+\-]?[0-9]+|[A-Za-z]|[+\-])")
# ``_{…}`` — explicit braces are unambiguous math.
_SUB_BRACE_RE = re.compile(r"_\s*\{([^{}]*)\}")

# ``x_1``, ``a_2`` — a *single* lowercase variable with a numeric index. The
# lookbehind keeps ``item_1`` and ``max_value_2`` intact.
_SUB_VAR_RE = re.compile(r"(?<![A-Za-z_])([a-zα-ωΑ-Ω])_(\d{1,3})(?!\d)")
# ``[H⁺]_2``, ``(C₁)_2`` — an index right after a closing bracket.
_SUB_AFTER_CLOSER_RE = re.compile(r"(?<=[\])])_(\d{1,3})(?!\d)")
# ``m_(ост)`` / ``m_((ост))`` — a parenthesised subscript.
_SUB_PAREN_RE = re.compile(r"(?<=[A-Za-zα-ωΑ-Ω\)\]])_\(\(?([^()]+?)\)?\)")
# ``K_eq``, ``T_max`` — an uppercase base with a short lowercase subscript.
_SUB_WORD_RE = re.compile(r"(?<![A-Za-z_])([A-Z][a-z]?)_([a-z]{1,4})(?![A-Za-z0-9_])")
# ``x_i``, ``d_x`` — a single-letter base with a single-letter subscript. Both
# lookbehind and the single-character shape keep ``user_name`` / ``a_bc`` safe,
# and unmappable letters (``a_b``) are left as they are.
_SUB_SINGLE_RE = re.compile(r"(?<![A-Za-z_])([A-Za-z])_([a-z])(?![A-Za-z0-9_])")

# ``\frac{a}{b}``, ``\dfrac``, ``\tfrac``.
_FRAC_RE = re.compile(
    r"\\[dt]?frac\s*(\{[^{}]*\}|[A-Za-z0-9])\s*(\{[^{}]*\}|[A-Za-z0-9])"
)
# ``\sqrt{x}``, ``\sqrt[3]{x}``.
_SQRT_RE = re.compile(r"\\sqrt\s*(?:\[(\d)\])?\s*(\{[^{}]*\}|[A-Za-z0-9])")

# ``\ce{…}`` / ``\pu{…}`` from mhchem.
_CHEM_RE = re.compile(r"\\(?:ce|pu)\s*\{([^{}]*)\}")

# Math delimiters.
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<![\$\\])\$([^$\n]+)\$(?!\$)|\\\((.+?)\\\)")
# ``$5 and $10`` is a price, not a formula: require a formula marker or a
# short variable name (``$c$``, ``$xy$``).
_MATH_MARKER_RE = re.compile(r"[\\^_={}]|[-+*/<>]=|[0-9][A-Za-z]|[A-Za-z][0-9]")
_VARIABLE_RE = re.compile(r"[A-Za-z]{1,3}")

# ASCII operators worth upgrading outside code.
_ASCII_OPS = (
    ("<=>", "⇌"), ("<-->", "↔"), ("<->", "↔"), ("-->", "→"), ("<--", "←"),
    ("=>", "⇒"), ("->", "→"), ("<-", "←"),
    ("<=", "≤"), (">=", "≥"), ("!=", "≠"), ("+-", "±"), ("...", "…"),
)

# Escaped LaTeX punctuation.
_ESCAPED_PUNCT = (
    (r"\%", "%"), (r"\$", "$"), (r"\&", "&"), (r"\#", "#"),
    (r"\{", "{"), (r"\}", "}"), (r"\_", "_"),
)

# Cheap pre-check: nothing to do when the text has no notation markers at all.
_NEEDS_WORK_RE = re.compile(r"[\\^_$~]|\{,\}|<=>|<->|-->|->|<-|=>|<=|>=|!=|\+-")

# Fenced code blocks and inline code, captured so they can be skipped.
_CODE_RE = re.compile(r"(```.*?(?:```|\Z)|`[^`\n]+`)", re.DOTALL)

# A token that may be a chemical formula: element symbols, digits, ``_``,
# parentheses (a leading stoichiometric coefficient is allowed).
_FORMULA_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z0-9][A-Za-z0-9_()]*)")
_PAREN_FORMULA_RE = re.compile(r"(?<=\()([A-Z][A-Za-z0-9_]*_\d[A-Za-z0-9_]*)(?=\))")
_FORMULA_SUB_RE = re.compile(r"_(\d{1,3})(?!\d)")


# --- Helpers ---------------------------------------------------------------


def _to_super(body: str) -> str | None:
    """Superscript ``body``, or None when a character has no equivalent."""
    if not body:
        return None
    mapped = [_SUPERSCRIPTS.get(ch) for ch in body]
    if any(item is None for item in mapped):
        return None
    return "".join(item for item in mapped if item is not None)


def _to_sub(body: str) -> str | None:
    """Subscript ``body``, or None when a character has no equivalent."""
    if not body:
        return None
    mapped = [_SUBSCRIPTS.get(ch) for ch in body]
    if any(item is None for item in mapped):
        return None
    return "".join(item for item in mapped if item is not None)


def _unwrap(body: str) -> str:
    """Drop one level of surrounding braces."""
    body = body.strip()
    if len(body) >= 2 and body[0] == "{" and body[-1] == "}":
        return body[1:-1].strip()
    return body


def _parenthesize(body: str) -> str:
    """Wrap in parentheses only when the body is not a plain atom."""
    if re.fullmatch(r"[A-Za-z0-9.,]+", body):
        return body
    return body if len(body) <= 1 else f"({body})"


def _is_formula(token: str) -> bool:
    """True when ``token`` parses as element symbols, digits, ``_`` and parens.

    Requires at least one element symbol and at least one ``_``, which is what
    separates ``H_2SO_4`` / ``Ca(OH)_2`` from ``user_name`` / ``MAX_VALUE_2`` /
    ``x86_64`` / ``2024_2025``.
    """
    if "_" not in token or not re.search(r"_\d", token):
        return False
    index = 0
    elements = 0
    length = len(token)
    while index < length:
        char = token[index]
        if char.isdigit() or char in "_()":
            index += 1
            continue
        pair = token[index : index + 2]
        if pair in ELEMENT_SYMBOLS:
            index += 2
            elements += 1
            continue
        if char in ELEMENT_SYMBOLS:
            index += 1
            elements += 1
            continue
        return False
    return elements > 0


# --- Conversion stages -----------------------------------------------------


def _strip_math_delimiters(text: str) -> str:
    """Remove ``$$…$$``, ``\\[…\\]``, ``\\(…\\)`` and math-looking ``$…$``."""

    def _display(match: re.Match[str]) -> str:
        return (match.group(1) or match.group(2) or "").strip()

    text = _DISPLAY_MATH_RE.sub(_display, text)

    def _inline(match: re.Match[str]) -> str:
        body = match.group(1) if match.group(1) is not None else match.group(2)
        if body is None:
            return match.group(0)
        stripped = body.strip()
        if match.group(1) is not None and not (
            _MATH_MARKER_RE.search(stripped)
            or _VARIABLE_RE.fullmatch(stripped)
            # ``$n(HCl)$``, ``$[ion]$``: tight on both sides, with a letter and
            # not a price range like ``$5 and $10``.
            or (
                body == stripped
                and re.search(r"[A-Za-zА-Яа-я]", body)
                and not (body[0].isdigit() and " " in body)
            )
        ):
            return match.group(0)
        return stripped

    return _INLINE_MATH_RE.sub(_inline, text)


def _expand_structures(text: str) -> str:
    """Resolve ``\\frac``, ``\\sqrt``, ``\\ce``, ``\\text`` and decorations."""
    for _ in range(4):  # nested constructs, bounded
        previous = text

        def _frac(match: re.Match[str]) -> str:
            numerator = _unwrap(match.group(1))
            denominator = _unwrap(match.group(2))
            return f"{_parenthesize(numerator)}/{_parenthesize(denominator)}"

        text = _FRAC_RE.sub(_frac, text)

        def _sqrt(match: re.Match[str]) -> str:
            index = match.group(1)
            body = _unwrap(match.group(2))
            root = {None: "√", "2": "√", "3": "∛", "4": "∜"}.get(index)
            if root is None:
                return f"{index}√{_parenthesize(body)}"
            return f"{root}{_parenthesize(body) if len(body) > 1 else body}"

        text = _SQRT_RE.sub(_sqrt, text)

        def _chem(match: re.Match[str]) -> str:
            return _chem_subscripts(_replace_symbols(match.group(1)))

        text = _CHEM_RE.sub(_chem, text)
        text = _TEXT_WRAPPER_RE.sub(lambda m: m.group(1), text)
        text = _LEFT_RE.sub(lambda m: "" if m.group(1) == "." else m.group(1), text)
        text = _RIGHT_RE.sub(lambda m: "" if m.group(1) == "." else m.group(1), text)

        def _decorate(match: re.Match[str]) -> str:
            mark = _DECORATIONS.get(match.group(1), "")
            body = match.group(2) if match.group(2) is not None else match.group(3)
            return f"{body}{mark}" if mark else body

        text = _DECORATION_RE.sub(_decorate, text)

        if text == previous:
            break
    return text


def _chem_subscripts(body: str) -> str:
    """Digits after a symbol become subscripts inside ``\\ce{…}`` (H2O → H₂O)."""
    return re.sub(
        r"([A-Za-z\)\]])\s*(\d+)",
        lambda m: m.group(1) + (_to_sub(m.group(2)) or m.group(2)),
        body,
    )


def _replace_symbols(text: str) -> str:
    """Replace LaTeX commands and ASCII operators with Unicode equivalents."""
    text = _DEGREE_RE.sub("°", text)

    def _symbol(match: re.Match[str]) -> str:
        name = match.group(1)
        replacement = _SYMBOLS.get(name)
        if replacement is None:
            return match.group(0)
        if not replacement.strip():
            return replacement  # spacing / newline command
        space = match.group(2)
        if name in _TIGHT_SYMBOLS:
            following = text[match.end() : match.end() + 1]
            if space and (following.isalnum() or following in "_^"):
                return replacement  # "\Delta G" -> "ΔG", "\sum_{i}" -> "Σᵢ"
            return replacement + space
        # Operators and relations read better surrounded by single spaces;
        # _convert_segment collapses the doubles this can produce.
        return f" {replacement} "

    text = _SYMBOL_RE.sub(_symbol, text)
    text = _SPACE_ESCAPE_RE.sub(" ", text)

    for source, target in _ASCII_OPS:
        if source in text:
            text = text.replace(source, f" {target} ")
    for source, target in _ESCAPED_PUNCT:
        if source in text:
            text = text.replace(source, target)
    return text


def _apply_scripts(text: str) -> str:
    """Convert ``^{…}`` / ``_{…}`` and chemical ``X_2`` style subscripts."""

    def _superscript(match: re.Match[str]) -> str:
        body = _unwrap(match.group(1))
        converted = _to_super(body)
        return converted if converted is not None else f"^({_parenthesize(body)})"

    text = _SUPER_RE.sub(_superscript, text)

    def _sub_brace(match: re.Match[str]) -> str:
        body = _unwrap(match.group(1))
        converted = _to_sub(body)
        return converted if converted is not None else f"_({_parenthesize(body)})"

    text = _SUB_BRACE_RE.sub(_sub_brace, text)

    def _sub_paren(match: re.Match[str]) -> str:
        body = match.group(1)
        return _to_sub(body) or f"({body})"  # Cyrillic etc. has no subscript glyphs

    text = _SUB_PAREN_RE.sub(_sub_paren, text)
    text = _SUB_AFTER_CLOSER_RE.sub(lambda m: _to_sub(m.group(1)) or m.group(0), text)

    def _formula(match: re.Match[str]) -> str:
        token = match.group(1)
        if not _is_formula(token):
            return token
        return _FORMULA_SUB_RE.sub(
            lambda m: _to_sub(m.group(1)) or m.group(0), token
        )

    text = _FORMULA_TOKEN_RE.sub(_formula, text)
    # ``m(BaCl_2)``: the whole token starts with a lowercase letter, so the
    # formula inside the parentheses is checked on its own.
    text = _PAREN_FORMULA_RE.sub(_formula, text)
    text = _SUB_VAR_RE.sub(
        lambda m: m.group(1) + (_to_sub(m.group(2)) or m.group(2)), text
    )

    def _sub_word(match: re.Match[str]) -> str:
        converted = _to_sub(match.group(2))
        return match.group(1) + converted if converted is not None else match.group(0)

    text = _SUB_WORD_RE.sub(_sub_word, text)
    return _SUB_SINGLE_RE.sub(_sub_word, text)


_FUNCTIONS_RE = re.compile(
    r"\\(lg|ln|log|sin|cos|tan|tg|cot|ctg|exp|min|max|lim|det|arcsin|arccos|arctan)(?![A-Za-z])"
)
_GLUED_OPERATOR_RE = re.compile(
    r"\\(cdot|times|approx|pm|Rightarrow|rightarrow|to)(?=[A-Za-z0-9])"
)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")


def _prepare(text: str) -> str:
    """Fix common model slips before the main conversion."""
    text = text.replace("{,}", ",")  # Russian decimal comma: 0{,}5 -> 0,5
    text = _BOXED_RE.sub(lambda m: f"**{m.group(1)}**", text)
    text = _GLUED_OPERATOR_RE.sub(lambda m: "\\" + m.group(1) + " ", text)  # \cdotx
    text = _FUNCTIONS_RE.sub(
        lambda m: m.group(1) + (" " if text[m.end():m.end() + 1].isalnum() else ""),
        text,
    )
    return text.replace("\\ ", " ")


def _convert_segment(text: str) -> str:
    """Normalize one non-code segment."""
    text = _prepare(text)
    text = _strip_math_delimiters(text)
    text = _expand_structures(text)
    text = _replace_symbols(text)
    text = _apply_scripts(text)
    # Collapse double spaces produced by padded operators; line indentation is
    # preserved (it is preceded by a newline, not by a non-space character).
    return re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)


def normalize_formulas(text: str | None) -> str | None:
    """Rewrite scientific notation as Unicode, leaving code untouched.

    Returns the input unchanged when there is nothing to convert, and never
    raises: formatting a reply must not be able to break it.
    """
    if not text:
        return text
    try:
        if not _NEEDS_WORK_RE.search(text):
            return text
        parts = _CODE_RE.split(text)
        return "".join(
            part if index % 2 else _convert_segment(part)
            for index, part in enumerate(parts)
        )
    except Exception as exc:  # noqa: BLE001 - must never break a reply
        logger.warning("Formula normalization failed; using raw text: %s", exc)
        return text
