from __future__ import annotations

import random

# ─── Star Wars themed responses ───────────────────────────────────────────────

_STAR_WARS_RESPONSES = [
    (
        "🛸 <b>These aren't the droids you're looking for.</b>\n\n"
        "The Imperial Senate has decreed that only authorized personnel "
        "may access this terminal. Your identification is not part of "
        "the Alliance's databanks.\n\n"
        "<i>— A mysterious hologram flickers and vanishes.</i>"
    ),
    (
        "⚡ <b>Restricted area — by order of the Galactic Empire.</b>\n\n"
        "The Death Star's targeting computer has locked onto your "
        "communication frequency. This is a restricted channel for "
        "allied forces only.\n\n"
        "<i>— The sound of a lightsaber ignites in the distance.</i>"
    ),
    (
        "🌌 <b>The Force is not strong with this one.</b>\n\n"
        "You are not a Jedi — nor a Sith. You are merely a traveler "
        "in a galaxy far, far away, and this communication channel "
        "is reserved for those who carry the admin's mark.\n\n"
        "<i>— R2-D2 beeps disapprovingly.</i>"
    ),
    (
        "🚀 <b>This is not the bot you're looking for.</b>\n\n"
        "Move along. Move along. Only those with the admin clearance "
        "code may proceed. Your credentials do not match the "
        "Millennium Falcon's encrypted transponder signal.\n\n"
        "<i>— Chewbacca growls in agreement.</i>"
    ),
    (
        "🪐 <b>Access denied — Hoth system is off-limits.</b>\n\n"
        "The Rebel Alliance has encrypted this channel. Without an "
        "admin's authentication code, you're as lost as a stormtrooper "
        "on Tatooine.\n\n"
        "<i>— A thermal detonator ticks ominously.</i>"
    ),
]

# ─── Hitchhiker's Guide to the Galaxy themed responses ───────────────────────

_HITCHHIKER_RESPONSES = [
    (
        "📖 <b>You are not authorized — and frankly, don't panic.</b>\n\n"
        "According to <i>The Hitchhiker's Guide to the Galaxy</i>, "
        "this bot is reserved for users with an admin-level "
        "improbability drive. Yours appears to be on the fritz.\n\n"
        "<i>— A small, depressed robot hands you a towel.</i>"
    ),
    (
        "🌍 <b>Access denied. Mostly harmless.</b>\n\n"
        "The Guide rates your clearance level as <i>\"mostly harmless\"</i> — "
        "which, in this case, means <i>\"not nearly enough.\"</i> "
        "Only beings with a Pan Galactic Gargle Blaster tolerance "
        "and admin privileges may pass.\n\n"
        "<i>— Ford Prefect shrugs and orders another drink.</i>"
    ),
    (
        "🐁 <b>The answer is 42. The question is: are you an admin?</b>\n\n"
        "Deep Thought has calculated the meaning of life, the universe, "
        "and everything — and it turns out you need admin access "
        "to use this bot. Sorry, the mice were very clear about this.\n\n"
        "<i>— Two white mice scurry away with the final answer.</i>"
    ),
    (
        "🚀 <b>Infinite Improbability Drive: offline.</b>\n\n"
        "Your request has encountered an infinitesimally improbable "
        "barrier — namely, that you lack admin privileges. "
        "The Heart of Gold's computer suggests you come back when "
        "you've obtained proper clearance.\n\n"
        "<i>— Marvin the Paranoid Android sighs heavily.</i>"
    ),
    (
        "📕 <b>Entry not found in The Guide.</b>\n\n"
        "<i>The Hitchhiker's Guide to the Galaxy</i> has no entry for "
        "your user ID. It does, however, note that you should "
        "probably contact an administrator. In the meantime, "
        "here's a towel. You'll need it.\n\n"
        "<i>— A Vogon poetry reading begins in the background. Run.</i>"
    ),
]

# ─── Combined pool ────────────────────────────────────────────────────────────

_ALL_RESPONSES = _STAR_WARS_RESPONSES + _HITCHHIKER_RESPONSES


def get_random_themed_response() -> str:
    """Return a random themed response (Star Wars or Hitchhiker's Guide)."""
    return random.choice(_ALL_RESPONSES)


def get_star_wars_response() -> str:
    """Return a random Star Wars themed response."""
    return random.choice(_STAR_WARS_RESPONSES)


def get_hitchhiker_response() -> str:
    """Return a random Hitchhiker's Guide themed response."""
    return random.choice(_HITCHHIKER_RESPONSES)
