from __future__ import annotations

import random

# ─── Star Wars themed responses ─────────────────────────────────────────────

_STAR_WARS_RESPONSES = [
    (
        "🛸 <b>These are not the droids you're looking for.</b>\n\n"
        "The Imperial Senate has decreed that only authorized personnel "
        "may access this terminal. Your identification does not appear in "
        "the Alliance data banks.\n\n"
        "<i>— A mysterious hologram flickers and fades.</i>"
    ),
    (
        "⚡ <b>Restricted area — by order of the Galactic Empire.</b>\n\n"
        "The Death Star's targeting computer has locked your "
        "communication frequency. This is a secure channel "
        "for allied forces only.\n\n"
        "<i>— A lightsaber ignites in the distance.</i>"
    ),
    (
        "🌌 <b>The Force is not with you.</b>\n\n"
        "You are neither Jedi nor Sith. You are merely a traveler "
        "in a galaxy far, far away, and this comm channel "
        "is reserved for those who bear the administrator's mark.\n\n"
        "<i>— R2-D2 beeps disapprovingly.</i>"
    ),
    (
        "🚀 <b>This is not the bot you're looking for.</b>\n\n"
        "Move along. Move along. Only those with the "
        "administrative clearance code may pass. Your credentials "
        "do not match the Millennium Falcon's encrypted transponder signal.\n\n"
        "<i>— Chewbacca growls in agreement.</i>"
    ),
    (
        "🪐 <b>Access denied — Hoth base is closed.</b>\n\n"
        "The Rebel Alliance has encrypted this channel. Without "
        "the administrator authentication code, you are as lost as a "
        "stormtrooper on Tatooine.\n\n"
        "<i>— A thermal detonator ticks ominously.</i>"
    ),
]

# ─── Hitchhiker's Guide to the Galaxy themed responses ──────────────────────

_HITCHHIKER_RESPONSES = [
    (
        "📖 <b>You are not authorized — and frankly, don't panic.</b>\n\n"
        "According to <i>The Hitchhiker's Guide to the Galaxy</i>, "
        "this bot is intended for users with an administrative "
        "Infinite Improbability Drive. Yours appears to be malfunctioning.\n\n"
        "<i>— A small, depressed robot hands you a towel.</i>"
    ),
    (
        "🌍 <b>Access denied. Mostly harmless.</b>\n\n"
        "The Guide rates your clearance level as "
        "<i>\"mostly harmless\"</i> — "
        "which in this case means <i>\"insufficient\"</i>. "
        "Only beings with a tolerance for Pan-Galactic Gargle Blasters "
        "and administrator rights may proceed.\n\n"
        "<i>— Ford Prefect shrugs and orders another drink.</i>"
    ),
    (
        "🐁 <b>The answer is 42. The question is: are you an admin?</b>\n\n"
        "Deep Thought has computed the meaning of life, the universe, "
        "and everything — and it turns out you need admin access "
        "to use this bot. Sorry, the mice were very clear on this.\n\n"
        "<i>— Two white mice scurry away with the ultimate answer.</i>"
    ),
    (
        "🚀 <b>Improbability Drive: offline.</b>\n\n"
        "Your request has encountered an infinitely improbable "
        "barrier — namely, your lack of administrator rights. "
        "The Heart of Gold's computer suggests returning "
        "when you have the proper clearance.\n\n"
        "<i>— Marvin, the paranoid android, sighs heavily.</i>"
    ),
    (
        "📕 <b>Entry not found in the Guide.</b>\n\n"
        "<i>The Hitchhiker's Guide to the Galaxy</i> has no entry "
        "for your user ID. It does, however, note that you should "
        "contact an administrator. In the meantime — "
        "here's a towel. You'll need it.\n\n"
        "<i>— Vogon poetry begins in the distance. Run.</i>"
    ),
]

# ─── Combined pool ──────────────────────────────────────────────────────────

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