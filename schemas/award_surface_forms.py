from dataclasses import dataclass

from schemas.enums import AwardCeremony


@dataclass(frozen=True)
class AwardPromptSurfaceForm:
    ceremony: AwardCeremony
    event_signals: tuple[str, ...]
    prize_names: tuple[str, ...]
    note: str | None = None


_SURFACE_FORMS: tuple[AwardPromptSurfaceForm, ...] = (
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.ACADEMY_AWARDS,
        event_signals=("Academy Awards", "Academy Awards ceremony"),
        prize_names=("Oscar",),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.GOLDEN_GLOBES,
        event_signals=("Golden Globes", "Golden Globe Awards"),
        prize_names=("Golden Globe",),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.BAFTA,
        event_signals=("BAFTA Awards", "at the BAFTAs"),
        prize_names=("BAFTA Film Award",),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.CANNES,
        event_signals=("Cannes", "Cannes Film Festival", "at Cannes"),
        prize_names=(
            "Palme d'Or",
            "Grand Jury Prize",
            "Un Certain Regard Award",
            "Jury Prize",
            "FIPRESCI Prize",
        ),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.VENICE,
        event_signals=("Venice", "Venice Film Festival", "at Venice"),
        prize_names=("Golden Lion", "Grand Jury Prize", "Silver Lion"),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.BERLIN,
        event_signals=("Berlin", "Berlinale", "Berlin International Film Festival"),
        prize_names=("Golden Berlin Bear", "Silver Berlin Bear", "Teddy"),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.SAG,
        event_signals=("SAG Awards", "Screen Actors Guild Awards", "at SAG"),
        prize_names=("Actor",),
        note='stored under ceremony "Actor Awards"',
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.CRITICS_CHOICE,
        event_signals=("Critics Choice Awards", "at Critics Choice"),
        prize_names=("Critics Choice Award",),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.SUNDANCE,
        event_signals=("Sundance", "Sundance Film Festival", "at Sundance"),
        prize_names=(),
        note="typically no prize-name distinction; use category tags or ceremony only",
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.RAZZIE,
        event_signals=("Razzie Awards", "Golden Raspberry Awards"),
        prize_names=("Razzie Award",),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.SPIRIT_AWARDS,
        event_signals=("Spirit Awards", "Independent Spirit Awards", "at the Spirit Awards"),
        prize_names=("Independent Spirit Award",),
    ),
    AwardPromptSurfaceForm(
        ceremony=AwardCeremony.GOTHAM,
        event_signals=("Gotham Awards", "at the Gothams"),
        prize_names=("Gotham Independent Film Award",),
    ),
)


def render_ceremony_mappings_for_prompt() -> str:
    lines: list[str] = []
    for entry in _SURFACE_FORMS:
        signals = " / ".join(f'"{signal}"' for signal in entry.event_signals)
        lines.append(f"  {signals} → {entry.ceremony.value}")
    return "\n".join(lines)


def render_award_name_surface_forms_for_prompt() -> str:
    lines: list[str] = []
    for entry in _SURFACE_FORMS:
        label = entry.ceremony.value
        if entry.note:
            label = f"{label} ({entry.note})"
        if entry.prize_names:
            prize_names = ", ".join(f'"{name}"' for name in entry.prize_names)
            noun = "prize names" if len(entry.prize_names) > 1 else "prize name"
            lines.append(f"{label} — {noun}: {prize_names}")
        else:
            lines.append(f"{label} — {entry.note}")
    return "\n".join(lines)
