# Additional objective notes - Streaming platform

## Target

Fire for watch-availability constraints on tracked streaming providers or
supported access modes.

## Decision Questions

- Is the user asking where/how it can be watched?
- Did they name a provider, or only a generic supported access mode?
- Is the provider word actually a studio/brand request?

## Boundaries

- "On Netflix" / "streaming on Hulu" fires here.
- "Disney movies" is studio/brand, not Disney+ availability.
- "Leaving soon", theatrical windows, and future availability no-fire unless
  the endpoint explicitly supports that data.
- Do not import current catalog knowledge; use only provider intent.

