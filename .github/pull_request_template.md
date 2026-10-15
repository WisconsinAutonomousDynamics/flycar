## Team

<!-- vehicle / world / vision / core / tools / leads -->

## What changed and why

<!-- One or two sentences. -->

## Which files changed

<!-- Should all be files your team owns (see .github/CODEOWNERS and docs/teams/). -->

-

## Did any default behaviour change?

<!-- New features must be opt-in: with default settings, everything behaves as before. -->

- [ ] No. `pytest tests/test_baseline.py` passes without re-recording anything.
- [ ] Yes, and it was approved in a lead-approved `api-change` issue: #

## Did you need anything from another team?

<!-- Link the cross-team-request (or api-change) issue, or write "No". -->

## Checklist

- [ ] `pytest` passes locally (headless; `FLYCAR_EYE_BACKEND=raycast pytest` to match CI)
- [ ] New settings are added only in my team's `flycar/config/<team>.py`, with defaults that keep today's behaviour
- [ ] My team page (`docs/teams/<team>.md`) is up to date
