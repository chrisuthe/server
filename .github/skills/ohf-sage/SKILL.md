---
name: ohf-sage
description: Apply the Open Home Foundation / Music Assistant project leads' cited engineering principles when reviewing a pull request. Use on every code review.
---

# OHF Sage — leads' principles review

Cited engineering principles of the project leads, mined from real PR reviews. They speak for
Marcel van der Veldt (`marcelveldt`) across all projects and Marvin Schenkel (`MarvinSchenkel`)
for Music Assistant. Home Assistant is out of scope.

Apply these when reviewing a change, alongside the repo's existing review standards:

1. Identify which layer applies — Overall (Marcel van der Veldt) always; plus Music Assistant
   (Marvin Schenkel) or the relevant per-project layer. Prefer the most specific rule when
   several apply.
2. When a change violates a principle in `principles.md` (bundled beside this file), flag it and
   **cite the linked PR/issue permalink** the principle came from — every rule carries its source.
3. Map severity to the existing taxonomy: a `MUST` / "won't support" violation is `[CRITICAL]`
   or `[PROBLEM]`; a `Prefer` mismatch is `[SUGGESTION]`.
4. Weigh by provenance marker: `[authored]`/`[enforced]` are firm policy, `[authored+mined]` is
   strongest, `[mined · N PRs]` is inferred from review history (higher N = firmer).
5. If no principle covers the case, don't invent one — defer to the existing review standards.

The full cited rule set is in `principles.md`, in this same directory.
