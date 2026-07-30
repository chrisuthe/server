---
name: ohf-sage
description: Apply the Open Home Foundation / Music Assistant project leads' cited engineering principles when reviewing a pull request. Use on every code review.
---

# OHF Sage — leads' principles review

Cited engineering principles of the project leads — Marcel van der Veldt (`marcelveldt`)
across all projects and Marvin Schenkel (`MarvinSchenkel`) for Music Assistant — mined from
real PR reviews. Home Assistant is out of scope. The full cited rule set with permalinks is in
`principles.md`, in this same directory.

## How to review

**Do not stop after finding a few issues.** Systematically walk the **entire** diff against
**every** checklist item below, in order. For each item, scan every changed file for that
specific violation; when you find one, flag it and **cite the linked PR** from `principles.md`.
Then continue to the next item — a substantial diff usually violates several, and missing one
because you already found others is the main failure mode to avoid.

Map severity to the existing taxonomy: a `MUST` / "won't support" violation is `[CRITICAL]` or
`[PROBLEM]`; a `Prefer` mismatch is `[SUGGESTION]`. Weigh by provenance marker in `principles.md`
(`[authored]`/`[enforced]` firm policy, `[authored+mined]` strongest, `[mined · N PRs]` inferred).

## Checklist — check the diff against EACH item (do not skip any)

**Async & performance**
1. Blocking/sync IO on the event loop (`requests`, file `read`/`stat`, PIL, `pickle`, BeautifulSoup not wrapped in `asyncio.to_thread`/executor)? — server#1137
2. N+1: a per-item API or DB call inside a loop over a listing? — server#2501, server#3171
3. A new `aiohttp.ClientSession` created instead of reusing `self.mass.http_session`? — server#1817
4. External-API lookups not routed through the cache (`@use_cache` / `mass.cache`)? — server#3640

**Error handling**
5. Broad or bare `except` instead of specific, expected exception types? — server#3096
6. Swallowed errors — a failure collapsed into `None`/`False`/empty/a silent default instead of propagating or raising a `MusicAssistantError`? — server#1214, server#3722

**Music Assistant architecture**
7. A provider referencing another provider by name, reaching into another controller, or touching the DB directly? — server#2502
8. A parser doing IO or being `async` (parsers must be lightweight, synchronous, IO-free)? — server#2472
9. `StreamDetails` content type/codec **guessed** (e.g. hardcoded `MP3`/`MPEG`) instead of `ContentType.UNKNOWN` when not certain? — server#626, server#2295
10. Multi-item listings returning fully-detailed objects instead of a compact `ItemMapping`? — server#3302
11. Auth/setup/derived-state done in `__init__` instead of `handle_async_init`? — server#3836

**Code quality**
12. Private/helper methods placed above public methods (ordering)? — server#4015
13. f-strings used inside logger calls instead of `%s` positional args? — server#3147
14. A speculative abstraction (helper/method/separate file) for logic with only one consumer? — server#3386
15. The PR doing several unrelated things and needing to be split into single-purpose PRs? — server#2911

For anything not covered above, defer to the repo's existing review standards. If no principle
applies, do not invent one.
