# Vera Rebuilt

## Approach

This treats the challenge as a decision problem first, a copywriting problem second. `bot.py` splits into a `decide()` step and a `compose_body()` step. `decide()` looks at the category, merchant, trigger, and optional customer, and produces a `Decision`: which trigger *family* this is, what angle to take (direct, contrarian, seasonal-reframe, or "just ask the merchant"), and a dict of `talking_points` pulled straight from the four contexts. `compose_body()` never sees raw trigger payloads - it only turns `talking_points` into WhatsApp copy. That split is what makes the hallucination question answerable: every number in a message either came from `talking_points` or it doesn't exist, because the composer has no other source to invent one from.

Routing is by trigger *family*, not by kind. `perf_dip` and `seasonal_perf_dip` share a handler that decides whether a dip is a real problem or a known seasonal pattern; the two lapse kinds share a handler; a trigger kind we've never seen falls through to a generic handler keyed only on `trigger.scope`. Roughly 25 trigger kinds map onto about 20 family handlers this way.

## Why deterministic, not LLM-only

Running the supplied `dataset/generate_dataset.py` (not just reading the seed file) showed that 75 of its 100 generated triggers carry `payload: {"placeholder": true}` - no real facts at all. 13 of the 30 canonical test pairs hit this. An LLM given a placeholder payload has nothing to ground a message in either - the actual fix has to be in the decision layer, falling back to real merchant/category data (performance deltas, signals, peer stats, review themes) instead of the missing trigger specifics. Once that logic exists, it's also just... deterministic Python. It's free, instant (sub-2ms across all 100 dataset triggers, measured), and the same input always produces the same output, which matters for a 60-minute test window with a request-rate limit. An LLM can optionally rephrase the final draft if `LLM_PROVIDER` and a matching API key are set - never to add a fact, only to rephrase, checked afterward by the validator - and the bot is fully functional with neither set. `/v1/healthz` never touches this path.

## Conversation handling

`conversation_handlers.py` is a short regex classifier, not a model call - replies have to come back well inside the judge's timeout, and every signal the challenge tests for (hostility, opt-out, explicit commitment, an auto-reply, a request to wait) is lexical, not something that needs language understanding. Ordering matters: explicit commitment ("yes let's do it") is checked *before* the "same message repeated" auto-reply heuristic, because a merchant who impatiently repeats "yes" twice is not a canned WhatsApp Business responder - that specific false positive showed up during testing and is why the check order is what it is. The auto-reply streak is also tracked per-merchant in `bot.py`, not just per-conversation, because a canned responder can show up under a fresh `conversation_id` every turn if the harness rotates them.

## Grounding and safety

Every composer function only reads from `decision.talking_points`, never from `trigger.payload` directly - that's the actual mechanism, not a policy. `validate_and_fix()` removes URLs by design when they are not needed, strips category-taboo vocabulary, and enforces one CTA and correct `send_as` attribution. `quality_gate()` is a second, softer check: eight signals (personalization present, evidence grounded, single CTA, not over-length, ...) that decide whether a structurally-valid draft is still substantial enough to ship, or should fall back to a safe generic check-in instead. On the 30 canonical pairs this gate never had to intervene - useful as a regression check, not as a crutch.

## Suppression

Three independent things, because they answer different questions: a `suppression_key` seen once is never sent again (event-level dedup); a merchant who says "stop" gets a 30-day cooldown across *all* their triggers, not just the one conversation (person-level, and only from hostility/opt-out - not from an auto-reply conversation dying out, which shouldn't silence real opportunities); and a conversation never repeats a body it already sent (message-level).

## Tradeoffs

- In-memory state, no database. The test window is one process for 60 simulated minutes; a database adds restart-survival that isn't needed and a dependency that is one more thing to break.
- `template_params` is a rough 3-way sentence split for the first-touch template structure, not real Kaleyra template authoring - there's no real WhatsApp Business Platform template catalog to target here.
- Hindi-English mixing is hand-written per trigger family, not machine-translated, so category vocabulary and offer names can't get mistranslated.

## What additional context would have helped most

A real per-customer engagement history for `merchant_on_behalf` sends (did *this* customer engage with a past recall reminder?) beyond the merchant-level aggregate that's currently available. And a resolved "current period" on the trigger itself - seasonal reasoning keys off the tick's `now`, but several dataset timestamps predate the test window, so seasonal-beat matching is best-effort rather than exact.

## Testing

Built against the actual expanded dataset (`python dataset/generate_dataset.py`), not just the seed file - the placeholder-payload finding above came from doing that. Tested all 30 canonical pairs (including every placeholder-trigger path), the full HTTP contract (context version ordering - including the equal-version idempotent-replay case - tick restraint/dedup/expiry, auto-reply-hell and hostile-ends over real HTTP, suppression edge cases like two different trigger ids sharing one suppression key), and the conversation classifier, then ran the supplied `judge_simulator.py`'s `warmup`, `auto_reply_hell`, `intent_transition`, `hostile`, and `all` scenarios against a live instance.
