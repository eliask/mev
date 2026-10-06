# Question-driven interface contract

Status: maintained design and acceptance target, 2026-10-11. This refines the
question-derived UI proposal; it does not claim the current renderer implements
every item. [PURPOSE.md](PURPOSE.md) owns the goal and [SPEC.md](SPEC.md) owns
evidence and admission. Implementation choices remain revisable through real
reader tasks.

## Reader task

A reader brings a person, issue or bounded question. They should learn what was
proposed, what a person actually did, which alternative the institution chose,
what changed, and which evidence supports that answer. Effects and correction
paths appear when the relevant evidence exists.

Support two entry routes: **issue → decision → person/action** and
**person → issue → decision**. Search and shortcuts operate within the declared
index over prepared journeys, matter titles/IDs, canonical actor pages and
declared dossier aliases. Results distinguish prepared AI-source-reviewed
answers from record-only hubs; unrecognized questions show available records and
the unresolved target — they do not generate a purportedly verified answer.
Exact actor identities and ambiguous name-only rows stay distinct. The public
reader can explore prepared answers without a live model or account.

## Minimal useful surface

Each view exists to answer a question, not to display its input records. The
project's questions concern political preferences, personal choices and agency,
policy change, evidence response, implementation, consequences and correction.
Show the supported answer to the selected question; other channels are optional.
A speaking record proves participation, not a position. An activity count is
not performance, influence or an answer about responsibility.

Prefer a few compact linked facts, a comparison table or an actual decision path.
Do not build a card feed of speeches, ballots, pipeline statuses or source IDs.
Repeated legal titles and actor/channel/action labels add no information.
For person–matter views, surface explicit approval/rejection pairings, personally
named alternatives/signatures and documented group departures. Other ballots
remain inspectable. Select a pairing by its literal source title, without calling
it enactment or inferring a position on every clause. Speeches without reviewed
interpretations form a secondary source index with original text, not empty
"Puhui" cards or fabricated stance summaries.

Put denominators, hashes, model codes and audit detail in a secondary source
view. Preserve exact alternatives, material conditions and limitations affecting
the answer. Every additional visible row must add a distinction that changes
what the reader can establish or give a needed next action.

Empty search shows the box, help and at most a few prepared shortcuts — never the
whole index. Route and question pages lead with an ordinary reader title and a
brief answer/status line; research prompts and raw candidate bodies stay closed.
A decisive already-bound source excerpt belongs next to the compact answer, not
only inside a long journey fold.

## Answer surface

One question → one bounded answer → the decisive choice or comparison → evidence.
Begin with the concrete problem and stakes when they are not already apparent.
Explain unfamiliar procedural terms before comparing them. A shorter sequence
of identifiers is not a clearer explanation. Select the nearby evidence for
the answer it supports; the first available quote may establish only a signature
and say nothing about a policy change. Bind explicit evidence selections to the
reviewed projection, and retain the complete selected passage.
This is not a required stack of sections. Do not repeat the answer as a takeaway,
summary, card feed and graph. Use one representation that makes its important
distinctions legible. Expand supporting records only on request. Where the data
cannot answer the question, state that particular gap; do not substitute a list
of activity as an apparent answer.

Keep one plain AI-review sentence visible. Review constants, projection status
codes, hashes and export remain a secondary audit view. Preserve genuine
time/coverage/attribution limits. On a phone, the answer and key choice should
precede long methodology or record lists. Keyboard navigation, query-state
deep links and useful static pages must work without JavaScript.

## First reusable components

Start with two consumers of one source-bound decision-journey presentation:

- **Branching policy history:** distinct initiative and government routes,
  formal outcomes, enacted provisions and dated successors. Preserve differing
  conditions and independently supported paths.
- **Alternative sequence:** who publicly proposed each alternative, what each
  pairing meant, which alternative won, and the separate institutional result.

Use existing reviewed route and tax examples for development. Apply the same
components to fresh matters for transfer evaluation. An actor-action summary and
nearby evidence disclosure support both. Additional components earn their place
through a real question; the wider widget catalogue is not a mandatory backlog.

## Visual claims are evidence claims

Every substantive node, status, edge and comparison identifies its supporting
claim/event, exact source version, relevant date, interpretation state and scope.
The maintained backend owns these relationships. The renderer arranges them and
does no semantic matching.

Reordering reviewed claims is presentation. Calling provisions equivalent,
depicting an amendment as a response, asserting a successor relationship or
attributing authorship introduces meaning. Bind and review that meaning before
displaying it as established. A review of prose does not automatically review a
new graph assembled from it. Changed analytical projections invalidate dependent
review according to SPEC.md.

Distinguish documented references, formal procedural progression, interpreted
policy relationships and causal hypotheses. An arrow must say which it means.
Document order, shared vocabulary and lexical adoption/rejection cues do not
establish institutional disposition or causal influence. Where a relationship is
unresolved, show the supported endpoints and the precise missing relationship.

## Evidence and attribution

Original excerpts should be within one interaction of the displayed claim, with
a readable source title, issuer, relevant page/date and original link. Structured
records appear as named vote choices or other human-readable fields, never raw
JSON. Hashes, parser locators, file names, acquisition notes and schema fields
belong in downloadable evidence packets, not anywhere in the reading UI—even
inside expanded disclosures. A disclosure must help check the claim, not expose
the transport format.
Review labels state who reviewed what, including AI versus human review. Source
binding, substantive support and coverage are separate checks.

An actor view preserves exact actions and alternatives, event-time group/role
context, and the institution's response. Signatory, drafter, committee participant,
decision-maker and causal contributor remain distinct. A recorded package ballot
does not establish support for every component. Enactment, legal effectiveness,
operational delivery and measured effects remain separate.

Keep acquisition failure, uninspected channels, bounded absence, contested
interpretation and genuinely unobservable influence distinguishable. Positive
documented actions survive an unresolved downstream effect.

## Acceptance and continuation

Technical acceptance exercises actual selected mobile/desktop pages, keyboard,
back/forward and deep links, nearby evidence, static fallbacks, exports and source
revision invalidation. Large matters load records by person/stage/page rather
than every detailed intervention at once.

Semantic acceptance uses fresh matters with independently source-grounded
reference obligations, material-error criteria and incomplete-source controls.
Development examples and fresh evaluation results are reported separately.

Reader acceptance asks whether actual participants can find the answer, identify
the person versus institutional choice, interpret the alternatives, state the
attribution limit and verify the decisive source. Compare reliable completion
and effort against existing records or competent direct research, including
follow-up tasks where reuse could help. Internal AI reading and browser tests
cannot establish this acceptance.

Continue when the component transfers and improves reliable task completion or
total research/verification effort. Simplify, remove or link outward when it does
not. Population accuracy, arbitrary-question competence and outside usefulness
remain unestablished until separately evaluated.
