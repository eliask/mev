# MeV / PAA — purpose and acceptance of useful work

Status: **maintained project goal and work-selection contract**, updated 2026-10-11. This is the reference for what the project is trying to achieve. It does not certify implemented features, deployment or analytical accuracy. [SPEC.md](SPEC.md) governs evidence and engineering acceptance; [UI.md](UI.md) governs the reader-facing answer; [INVESTIGATION_PROCESS.md](INVESTIGATION_PROCESS.md) governs investigation and stopping.

## Project goal

**PAA helps citizens, journalists and public decision-makers answer consequential questions about who publicly proposed, supported, opposed or changed a choice, what the institution decided, and what the available evidence establishes.** It turns dispersed public records into reusable, inspectable answers with exact sources, dates, alternatives and clear attribution boundaries. Implementation, effects and correction routes are investigated when the question and evidence warrant them. Success means materially better supported understanding or less effort to obtain and verify a reliable answer than competent direct research. Users supply the political values and judgments.

The long-term purpose is to **make public power intelligible, attributable and correctable**. The immediate product is a question-driven political memory: a reader can follow an issue or person into a documented decision, understand the strongest supported answer, and inspect its decisive evidence. Record and page counts measure inventory, not achievement of this goal.

## Research workflow and acceptance

Start from an ordinary person-and-issue question, discover the needed sources,
interpret substantive positions and actions, and retain a useful answer through
the existing tools. A supported subanswer must not replace the parent question.
Pursue material obtainable evidence; distinguish unfinished acquisition from
unidentifiable influence. Test changed participants and follow-ups on fresh
matters, counting source selection, preparation, review, corrections and owner
steering across the whole task.

Use a capable research agent as the reasoning component. Meaningful access,
verification or repeat-use savings can justify PAA without novel facts or a
proprietary reasoning advantage. Ordinary name/issue lookup should reach retained
acts and prepared answers, including co-participants under their own names.
Record-only hubs must not be presented as interpreted answers.

Engineering checks, AI source review, and reader usefulness are distinct tests.
General automated accuracy and outside-reader usefulness remain unestablished.
External testing evaluates a completed task; it does not substitute for finishing
missing research. Continue a component only when it improves a reliable answer
or reduces its acquisition or verification cost.

## What are we doing?

Turn scattered records of public decisions into usable answers about **what someone said or proposed; who had the power, duty and opportunity to act; which actions were actually taken; where a rule or decision changed; what was enacted and implemented; what happened in the world; and who can now answer for or change the relevant mechanism**.

The unit of inquiry is a **consequential question about an actor, a choice, a public mechanism or a result**. It is *not* an arbitrary document, politician profile, source-anchored paragraph, or interesting-looking statistic.

The interface earns its space by answering that inquiry. A person–issue view
shows their specific choices and proposals alongside the institution's recorded
result. A policy view shows the consequential difference between documented
versions. A response view connects the exact objection to what changed or was
answered. Raw speeches, counts and provenance support those answers; a feed of
activity does not substitute for an interpretation. If a speech has not been
interpreted and reviewed, offer its original text rather than inventing a stance.

In Finnish, the public-facing promise can be:

> **Kuka päätti ja teki mitä, millä perusteilla, mitä siitä seurasi ja mitä voidaan korjata?**
>
> Julkiset asiakirjat, äänestykset ja puheet yhdistetään dokumentoiduiksi päätöspoluiksi. Toteutusta, vaikutuksia ja korjauskeinoja selvitetään silloin, kun lähteet sen mahdollistavat. Näytetty teko erotetaan väitetystä vaikutuksesta ja avoimet tiedonaukot pidetään näkyvissä.

This product does not have political authority, and an accurate analysis does not itself repair a public mechanism. The attainable contribution today is **publicly usable evidence, distinctions, analysis and correction routes**. A possible future MeV institution would need separate powers and duties for response, implementation and re-entry.

## Whose decisions can this improve?

| User / task | Question they can bring | A useful result changes |
| --- | --- | --- |
| Voter or citizen | What did this person actually do on issue X, as distinct from their group and their promises? | Their *evidence base* for a value-dependent political decision; the system does not select a candidate for them. |
| Journalist or civic investigator | Where did a contested provision come from, who supported/changed it, which explanation survives the sources? | What they can accurately report, challenge or pursue. |
| MP, ministry, local official or affected party | Did objection A survive scrutiny, what rule is causing failure, and which authority can change it? | A specific response, alternative, request, test or correction path. |
| Researcher or another civic-tech project | Which events, source versions and actors are reliably linked? | Reusable public evidence and less duplicated investigation effort. |

The same factual answer can serve more than one user. But **"a general reader might find it interesting" is not a sufficient investigation mandate**. The investigator must name the concrete decision or question the work could improve.

## Five question families, one evidence substrate

1. **Selection/credibility.** What observable past conduct and relevant constraints bear on how an actor might act in a future issue? Do not invent probabilities, generic trust scores or voting instructions; expose evidence and its limits.
2. **Agency and attribution.** What did the person, party, coalition, ministry or institution actually say, propose, sign, vote for, oppose, decide and implement? Which formal duties and feasible options belonged to which actor at the time? **Personal actions are not dissolved into party aggregates; collective decisions are not silently assigned to individual members.**
3. **Policy transmission.** How did an objective, instrument, threshold, allocation, exception, obligation or deadline change from preparation through enactment and subsequent legal/administrative state? What caused the documented change, if anything can be established?
4. **Mechanism and outcomes.** What causal machine did a policy install? How were actors expected to respond, how did they actually respond, what costs or stocks absorbed pressure, and did the policy produce its intended and other important consequences?
5. **Correction and feedback.** When a warning, observed failure or alternative entered the record, who received it, who answered, what was amended or rejected, what was implemented, and what would trigger further correction?

A case may answer only a bounded part of its family. It **must not imply closure of later stages** merely because an earlier stage is documented.

## What PAA and MeV each contribute

- **PAA / Poliittinen muisti:** a broad, time-aware *public action and decision record* plus selective evidence-bound investigations. Its differentiator is the actor–opportunity–action–decision chain, without fictional personal causal attribution.
- **MeV mechanism analysis:** reconstruction and adversarial testing of the *rules and incentive mechanisms* behind proposals or enacted policies, including alternative mechanisms and an owner/trigger/measurement for repair. This tests a causal design, not someone's moral character.
- **LawVM:** where available and relevant, a time-specific *legal state* and provenance layer. Legal text is not implementation, and implementation is not impact.

They compose when the user question requires them. They do not all have to run for a routine records lookup. A mechanism defect can exist even when every official faithfully discharged a narrow duty; an attributable individual action can matter even when no causal outcome can be proven.

## The two operating scales

**Scale A — broad, quiet infrastructure:** Acquire and reconcile public source records across a defined parliamentary period. Preserve identities, document structure (majority, dissent, quoted expert), role and time, exact vote alternatives, filing versus signature, coverage and errors. Index possible relations with local inference as **proposals**. High-volume completeness is useful for arbitrary future lookups, but does not by itself constitute an investigation.

**Scale B — selective, deep investigations:** Spend inference, source acquisition, modeling and editorial review on *important questions*. Choose a hypothesis that can change a user's understanding or next action; look for a discriminating observation; perform actor attribution, mechanism analysis or both; publish an intelligible answer with source pointers, alternatives, uncertainty and (when relevant) a correction path.

Do **not** attempt to write a narrative case page for every vote. Do **not** let the ease of producing pages dictate what is considered important.

## The worth-doing / worth-publishing test

Before launching substantial case research, the owner or agent must be able to answer, in ordinary language:

1. **Who might use this?** Name an actual audience and the decision, dispute, action or important historical explanation at stake.
2. **What is not already resolved?** Write two materially different possible answers (including a credible benign alternative). What would finding A rather than B change?
3. **What can the available evidence distinguish?** Identify specific sources and gaps; no top-k search becomes proof of absence, and no repeated LLM agreement becomes proof of truth.
4. **What does the investigation add?** Compared with the best existing official summary or an ordinary same-source research answer, will it recover a missing action/decision, distinguish responsibility, resolve a contradiction, expose a neglected mechanism, identify a new correction lever, or appreciably lower reliable research cost?
5. **What is a useful stopping point?** A bounded positive/negative result, an explicit unidentifiability proof, or a named missing observation that a real reader can act on—not simply "more research needed".

A case **does not need scandal or a negative finding**. A correct institutional repair, an exonerating source, or proof that a purported conflict is illusory can be useful when that distinction affects a live understanding or decision. Consolidating dispersed primary evidence can be valuable even when the facts are individually public. But **mere correctness, caution or novelty of wording is insufficient**.

If both possible answers leave the audience with essentially the same understanding and no meaningful research-cost savings, **do not commission a deep public investigation**. Keep it as a regression fixture, source annotation or private research note if useful.

## What counts as attribution?

Attribution is **a vector of separately justified observations**, never a universal blame/credit score:

- *Speech / carrier:* whose statement is it, and is it personal, collective, institutional, quoted or dissenting?
- *Formal capability and duty:* what authority/role existed at the relevant time, with what actual constraints?
- *Observable opportunity:* which real channels and alternatives were available; did an election or procedural condition hold?
- *Personal or collective act:* filed, amended, signed, voted, asked, answered, approved, dissented, implemented; exact object and stage.
- *Congruence:* what part of a particular commitment or position does that act bear on, if any?
- *Institutional disposition and state:* what the competent body decided, enacted, operated or measured.
- *Causal contribution or pivotality:* evidence of a contribution and a justified counterfactual, if the record can sustain it. **Not assumed** from rank, membership, sequence or agreement with a party.
- *Evaluative responsibility:* a separate, transparent normative standard (duty, knowledge, feasible alternatives, chosen act, consequences). The user decides which values and burdens matter; the application reports the relevant evidence and rival readings.

An MP's individually recorded vote remains their action even if changing that one vote would not change the tally. Equally, voting for a bundled proposal does not prove assent to each component. Formal assignment of a function does not prove private authorship or outcome causation. Distinguish absence of a documented action from failure to search the relevant action channel.

## Public result shape — answer first, audit trail second

A publishable page should tell a cold reader, without assuming knowledge of PAA:

**A question, its material answer and the decisive evidence.** Show the choice,
comparison or change that establishes the answer, and the specific unresolved
fact that matters. These are requirements for useful content, not a mandatory
stack of headings: do not repeat the answer as a summary, cards and a graph.

The answer need not claim causation when the evidence supports only a choice or document history. It also should not bury a substantial finding under generic disclaimers, repeat the same caveat in every paragraph, or expose dozens of low-level source IDs before the user knows why to care.

Different outputs can be appropriate: a factual profile, a comparison table, a timeline, a scoped answer, a documented counterexample, an implementation audit, a mechanism test or a correction specification. **Visualization is justified by a better answer, not by graphic novelty.**

## Success, failure and stopping

Success is **correct, consequential, transferable answers with inspectable evidence**, and their uptake in actual inquiries or changes. Track by category: independently source-correct results; consequential corrections or exonerations; repeat research time saved; external reuse; decisions or public claims that became more precise; and transfer to *unseen* matters. Report publication selection bias, missed cases and failure modes.

Failure includes confident false attribution, a researched question nobody needs answered, a dossier that merely restates one official report's caveat, and public claims that implementation or causation was proven from a procedural artifact.

**Stop** when another source, prompt, iteration, or layer of review has no credible path to changing a material answer, decision, or reusable capability. Archive the case with its reason. An unidentifiable question can be a legitimate termination, not a permanent assignment to produce more prose.

## Conceptual relation to existing work

This contract is an operational product proposal, not a claim that these essays independently prove any particular political case:

- [The Mechanism Analysis](https://kunnas.com/articles/mechanism-analysis): pre-decision causal object and repair test.
- [The Corrigible State](https://kunnas.com/articles/the-corrigible-state): a challenge–authority–implementation–verification–reopening path, with no magical single corrector.
- [How I Work](https://kunnas.com/articles/how-i-work): a public distinction earns its place when someone else can use it to decide, test, build, repair or route something.
- [The Layer Walk](https://kunnas.com/articles/layer-walk): a separate diagnostic protocol for complex mechanism failures; invoke only when needed, not as boilerplate for every PAA records query.
