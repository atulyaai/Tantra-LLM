# Response inspection at update 1,950

Read-only probe evaluation of the latest saved 64K checkpoint (1,950 updates,
7,987,200 targets in the 19M stage). Nine questions cover greetings, developer
identity and uncertainty in English, Hindi and Hinglish. References are AI-authored;
Hindi/Hinglish fluent review is pending. This is a small diagnostic, not a release
accuracy estimate. Exact matches were zero; actual answers were manually inspected
by the coding assistant rather than equating exact mismatch with incorrectness.

Confirmed observable failures:
- English greeting repeats "a number of a number" rather than greeting.
- Developer identity does not name Tantra/Atulya AI; unrelated article-style text.
- Hindi greeting includes a references heading and unrelated names.
- Hinglish questions produce English article fragments and repeated compounds.
- Generated responses do not establish instruction following or coherent dialogue.

Reference-answer loss at update 190 -> 1950:
English 6.220 -> 7.604; Hindi 4.959 -> 4.165; Hinglish 8.843 -> 9.574.
The same suite is used, but different CPU inference parallelism can introduce
small FP32 variations. These losses are diagnostics, not correctness percentages.
General text validation at update 1950: Hindi 2.903, English 6.010. Improved
encyclopedia prediction does not imply improved conversation; the monitored
English/Hinglish responses demonstrate the distinction.

This stage trains plain Wikipedia next-token prediction without chat supervision
or revalidated 64K conversational adapters. That objective mismatch is a plausible
contributor, not proof that all architecture/tokenizer choices are optimal or
that one code change would repair outputs. More vocabulary or RAM alone will not
supply identity facts and dialogue behavior.

Next required work: distinct reviewed conversation families; balanced Hindi,
English and Hinglish coverage; supervised conversation/identity tuning paired to
the exact 64K base; regression tests; independently held-out answer and routing
evaluation. Keep reserved evaluations out of training. No new quality promotion,
checkpoint overwrite or automatic training stop was performed by this review.
The current base is not a usable conversational release.

Fixed dashboard issue: later loss-only probe entries previously hid earlier
generated samples. The viewer now keeps the highest-update generated answers
across available histories and labels their update explicitly. Three dashboard
tests passed, including the loss-only-history regression.
