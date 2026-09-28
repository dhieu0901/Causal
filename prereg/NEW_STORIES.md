# NEW_STORIES: CLadder's questions told in new stories

Nguyễn Dương Hiếu, Nguyễn Đại Quân, Trương Hoàng Tùng, Triệu Hải Đăng Trình. 28 September 2026.

Registered before the first call of this study: this file, the stories and the
code that tells each question in them (`src/new_stories.py`, the lexicon
`NEWSTORY` of `scripts/pilot.py`), the run script (`scripts/run_new_stories.sh`)
and the analysis (`scripts/analyze_new_stories.py`) are committed and pushed
together. Nothing here is changed after that commit; any deviation is reported
as one.

## 1. Question

Real names beat letters by 11 to 13 points on CLadder in three model families
(H3b of Study 1). CLadder's text has been public since 2023, so part of that
gain could be recall of the benchmark's own wording rather than reasoning over
coherent text. An exploratory split also found the gain as large on CLadder's
anticommonsense stories as on its commonsense ones, which rests on CLadder's 37
stories. Both questions can be put to text no CLadder question contains.

## 2. The new stories

Each of the 37 stories of Study 1 is moved to a new domain whose variables play
the same roles in the same graph (a confounder stays a confounder, a mediator a
mediator, an instrument an instrument), for example CEO, director, manager and
employee firing become principal, vice principal, class teacher and student
expulsion. Every story-specific word of a question is replaced: the names and
values of CLadder's variable mapping, the unit nouns ("patients", "farms"), the
grammatical variants ("receiving", "blown out") and the hand-written phrases of
the counterfactual questions. The graph, the probabilities, the query, the
wording of the template and the gold label do not change.

CLadder's anticommonsense versions replace the treatment or the outcome by an
unrelated phrase ("having a sister", "freckles"); their new versions do the same
with other unrelated phrases ("having a garden", "dimples"). A new question is
therefore plausible or implausible exactly when its CLadder original is, by
construction, as CLadder's own labels are.

Checks, all passed on the 484 questions before this commit: no story-specific
word of the original survives (every word that is not common to CLadder's
templates, apart from a listed set of generic words and, for six stories, a
listed set of generic words the new phrases reuse); every rewritten question
still states its graph in sentences that parse to the family's structure; and
every distinct sentence of the rewritten questions was read. The stories were
written after Study 1's results were known; they are fixed by this commit.

## 3. Sample, models, conditions

- Questions: Study 1's 484 CLadder questions (the draws of
  `scripts/run_confirmatory.sh` and `scripts/run_b5_llama.sh`).
- New condition: `NEWSTORY` under `RAW` (no graph).
- Reused: `KEEP` and `SYMBOL` under `RAW`, Study 1's own answers.
- Models, registered as two families with separate verdicts: the GPT-4.1 family
  (`gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1`, temperature 0, 700 output tokens,
  key `back_up`), and Llama 3.3 70B Instruct through OpenRouter pinned to
  Novita (bf16), temperature 0, 700 output tokens, answers cut off at 700
  asked again at 1,500 into a separate file. One user message, no system
  prompt, as in Study 1.

## 4. Tests

Per family; every difference is taken per (model, question) on parsed answers
and averaged over every cell, as H3b is.

| Test | Quantity | Prediction |
|---|---|---|
| S1 | `NEWSTORY` - `SYMBOL` | positive |
| S2 | `NEWSTORY` - `KEEP` | within +-5 points (two one-sided tests; 90% interval) |
| S3 | balanced accuracy of `NEWSTORY` - `SYMBOL`, commonsense minus anticommonsense questions | within +-5 points (TOST; 90% interval) |

Balanced accuracy is the mean of the change on gold-Yes and on gold-No
questions, because the two groups differ in their share of Yes labels.

## 5. Decision rule

- Bootstrap: stories resampled with all their questions (both versions of each
  story together), 4,000 draws, percentile intervals; two-sided bootstrap p for
  S1 (`stats.boot_p`), the larger of the two one-sided p for S2 and S3.
- Holm across the three tests of a family, family-wise alpha 0.05. S1 is
  confirmed when its adjusted p is below 0.05 with a positive estimate and
  contradicted with a negative one; S2 and S3 are confirmed when their adjusted
  p is below 0.05, and not confirmed otherwise.
- Seeds 20260907 (primary), 1, 2, 3, 4; a verdict that changes across them is
  reported as borderline.
- Sensitivity, beside the primary analysis: questions resampled instead of
  stories; unparsed answers scored as wrong; for Llama, cut-off answers replaced
  by their re-ask at 1,500 tokens.
- Before reading the new files, `analyze_new_stories.py` must reproduce each
  family's published H3b from the reused `KEEP` and `SYMBOL` answers, or stop.

## 6. How the result is read

- S1 and S2 confirmed: real names help as much on text the model cannot have
  seen as on CLadder's own; the gain of CLadder's names is not recall of its
  text.
- S2 not confirmed with `NEWSTORY` below `KEEP`: CLadder's own wording helps
  beyond new real names, which recall would produce, and so would CLadder's
  stories simply being clearer than ours; the study cannot separate the two.
- S3 confirmed: on new stories, too, the gain does not depend on whether the
  named effect is plausible. S3 not confirmed: the test is inconclusive at 37
  stories, which the exploratory interval (about +-6 points) already suggested.

Descriptive, no verdicts: every cell's accuracy, share of Yes answers and
balanced accuracy; S1 and S2 on gold-Yes and gold-No questions, as a change in
the share of Yes answers, by commonsense label and, for the GPT-4.1 family, by
model.

## 7. Cost

Estimated from the measured tokens of Study 1's `KEEP` answers under `RAW` on the
same questions: Llama about 0.07 USD (hard cap 0.15); GPT-4.1 family about
1.18 USD (nano 0.07, mini 0.20, `gpt-4.1` 0.91; hard cap 1.40, every answer at
700 tokens would cost 3.55). Caps are enforced by the run script.

## 8. Limits known in advance

- One set of new stories, fixed after Study 1; another set could differ.
- The template of CLadder's questions is kept, so the test separates recall of
  CLadder's stories from reasoning over coherent stories, not recall of its
  template.
- The new stories keep CLadder's awkward phrasings where they are part of the
  template ("Would the patient has ..."), so fluency is not improved over
  CLadder.
