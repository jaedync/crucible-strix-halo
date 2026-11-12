# Session replay: what changes for you

Session replay is the default when the server has a conversation cache
(`--conversation-cache-mib N`). It makes later turns of a long session start faster.

## What it does

- In an agent session, each new prompt repeats the earlier prompt and the answer that the model wrote.
- Without replay, the server keeps only states at fixed grid positions (every 4,096 tokens).
  It must compute again everything after the last grid position, including the earlier answer.
- With replay, the server also keeps the state at the end of each answer.
  The next turn starts from that state and computes only the new tokens.

## Measured effect (3.05 pack, 30K-token session, 1,500 new tokens per turn)

| Turn type | Without replay | With replay |
|---|---|---|
| Later turn, time to first token | 2.0 to 3.9 s | 1.36 to 1.41 s |

On the 4.05 pack, later turns take 1.45 to 1.56 s with replay and 2.1 to 3.9 s without it.

## Quality

- The model output stays exact in quality. The answer rows of the earlier turn keep the numbers that
  the decode path computed. The fp64 reference shows the same error as for the normal prompt path
  (relative error 1.92e-4 with replay, 1.99e-4 without, 3.05 pack).
- The next-token distributions differ from the path without replay by about as much as a different chunk
  schedule makes them differ (3.05: median KL about 1.5e-4; 4.05: median KL about 3.8e-4).
  The most probable next token was the same in all 12 test prompts.

## What changes in the output

- A later turn with replay can give different tokens than the same prompt sent alone to a new server.
  The cause is the different arithmetic order of the earlier answer rows, not an error.
- The same session history gives the same tokens in every server mode: serial or batch,
  alone or with other sessions active at the same time.
- One exception: slot checkpoints (batch serve, `--slot-checkpoints 1`). A turn that resumes from a checkpoint can
  give other tokens than serial serve, and the checkpoint stays only while no other session takes the slot.
  This is the accepted contract A. A think-capped turn is one example.
- A resume from a position that is not a multiple of 2,048 rows is not bitwise equal to a cold prefill of the same
  tokens. Top-1 is equal in all measured cases, but a greedy output can change at a near tie.
  `CRUCIBLE_EXACT_RESUME=1` makes the resume of a longer prompt exact; it costs at
  most 20 rows of extra prefill for a resume and one small launch for each prompt.

## Limits

- Replay needs the saved state of the previous answer. If the cache removed that state (many sessions,
  small cache) or the server restarted, the turn uses the normal path. The tokens of that turn can then
  differ from a run where the state stayed in the cache.
- Replay applies only when the new prompt repeats the earlier answer token for token.
  A client that changes or removes the earlier answer uses the normal path.
- In batch serve, the state of the previous answer also stays on the session's slot until another request
  uses that slot. A later turn that repeats it resumes there, with no copy from the host cache. The tokens
  are the same as after a copy from the host cache (`--slot-resume 0` turns this off).
- A turn with thinking keeps its saved state only beside the older state, never in place of it, because
  many clients do not send the reasoning back.
- The model can write a token split that the tokenizer does not make (for example `:` and `<` for `:<`).
  The client sends the answer back as text, so the new prompt differs from the saved tail at that token.
  The turn then starts from the previous tail. The server log line `session_replay diverged` gives the
  position (numbers only). A turn whose think block the server closes uses the tokenizer's split for its reasoning.

## How to turn it off

- Start the server with `--session-replay 0`.
- Use this setting when you must reproduce a request exactly as if it were sent alone,
  for example in tests or comparisons.
