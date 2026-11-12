"""The configuration that the public README speed table measured: the one table that ./crucible start and the Docker run
line (docs/DOCKER.md) read. tests/test_service_config.py pins this table, the service and DOCKER.md to the README words.

Engine words (environment). Each one is a switch of engine/switches.h. The draft words change drafts only: the verify pass
checks every draft, so the output stays (token gate PASS on both packs, public files, no tuned draft head file; CHANGELOG
v0.32.0). CRUCIBLE_PF_LAUNCH_ROWS=16384+128 is the engine default: the stage arena is 2,174 MiB and the
launch fits 4 sessions of 262,144 tokens with --vision on the 4.05 pack (CHANGELOG v0.32.0). It is set here by name so that
the engine default can change without a change of the measured run."""

ENV = {
    "CRUCIBLE_PLE_HOST": "1",
    "CRUCIBLE_COMPACT_STATE": "1",
    "CRUCIBLE_PLE_FETCH": "1",
    "CRUCIBLE_DRAFT_HEAD": "q2",
    "CRUCIBLE_DRAFT_CHAIN": "hc8,lead",
    "CRUCIBLE_PF_LAUNCH_ROWS": "16384+128",
}

# Server options. --streams and --ctx are the defaults of the service (the user may change them at setup).
SERVER_FLAGS = {
    "--streams": "4",
    "--ctx": "262144",
    "--prefill": "4096",
    "--conversation-cache-mib": "9700",
    "--host-reserve-mib": "11264",
}

STREAMS = int(SERVER_FLAGS["--streams"])
CTX = int(SERVER_FLAGS["--ctx"])

# The named session layouts of ./crucible setup --layout: (streams, ctx, short streams, short ctx). The default is the measured
# one above. A ctx above 262,144 starts the engine with YaRN (factor 2, up to 524,288 positions); each layout needs about
# the same GPU memory for the session state as the default (about 28 GB).
LAYOUTS = {
    "default": (STREAMS, CTX, 0, 0),
    "long": (3, 524288, 2, 262144),     # 1 session of 524,288 tokens and 2 of 262,144
    "two-long": (2, 524288, 0, 0),      # 2 sessions of 524,288 tokens
}
# Above this context the server runs YaRN and refuses --state-dir (first version): the conversation cache stays in memory.
PLAIN_CTX = 262144
