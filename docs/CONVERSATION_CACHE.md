# The conversation cache: where a conversation's state lives

Crucible keeps the attention state of each conversation, so a later turn pays only for its new tokens.

## Three places

| Place | What it holds | Size |
|---|---|---|
| Session slot (GPU memory) | The live state of a running conversation | Reserved at start: about 6.85 GB for a slot of 262,144 tokens |
| Host cache (`--conversation-cache-mib`) | Resume points of recent conversations | By real size: about 26.5 KB a token plus about 150 MB a state |
| Disk cache (`--state-dir`) | States that left the host cache or a slot, also across a restart | By real size, up to `--state-disk-gib` |

## When the disk is written

Nothing is written at each turn. A state goes to the disk when it leaves the host cache or a slot, and when the
server stops. Only states of 16,384 tokens or more go to the disk. A state is one small file plus block files of
4,096 tokens. States with equal blocks share the files, so a later turn of the same conversation adds only its new
blocks.

## What a slow disk does

The GPU never waits for the disk. A write waits in host memory, in a queue of at most half of `--host-reserve-mib`
(5.5 GiB with the setup default). When the queue is full, the next write is refused, and a later turn that needs
that state computes its prompt again. A turn waits at most 30 seconds for a disk read, then it starts cold.
A slow disk gives slow starts, never a stopped reply or a wrong state.

## How fast the disk must be

The engine makes new state at 48 MB/s at most. Reads decide what you feel:

| Read speed | Restore of a 100K-token state (2.8 GB) | Restore of a 250K-token state (6.6 GB) |
|---|---:|---:|
| 5 GB/s (NVMe) | about 0.6 s | about 1.3 s |
| 0.5 GB/s (SATA SSD) | about 6 s | about 13 s |
| No disk cache: compute again | about 55 s | about 140 s |

The table is arithmetic from the state sizes and the prefill rate, not a measurement of other drives.

## Shared blocks

States with equal blocks share the block files, so the disk gets fewer bytes than the total size of the states.

## The disk limit covers the whole state folder

- `--state-disk-gib` limits all the states in the state folder, not only the states of the running engine.
- Each state identity (state epoch, pack and state settings) has its own folder in the state folder. An update or a
  changed setting makes a new folder. The old folder stays while the limit has room, so a rollback finds its states.
- When the total is above the limit, the engine removes the folders of other identities first, oldest first. It does
  this at start and before each state write. Only then does it remove its own oldest states.
- The engine never removes a folder that another running engine uses. That folder does not count against the limit
  of this engine. Thus two servers on one state folder use at most the sum of their two limits.
- The engine removes only the files that it writes. The `write-ledger` file and other files stay.
- Each removed folder gives one log line with its size, its file count and the age of its newest file:
  `state_disk remove_dir dir=<folder> reason=cap bytes=<B> files=<N> newest_age_s=<S>`.
- When a file cannot be removed, the line also gives `left_files=<N> errno=<E>`. The bytes that stay still count, so
  the engine removes more of its own states. The engine tries that folder again after one hour.
- When a folder cannot be read, the engine logs `state_disk walk_error dir=<folder> errno=<E>` and continues. It does
  not count or remove that folder.
- A folder of another identity also leaves at start when it was not used for 7 days. The folder that was used last
  before this start stays under that rule.
- The removal changes no reply. A state is read only by an engine of the same identity, so this engine can never read
  the states that it removes.

## SSD wear

- A guard limits the state writes to 300 GiB (322 GB) in one local calendar day
  (server option `--state-disk-write-gib-per-day`, 0 to 65536, default 300). The count is in the file `write-ledger`
  in the state folder, so a restart does not reset it. Above the limit every state write is refused, also at a stop.
  A refused state costs one cold prompt later. `0` makes the folder read only.
- The largest possible wear is 300 GiB x 365 = 118 TB a year. Compare it with the rating of your SSD (TBW divided by
  the warranty years). A 2 TB drive rated 1,200 TBW over 5 years allows 240 TB a year: the limit is 49% of it.
  For a drive rated 600 TBW, lower the limit (for example to 100).
- The guard counts one state folder. Two servers with two folders have two budgets.

## Where the state folder can be

The state folder can be on any disk, also outside your home folder (for example a second, faster SSD).
Set it with `./crucible setup --state-dir DIR`, or start the server with `--state-dir DIR`.
The engine, setup, doctor and uninstall use the same rule:

1. The path is absolute.
2. Links in the parent folders are resolved. A parent link to a folder that is not there (for example a disk that
   is not mounted) is refused. The last name must not be a link.
3. System folders are refused: `/etc`, `/usr`, `/proc`, `/sys`, `/dev`, `/boot`, `/efi`, `/bin`, `/sbin`,
   `/lib`, `/lib32`, `/lib64`, `/var/lib` and all folders in them. The shared top folders `/`, `/run`, `/tmp`,
   `/var`, `/var/tmp`, `/home`, `/mnt`, `/media`, `/opt`, `/srv` and `/root` are refused as the state folder itself.
   They are accepted as parent folders.
4. Your home folder itself is refused. Use a folder of its own in it, for example `~/.cache/crucible/states`.
5. Each parent folder that exists is a real folder of root or of you, and the engine can read it. Write access for all
   users is refused. Group write access is accepted only when the group of the folder is your main group. On a system
   where all users have the same main group (for example `users`), use parent folders with no group write access.
   A folder with the sticky bit (as `/tmp`) is accepted as a parent, also when other users can write it.
6. The state folder belongs to you, is not shared (no sticky bit), and holds only the `write-ledger` file and
   state identity folders. The engine sets its mode to 0700 and makes missing parent folders with mode 0700.
   Then it checks the path again. When the real path changed between the two checks, the server does not start.
7. The engine and setup make and open the folder one name at a time from `/`. They do not follow a link at any
   depth, and they check each parent folder again with rule 5. Thus a link that another user puts in a shared folder
   (for example `/tmp`) after the check stops the start, and the error names the link. A folder of another user
   never gets a mode change. A parent folder needs only the search permission (`x`), as before. Thus a parent such
   as `/home` with mode 0711 is accepted.

Why: the folder holds the state of your conversations. Another user must not be able to read it, or move or
replace it while the server runs. The rule keeps that promise without a fixed place in the home folder.
Inside a user namespace (bwrap, a rootless container), each host user that the namespace does not map shows as the
overflow user (65534). Root of the host is one of them. The rule accepts that owner for parent folders there, so in a
namespace it also accepts a parent folder of another host user that the namespace does not map.

`./crucible doctor` shows the result in the row "State folder". A path refusal fails the check, also a parent folder
that cannot be read. A folder with files that Crucible did not make, or a folder whose list cannot be read, gives a
warning, because the server stops at start.
`./crucible uninstall --remove-models` removes the state folder only when setup made it, the rule still accepts it,
and it holds only Crucible state. It never removes the parent folder.

## Limits

- With more than 262,144 tokens a session (YaRN), setup starts the server with no state folder in this version.
- A state is read only by an engine with the same state identity (state epoch, pack and state settings). After an
  update that changes it, the first turn of an old conversation computes again one time.
