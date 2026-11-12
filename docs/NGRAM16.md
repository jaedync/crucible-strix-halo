# 16-bit n-gram table (opt-in, experimental)

This is an opt-in, experimental setting. It is not the default.
The model has an n-gram embedding table. A pack stores this table at low precision.
The 16-bit setting makes a BF16 copy of the table from the original `Qwen/Qwen3.8-Flash-Next` weights
(the `ngram_embedding.shard_*` tensors). The engine then reads rows from this copy instead of the pack's own table.
There is no download of the finished file. You build it with one command:

```sh
python -m pip install "huggingface_hub>=0.23"
python tools/ngram16/fetch.py --out /path/to/ngram16
```

The command does these steps:

1. It reads `model.safetensors.index.json` of `Qwen/Qwen3.8-Flash-Next` at revision
   `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
   It picks only the files that hold n-gram tensors: 33 files, `model-00005-of-00131.safetensors`
   to `model-00037-of-00131.safetensors`. It prints the list and the size.
2. It checks the free disk space and the free memory before it starts. It stops with a message (exit code 2)
   if either is too small.
3. It downloads the 33 files (a stopped download continues when you run the command again),
   builds the BF16 file with `tools/ngram16/convert.py`, and checks the result.
4. It prints the two lines to add to the environment of the engine.

Use `--dry-run` to see the list and the sizes without a download. `--rate-mb` sets the write rate cap of the build
(default 100). A stopped download or build continues when you run the command again.

The command does not delete the downloaded files. When it ends, it tells you how much space they use.
To remove them, run the command again with `--delete-source`. This run checks the size and the stored sha256 of
the finished file again and then deletes only the downloaded files. It needs no free disk space.

What it costs:

- Download: 104.3 GB (33 files).
- Disk: about 207 GB at the peak (the downloaded files and the 102.4 GB output). After the build, the
  output alone needs 102.4 GB.
- Time: the download takes about 17 minutes at 100 MB/s. The build took 26 minutes on the test system
  (66 MB/s, with the write cap at 100 MB/s).
- Memory: the command stops (exit code 2) when `MemAvailable` is below 15,625,000 kB. During the build,
  `convert.py` pauses when it falls below 13,600,000 kB.
- Cache: the Hugging Face client can keep a download cache in `HF_HOME` (default `~/.cache/huggingface`).
  If that disk is small, set `HF_HOME` to a larger disk before you run the command.

The output is the file `ngram_embedding_bf16.safetensors` in the `--out` folder. It must be exactly 102,400,496,832 bytes,
sha256 `71205584be130551594a3e1f43949f47c576a3cc65486f1e51f4e7ba45fa42be`.
The command compares the size of the file and the sha256 that the build computed while it wrote the bytes.
It refuses a file that differs. It does not read the finished file a second time.

To use the file, start the engine with these two variables. `CRUCIBLE_PLE_BF16` takes the path of the file,
and it needs `CRUCIBLE_PLE_HOST=1`:

```sh
CRUCIBLE_PLE_HOST=1 CRUCIBLE_PLE_BF16=/path/to/ngram16/ngram_embedding_bf16.safetensors
```

The engine reads the file from the SSD during decode, so there are constant disk reads.
The file is 102.4 GB, which does not fit in memory with the model, so the engine reads the rows it needs from the disk.
Do not change or move the file while the engine runs.

Measured effect on the 4.05 pack (the same binary, 5 A/B pairs for the speed rows):

| Measure | Own table (default) to BF16 table |
|---|---|
| KL against the BF16 reference | 5.7% lower (95% interval 1.2 to 9.9%) |
| 4 streams at 32K, decode | 1.3% slower |
| Code with speculation | 2.2% faster |
| Serial decode | flat (0.1% lower) |
| Cold 32K prefill | flat (0.4% higher) |

The quality gain is small and the speed change goes both ways. Turn it on only if you want to test it.
