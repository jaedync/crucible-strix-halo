# Results history

Made from [results.tsv](results.tsv) by the release tooling. Do not edit it by hand.
Method and caveats: [Results](RESULTS.md). Each release ran with the settings of its own time.
[Results](RESULTS.md#release-ladder) has every ladder run of each build. Where a release ran more than once, this
page holds the run of the date in its row.

## Release ladder, 4.05 bpw pack

| Release | Measured | Cold 32K prompt: first token (s) | Cold 32K prompt: prefill (tok/s) | Cold 130K prompt: first token (s) | Cold 130K prompt: prefill (tok/s) | Same 32K prompt again: first token (s) | Decode at 32K, 1 session, code (tok/s) | Decode at 32K, 1 session, prose (tok/s) | Decode, 4 sessions, total (tok/s) | Serial decode, no speculation (tok/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.38.1 | Nov 22, 2026 | 16.6 | 1,883 ¹ | 70.0 |  | 0.009 | 72.8 | 58.8 | 112.2 ¹ |  |
| v0.38.0 | Nov 19, 2026 | 16.2 |  | 69.3 |  | 0.017 | 72.9 | 58.3 | 112.8 ¹ |  |
| v0.37.3 | Nov 17, 2026 | 16.3 |  | 69.7 |  | 0.018 | 72.1 | 54.8 | 115.0 ¹ |  |
| v0.34.0 | Nov 7, 2026 | 17.1 | 1,831 | 72.4 | 1,782 | 0.019 | 71.2 | 58.4 | 118.3 | 34.2 |
| v0.33.0 | Nov 3, 2026 | 16.9 | 1,851 | 72.1 | 1,789 | 0.020 | 71.7 ¹ | 56.3 | 113.8 | 34.1 |
| v0.32.0 | Nov 3, 2026 | 17.0 | 1,841 | 74.3 | 1,736 | 0.021 | 71.8 ¹ | 57.6 | no value ¹ | 34.2 |
| v0.31.0 | Nov 1, 2026 | 17.7 | 1,763 | 75.6 | 1,707 | 0.021 | 69.0 ¹ | 55.2 | 117.6 | 34.1 |
| v0.30.0 | Oct 31, 2026 | 17.9 | 1,741 | 77.3 | 1,669 | 0.023 | 68.4 ¹ | 53.3 | 102.4 | 34.3 |
| v0.26.0 | Oct 27 to 29, 2026 | 19.4 | 1,613 | 82.0 | 1,574 | 0.014 | 58.4 ¹ | 50.5 | 106.9 | 32.1 |
| v0.22.0 | Oct 27 to 29, 2026 | 19.7 | 1,585 | 84.5 | 1,527 | 1.870 | 64.6 ¹ | 50.4 | 106.7 | 31.0 |

## Release ladder, 3.05 bpw pack

| Release | Measured | Cold 32K prompt: first token (s) | Cold 32K prompt: prefill (tok/s) | Cold 130K prompt: first token (s) | Cold 130K prompt: prefill (tok/s) | Same 32K prompt again: first token (s) | Decode at 32K, 1 session, code (tok/s) | Decode at 32K, 1 session, prose (tok/s) | Decode, 4 sessions, total (tok/s) | Serial decode, no speculation (tok/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.38.1 | Nov 22, 2026 | 16.5 | 1,903 ¹ | 69.4 |  | 0.009 | 79.1 | 63.4 | 127.8 |  |
| v0.38.0 | Nov 19, 2026 | 16.2 |  | 69.3 |  | 0.017 | 79.1 | 63.4 | 127.9 |  |
| v0.37.3 | Nov 17, 2026 | 16.1 |  | 68.7 |  | 0.018 | 77.0 | 61.2 | 129.3 |  |
| v0.34.0 | Nov 7, 2026 | 16.6 | 1,877 | 71.1 | 1,814 | 0.015 | 76.8 | 60.7 | 129.0 | 39.1 |
| v0.33.0 | Nov 3, 2026 | 16.7 | 1,870 | 71.3 | 1,809 | 0.020 | 73.9 | 60.3 | 118.3 | 39.3 |
| v0.32.0 | Nov 3, 2026 | 16.7 | 1,867 | 72.4 | 1,783 | 0.023 | 74.0 | 58.9 | 130.2 ¹ | 39.3 |
| v0.31.0 | Nov 1 to 2, 2026 | 16.8 | 1,858 | 72.8 | 1,771 | 0.017 | 75.4 | 57.2 | 128.1 | 39.2 |
| v0.30.0 | Nov 1 to 2, 2026 | 16.9 | 1,843 | 74.2 | 1,739 | 0.017 | 76.7 | 59.4 | 127.2 | 39.5 |
| v0.26.0 | Nov 1 to 2, 2026 | 19.2 | 1,628 | 81.5 | 1,583 | 0.025 | 70.8 | 58.4 | 125.8 | 36.5 |
| v0.22.0 | Nov 1 to 2, 2026 | 18.9 | 1,656 | 81.0 | 1,594 | 1.736 | 71.5 | 56.3 | 128.3 | 35.4 |

## Cold prefill by context length (tok/s)

| Pack | Release | Measured | 32K | 64K | 128K | 192K | 250K |
|---|---|---|---:|---:|---:|---:|---:|
| 4.05 | v0.38.1 | Nov 22, 2026 | 1,888 | 1,881 | 1,845 | 1,805 | 1,701 |
| 4.05 | v0.38.0 | Nov 22, 2026 | 1,883 | 1,867 | 1,834 | 1,786 | 1,676 |
| 4.05 | v0.33.0 | Nov 3, 2026 | 1,830 | 1,814 | 1,752 | 1,692 | 1,622 |
| 4.05 | v0.32.0 | Nov 3, 2026 | 1,808 | 1,801 | 1,760 | 1,723 | 1,592 |
| 4.05 | v0.31.0 | Nov 1 to 2, 2026 | 1,749 | 1,754 | 1,723 | 1,683 | 1,576 |
| 4.05 | v0.30.0 | Oct 31, 2026 | 1,700 | 1,701 | 1,664 | 1,624 | 1,536 |
| 4.05 | v0.26.0 | Oct 28 to 29, 2026 | 1,626 | 1,604 | 1,567 | 1,526 | 1,448 |
| 4.05 | v0.22.0 | Oct 28 to 29, 2026 | 1,578 | 1,564 | 1,526 | 1,490 | 1,416 |
| 3.05 | v0.38.1 | Nov 22, 2026 | 1,910 | 1,895 | 1,862 | 1,823 | 1,705 |
| 3.05 | v0.38.0 | Nov 22, 2026 | 1,898 | 1,886 | 1,853 | 1,813 | 1,696 |
| 3.05 | v0.33.0 | Nov 3, 2026 | 1,855 | 1,840 | 1,807 | 1,767 | 1,648 |
| 3.05 | v0.32.0 | Nov 3, 2026 | 1,828 | 1,802 | 1,761 | 1,731 | 1,620 |
| 3.05 | v0.31.0 | Nov 2, 2026 | 1,846 | 1,824 | 1,795 | 1,755 | 1,644 |
| 3.05 | v0.30.0 | Nov 1, 2026 | 1,799 | 1,767 | 1,722 | 1,677 | 1,589 |
| 3.05 | v0.26.0 | Nov 1 to 2, 2026 | 1,598 | 1,601 | 1,567 | 1,530 | 1,448 |
| 3.05 | v0.22.0 | Oct 28 to 29, 2026 | 1,661 | 1,640 | 1,607 | 1,580 | 1,494 |

## Decode by context length (tok/s)

| Pack | Release | Measured | 32K | 64K | 128K | 192K | 250K |
|---|---|---|---:|---:|---:|---:|---:|
| 4.05 | v0.38.1 | Nov 22, 2026 | 73.0 | 70.6 | 68.9 | 67.1 | 67.4 |
| 4.05 | v0.38.0 | Nov 22, 2026 | 71.5 | 69.3 | 72.3 | 66.5 | 67.4 |
| 4.05 | v0.33.0 | Nov 3, 2026 | 71.7 | 68.8 | 66.9 | 64.7 | 62.2 |
| 4.05 | v0.32.0 | Nov 3, 2026 | 71.8 | 68.6 | 66.8 | 64.7 | 64.3 |
| 4.05 | v0.31.0 | Nov 1, 2026 | 69.0 | 66.6 | 63.9 | 62.9 | 61.9 |
| 4.05 | v0.30.0 | Oct 31, 2026 | 68.4 | 66.1 | 63.3 | 61.9 | 61.2 |
| 4.05 | v0.26.0 | Oct 29, 2026 | 58.4 | 55.4 | 53.8 | 53.9 | 57.9 |
| 4.05 | v0.22.0 | Oct 29, 2026 | 64.6 | 56.1 | 57.3 | 54.0 | 58.1 |
| 3.05 | v0.38.1 | Nov 22, 2026 | 79.4 | 78.1 | 73.6 | 73.3 | 67.5 |
| 3.05 | v0.38.0 | Nov 22, 2026 | 79.6 | 78.3 | 73.4 | 87.0 | 67.7 |
| 3.05 | v0.33.0 | Nov 3, 2026 | 77.0 | 74.0 | 73.9 | 71.2 | 69.1 |
| 3.05 | v0.32.0 | Nov 3, 2026 | 75.7 | 72.5 | 71.8 | 69.6 | 67.3 |
| 3.05 | v0.31.0 | Nov 2, 2026 | 73.8 | 71.0 | 70.9 | 68.8 | 66.7 |
| 3.05 | v0.30.0 | Nov 1, 2026 | 74.0 | 71.2 | 71.0 | 68.8 | 66.5 |
| 3.05 | v0.26.0 | Nov 1 to 2, 2026 | 73.2 | 70.3 | 71.0 | 67.4 | 66.4 |

## Long context (configuration `long`)

| Release | Pack | Measured | Row | Tokens | Value |
|---|---|---|---|---:|---:|
| v0.37.0 | 4.05 | Nov 11, 2026 | Cold prompt: first token | 250,000 | 149.0 s |
| v0.37.0 | 4.05 | Nov 11, 2026 | Cold prompt: first token | 490,000 | 340.4 s |
| v0.37.0 | 4.05 | Nov 11, 2026 | Cold prompt: prefill | 250,000 | 1,677.6 tok/s |
| v0.37.0 | 4.05 | Nov 11, 2026 | Cold prompt: prefill | 490,000 | 1,436.2 tok/s |
| v0.37.0 | 4.05 | Nov 11, 2026 | Decode, code | 250,000 | 67.7 tok/s |
| v0.37.0 | 4.05 | Nov 11, 2026 | Decode, code | 490,000 | 59.8 tok/s |
| v0.37.0 | 4.05 | Nov 11, 2026 | Decode, tool call | 250,000 | 97.0 tok/s |
| v0.37.0 | 4.05 | Nov 11, 2026 | Decode, tool call | 490,000 | 100.2 tok/s |
| v0.38.0 | 4.05 | Nov 22, 2026 | Cold prompt: first token | 250,000 | 147.6 s |
| v0.38.0 | 4.05 | Nov 22, 2026 | Cold prompt: first token | 490,000 | 338.2 s |
| v0.38.0 | 4.05 | Nov 22, 2026 | Cold prompt: prefill | 250,000 | 1,693.5 tok/s |
| v0.38.0 | 4.05 | Nov 22, 2026 | Cold prompt: prefill | 490,000 | 1,445.6 tok/s |
| v0.38.0 | 4.05 | Nov 22, 2026 | Decode, code | 250,000 | 64.3 tok/s |
| v0.38.0 | 4.05 | Nov 22, 2026 | Decode, code | 490,000 | 63.6 tok/s |
| v0.38.0 | 4.05 | Nov 22, 2026 | Decode, tool call | 250,000 | 105.1 tok/s |
| v0.38.0 | 4.05 | Nov 22, 2026 | Decode, tool call | 490,000 | 99.1 tok/s |
| v0.38.1 | 4.05 | Nov 22, 2026 | Cold prompt: first token | 250,000 | 148.0 s |
| v0.38.1 | 4.05 | Nov 22, 2026 | Cold prompt: first token | 490,000 | 342.4 s |
| v0.38.1 | 4.05 | Nov 22, 2026 | Cold prompt: prefill | 250,000 | 1,688.9 tok/s |
| v0.38.1 | 4.05 | Nov 22, 2026 | Cold prompt: prefill | 490,000 | 1,427.8 tok/s |
| v0.38.1 | 4.05 | Nov 22, 2026 | Decode, code | 250,000 | 64.1 tok/s |
| v0.38.1 | 4.05 | Nov 22, 2026 | Decode, code | 490,000 | 63.4 tok/s |
| v0.38.1 | 4.05 | Nov 22, 2026 | Decode, tool call | 250,000 | 104.9 tok/s |
| v0.38.1 | 4.05 | Nov 22, 2026 | Decode, tool call | 490,000 | 98.9 tok/s |

¹ Notes:

- v0.33.0, 4.05 pack, Decode at 32K, 1 session, code: from the run by context length, not from the ladder run
- v0.32.0, 4.05 pack, Decode at 32K, 1 session, code: from the run by context length, not from the ladder run
- v0.31.0, 4.05 pack, Decode at 32K, 1 session, code: from the run by context length, not from the ladder run
- v0.30.0, 4.05 pack, Decode at 32K, 1 session, code: from the run by context length, not from the ladder run
- v0.26.0, 4.05 pack, Decode at 32K, 1 session, code: 2 low samples; from the run by context length, not from the ladder run
- v0.22.0, 4.05 pack, Decode at 32K, 1 session, code: from the run by context length, not from the ladder run
- v0.32.0, 4.05 pack, Decode, 4 sessions, total: no value: one session of each run stopped after 13 tokens
- v0.32.0, 3.05 pack, Decode, 4 sessions, total: 1 run
- v0.37.0, 4.05 pack, Cold prompt: first token, 250K: plain positions
- v0.37.0, 4.05 pack, Cold prompt: first token, 490K: YaRN
- v0.37.0, 4.05 pack, Cold prompt: prefill, 250K: plain positions
- v0.37.0, 4.05 pack, Cold prompt: prefill, 490K: YaRN
- v0.37.0, 4.05 pack, Decode, code, 250K: plain positions
- v0.37.0, 4.05 pack, Decode, code, 490K: YaRN
- v0.37.0, 4.05 pack, Decode, tool call, 250K: plain positions
- v0.37.0, 4.05 pack, Decode, tool call, 490K: YaRN
- v0.26.0, 3.05 pack, Cold prefill by context length, 32K: the number of runs is not stated for this line
- v0.26.0, 3.05 pack, Cold prefill by context length, 64K: the number of runs is not stated for this line
- v0.26.0, 3.05 pack, Cold prefill by context length, 128K: the number of runs is not stated for this line
- v0.26.0, 3.05 pack, Cold prefill by context length, 192K: the number of runs is not stated for this line
- v0.26.0, 3.05 pack, Cold prefill by context length, 250K: the number of runs is not stated for this line
- v0.22.0, 4.05 pack, Decode by context length, 32K: 4 samples
- v0.22.0, 4.05 pack, Decode by context length, 64K: 3 samples
- v0.22.0, 4.05 pack, Decode by context length, 128K: 4 samples
- v0.22.0, 4.05 pack, Decode by context length, 192K: 3 samples
- v0.22.0, 4.05 pack, Decode by context length, 250K: 4 samples
- v0.26.0, 4.05 pack, Decode by context length, 32K: 4 samples
- v0.26.0, 4.05 pack, Decode by context length, 64K: 3 samples
- v0.26.0, 4.05 pack, Decode by context length, 128K: 4 samples
- v0.26.0, 4.05 pack, Decode by context length, 192K: 3 samples
- v0.26.0, 4.05 pack, Decode by context length, 250K: 4 samples
- v0.30.0, 4.05 pack, Decode by context length, 32K: 6 samples
- v0.30.0, 4.05 pack, Decode by context length, 64K: 5 samples
- v0.30.0, 4.05 pack, Decode by context length, 128K: 6 samples
- v0.30.0, 4.05 pack, Decode by context length, 192K: 5 samples
- v0.30.0, 4.05 pack, Decode by context length, 250K: 6 samples
- v0.31.0, 4.05 pack, Decode by context length, 32K: 6 samples
- v0.31.0, 4.05 pack, Decode by context length, 64K: 5 samples
- v0.31.0, 4.05 pack, Decode by context length, 128K: 6 samples
- v0.31.0, 4.05 pack, Decode by context length, 192K: 5 samples
- v0.31.0, 4.05 pack, Decode by context length, 250K: 6 samples
- v0.32.0, 4.05 pack, Decode by context length, 32K: 6 samples
- v0.32.0, 4.05 pack, Decode by context length, 64K: 5 samples
- v0.32.0, 4.05 pack, Decode by context length, 128K: 6 samples
- v0.32.0, 4.05 pack, Decode by context length, 192K: 5 samples
- v0.32.0, 4.05 pack, Decode by context length, 250K: 6 samples
- v0.33.0, 4.05 pack, Decode by context length, 32K: 6 samples
- v0.33.0, 4.05 pack, Decode by context length, 64K: 5 samples
- v0.33.0, 4.05 pack, Decode by context length, 128K: 6 samples
- v0.33.0, 4.05 pack, Decode by context length, 192K: 5 samples
- v0.33.0, 4.05 pack, Decode by context length, 250K: 6 samples
- v0.26.0, 3.05 pack, Decode by context length, 32K: 4 samples
- v0.26.0, 3.05 pack, Decode by context length, 64K: 4 samples
- v0.26.0, 3.05 pack, Decode by context length, 128K: 6 samples
- v0.26.0, 3.05 pack, Decode by context length, 192K: 5 samples
- v0.26.0, 3.05 pack, Decode by context length, 250K: 5 samples
- v0.30.0, 3.05 pack, Decode by context length, 32K: 4 samples
- v0.30.0, 3.05 pack, Decode by context length, 64K: 4 samples
- v0.30.0, 3.05 pack, Decode by context length, 128K: 6 samples
- v0.30.0, 3.05 pack, Decode by context length, 192K: 5 samples
- v0.30.0, 3.05 pack, Decode by context length, 250K: 5 samples
- v0.31.0, 3.05 pack, Decode by context length, 32K: 4 samples
- v0.31.0, 3.05 pack, Decode by context length, 64K: 4 samples
- v0.31.0, 3.05 pack, Decode by context length, 128K: 6 samples
- v0.31.0, 3.05 pack, Decode by context length, 192K: 5 samples
- v0.31.0, 3.05 pack, Decode by context length, 250K: 5 samples
- v0.32.0, 3.05 pack, Decode by context length, 32K: 4 samples
- v0.32.0, 3.05 pack, Decode by context length, 64K: 4 samples
- v0.32.0, 3.05 pack, Decode by context length, 128K: 6 samples
- v0.32.0, 3.05 pack, Decode by context length, 192K: 5 samples
- v0.32.0, 3.05 pack, Decode by context length, 250K: 5 samples
- v0.33.0, 3.05 pack, Decode by context length, 32K: 4 samples
- v0.33.0, 3.05 pack, Decode by context length, 64K: 4 samples
- v0.33.0, 3.05 pack, Decode by context length, 128K: 6 samples
- v0.33.0, 3.05 pack, Decode by context length, 192K: 5 samples
- v0.33.0, 3.05 pack, Decode by context length, 250K: 5 samples
- v0.37.3, 4.05 pack, Decode, 4 sessions, total: 5 of 6 texts gave a value
- v0.38.0, 4.05 pack, Decode, 4 sessions, total: 5 of 6 texts gave a value
- v0.38.1, 4.05 pack, Decode, 4 sessions, total: 5 of 6 texts gave a value
- v0.38.0, 4.05 pack, Decode by context length, 32K: 5 samples
- v0.38.0, 4.05 pack, Decode by context length, 64K: 5 samples
- v0.38.0, 4.05 pack, Decode by context length, 128K: 6 samples
- v0.38.0, 4.05 pack, Decode by context length, 192K: 6 samples
- v0.38.0, 4.05 pack, Decode by context length, 250K: 6 samples
- v0.38.0, 3.05 pack, Decode by context length, 32K: 4 samples
- v0.38.0, 3.05 pack, Decode by context length, 64K: 4 samples
- v0.38.0, 3.05 pack, Decode by context length, 128K: 6 samples
- v0.38.0, 3.05 pack, Decode by context length, 192K: 4 samples
- v0.38.0, 3.05 pack, Decode by context length, 250K: 6 samples
- v0.38.0, 4.05 pack, Cold prompt: first token, 250K: plain positions
- v0.38.0, 4.05 pack, Cold prompt: first token, 490K: YaRN
- v0.38.0, 4.05 pack, Cold prompt: prefill, 250K: plain positions
- v0.38.0, 4.05 pack, Cold prompt: prefill, 490K: YaRN
- v0.38.0, 4.05 pack, Decode, code, 250K: plain positions
- v0.38.0, 4.05 pack, Decode, code, 490K: YaRN
- v0.38.0, 4.05 pack, Decode, tool call, 250K: plain positions
- v0.38.0, 4.05 pack, Decode, tool call, 490K: YaRN
- v0.38.1, 4.05 pack, Decode by context length, 32K: 6 samples
- v0.38.1, 4.05 pack, Decode by context length, 64K: 5 samples
- v0.38.1, 4.05 pack, Decode by context length, 128K: 6 samples
- v0.38.1, 4.05 pack, Decode by context length, 192K: 5 samples
- v0.38.1, 4.05 pack, Decode by context length, 250K: 6 samples
- v0.38.1, 3.05 pack, Decode by context length, 32K: 4 samples
- v0.38.1, 3.05 pack, Decode by context length, 64K: 4 samples
- v0.38.1, 3.05 pack, Decode by context length, 128K: 5 samples
- v0.38.1, 3.05 pack, Decode by context length, 192K: 4 samples
- v0.38.1, 3.05 pack, Decode by context length, 250K: 6 samples
- v0.38.1, 4.05 pack, Cold prompt: first token, 250K: plain positions
- v0.38.1, 4.05 pack, Cold prompt: first token, 490K: YaRN
- v0.38.1, 4.05 pack, Cold prompt: prefill, 250K: plain positions
- v0.38.1, 4.05 pack, Cold prompt: prefill, 490K: YaRN
- v0.38.1, 4.05 pack, Decode, code, 250K: plain positions
- v0.38.1, 4.05 pack, Decode, code, 490K: YaRN
- v0.38.1, 4.05 pack, Decode, tool call, 250K: plain positions
- v0.38.1, 4.05 pack, Decode, tool call, 490K: YaRN
- v0.38.1, 4.05 pack, Cold 32K prompt: prefill: release ladder (prompts of the v0.33.0 ladder), not the standing eval texts
- v0.38.1, 3.05 pack, Cold 32K prompt: prefill: release ladder (prompts of the v0.33.0 ladder), not the standing eval texts
