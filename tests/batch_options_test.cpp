#include "../engine/cli/batch_options.h"

#include <cassert>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <vector>

crucible::BatchOptions Parse(std::vector<std::string> options) {
  crucible::BatchOptions out;
  std::vector<char*> argv{(char*)"crucible", (char*)"gen", (char*)"checkpoint"};
  for (auto& arg : options) argv.push_back(arg.data());
  const auto error = crucible::ParseBatchOptions((int)argv.size(), argv.data(), out);
  if (!error.empty()) throw std::invalid_argument(error);
  return out;
}

int main() {
  char path[] = "/tmp/crucible-batch-options-XXXXXX";
  const int fd = mkstemp(path);
  assert(fd >= 0);
  close(fd);
  {
    std::ofstream list(path);
    list << "1,2\n\n3,248044\n";
  }
  const auto list = Parse({"--batch-list", path, "--spec", "0"});
  assert(list.streams == 2 && list.prompts[1] == std::vector<int>({3, 248044}));
  {
    std::ofstream ids(path);
    ids << "1,2,3\n";
  }
  const auto file = Parse({"--ids-file", path, "--streams", "1"});
  assert(file.prompts.size() == 1 && file.prompts[0] == std::vector<int>({1, 2, 3}));
  std::remove(path);
  const auto good = Parse({"--streams", "8", "--ids", "1,2,248044", "--n", "16"});
  assert(good.streams == 8 && good.prompts.size() == 8 && good.n == 16);
  assert(good.prompts[3] == std::vector<int>({1, 2, 248044}));
  assert(Parse({"--streams", "2", "--ids", "1", "--prefill", "4096"}).prefill_rows == 4096);
  assert(Parse({"--streams", "2", "--ids", "1", "--prefill", "0"}).prefill_rows == 0);
  for (const auto& options : std::vector<std::vector<std::string>>{
           {"--streams", "0", "--ids", "1"}, {"--streams", "9", "--ids", "1"},
           {"--streams", "2", "--ids", "-1"}, {"--streams", "2", "--ids", "248320"},
           {"--streams", "2", "--ids", "1,,2"}, {"--streams", "2", "--ids", "1x"},
           {"--streams", "2", "--ids", "1", "--spec", "2"},
           {"--streams", "2", "--ids", "1", "--lookup", "1"},
           {"--streams", "2", "--ids", "1", "--n", "2x"},
           {"--streams", "2", "--ids", "1", "--n", "0"},
           {"--streams", "2", "--ids", "1", "--ctx", "1", "--n", "2"},
           {"--streams", "2", "--ids", "1", "--chunk", "9"},
           {"--streams", "2", "--ids", "1", "--prefill", "-1"},
           {"--streams", "2", "--ids", "1", "--prefill", "4097"},
           {"--streams", "2", "--ids", "1", "--ctx", "1048577"},
           {"--streams", "2", "--ids", "1", "--n", "999999999999999"},
           {"--batch-list", "missing", "--streams", "2"},
           {"--streams", "2", "--ids", "1", "--ids", "2"},
           {"--batch-list", "/missing/crucible/batch-list"}, {"--streams", "2"},
           {"--streams", "2", "--ids", "1", "--n"}}) {
    bool rejected = false;
    try { Parse(options); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  {
    // Leak guard (M5): the message names the file, never its folder (a list may sit in a private folder).
    std::string what;
    try { Parse({"--batch-list", "/missing/crucible/batch-list"}); } catch (const std::invalid_argument& error) { what = error.what(); }
    assert(what.find("batch-list") != std::string::npos);
    assert(what.find("/missing") == std::string::npos && what.find("crucible/") == std::string::npos);
  }
}
