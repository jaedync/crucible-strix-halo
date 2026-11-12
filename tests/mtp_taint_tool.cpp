// A CPU stand-in for mtp_dump (tests/test_mtp_dump_taint.py): the same taint code (bench/mtp_taint.h) and the same
// output calls, with stack rows of fake bytes instead of the engine. Usage: mtp_taint_tool LIST OUTDIR [--check 1]
#include "../bench/mtp_taint.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (const char* bad = mtp_taint::DiagnosticSwitchInEnvironment()) {
    std::fprintf(stderr, "mtp_taint_tool: %s is a diagnostic switch\n", bad);
    return 2;
  }
  if (argc < 3) return 2;
  const bool check = argc >= 5 && std::string(argv[3]) == "--check" && std::string(argv[4]) != "0";
  const auto provenance = mtp_taint::ListProvenance(argv[1]);
  if (!mtp_taint::CheckOutputAllowed(provenance, check)) return 2;
  std::ifstream list(argv[1]);
  std::vector<std::pair<std::string, std::vector<int>>> names;
  for (std::string line; std::getline(list, line);) {
    const size_t sp = line.find(' ');
    if (line.empty() || line[0] == '#' || sp == std::string::npos) continue;
    std::vector<int> ids;
    std::stringstream rest(line.substr(sp + 1));
    for (std::string v; std::getline(rest, v, ',');) ids.push_back(std::stoi(v));
    std::string rows(ids.size() * 8, '\0');
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = char(i * 7 + ids.size());
    const std::string name = line.substr(0, sp);
    names.emplace_back(name, ids);
    if (!mtp_taint::WriteStk(std::string(argv[2]) + "/" + name + ".stk", rows.data(), rows.size(), provenance) ||
        !mtp_taint::WriteIds(std::string(argv[2]) + "/" + name + ".ids", ids, provenance))
      return 1;
  }
  return mtp_taint::FinishFolder(argv[2], provenance, names) ? 0 : 1;
}
