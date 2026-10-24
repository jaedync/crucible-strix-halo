import hashlib
import pathlib
import subprocess
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]


class SharedFirstTests(unittest.TestCase):
    def test_rotation_keeps_all_fields_and_old_input(self):
        source=r'''
#include "engine/layers/moe_shared_first.h"
#include <cassert>
struct Item {int job,s,b0,nb;bool operator==(const Item&)const=default;};
int main() {
  for(int jobs:{2,22,82,162,324})for(int shared:{1,2}) {
    std::vector<Item> original;
    for(int job=0;job<jobs;++job)for(int b=0;b<40;b+=16)
      original.push_back({job,1,b,std::min(16,40-b)});
    const auto before=original;
    const int first=jobs-shared;
    const auto result=crucible::MoeSharedFirstItems(original,first);
    std::vector<Item> expected;
    for(const auto& item:original)if(item.job>=first)expected.push_back(item);
    for(const auto& item:original)if(item.job<first)expected.push_back(item);
    assert(result==expected);assert(original==before);
    assert(crucible::MoeSharedFirstItems(original,-1)==original);
    assert(crucible::MoeSharedFirstItems(original,jobs)==original);
    assert(crucible::MoeSharedFirstItems(original,0)==original);
  }
}
'''
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);(p/"test.cc").write_text(source)
            result=subprocess.run(["g++","-std=c++20","-O2","-I",str(ROOT),str(p/"test.cc"),"-o",str(p/"test")],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            subprocess.run([str(p/"test")],check=True)

    def test_device_header_matches_pinned_abliterate(self):
        # A moving remote ref cannot identify the device code used by this packet.
        # Repinned on purpose for the StoreY switch; the stock
        # instantiation passed tools/gates.py against crucible-t8 (gates-stock-c8).
        # Repinned on purpose by the decode-kernel work (input slots, chunked staging,
        # packed passes, padded last shared chunk); its gates run against
        # crucible-t12. Repinned on purpose by dead-kern104: the bench-only
        # kernels and their helpers moved to bench/kernels/moe_rows.h; every
        # remaining kernel keeps its ISA. Repinned on purpose
        # by packs: MoeFusedKernelWide (KS 7/8) and TileDwords; isa_same against train 23
        # shows every MoeFusedKernel unchanged. Repinned on purpose by packs
        # (KS 6 in MoeFusedKernelWide for 2.18 and 2.68).
        expected="3ea916587068bbe6f02c6d26497625c8f3f76149fc7408a3dfd7f99931d72685"
        self.assertEqual(hashlib.sha256((ROOT/"engine/layers/moe_rows.h").read_bytes()).hexdigest(),expected)


if __name__=="__main__":unittest.main()
