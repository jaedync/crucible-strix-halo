// CPU checks of the master numerics switch: CRUCIBLE_EXACT=1
// hides every numerics switch, and a bad value fails instead of keeping option B.
#include "../engine/layers/numerics.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace crucible;
// Fixture names stay outside the switch registry (engine/switches.h).
constexpr SwitchName Fixture(const char* name) { return SwitchName::Unchecked(name); }
template<class F> void Reject(F run) {
  bool rejected=false;
  try { run(); } catch (const std::invalid_argument&) { rejected=true; }
  assert(rejected);
}
int main() {
  unsetenv("CRUCIBLE_EXACT");
  setenv("CRUCIBLE_TEST_SWITCH","1",1);
  setenv("CRUCIBLE_TEST_INT","0",1);
  setenv("CRUCIBLE_TEST_LANDED","0",1);
  setenv("CRUCIBLE_TEST_ZERO","0",1);
  setenv("CRUCIBLE_TEST_EMPTY","",1);
  unsetenv("CRUCIBLE_TEST_UNSET");
  for (const char* b : {static_cast<const char*>(nullptr),"0"}) {
    if (b) setenv("CRUCIBLE_EXACT",b,1);
    assert(!NumericsExact());
    assert(std::strcmp(NumericsEnv(Fixture("CRUCIBLE_TEST_SWITCH")),"1")==0);
    assert(NumericsOn(Fixture("CRUCIBLE_TEST_SWITCH")));
    // An opt-in switch set to 0 (or empty) is off, like an unset one.
    assert(!NumericsOn(Fixture("CRUCIBLE_TEST_ZERO")) && !NumericsOn(Fixture("CRUCIBLE_TEST_EMPTY")) && !NumericsOn(Fixture("CRUCIBLE_TEST_UNSET")));
    assert(NumericsInt(Fixture("CRUCIBLE_TEST_INT"),1,0,1)==0);
    assert(!NumericsDefaultOn(Fixture("CRUCIBLE_TEST_LANDED")));
    assert(NumericsDefaultOn(Fixture("CRUCIBLE_TEST_UNSET")));
    assert(std::strncmp(NumericsModeText(),"B ",2)==0);
  }
  setenv("CRUCIBLE_EXACT","1",1);
  assert(NumericsExact());
  assert(NumericsEnv(Fixture("CRUCIBLE_TEST_SWITCH"))==nullptr);
  assert(!NumericsOn(Fixture("CRUCIBLE_TEST_SWITCH")));
  assert(NumericsInt(Fixture("CRUCIBLE_TEST_INT"),1,0,1)==1);
  // Option A ignores an out-of-range value too: the switch is not read.
  setenv("CRUCIBLE_TEST_INT","9",1);
  assert(NumericsInt(Fixture("CRUCIBLE_TEST_INT"),1,0,1)==1);
  assert(!NumericsDefaultOn(Fixture("CRUCIBLE_TEST_LANDED")));
  assert(!NumericsDefaultOn(Fixture("CRUCIBLE_TEST_UNSET")));
  assert(std::strncmp(NumericsModeText(),"A ",2)==0);
  for (const char* bad : {"","yes","true","2"," 1"}) {
    setenv("CRUCIBLE_EXACT",bad,1);
    Reject([] { NumericsExact(); });
    Reject([] { NumericsEnv(Fixture("CRUCIBLE_TEST_SWITCH")); });
  }
  unsetenv("CRUCIBLE_EXACT");
  // An opt-in switch with a typo must fail, not stay off without a message.
  for (const char* bad : {"yes","true","2"," 1","on"}) {
    setenv("CRUCIBLE_TEST_SWITCH",bad,1);
    Reject([] { NumericsOn(Fixture("CRUCIBLE_TEST_SWITCH")); });
  }
  setenv("CRUCIBLE_TEST_SWITCH","1",1);
  assert(NumericsOn(Fixture("CRUCIBLE_TEST_SWITCH")));
  setenv("CRUCIBLE_TEST_INT","9",1);
  Reject([] { NumericsInt(Fixture("CRUCIBLE_TEST_INT"),1,0,1); });
  // Startup check: a bad value of any opt-in switch stops the process before a request, also in option A and for the
  // switches that a function-local static reads at the first long prompt.
  const char* opt_in[] = {"CRUCIBLE_PERSISTENT_GDN","CRUCIBLE_PF_EXACT_SPLIT","CRUCIBLE_PF_GDN_EAGER",
                          "CRUCIBLE_PF_GDN_MICRO","CRUCIBLE_PF_HC_FUSED","CRUCIBLE_PF_QSA_ROWS"};
  unsetenv("CRUCIBLE_TEST_SWITCH"); unsetenv("CRUCIBLE_TEST_INT");
  ValidateNumericsSwitches();
  for (const char* name : opt_in) {
    for (const char* exact : {"","1"}) {
      if (*exact) setenv("CRUCIBLE_EXACT",exact,1); else unsetenv("CRUCIBLE_EXACT");
      for (const char* good : {"","0","1"}) { setenv(name,good,1); ValidateNumericsSwitches(); }
      for (const char* bad : {"true","2"," 1","on","yes"}) {
        setenv(name,bad,1);
        Reject([] { ValidateNumericsSwitches(); });
      }
    }
    unsetenv(name);
  }
  unsetenv("CRUCIBLE_EXACT");
  // A landed option-B integer default (CRUCIBLE_HC_WMMA 1 -> 2): unset gives the new default, a set value applies,
  // and option A gives the baseline's value without a read of the switch.
  unsetenv("CRUCIBLE_TEST_INT");
  assert(NumericsDefaultInt(Fixture("CRUCIBLE_TEST_INT"),2,1,0,2)==2);
  setenv("CRUCIBLE_TEST_INT","1",1);
  assert(NumericsDefaultInt(Fixture("CRUCIBLE_TEST_INT"),2,1,0,2)==1);
  setenv("CRUCIBLE_TEST_INT","0",1);
  assert(NumericsDefaultInt(Fixture("CRUCIBLE_TEST_INT"),2,1,0,2)==0);
  setenv("CRUCIBLE_TEST_INT","9",1);
  Reject([] { NumericsDefaultInt(Fixture("CRUCIBLE_TEST_INT"),2,1,0,2); });
  setenv("CRUCIBLE_EXACT","1",1);
  assert(NumericsDefaultInt(Fixture("CRUCIBLE_TEST_INT"),2,1,0,2)==1);
  unsetenv("CRUCIBLE_TEST_INT");
  assert(NumericsDefaultInt(Fixture("CRUCIBLE_TEST_INT"),2,1,0,2)==1);
  unsetenv("CRUCIBLE_EXACT");
  ValidateNumericsSwitches();
  return 0;
}
