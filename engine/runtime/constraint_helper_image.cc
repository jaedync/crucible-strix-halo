// The helper image of constraint_helper_image.S (constraint_helper.h).
#include "constraint_helper.h"

extern "C" const unsigned char crucible_grammar_helper_image_begin[];
extern "C" const unsigned char crucible_grammar_helper_image_end[];

namespace crucible {
std::span<const unsigned char> ConstraintHelperImage() {
  return {crucible_grammar_helper_image_begin, crucible_grammar_helper_image_end};
}
}  // namespace crucible
