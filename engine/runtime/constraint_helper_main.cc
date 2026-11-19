// The grammar compile helper program (constraint_helper.h). The engine library holds its image and starts it; the
// program itself holds no image.
#include "constraint_helper.h"

namespace crucible {
std::span<const unsigned char> ConstraintHelperImage() { return {}; }
}  // namespace crucible

int main() { return crucible::ConstraintHelperMain(0); }
