// The grammar compile helper program (constraint_helper.h) as bytes in the engine library. The Makefile passes the
// path of the built helper as GRAMMAR_HELPER_PATH.
  .section .rodata
  .balign 64
  .globl crucible_grammar_helper_image_begin
crucible_grammar_helper_image_begin:
  .incbin GRAMMAR_HELPER_PATH
  .globl crucible_grammar_helper_image_end
crucible_grammar_helper_image_end:
  .section .note.GNU-stack,"",@progbits
