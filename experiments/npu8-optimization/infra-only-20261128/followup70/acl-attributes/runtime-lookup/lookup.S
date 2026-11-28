// AArch64 no-device microbenchmark. Both wrappers use the same caller frame.
// Only the positively identified non-null branch is represented; null is rejected.
.text
.p2align 4
.global original_lookup
.type original_lookup,%function
original_lookup:
    stp x29,x30,[sp,#-16]!
    mov x29,sp
    adrp x3,:got:runtime_word
    ldr x3,[x3,:got_lo12:runtime_word]
    ldr x0,[x3]
    bl lookup_nonnull
    ldp x29,x30,[sp],#16
    ret
.size original_lookup,.-original_lookup

.p2align 4
.global direct_lookup
.type direct_lookup,%function
direct_lookup:
    stp x29,x30,[sp,#-16]!
    mov x29,sp
    cbz x1,invalid
    mov x0,x1
    ldp x29,x30,[sp],#16
    ret
.size direct_lookup,.-direct_lookup

.p2align 4
.type lookup_nonnull,%function
lookup_nonnull:
    stp x29,x30,[sp,#-96]!
    mov x29,sp
    stp x19,x20,[sp,#16]
    adrp x19,:got:__stack_chk_guard
    str x21,[sp,#32]
    mov x21,x0
    ldr x0,[x19,:got_lo12:__stack_chk_guard]
    mov x20,x1
    ldr x1,[x0]
    str x1,[x29,#88]
    mov x1,#0
    cbz x20,invalid
    ldr x19,[x19,:got_lo12:__stack_chk_guard]
    mov x0,x20
    ldr x2,[x29,#88]
    ldr x1,[x19]
    eor x1,x2,x1
    cbnz x1,invalid
    ldp x19,x20,[sp,#16]
    ldr x21,[sp,#32]
    ldp x29,x30,[sp],#96
    ret
.size lookup_nonnull,.-lookup_nonnull
invalid:
    brk #0
.section .note.GNU-stack,"",@progbits
