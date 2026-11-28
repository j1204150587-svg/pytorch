
bench:     file format elf64-littleaarch64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .text:

0000000000000ca0 <original_lookup>:
 ca0:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
 ca4:	910003fd 	mov	x29, sp
 ca8:	b0000083 	adrp	x3, 11000 <__FRAME_END__+0x10124>
 cac:	f947f063 	ldr	x3, [x3, #4064]
 cb0:	f9400060 	ldr	x0, [x3]
 cb4:	9400000b 	bl	ce0 <lookup_nonnull>
 cb8:	a8c17bfd 	ldp	x29, x30, [sp], #16
 cbc:	d65f03c0 	ret

Disassembly of section .fini:

bench:     file format elf64-littleaarch64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .text:

0000000000000cc0 <direct_lookup>:
 cc0:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
 cc4:	910003fd 	mov	x29, sp
 cc8:	b4000381 	cbz	x1, d38 <invalid>
 ccc:	aa0103e0 	mov	x0, x1
 cd0:	a8c17bfd 	ldp	x29, x30, [sp], #16
 cd4:	d65f03c0 	ret

Disassembly of section .fini:

bench:     file format elf64-littleaarch64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .text:

0000000000000ce0 <lookup_nonnull>:
 ce0:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
 ce4:	910003fd 	mov	x29, sp
 ce8:	a90153f3 	stp	x19, x20, [sp, #16]
 cec:	b0000093 	adrp	x19, 11000 <__FRAME_END__+0x10124>
 cf0:	f90013f5 	str	x21, [sp, #32]
 cf4:	aa0003f5 	mov	x21, x0
 cf8:	f947ee60 	ldr	x0, [x19, #4056]
 cfc:	aa0103f4 	mov	x20, x1
 d00:	f9400001 	ldr	x1, [x0]
 d04:	f9002fa1 	str	x1, [x29, #88]
 d08:	d2800001 	mov	x1, #0x0                   	// #0
 d0c:	b4000174 	cbz	x20, d38 <invalid>
 d10:	f947ee73 	ldr	x19, [x19, #4056]
 d14:	aa1403e0 	mov	x0, x20
 d18:	f9402fa2 	ldr	x2, [x29, #88]
 d1c:	f9400261 	ldr	x1, [x19]
 d20:	ca010041 	eor	x1, x2, x1
 d24:	b50000a1 	cbnz	x1, d38 <invalid>
 d28:	a94153f3 	ldp	x19, x20, [sp, #16]
 d2c:	f94013f5 	ldr	x21, [sp, #32]
 d30:	a8c67bfd 	ldp	x29, x30, [sp], #96
 d34:	d65f03c0 	ret

Disassembly of section .fini:
