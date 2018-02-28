/*
 * Mesa 3-D graphics library
 *
 * Copyright 2012 Intel Corporation
 * Copyright 2013 Google
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Chad Versace <chad.versace@linux.intel.com>
 *    Frank Henigman <fjhenigman@google.com>
 */

#include <string.h>

#include "util/macros.h"

#include "brw_context.h"
#include "intel_tiled_memcpy.h"

#if defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

#define FILE_DEBUG_FLAG DEBUG_TEXTURE

#define ALIGN_DOWN(a, b) ROUND_DOWN_TO(a, b)
#define ALIGN_UP(a, b) ALIGN(a, b)

/* Tile dimensions.  Width and span are in bytes, height is in pixels (i.e.
 * unitless).  A "span" is the most number of bytes we can copy from linear
 * to tiled without needing to calculate a new destination address.
 */
static const uint32_t xtile_width = 512;
static const uint32_t xtile_height = 8;
static const uint32_t xtile_span = 64;
static const uint32_t ytile_width = 128;
static const uint32_t ytile_height = 32;
static const uint32_t ytile_span = 16;
static const uint32_t std_ytile128_width = 256;
static const uint32_t std_ytile128_height = 16;
static const uint32_t std_ytile32_width = 128;
static const uint32_t std_ytile32_height = 32;
static const uint32_t std_ytile8_width = 64;
static const uint32_t std_ytile8_height = 64;

static inline uint32_t
ror(uint32_t n, uint32_t d)
{
   return (n >> d) | (n << (32 - d));
}

static inline uint32_t
bswap32(uint32_t n)
{
#if defined(HAVE___BUILTIN_BSWAP32)
   return __builtin_bswap32(n);
#else
   return (n >> 24) |
          ((n >> 8) & 0x0000ff00) |
          ((n << 8) & 0x00ff0000) |
          (n << 24);
#endif
}

/**
 * Copy RGBA to BGRA - swap R and B.
 */
static inline void *
rgba8_copy(void *dst, const void *src, size_t bytes)
{
   uint32_t *d = dst;
   uint32_t const *s = src;

   assert(bytes % 4 == 0);

   while (bytes >= 4) {
      *d = ror(bswap32(*s), 8);
      d += 1;
      s += 1;
      bytes -= 4;
   }
   return dst;
}

#ifdef __SSSE3__
static const uint8_t rgba8_permutation[16] =
   { 2,1,0,3, 6,5,4,7, 10,9,8,11, 14,13,12,15 };

static inline void
rgba8_copy_16_aligned_dst(void *dst, const void *src)
{
   _mm_store_si128(dst,
                   _mm_shuffle_epi8(_mm_loadu_si128(src),
                                    *(__m128i *)rgba8_permutation));
}

static inline void
rgba8_copy_16_aligned_src(void *dst, const void *src)
{
   _mm_storeu_si128(dst,
                    _mm_shuffle_epi8(_mm_load_si128(src),
                                     *(__m128i *)rgba8_permutation));
}

#elif defined(__SSE2__)
static inline void
rgba8_copy_16_aligned_dst(void *dst, const void *src)
{
   __m128i srcreg, dstreg, agmask, ag, rb, br;

   agmask = _mm_set1_epi32(0xFF00FF00);
   srcreg = _mm_loadu_si128((__m128i *)src);

   rb = _mm_andnot_si128(agmask, srcreg);
   ag = _mm_and_si128(agmask, srcreg);
   br = _mm_shufflehi_epi16(_mm_shufflelo_epi16(rb, _MM_SHUFFLE(2, 3, 0, 1)),
                            _MM_SHUFFLE(2, 3, 0, 1));
   dstreg = _mm_or_si128(ag, br);

   _mm_store_si128((__m128i *)dst, dstreg);
}

static inline void
rgba8_copy_16_aligned_src(void *dst, const void *src)
{
   __m128i srcreg, dstreg, agmask, ag, rb, br;

   agmask = _mm_set1_epi32(0xFF00FF00);
   srcreg = _mm_load_si128((__m128i *)src);

   rb = _mm_andnot_si128(agmask, srcreg);
   ag = _mm_and_si128(agmask, srcreg);
   br = _mm_shufflehi_epi16(_mm_shufflelo_epi16(rb, _MM_SHUFFLE(2, 3, 0, 1)),
                            _MM_SHUFFLE(2, 3, 0, 1));
   dstreg = _mm_or_si128(ag, br);

   _mm_storeu_si128((__m128i *)dst, dstreg);
}
#endif

/**
 * Copy RGBA to BGRA - swap R and B, with the destination 16-byte aligned.
 */
static inline void *
rgba8_copy_aligned_dst(void *dst, const void *src, size_t bytes)
{
   assert(bytes == 0 || !(((uintptr_t)dst) & 0xf));

#if defined(__SSSE3__) || defined(__SSE2__)
   if (bytes == 64) {
      rgba8_copy_16_aligned_dst(dst +  0, src +  0);
      rgba8_copy_16_aligned_dst(dst + 16, src + 16);
      rgba8_copy_16_aligned_dst(dst + 32, src + 32);
      rgba8_copy_16_aligned_dst(dst + 48, src + 48);
      return dst;
   }

   while (bytes >= 16) {
      rgba8_copy_16_aligned_dst(dst, src);
      src += 16;
      dst += 16;
      bytes -= 16;
   }
#endif

   rgba8_copy(dst, src, bytes);

   return dst;
}

/**
 * Copy RGBA to BGRA - swap R and B, with the source 16-byte aligned.
 */
static inline void *
rgba8_copy_aligned_src(void *dst, const void *src, size_t bytes)
{
   assert(bytes == 0 || !(((uintptr_t)src) & 0xf));

#if defined(__SSSE3__) || defined(__SSE2__)
   if (bytes == 64) {
      rgba8_copy_16_aligned_src(dst +  0, src +  0);
      rgba8_copy_16_aligned_src(dst + 16, src + 16);
      rgba8_copy_16_aligned_src(dst + 32, src + 32);
      rgba8_copy_16_aligned_src(dst + 48, src + 48);
      return dst;
   }

   while (bytes >= 16) {
      rgba8_copy_16_aligned_src(dst, src);
      src += 16;
      dst += 16;
      bytes -= 16;
   }
#endif

   rgba8_copy(dst, src, bytes);

   return dst;
}

/**
 * Each row from y0 to y1 is copied in three parts: [x0,x1), [x1,x2), [x2,x3).
 * These ranges are in bytes, i.e. pixels * bytes-per-pixel.
 * The first and last ranges must be shorter than a "span" (the longest linear
 * stretch within a tile) and the middle must equal a whole number of spans.
 * Ranges may be empty.  The region copied must land entirely within one tile.
 * 'dst' is the start of the tile and 'src' is the corresponding
 * address to copy from, though copying begins at (x0, y0).
 * To enable swizzling 'swizzle_bit' must be 1<<6, otherwise zero.
 * Swizzling flips bit 6 in the copy destination offset, when certain other
 * bits are set in it.
 */
typedef void (*tile_copy_fn)(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                             uint32_t y0, uint32_t y1,
                             char *dst, const char *src,
                             int32_t linear_pitch,
                             uint32_t swizzle_bit,
                             enum isl_tiling tiling,
                             int cpp,
                             mem_copy_fn mem_copy);

/* a tile_addr_fn returns the address of a 4kbyte tile (or a 4kbyte sub-tile
 * in larger tiling formats) which contains the points (x, y) within the given
 * tiled image.
 */
typedef char* (*tile_addr_fn)(uint32_t x, uint32_t y, char *src,
                              uint32_t src_pitch);

static char *
xtile_addr(uint32_t x, uint32_t y, char *src, uint32_t src_pitch)
{
   return src + (((y >> 3) * (src_pitch >> 9) + (x >> 9)) << 12);
}

static char *
ytile_addr(uint32_t x, uint32_t y, char *src, uint32_t src_pitch)
{
   return src + (((y >> 5) * (src_pitch >> 7) + (x >> 7)) << 12);
}

static char *
yf128_addr(uint32_t x, uint32_t y, char *src, uint32_t src_pitch)
{
   return src + (((y >> 4) * (src_pitch >> 8) + (x >> 8)) << 12);
}

static char *
yf32_addr(uint32_t x, uint32_t y, char *src, uint32_t src_pitch)
{
   return src + (((y >> 5) * (src_pitch >> 7) + (x >> 7)) << 12);
}

static char *
yf8_addr(uint32_t x, uint32_t y, char *src, uint32_t src_pitch)
{
   return src + (((y >> 6) * (src_pitch >> 6) + (x >> 6)) << 12);
}

static char *
ys128_addr(uint32_t x, uint32_t y, char *src, uint32_t src_pitch)
{
   return src + (((y & 0x10) << 8) + ((y & 0x20) << 9) + ((x & 0x100) << 5) +
                 ((x & 0x200) << 6) +
                 (((y >> 6) * (src_pitch >> 10) + (x >> 10)) << 16));
}

static char *
ys32_addr(uint32_t x, uint32_t y, char *src, uint32_t src_pitch)
{
   return src + (((y & 0x20) << 7) + ((y & 0x40) << 8) + ((x & 0x80) << 6) +
                 ((x & 0x100) << 7) +
                 (((y >> 7) * (src_pitch >> 9) + (x >> 9)) << 16));
}

static char *
ys8_addr(uint32_t x, uint32_t y, char *src, uint32_t src_pitch)
{
   return src + (((y & 0x40) << 6) + ((y & 0x80) << 7) + ((x & 0x40) << 7) +
                 ((x & 0x80) << 8) +
                 (((y >> 8) * (src_pitch >> 8) + (x >> 8)) << 16));
}

/**
 * Copy texture data from linear to X tile layout.
 *
 * \copydoc tile_copy_fn
 *
 * The mem_copy parameters allow the user to specify an alternative mem_copy
 * function that, for instance, may do RGBA -> BGRA swizzling.  The first
 * function must handle any memory alignment while the second function must
 * only handle 16-byte alignment in whichever side (source or destination) is
 * tiled.
 */
static inline void
linear_to_xtiled(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                 uint32_t y0, uint32_t y1,
                 char *dst, const char *src,
                 int32_t src_pitch,
                 uint32_t swizzle_bit,
                 mem_copy_fn mem_copy,
                 mem_copy_fn mem_copy_align16)
{
   /* The copy destination offset for each range copied is the sum of
    * an X offset 'x0' or 'xo' and a Y offset 'yo.'
    */
   uint32_t xo, yo;

   src += (ptrdiff_t)y0 * src_pitch;

   for (yo = y0 * xtile_width; yo < y1 * xtile_width; yo += xtile_width) {
      /* Bits 9 and 10 of the copy destination offset control swizzling.
       * Only 'yo' contributes to those bits in the total offset,
       * so calculate 'swizzle' just once per row.
       * Move bits 9 and 10 three and four places respectively down
       * to bit 6 and xor them.
       */
      uint32_t swizzle = ((yo >> 3) ^ (yo >> 4)) & swizzle_bit;

      mem_copy(dst + ((x0 + yo) ^ swizzle), src + x0, x1 - x0);

      for (xo = x1; xo < x2; xo += xtile_span) {
         mem_copy_align16(dst + ((xo + yo) ^ swizzle), src + xo, xtile_span);
      }

      mem_copy_align16(dst + ((xo + yo) ^ swizzle), src + x2, x3 - x2);

      src += src_pitch;
   }
}

/**
 * Copy texture data from linear to Y tile layout. This function tiles a
 * single 4KB portion of the tiling (even for the 64KB tiling variants)
 *
 * \copydoc tile_copy_fn
 */
static inline void
linear_to_ytiled(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                 uint32_t y0, uint32_t y3,
                 char *dst, const char *src,
                 int32_t src_pitch,
                 uint32_t swizzle_bit,
                 enum isl_tiling tiling,
                 int cpp,
                 mem_copy_fn mem_copy,
                 mem_copy_fn mem_copy_align16)
{
   /* The Y tilings are a family of different tilings with the following
    * linear-to-tiled address mapping for the low 12-bits of the tiled
    * addresses:
    *
    * Tiling        bpp        11 10  9  8  7  6  5  4  3  2  1  0
    * ------------------------------------------------------------
    * TileYF/TileYS 64 & 128   u7 v3 u6 v2 u5 u4 v1 v0 u3 u2 u1 u0
    * TileYF/TileYS 16 &  32   u6 v4 u5 v3 u4 v2 v1 v0 u3 u2 u1 u0
    * TileYF/TileYS        8   u5 v5 u4 v4 v3 v2 v1 v0 u3 u2 u1 u0
    * TileY                    u6 u5 u4 v4 v3 v2 v1 v0 u3 u2 u1 u0
    *
    * The low 6-bits (addressing 64B, one cache line) of the tiling is common
    * across all variants.
    */
   const uint32_t column_width = ytile_span;

   uint32_t y1 = MIN2(y3, ALIGN_UP(y0, 4));
   uint32_t y2 = MAX2(y1, ALIGN_DOWN(y3, 4));

   uint32_t xinc_16, x_mask;
   uint32_t yinc_1, yinc_4, y_mask;

   uint32_t xo0, xo1, xo2;
   uint32_t yo0, yo1, yo2;

#define YF_128_X(x) (((x) & 0xF) | (((x) & 0x30) << 2) | (((x) & 0x40) << 3) | (((x) & 0x80) << 4))
#define YF_32_X(x)  (((x) & 0xF) | (((x) & 0x10) << 3) | (((x) & 0x20) << 4) | (((x) & 0x40) << 5))
#define YF_8_X(x)   (((x) & 0xF) | (((x) & 0x10) << 5) | (((x) & 0x20) << 6))
#define Y0_X(x)     (((x) & 0xF) | (((x) & 0x70) << 5))

#define YF_128_Y(y) ((((y) & 0x03) << 4) | (((y) & 0x04) << 6) | (((y) & 0x08) << 7))
#define YF_32_Y(y)  ((((y) & 0x07) << 4) | (((y) & 0x08) << 5) | (((y) & 0x10) << 6))
#define YF_8_Y(y)   ((((y) & 0x1F) << 4) | (((y) & 0x20) << 5))
#define Y0_Y(y)     ((((y) & 0x1F) << 4))

#define TILING_INIT(TILING)                           \
   do {                                               \
      x_mask = TILING ## _X(~0);                      \
      y_mask = TILING ## _Y(~0);                      \
      xo0 = TILING ## _X(x0);                         \
      xo1 = TILING ## _X(x1);                         \
      xo2 = TILING ## _X(x2);                         \
      yo0 = TILING ## _Y(y0);                         \
      yo1 = TILING ## _Y(y1);                         \
      yo2 = TILING ## _Y(y2);                         \
      xinc_16 = (TILING ## _X(16) | ~x_mask) & 0xFFF; \
      yinc_1 = (TILING ## _Y(1) | ~y_mask) & 0xFFF;   \
      yinc_4 = (TILING ## _Y(4) | ~y_mask) & 0xFFF;   \
   } while (0)

   if (tiling == ISL_TILING_Y0)
      TILING_INIT(Y0);
   else if (cpp == 16 || cpp == 8)
      TILING_INIT(YF_128);
   else if (cpp == 4 || cpp == 2)
      TILING_INIT(YF_32);
   else if (cpp == 1)
      TILING_INIT(YF_8);
   else
      unreachable("not reached");

   /* Bit 9 of the destination offset control swizzling.
    * Only the X offset contributes to bit 9 of the total offset,
    * so swizzle can be calculated in advance for these X positions.
    * Move bit 9 three places down to bit 6.
    */
   uint32_t swizzle0 = (xo0 >> 3) & swizzle_bit;
   uint32_t swizzle1 = (xo1 >> 3) & swizzle_bit;
   if (tiling != ISL_TILING_Y0)
      swizzle0 = swizzle1 = swizzle_bit = 0;

   uint32_t x, y, yo;

   src += (ptrdiff_t)y0 * src_pitch;

   if (y0 != y1) {
      for (y = y0, yo = yo0; y < y1; y++, yo = (yo + yinc_1) & y_mask) {
         uint32_t xo = xo1;
         uint32_t swizzle = swizzle1;

         mem_copy(dst + ((xo0 + yo) ^ swizzle0), src + x0, x1 - x0);

         /* Step by spans/columns.  As it happens, the swizzle bit flips
          * at each step so we don't need to calculate it explicitly.
          */
         for (x = x1; x < x2; x += ytile_span) {
            mem_copy_align16(dst + ((xo + yo) ^ swizzle), src + x, ytile_span);
            xo = (xo + xinc_16) & x_mask;
            swizzle ^= swizzle_bit;
         }

         mem_copy_align16(dst + ((xo + yo) ^ swizzle), src + x2, x3 - x2);

         src += src_pitch;
      }
   }

   for (y = y1, yo = yo1; y < y2; y += 4, yo = (yo + yinc_4) & y_mask) {
      uint32_t xo = xo1;
      uint32_t swizzle = swizzle1;

      if (x0 != x1) {
         mem_copy(dst + ((xo0 + yo + 0 * column_width) ^ swizzle0), src + x0 + 0 * src_pitch, x1 - x0);
         mem_copy(dst + ((xo0 + yo + 1 * column_width) ^ swizzle0), src + x0 + 1 * src_pitch, x1 - x0);
         mem_copy(dst + ((xo0 + yo + 2 * column_width) ^ swizzle0), src + x0 + 2 * src_pitch, x1 - x0);
         mem_copy(dst + ((xo0 + yo + 3 * column_width) ^ swizzle0), src + x0 + 3 * src_pitch, x1 - x0);
      }

      /* Step by spans/columns.  As it happens, the swizzle bit flips
       * at each step so we don't need to calculate it explicitly.
       */
      for (x = x1; x < x2; x += ytile_span) {
         mem_copy_align16(dst + ((xo + yo + 0 * column_width) ^ swizzle), src + x + 0 * src_pitch, ytile_span);
         mem_copy_align16(dst + ((xo + yo + 1 * column_width) ^ swizzle), src + x + 1 * src_pitch, ytile_span);
         mem_copy_align16(dst + ((xo + yo + 2 * column_width) ^ swizzle), src + x + 2 * src_pitch, ytile_span);
         mem_copy_align16(dst + ((xo + yo + 3 * column_width) ^ swizzle), src + x + 3 * src_pitch, ytile_span);
         xo = (xo + xinc_16) & x_mask;
         swizzle ^= swizzle_bit;
      }

      if (x2 != x3) {
         mem_copy_align16(dst + ((xo + yo + 0 * column_width) ^ swizzle), src + x2 + 0 * src_pitch, x3 - x2);
         mem_copy_align16(dst + ((xo + yo + 1 * column_width) ^ swizzle), src + x2 + 1 * src_pitch, x3 - x2);
         mem_copy_align16(dst + ((xo + yo + 2 * column_width) ^ swizzle), src + x2 + 2 * src_pitch, x3 - x2);
         mem_copy_align16(dst + ((xo + yo + 3 * column_width) ^ swizzle), src + x2 + 3 * src_pitch, x3 - x2);
      }

      src += 4 * src_pitch;
   }

   if (y2 != y3) {
      for (y = y2, yo = yo2; y < y3; y++, yo = (yo + yinc_1) & y_mask) {
         uint32_t xo = xo1;
         uint32_t swizzle = swizzle1;

         mem_copy(dst + ((xo0 + yo) ^ swizzle0), src + x0, x1 - x0);

         /* Step by spans/columns.  As it happens, the swizzle bit flips
          * at each step so we don't need to calculate it explicitly.
          */
         for (x = x1; x < x2; x += ytile_span) {
            mem_copy_align16(dst + ((xo + yo) ^ swizzle), src + x, ytile_span);
            xo = (xo + xinc_16) & x_mask;
            swizzle ^= swizzle_bit;
         }

         mem_copy_align16(dst + ((xo + yo) ^ swizzle), src + x2, x3 - x2);

         src += src_pitch;
      }
   }
}

/**
 * Copy texture data from X tile layout to linear.
 *
 * \copydoc tile_copy_fn
 */
static inline void
xtiled_to_linear(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                 uint32_t y0, uint32_t y1,
                 char *dst, const char *src,
                 int32_t dst_pitch,
                 uint32_t swizzle_bit,
                 mem_copy_fn mem_copy,
                 mem_copy_fn mem_copy_align16)
{
   /* The copy destination offset for each range copied is the sum of
    * an X offset 'x0' or 'xo' and a Y offset 'yo.'
    */
   uint32_t xo, yo;

   dst += (ptrdiff_t)y0 * dst_pitch;

   for (yo = y0 * xtile_width; yo < y1 * xtile_width; yo += xtile_width) {
      /* Bits 9 and 10 of the copy destination offset control swizzling.
       * Only 'yo' contributes to those bits in the total offset,
       * so calculate 'swizzle' just once per row.
       * Move bits 9 and 10 three and four places respectively down
       * to bit 6 and xor them.
       */
      uint32_t swizzle = ((yo >> 3) ^ (yo >> 4)) & swizzle_bit;

      mem_copy(dst + x0, src + ((x0 + yo) ^ swizzle), x1 - x0);

      for (xo = x1; xo < x2; xo += xtile_span) {
         mem_copy_align16(dst + xo, src + ((xo + yo) ^ swizzle), xtile_span);
      }

      mem_copy_align16(dst + x2, src + ((xo + yo) ^ swizzle), x3 - x2);

      dst += dst_pitch;
   }
}

 /**
 * Copy texture data from Y tile layout to linear.
 *
 * \copydoc tile_copy_fn
 */
static inline void
ytiled_to_linear(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                 uint32_t y0, uint32_t y3,
                 char *dst, const char *src,
                 int32_t dst_pitch,
                 uint32_t swizzle_bit,
                 enum isl_tiling tiling,
                 int cpp,
                 mem_copy_fn mem_copy,
                 mem_copy_fn mem_copy_align16)
{
   /* See comments in linear_to_ytiled about the theory of operation for Y
    * tilings and the definition of the TILING_INIT macro used here.
    */
   const uint32_t column_width = ytile_span;

   uint32_t y1 = MIN2(y3, ALIGN_UP(y0, 4));
   uint32_t y2 = MAX2(y1, ALIGN_DOWN(y3, 4));

   uint32_t xinc_16, x_mask;
   uint32_t yinc_1, yinc_4, y_mask;

   uint32_t xo0, xo1, xo2;
   uint32_t yo0, yo1, yo2;

   if (tiling == ISL_TILING_Y0)
      TILING_INIT(Y0);
   else if (cpp == 16 || cpp == 8)
      TILING_INIT(YF_128);
   else if (cpp == 4 || cpp == 2)
      TILING_INIT(YF_32);
   else if (cpp == 1)
      TILING_INIT(YF_8);
   else
      unreachable("not reached");

   /* Bit 9 of the destination offset control swizzling.
    * Only the X offset contributes to bit 9 of the total offset,
    * so swizzle can be calculated in advance for these X positions.
    * Move bit 9 three places down to bit 6.
    */
   uint32_t swizzle0 = (xo0 >> 3) & swizzle_bit;
   uint32_t swizzle1 = (xo1 >> 3) & swizzle_bit;
   if (tiling != ISL_TILING_Y0)
      swizzle0 = swizzle1 = swizzle_bit = 0;

   uint32_t x, y, yo;

   dst += (ptrdiff_t)y0 * dst_pitch;

   if (y0 != y1) {
      for (y = y0, yo = yo0; y < y1; y++, yo = (yo + yinc_1) & y_mask) {
         uint32_t xo = xo1;
         uint32_t swizzle = swizzle1;

         mem_copy(dst + x0, src + ((xo0 + yo) ^ swizzle0), x1 - x0);

         /* Step by spans/columns.  As it happens, the swizzle bit flips
          * at each step so we don't need to calculate it explicitly.
          */
         for (x = x1; x < x2; x += ytile_span) {
            mem_copy_align16(dst + x, src + ((xo + yo) ^ swizzle), ytile_span);
            xo = (xo + xinc_16) & x_mask;
            swizzle ^= swizzle_bit;
         }

         mem_copy_align16(dst + x2, src + ((xo + yo) ^ swizzle), x3 - x2);

         dst += dst_pitch;
      }
   }

   for (y = y1, yo = yo1; y < y2; y += 4, yo = (yo + yinc_4) & y_mask) {
      uint32_t xo = xo1;
      uint32_t swizzle = swizzle1;

      if (x0 != x1) {
         mem_copy(dst + x0 + 0 * dst_pitch, src + ((xo0 + yo + 0 * column_width) ^ swizzle0), x1 - x0);
         mem_copy(dst + x0 + 1 * dst_pitch, src + ((xo0 + yo + 1 * column_width) ^ swizzle0), x1 - x0);
         mem_copy(dst + x0 + 2 * dst_pitch, src + ((xo0 + yo + 2 * column_width) ^ swizzle0), x1 - x0);
         mem_copy(dst + x0 + 3 * dst_pitch, src + ((xo0 + yo + 3 * column_width) ^ swizzle0), x1 - x0);
      }

      /* Step by spans/columns.  As it happens, the swizzle bit flips
       * at each step so we don't need to calculate it explicitly.
       */
      for (x = x1; x < x2; x += ytile_span) {
         mem_copy_align16(dst + x + 0 * dst_pitch, src + ((xo + yo + 0 * column_width) ^ swizzle), ytile_span);
         mem_copy_align16(dst + x + 1 * dst_pitch, src + ((xo + yo + 1 * column_width) ^ swizzle), ytile_span);
         mem_copy_align16(dst + x + 2 * dst_pitch, src + ((xo + yo + 2 * column_width) ^ swizzle), ytile_span);
         mem_copy_align16(dst + x + 3 * dst_pitch, src + ((xo + yo + 3 * column_width) ^ swizzle), ytile_span);
         xo = (xo + xinc_16) & x_mask;
         swizzle ^= swizzle_bit;
      }

      if (x2 != x3) {
         mem_copy_align16(dst + x2 + 0 * dst_pitch, src + ((xo + yo + 0 * column_width) ^ swizzle), x3 - x2);
         mem_copy_align16(dst + x2 + 1 * dst_pitch, src + ((xo + yo + 1 * column_width) ^ swizzle), x3 - x2);
         mem_copy_align16(dst + x2 + 2 * dst_pitch, src + ((xo + yo + 2 * column_width) ^ swizzle), x3 - x2);
         mem_copy_align16(dst + x2 + 3 * dst_pitch, src + ((xo + yo + 3 * column_width) ^ swizzle), x3 - x2);
      }

      dst += 4 * dst_pitch;
   }

   if (y2 != y3) {
      for (y = y2, yo = yo2; y < y3; y++, yo = (yo + yinc_1) & y_mask) {
         uint32_t xo = xo1;
         uint32_t swizzle = swizzle1;

         mem_copy(dst + x0, src + ((xo0 + yo) ^ swizzle0), x1 - x0);

         /* Step by spans/columns.  As it happens, the swizzle bit flips
          * at each step so we don't need to calculate it explicitly.
          */
         for (x = x1; x < x2; x += ytile_span) {
            mem_copy_align16(dst + x, src + ((xo + yo) ^ swizzle), ytile_span);
            xo = (xo + xinc_16) & x_mask;
            swizzle ^= swizzle_bit;
         }

         mem_copy_align16(dst + x2, src + ((xo + yo) ^ swizzle), x3 - x2);

         dst += dst_pitch;
      }
   }
}


/**
 * Copy texture data from linear to X tile layout, faster.
 *
 * Same as \ref linear_to_xtiled but faster, because it passes constant
 * parameters for common cases, allowing the compiler to inline code
 * optimized for those cases.
 *
 * \copydoc tile_copy_fn
 */
static FLATTEN void
linear_to_xtiled_faster(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                        uint32_t y0, uint32_t y1,
                        char *dst, const char *src,
                        int32_t src_pitch,
                        uint32_t swizzle_bit,
                        UNUSED enum isl_tiling tiling,
                        UNUSED int cpp,
                        mem_copy_fn mem_copy)
{
   if (x0 == 0 && x3 == xtile_width && y0 == 0 && y1 == xtile_height) {
      if (mem_copy == memcpy)
         return linear_to_xtiled(0, 0, xtile_width, xtile_width, 0, xtile_height,
                                 dst, src, src_pitch, swizzle_bit, memcpy, memcpy);
      else if (mem_copy == rgba8_copy)
         return linear_to_xtiled(0, 0, xtile_width, xtile_width, 0, xtile_height,
                                 dst, src, src_pitch, swizzle_bit,
                                 rgba8_copy, rgba8_copy_aligned_dst);
      else
         unreachable("not reached");
   } else {
      if (mem_copy == memcpy)
         return linear_to_xtiled(x0, x1, x2, x3, y0, y1,
                                 dst, src, src_pitch, swizzle_bit,
                                 memcpy, memcpy);
      else if (mem_copy == rgba8_copy)
         return linear_to_xtiled(x0, x1, x2, x3, y0, y1,
                                 dst, src, src_pitch, swizzle_bit,
                                 rgba8_copy, rgba8_copy_aligned_dst);
      else
         unreachable("not reached");
   }
   linear_to_xtiled(x0, x1, x2, x3, y0, y1,
                    dst, src, src_pitch, swizzle_bit, mem_copy, mem_copy);
}

/**
 * Copy texture data from linear to Y tile layout, faster.
 *
 * Same as \ref linear_to_ytiled but faster, because it passes constant
 * parameters for common cases, allowing the compiler to inline code
 * optimized for those cases.
 *
 * \copydoc tile_copy_fn
 */
static FLATTEN void
linear_to_ytiled_faster(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                        uint32_t y0, uint32_t y1,
                        char *dst, const char *src,
                        int32_t src_pitch,
                        uint32_t swizzle_bit,
                        enum isl_tiling tiling,
                        int cpp,
                        mem_copy_fn mem_copy)
{
   if (x0 == 0 && x3 == ytile_width && y0 == 0 && y1 == ytile_height) {
      if (mem_copy == memcpy)
         return linear_to_ytiled(0, 0, ytile_width, ytile_width, 0,
                                 ytile_height, dst, src, src_pitch, swizzle_bit,
                                 tiling, cpp, memcpy, memcpy);
      else if (mem_copy == rgba8_copy)
         return linear_to_ytiled(0, 0, ytile_width, ytile_width, 0,
                                 ytile_height, dst, src, src_pitch, swizzle_bit,
                                 tiling, cpp, rgba8_copy,
                                 rgba8_copy_aligned_dst);
      else
         unreachable("not reached");
   } else {
      if (mem_copy == memcpy)
         return linear_to_ytiled(x0, x1, x2, x3, y0, y1, dst, src, src_pitch,
                                 swizzle_bit, tiling, cpp, memcpy, memcpy);
      else if (mem_copy == rgba8_copy)
         return linear_to_ytiled(x0, x1, x2, x3, y0, y1, dst, src, src_pitch,
                                 swizzle_bit, tiling, cpp, rgba8_copy,
                                 rgba8_copy_aligned_dst);
      else
         unreachable("not reached");
   }
   linear_to_ytiled(x0, x1, x2, x3, y0, y1, dst, src, src_pitch, swizzle_bit,
                    tiling, cpp, mem_copy, mem_copy);
}

/**
 * Copy texture data from X tile layout to linear, faster.
 *
 * Same as \ref xtile_to_linear but faster, because it passes constant
 * parameters for common cases, allowing the compiler to inline code
 * optimized for those cases.
 *
 * \copydoc tile_copy_fn
 */
static FLATTEN void
xtiled_to_linear_faster(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                        uint32_t y0, uint32_t y1,
                        char *dst, const char *src,
                        int32_t dst_pitch,
                        uint32_t swizzle_bit,
                        UNUSED enum isl_tiling tiling,
                        UNUSED int cpp,
                        mem_copy_fn mem_copy)
{
   if (x0 == 0 && x3 == xtile_width && y0 == 0 && y1 == xtile_height) {
      if (mem_copy == memcpy)
         return xtiled_to_linear(0, 0, xtile_width, xtile_width, 0, xtile_height,
                                 dst, src, dst_pitch, swizzle_bit, memcpy, memcpy);
      else if (mem_copy == rgba8_copy)
         return xtiled_to_linear(0, 0, xtile_width, xtile_width, 0, xtile_height,
                                 dst, src, dst_pitch, swizzle_bit,
                                 rgba8_copy, rgba8_copy_aligned_src);
      else
         unreachable("not reached");
   } else {
      if (mem_copy == memcpy)
         return xtiled_to_linear(x0, x1, x2, x3, y0, y1,
                                 dst, src, dst_pitch, swizzle_bit, memcpy, memcpy);
      else if (mem_copy == rgba8_copy)
         return xtiled_to_linear(x0, x1, x2, x3, y0, y1,
                                 dst, src, dst_pitch, swizzle_bit,
                                 rgba8_copy, rgba8_copy_aligned_src);
      else
         unreachable("not reached");
   }
   xtiled_to_linear(x0, x1, x2, x3, y0, y1,
                    dst, src, dst_pitch, swizzle_bit, mem_copy, mem_copy);
}

/**
 * Copy texture data from Y tile layout to linear, faster.
 *
 * Same as \ref ytile_to_linear but faster, because it passes constant
 * parameters for common cases, allowing the compiler to inline code
 * optimized for those cases.
 *
 * \copydoc tile_copy_fn
 */
static FLATTEN void
ytiled_to_linear_faster(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                        uint32_t y0, uint32_t y1,
                        char *dst, const char *src,
                        int32_t dst_pitch,
                        uint32_t swizzle_bit,
                        enum isl_tiling tiling,
                        int cpp,
                        mem_copy_fn mem_copy)
{
   if (x0 == 0 && x3 == ytile_width && y0 == 0 && y1 == ytile_height) {
      if (mem_copy == memcpy)
         return ytiled_to_linear(0, 0, ytile_width, ytile_width, 0,
                                 ytile_height, dst, src, dst_pitch, swizzle_bit,
                                 tiling, cpp, memcpy, memcpy);
      else if (mem_copy == rgba8_copy)
         return ytiled_to_linear(0, 0, ytile_width, ytile_width, 0,
                                 ytile_height, dst, src, dst_pitch, swizzle_bit,
                                 tiling, cpp, rgba8_copy,
                                 rgba8_copy_aligned_src);
      else
         unreachable("not reached");
   } else {
      if (mem_copy == memcpy)
         return ytiled_to_linear(x0, x1, x2, x3, y0, y1, dst, src, dst_pitch,
                                 swizzle_bit, tiling, cpp, memcpy, memcpy);
      else if (mem_copy == rgba8_copy)
         return ytiled_to_linear(x0, x1, x2, x3, y0, y1, dst, src, dst_pitch,
                                 swizzle_bit, tiling, cpp, rgba8_copy,
                                 rgba8_copy_aligned_src);
      else
         unreachable("not reached");
   }
   ytiled_to_linear(x0, x1, x2, x3, y0, y1, dst, src, dst_pitch, swizzle_bit,
                    tiling, cpp, mem_copy, mem_copy);
}

/**
 * Copy from linear to tiled texture.
 *
 * Divide the region given by X range [xt1, xt2) and Y range [yt1, yt2) into
 * pieces that do not cross tile boundaries and copy each piece with a tile
 * copy function (\ref tile_copy_fn).
 * The X range is in bytes, i.e. pixels * bytes-per-pixel.
 * The Y range is in pixels (i.e. unitless).
 * 'dst' is the address of (0, 0) in the destination tiled texture.
 * 'src' is the address of (xt1, yt1) in the source linear texture.
 */
void
linear_to_tiled(uint32_t xt1, uint32_t xt2,
                uint32_t yt1, uint32_t yt2,
                char *dst, const char *src,
                uint32_t dst_pitch, int32_t src_pitch,
                bool has_swizzling,
                enum isl_tiling tiling,
                int cpp,
                mem_copy_fn mem_copy)
{
   tile_addr_fn tile_addr;
   tile_copy_fn tile_copy;
   uint32_t xt0, xt3;
   uint32_t yt0, yt3;
   uint32_t xt, yt;
   uint32_t tw, th, span;
   uint32_t swizzle_bit = has_swizzling ? 1<<6 : 0;

   if (tiling == ISL_TILING_X) {
      tw = xtile_width;
      th = xtile_height;
      span = xtile_span;
      tile_copy = linear_to_xtiled_faster;
      tile_addr = xtile_addr;
   } else if (tiling == ISL_TILING_Y0) {
      tw = ytile_width;
      th = ytile_height;
      span = ytile_span;
      tile_copy = linear_to_ytiled_faster;
      tile_addr = ytile_addr;
   } else if (isl_tiling_is_std_y(tiling)) {
      if (cpp == 16 || cpp == 8) {
         tw = std_ytile128_width;
         th = std_ytile128_height;
         tile_addr = tiling == ISL_TILING_Ys ? ys128_addr : yf128_addr;
      } else if (cpp == 4 || cpp == 2) {
         tw = std_ytile32_width;
         th = std_ytile32_height;
         tile_addr = tiling == ISL_TILING_Ys ? ys32_addr : yf32_addr;
      } else if (cpp == 1) {
         tw = std_ytile8_width;
         th = std_ytile8_height;
         tile_addr = tiling == ISL_TILING_Ys ? ys8_addr : yf8_addr;
      } else {
         unreachable("not reached");
      }
      span = ytile_span;
      tile_copy = linear_to_ytiled_faster;
   } else {
      unreachable("unsupported tiling");
   }

   /* Round out to tile boundaries. */
   xt0 = ALIGN_DOWN(xt1, tw);
   xt3 = ALIGN_UP  (xt2, tw);
   yt0 = ALIGN_DOWN(yt1, th);
   yt3 = ALIGN_UP  (yt2, th);

   /* Loop over all tiles to which we have something to copy.
    * 'xt' and 'yt' are the origin of the destination tile, whether copying
    * copying a full or partial tile.
    * tile_copy() copies one tile or partial tile.
    * Looping x inside y is the faster memory access pattern.
    */
   for (yt = yt0; yt < yt3; yt += th) {
      for (xt = xt0; xt < xt3; xt += tw) {
         /* The area to update is [x0,x3) x [y0,y1).
          * May not want the whole tile, hence the min and max.
          */
         uint32_t x0 = MAX2(xt1, xt);
         uint32_t y0 = MAX2(yt1, yt);
         uint32_t x3 = MIN2(xt2, xt + tw);
         uint32_t y1 = MIN2(yt2, yt + th);

         /* [x0,x3) is split into [x0,x1), [x1,x2), [x2,x3) such that
          * the middle interval is the longest span-aligned part.
          * The sub-ranges could be empty.
          */
         uint32_t x1, x2;
         x1 = ALIGN_UP(x0, span);
         if (x1 > x3)
            x1 = x2 = x3;
         else
            x2 = ALIGN_DOWN(x3, span);

         assert(x0 <= x1 && x1 <= x2 && x2 <= x3);
         assert(x1 - x0 < span && x3 - x2 < span);
         assert(x3 - x0 <= tw);
         assert((x2 - x1) % span == 0);

         /* Translate by (xt,yt) for single-tile copier. */
         tile_copy(x0-xt, x1-xt, x2-xt, x3-xt,
                   y0-yt, y1-yt,
                   tile_addr(xt, yt, dst, dst_pitch),
                   src + (ptrdiff_t)xt - xt1 + ((ptrdiff_t)yt - yt1) * src_pitch,
                   src_pitch,
                   swizzle_bit,
                   tiling,
                   cpp,
                   mem_copy);
      }
   }
}

/**
 * Copy from tiled to linear texture.
 *
 * Divide the region given by X range [xt1, xt2) and Y range [yt1, yt2) into
 * pieces that do not cross tile boundaries and copy each piece with a tile
 * copy function (\ref tile_copy_fn).
 * The X range is in bytes, i.e. pixels * bytes-per-pixel.
 * The Y range is in pixels (i.e. unitless).
 * 'dst' is the address of (xt1, yt1) in the destination linear texture.
 * 'src' is the address of (0, 0) in the source tiled texture.
 */
void
tiled_to_linear(uint32_t xt1, uint32_t xt2,
                uint32_t yt1, uint32_t yt2,
                char *dst, const char *src,
                int32_t dst_pitch, uint32_t src_pitch,
                bool has_swizzling,
                enum isl_tiling tiling,
                int cpp,
                mem_copy_fn mem_copy)
{
   tile_addr_fn tile_addr;
   tile_copy_fn tile_copy;
   uint32_t xt0, xt3;
   uint32_t yt0, yt3;
   uint32_t xt, yt;
   uint32_t tw, th, span;
   uint32_t swizzle_bit = has_swizzling ? 1<<6 : 0;

   if (tiling == ISL_TILING_X) {
      tw = xtile_width;
      th = xtile_height;
      span = xtile_span;
      tile_copy = xtiled_to_linear_faster;
      tile_addr = xtile_addr;
   } else if (tiling == ISL_TILING_Y0) {
      tw = ytile_width;
      th = ytile_height;
      span = ytile_span;
      tile_copy = ytiled_to_linear_faster;
      tile_addr = ytile_addr;
   } else if (isl_tiling_is_std_y(tiling)) {
      if (cpp == 16 || cpp == 8) {
         tw = std_ytile128_width;
         th = std_ytile128_height;
         tile_addr = tiling == ISL_TILING_Ys ? ys128_addr : yf128_addr;
      } else if (cpp == 4 || cpp == 2) {
         tw = std_ytile32_width;
         th = std_ytile32_height;
         tile_addr = tiling == ISL_TILING_Ys ? ys32_addr : yf32_addr;
      } else if (cpp == 1) {
         tw = std_ytile8_width;
         th = std_ytile8_height;
         tile_addr = tiling == ISL_TILING_Ys ? ys8_addr : yf8_addr;
      } else {
         unreachable("not reached");
      }
      span = ytile_span;
      tile_copy = linear_to_ytiled_faster;
   } else {
      unreachable("unsupported tiling");
   }

   /* Round out to tile boundaries. */
   xt0 = ALIGN_DOWN(xt1, tw);
   xt3 = ALIGN_UP  (xt2, tw);
   yt0 = ALIGN_DOWN(yt1, th);
   yt3 = ALIGN_UP  (yt2, th);

   /* Loop over all tiles to which we have something to copy.
    * 'xt' and 'yt' are the origin of the destination tile, whether copying
    * copying a full or partial tile.
    * tile_copy() copies one tile or partial tile.
    * Looping x inside y is the faster memory access pattern.
    */
   for (yt = yt0; yt < yt3; yt += th) {
      for (xt = xt0; xt < xt3; xt += tw) {
         /* The area to update is [x0,x3) x [y0,y1).
          * May not want the whole tile, hence the min and max.
          */
         uint32_t x0 = MAX2(xt1, xt);
         uint32_t y0 = MAX2(yt1, yt);
         uint32_t x3 = MIN2(xt2, xt + tw);
         uint32_t y1 = MIN2(yt2, yt + th);

         /* [x0,x3) is split into [x0,x1), [x1,x2), [x2,x3) such that
          * the middle interval is the longest span-aligned part.
          * The sub-ranges could be empty.
          */
         uint32_t x1, x2;
         x1 = ALIGN_UP(x0, span);
         if (x1 > x3)
            x1 = x2 = x3;
         else
            x2 = ALIGN_DOWN(x3, span);

         assert(x0 <= x1 && x1 <= x2 && x2 <= x3);
         assert(x1 - x0 < span && x3 - x2 < span);
         assert(x3 - x0 <= tw);
         assert((x2 - x1) % span == 0);

         /* Translate by (xt,yt) for single-tile copier. */
         tile_copy(x0-xt, x1-xt, x2-xt, x3-xt,
                   y0-yt, y1-yt,
                   dst + (ptrdiff_t)xt - xt1 + ((ptrdiff_t)yt - yt1) * dst_pitch,
                   tile_addr(xt, yt, (char*)src, src_pitch),
                   dst_pitch,
                   swizzle_bit,
                   tiling,
                   cpp,
                   mem_copy);
      }
   }
}


/**
 * Determine which copy function to use for the given format combination
 *
 * The only two possible copy functions which are ever returned are a
 * direct memcpy and a RGBA <-> BGRA copy function.  Since RGBA -> BGRA and
 * BGRA -> RGBA are exactly the same operation (and memcpy is obviously
 * symmetric), it doesn't matter whether the copy is from the tiled image
 * to the untiled or vice versa.  The copy function required is the same in
 * either case so this function can be used.
 *
 * \param[in]  tiledFormat The format of the tiled image
 * \param[in]  format      The GL format of the client data
 * \param[in]  type        The GL type of the client data
 * \param[out] mem_copy    Will be set to one of either the standard
 *                         library's memcpy or a different copy function
 *                         that performs an RGBA to BGRA conversion
 * \param[out] cpp         Number of bytes per channel
 *
 * \return true if the format and type combination are valid
 */
bool intel_get_memcpy(mesa_format tiledFormat, GLenum format,
                      GLenum type, mem_copy_fn *mem_copy, uint32_t *cpp)
{
   if (type == GL_UNSIGNED_INT_8_8_8_8_REV &&
       !(format == GL_RGBA || format == GL_BGRA))
      return false; /* Invalid type/format combination */

   if ((tiledFormat == MESA_FORMAT_L_UNORM8 && format == GL_LUMINANCE) ||
       (tiledFormat == MESA_FORMAT_A_UNORM8 && format == GL_ALPHA)) {
      *cpp = 1;
      *mem_copy = memcpy;
   } else if ((tiledFormat == MESA_FORMAT_B8G8R8A8_UNORM) ||
              (tiledFormat == MESA_FORMAT_B8G8R8X8_UNORM) ||
              (tiledFormat == MESA_FORMAT_B8G8R8A8_SRGB) ||
              (tiledFormat == MESA_FORMAT_B8G8R8X8_SRGB)) {
      *cpp = 4;
      if (format == GL_BGRA) {
         *mem_copy = memcpy;
      } else if (format == GL_RGBA) {
         *mem_copy = rgba8_copy;
      }
   } else if ((tiledFormat == MESA_FORMAT_R8G8B8A8_UNORM) ||
              (tiledFormat == MESA_FORMAT_R8G8B8X8_UNORM) ||
              (tiledFormat == MESA_FORMAT_R8G8B8A8_SRGB) ||
              (tiledFormat == MESA_FORMAT_R8G8B8X8_SRGB)) {
      *cpp = 4;
      if (format == GL_BGRA) {
         /* Copying from RGBA to BGRA is the same as BGRA to RGBA so we can
          * use the same function.
          */
         *mem_copy = rgba8_copy;
      } else if (format == GL_RGBA) {
         *mem_copy = memcpy;
      }
   }

   if (!(*mem_copy))
      return false;

   return true;
}
