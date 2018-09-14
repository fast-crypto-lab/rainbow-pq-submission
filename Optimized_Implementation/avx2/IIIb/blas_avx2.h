#ifndef _BLAS_AVX2_H_
#define _BLAS_AVX2_H_

#include "blas.h"

#include "gf31_sse.h"

#include <immintrin.h>


#ifdef  __cplusplus
extern  "C" {
#endif


static inline
__m256i _gf31v_u8_to_u16_avx2( __m128i a ) {
	__m128i zero = _mm_setzero_si128();
	__m128i a0 = _mm_unpacklo_epi8( a , zero );
	__m128i a1 = _mm_unpackhi_epi8( a , zero );

	return _mm256_insertf128_si256 ( _mm256_castsi128_si256(a0) , a1 , 1 );
}


static inline
void gf31v_u8_to_u16( uint8_t * a16 , const uint8_t * a8 , unsigned n ) {
	uint16_t * a = (uint16_t *)a16;
	while( 16 <= n ){
		__m128i pa = _mm_loadu_si128( (const __m128i*)a8 );
		__m256i pr = _gf31v_u8_to_u16_avx2( pa );
		_mm256_storeu_si256( (__m256i*) a , pr );
		a += 16;
		a8 += 16;
		n -= 16;

	}
	if( 0 == n ) return;

	uint8_t temp[32] __attribute__((aligned(32))) = {0};
	uint16_t * temp_u16 = (uint16_t*)temp;
	for(unsigned i=0;i<n;i++) temp[i] = a8[i];
	__m128i pa = _mm_load_si128( (const __m128i*)temp );
	__m256i pr = _gf31v_u8_to_u16_avx2(pa);
	_mm256_store_si256( (__m256i*) temp , pr );
	for(unsigned i=0;i<n;i++) a[i] = temp_u16[i];
}


static inline
__m128i _gf31v_u16_to_u8_avx2( __m256i a ) {
	__m128i a1 = _mm256_extractf128_si256( a , 1 );
	__m128i a0 = _mm256_castsi256_si128( a );
	return _mm_packus_epi16( a0 , a1 );
}


static inline
void gf31v_u16_to_u8( uint8_t * a8 , const uint8_t * a16 , unsigned n ) {
	const uint16_t * a = (const uint16_t *)a16;
	while( 16 <= n ){
		__m256i pa = _mm256_loadu_si256( (const __m256i*)a );
		__m128i pr = _gf31v_u16_to_u8_avx2( pa );
		_mm_storeu_si128( (__m128i*) a8 , pr );
		a += 16;
		a8 += 16;
		n -= 16;

	}
	if( 0 == n ) return;

	uint8_t temp[32] __attribute__((aligned(32))) = {0};
	uint16_t * temp16 = (uint16_t *) &temp[0];

	for(unsigned i=0;i<n;i++) temp16[i] = a[i];
	__m256i pa = _mm256_load_si256( (const __m256i*)temp );
	__m128i pr = _gf31v_u16_to_u8_avx2(pa);
	_mm_store_si128( (__m128i*) temp , pr );
	for(unsigned i=0;i<n;i++) a8[i] = temp[i];
}



static inline
void gf31v_mul_scalar_u16_avx2( uint16_t * r , const uint16_t * a , uint16_t b , unsigned n ) {
	__m256i mb = _mm256_set1_epi16( b );
	while( 16 <= n ){
		__m256i pa = _mm256_loadu_si256( (const __m256i*)a );
		__m256i pr = _mm256_mullo_epi16( pa , mb );
		_mm256_storeu_si256( (__m256i*) r , pr );

		a += 16;
		r += 16;
		n -= 16;

	}
	if( 0 == n ) return;

	uint16_t temp[16] __attribute__((aligned(32))) = {0};

	for(unsigned i=0;i<n;i++) temp[i] = a[i];
	__m256i pa = _mm256_load_si256( (const __m256i*)temp );
	__m256i pr = _mm256_mullo_epi16( pa , mb );
	_mm256_store_si256( (__m256i*) temp , pr );
	for(unsigned i=0;i<n;i++) r[i] = temp[i];
}


static inline
__m256i _gf31v_reduce_u16_avx2( __m256i a ){
	__m256i mask = _mm256_set1_epi16( 31 );
	return _mm256_add_epi16( a&mask , _mm256_srli_epi16(a,5) );
}



static inline
void gf31v_reduce_u16_avx2( uint16_t * a , unsigned n ) {
	while( 16 <= n ){
		__m256i pa = _mm256_loadu_si256( (const __m256i*)a );
		pa = _gf31v_reduce_u16_avx2( pa );
		_mm256_storeu_si256( (__m256i*) a , pa );
		a += 16;
		n -= 16;

	}
	if( 0 == n ) return;

	uint16_t temp[16] __attribute__((aligned(32))) = {0};

	for(unsigned i=0;i<n;i++) temp[i] = a[i];
	__m256i pa = _mm256_load_si256( (const __m256i*)temp );
	__m256i pr = _gf31v_reduce_u16_avx2(pa);
	_mm256_store_si256( (__m256i*) temp , pr );
	for(unsigned i=0;i<n;i++) a[i] = temp[i];
}


static inline
__m256i _gf31v_reduce_u8_avx2( __m256i a ){
	__m256i mask_31 = _mm256_set1_epi8( 31 );
	__m256i mask_30 = _mm256_set1_epi8( 30 );
	__m256i r1 = _mm256_sub_epi8( a , _mm256_cmpgt_epi8(a,mask_30)&mask_31 );
	__m256i r2 = _mm256_sub_epi8( r1 , _mm256_cmpgt_epi8(r1,mask_30)&mask_31 );

	return r2;
}


static inline
void gf31v_reduce_u8_avx2( uint8_t * a , unsigned n ) {
	while( 32 <= n ){
		__m256i pa = _mm256_loadu_si256( (const __m256i*)a );
		pa = _gf31v_reduce_u8_avx2( pa );
		_mm256_storeu_si256( (__m256i*) a , pa );
		a += 32;
		n -= 32;

	}
	if( 0 == n ) return;

	uint8_t temp[32] __attribute__((aligned(32))) = {0};

	for(unsigned i=0;i<n;i++) temp[i] = a[i];
	__m256i pa = _mm256_load_si256( (const __m256i*)temp );
	__m256i pr = _gf31v_reduce_u8_avx2(pa);
	_mm256_store_si256( (__m256i*) temp , pr );
	for(unsigned i=0;i<n;i++) a[i] = temp[i];
}



static inline
void gf31v_add_avx2( uint8_t * accu_b, const uint8_t * a , unsigned n ) {
	__m256i mask_31 = _mm256_set1_epi8( 31 );
	__m256i mask_30 = _mm256_set1_epi8( 30 );

	while( 32 <= n ){
		__m256i pa = _mm256_loadu_si256( (const __m256i*)a );
		__m256i pb = _mm256_loadu_si256( (const __m256i*)accu_b );
		__m256i pc = _mm256_add_epi8( pa , pb );
		__m256i r1 = _mm256_sub_epi8( pc , _mm256_cmpgt_epi8(pc,mask_30)&mask_31 );
		_mm256_storeu_si256( (__m256i*) accu_b , r1 );
		a += 32;
		accu_b += 32;
		n -= 32;

	}
	if( 0 == n ) return;

	uint8_t temp[32] __attribute__((aligned(32))) = {0};

	for(unsigned i=0;i<n;i++) temp[i] = a[i];
	__m256i pa = _mm256_load_si256( (const __m256i*)temp );
	for(unsigned i=0;i<n;i++) temp[i] = accu_b[i];
	__m256i pb = _mm256_load_si256( (const __m256i*)temp );
	__m256i pc = _mm256_add_epi8( pa , pb );
	__m256i r1 = _mm256_sub_epi8( pc , _mm256_cmpgt_epi8(pc,mask_30)&mask_31 );
	_mm256_store_si256( (__m256i*) temp , r1 );

	for(unsigned i=0;i<n;i++) accu_b[i] = temp[i];
}

static inline
void gf31v_sub_avx2( uint8_t * accu_b, const uint8_t * a , unsigned n ) {
	__m256i mask_31 = _mm256_set1_epi8( 31 );
	__m256i mask_30 = _mm256_set1_epi8( 30 );

	while( 32 <= n ){
		__m256i pa = _mm256_loadu_si256( (const __m256i*)a );
		__m256i pb = _mm256_loadu_si256( (const __m256i*)accu_b );
		__m256i pc = _mm256_add_epi8( pb , _mm256_sub_epi8(mask_31,pa) );
		__m256i r1 = _mm256_sub_epi8( pc , _mm256_cmpgt_epi8(pc,mask_30)&mask_31 );
		_mm256_storeu_si256( (__m256i*) accu_b , r1 );
		a += 32;
		accu_b += 32;
		n -= 32;

	}
	if( 0 == n ) return;

	uint8_t temp[32] __attribute__((aligned(32))) = {0};

	for(unsigned i=0;i<n;i++) temp[i] = a[i];
	__m256i pa = _mm256_load_si256( (const __m256i*)temp );
	for(unsigned i=0;i<n;i++) temp[i] = accu_b[i];
	__m256i pb = _mm256_load_si256( (const __m256i*)temp );
	__m256i pc = _mm256_add_epi8( pb , _mm256_sub_epi8(mask_31,pa) );
	__m256i r1 = _mm256_sub_epi8( pc , _mm256_cmpgt_epi8(pc,mask_30)&mask_31 );
	_mm256_store_si256( (__m256i*) temp , r1 );

	for(unsigned i=0;i<n;i++) accu_b[i] = temp[i];
}


static inline
__m256i _gf31v_mul_u8_avx2( __m256i a , uint16_t b ){
	__m256i zero = _mm256_setzero_si256();
	__m256i a0 = _mm256_unpacklo_epi8( a , zero );
	__m256i a1 = _mm256_unpackhi_epi8( a , zero );
	__m256i bb = _mm256_set1_epi16( b );

	a0 = _mm256_mullo_epi16( a0 , bb );
	a1 = _mm256_mullo_epi16( a1 , bb );

	a0 = _gf31v_reduce_u16_avx2( a0 );
	a1 = _gf31v_reduce_u16_avx2( a1 );

	__m256i r = _mm256_packs_epi16( a0 , a1 );
	return _gf31v_reduce_u8_avx2( r );
}




//////////////////////////////////////////////



void gf31mat_prod_avx2( uint8_t * c , const uint8_t * mat , unsigned n_mat_h , unsigned n_mat_w , const uint8_t * b );

unsigned _gf31mat_gauss_elim_avx2( uint8_t * mat , unsigned h , unsigned w );


#ifdef  __cplusplus
}
#endif



#endif

