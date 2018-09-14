#ifndef _BLAS_AVX2_H_
#define _BLAS_AVX2_H_

#include "gf16.h"

#include <immintrin.h>

#include "blas_config.h"
#include "assert.h"

#include "gf16_avx2.h"

#include "blas_sse.h"


#ifdef  __cplusplus
extern  "C" {
#endif






static inline
__m256i _load_ymm( const uint8_t *a , unsigned _num_byte ) {
	uint8_t temp[32] __attribute__((aligned(32)));
	assert( 32 >= _num_byte );
	assert( 0 < _num_byte );
	for(unsigned i=0;i<_num_byte;i++) temp[i] = a[i];
	return _mm256_load_si256((__m256i*)temp);
}

static inline
void _store_ymm( uint8_t *a , unsigned _num_byte , __m256i data ) {
	uint8_t temp[32] __attribute__((aligned(32)));
	assert( 32 >= _num_byte );
	assert( 0 < _num_byte );
	_mm256_store_si256((__m256i*)temp,data);
	for(unsigned i=0;i<_num_byte;i++) a[i] = temp[i];
}



static inline
void linearmap_8x8_ymm( uint8_t * a , __m256i ml , __m256i mh , __m256i mask , unsigned _num_byte ) {
	unsigned n_32 = _num_byte>>5;
	for(unsigned i=0;i<n_32;i++) {
		__m256i inp = _mm256_loadu_si256( (__m256i*)(a+i*32) );
		__m256i r0 = linear_transform_8x8_256b( ml , mh , inp , mask );
		_mm256_storeu_si256( (__m256i*)(a+i*32) , r0 );
	}
	unsigned rem = _num_byte&31;
	if( rem ) linearmap_8x8_sse( a+n_32*32 , _mm256_castsi256_si128(ml) , _mm256_castsi256_si128(mh) , _mm256_castsi256_si128(mask) , rem );
}


static inline
void linearmap_8x8_accu_ymm( uint8_t * accu_c , const uint8_t * a ,  __m256i ml , __m256i mh , __m256i mask , unsigned _num_byte ) {
	unsigned n_32 = _num_byte>>5;
	for(unsigned i=0;i<n_32;i++) {
		__m256i inp = _mm256_loadu_si256( (__m256i*)(a+i*32) );
		__m256i out = _mm256_loadu_si256( (__m256i*)(accu_c+i*32) );
		__m256i r0 = linear_transform_8x8_256b( ml , mh , inp , mask );
		r0 ^= out;
		_mm256_storeu_si256( (__m256i*)(accu_c+i*32) , r0 );
	}
	unsigned rem = _num_byte&31;
	if( rem ) linearmap_8x8_accu_sse( accu_c+n_32*32 , a+n_32*32 , _mm256_castsi256_si128(ml) , _mm256_castsi256_si128(mh) , _mm256_castsi256_si128(mask) , rem );
}




///////////////////////   basic functions   /////////////////////////////////////


static inline
void gf256v_add_avx2( uint8_t * accu_b, const uint8_t * a , unsigned _num_byte ) {
	//uint8_t temp[32] __attribute__((aligned(32)));
	unsigned n_ymm = (_num_byte)>>5;
	unsigned i=0;
	for(;i<n_ymm;i++) {
		__m256i inp = _mm256_loadu_si256( (__m256i*) (a+i*32) );
		__m256i out = _mm256_loadu_si256( (__m256i*) (accu_b+i*32) );
		out ^= inp;
		_mm256_storeu_si256( (__m256i*) (accu_b+i*32) , out );
	}
	if( 0 != (_num_byte&0x1f) ) {
		n_ymm <<= 5;
		gf256v_add_sse( accu_b + n_ymm , a + n_ymm , _num_byte&0x1f );
	}
}




#define _AVX2_USE_TOOL_FUNCS_



static inline
void gf16v_mul_scalar_avx2( uint8_t * a, uint8_t gf16_b , unsigned _num_byte ) {
	unsigned b = gf16_b&0xf;
	__m256i m_tab = _mm256_load_si256( (__m256i*) (__gf16_mul + 32*b) );
	__m256i ml = _mm256_permute2x128_si256( m_tab , m_tab , 0 );
	__m256i mh = _mm256_permute2x128_si256( m_tab , m_tab , 0x11 );
	__m256i mask = _mm256_load_si256( (__m256i*) __mask_low );

#ifdef _AVX2_USE_TOOL_FUNCS_
	linearmap_8x8_ymm( a , ml , mh , mask , _num_byte );
#else
	unsigned n_32 = _num_byte>>5;
	for(unsigned i=0;i<n_32;i++) {
		__m256i inp = _mm256_loadu_si256( (__m256i*)(a+i*32) );
		__m256i r0 = _mm256_shuffle_epi8(ml, inp&mask );
		__m256i r1 = _mm256_shuffle_epi8(mh, _mm256_srli_epi16(_mm256_andnot_si256(mask,inp),4) );
		r0 ^= r1;
		_mm256_storeu_si256( (__m256i*)(a+i*32) , r0 );
	}

	unsigned rem = _num_byte&31;
	if( rem ) {
		a += (n_32<<5);
		uint8_t temp[32] __attribute__((aligned(32)));
		for(unsigned i=0;i<rem;i++) temp[i] = a[i];
		__m256i inp = _mm256_load_si256( (__m256i*)(temp) );

		__m256i r0 = _mm256_shuffle_epi8(ml, inp&mask );
		__m256i r1 = _mm256_shuffle_epi8(mh, _mm256_srli_epi16(_mm256_andnot_si256(mask,inp),4) );
		r0 ^= r1;

		_mm256_store_si256( (__m256i*)(temp) , r0 );
		for(unsigned i=0;i<rem;i++) a[i] = temp[i];
	}
#endif
}



static inline
void gf16v_madd_avx2( uint8_t * accu_c, const uint8_t * a , uint8_t gf16_b, unsigned _num_byte ) {
	unsigned b = gf16_b&0xf;
	__m256i m_tab = _mm256_load_si256( (__m256i*) (__gf16_mul + 32*b) );
	__m256i ml = _mm256_permute2x128_si256( m_tab , m_tab , 0 );
	__m256i mh = _mm256_permute2x128_si256( m_tab , m_tab , 0x11 );
	__m256i mask = _mm256_load_si256( (__m256i*) __mask_low );

#ifdef _AVX2_USE_TOOL_FUNCS_
	linearmap_8x8_accu_ymm( accu_c , a , ml , mh , mask , _num_byte );
#else
	unsigned n_32 = _num_byte>>5;
	for(unsigned i=0;i<n_32;i++) {
		__m256i inp = _mm256_loadu_si256( (__m256i*)(a+i*32) );
		__m256i out = _mm256_loadu_si256( (__m256i*)(accu_c+i*32) );
		__m256i r0 = _mm256_shuffle_epi8(ml, inp&mask );
		__m256i r1 = _mm256_shuffle_epi8(mh, _mm256_srli_epi16(_mm256_andnot_si256(mask,inp),4) );
		r0 ^= r1^out;
		_mm256_storeu_si256( (__m256i*)(accu_c+i*32) , r0 );
	}

	unsigned rem = _num_byte&31;
	if( rem ) {
		a += (n_32<<5);
		accu_c += (n_32<<5);
		uint8_t temp[32] __attribute__((aligned(32)));
		for(unsigned i=0;i<rem;i++) temp[i] = a[i];
		__m256i inp = _mm256_load_si256( (__m256i*)(temp) );
		for(unsigned i=0;i<rem;i++) temp[i] = accu_c[i];
		__m256i out = _mm256_load_si256( (__m256i*)(temp) );

		__m256i r0 = _mm256_shuffle_epi8(ml, inp&mask );
		__m256i r1 = _mm256_shuffle_epi8(mh, _mm256_srli_epi16(_mm256_andnot_si256(mask,inp),4) );
		r0 ^= r1^out;

		_mm256_store_si256( (__m256i*)(temp) , r0 );
		for(unsigned i=0;i<rem;i++) accu_c[i] = temp[i];
	}
#endif
}





static inline
void gf256v_mul_scalar_avx2( uint8_t * a, uint8_t _b , unsigned _num_byte ) {
	unsigned b = _b;
	__m256i m_tab = _mm256_load_si256( (__m256i*) (__gf256_mul + 32*b) );
	__m256i ml = _mm256_permute2x128_si256( m_tab , m_tab , 0 );
	__m256i mh = _mm256_permute2x128_si256( m_tab , m_tab , 0x11 );
	__m256i mask = _mm256_load_si256( (__m256i*) __mask_low );

#ifdef _AVX2_USE_TOOL_FUNCS_
	linearmap_8x8_ymm( a , ml , mh , mask , _num_byte );
#else
	unsigned n_32 = _num_byte>>5;
	for(unsigned i=0;i<n_32;i++) {
		__m256i inp = _mm256_loadu_si256( (__m256i*)(a+i*32) );
		__m256i r0 = _mm256_shuffle_epi8(ml, inp&mask );
		__m256i r1 = _mm256_shuffle_epi8(mh, _mm256_srli_epi16(_mm256_andnot_si256(mask,inp),4) );
		r0 ^= r1;
		_mm256_storeu_si256( (__m256i*)(a+i*32) , r0 );
	}

	unsigned rem = _num_byte&31;
	if( rem ) {
		a += (n_32<<5);
		uint8_t temp[32] __attribute__((aligned(32)));
		for(unsigned i=0;i<rem;i++) temp[i] = a[i];
		__m256i inp = _mm256_load_si256( (__m256i*)(temp) );

		__m256i r0 = _mm256_shuffle_epi8(ml, inp&mask );
		__m256i r1 = _mm256_shuffle_epi8(mh, _mm256_srli_epi16(_mm256_andnot_si256(mask,inp),4) );
		r0 ^= r1;

		_mm256_store_si256( (__m256i*)(temp) , r0 );
		for(unsigned i=0;i<rem;i++) a[i] = temp[i];
	}
#endif
}



static inline
void gf256v_madd_avx2( uint8_t * accu_c, const uint8_t * a , uint8_t _b, unsigned _num_byte ) {
	unsigned b = _b;
	__m256i m_tab = _mm256_load_si256( (__m256i*) (__gf256_mul + 32*b) );
	__m256i ml = _mm256_permute2x128_si256( m_tab , m_tab , 0 );
	__m256i mh = _mm256_permute2x128_si256( m_tab , m_tab , 0x11 );
	__m256i mask = _mm256_load_si256( (__m256i*) __mask_low );

#ifdef _AVX2_USE_TOOL_FUNCS_
	linearmap_8x8_accu_ymm( accu_c , a , ml , mh , mask , _num_byte );
#else
	unsigned n_32 = _num_byte>>5;
	for(unsigned i=0;i<n_32;i++) {
		__m256i inp = _mm256_loadu_si256( (__m256i*)(a+i*32) );
		__m256i out = _mm256_loadu_si256( (__m256i*)(accu_c+i*32) );
		__m256i r0 = _mm256_shuffle_epi8(ml, inp&mask );
		__m256i r1 = _mm256_shuffle_epi8(mh, _mm256_srli_epi16(_mm256_andnot_si256(mask,inp),4) );
		r0 ^= r1^out;
		_mm256_storeu_si256( (__m256i*)(accu_c+i*32) , r0 );
	}

	unsigned rem = _num_byte&31;
	if( rem ) {
		a += (n_32<<5);
		accu_c += (n_32<<5);
		uint8_t temp[32] __attribute__((aligned(32)));
		for(unsigned i=0;i<rem;i++) temp[i] = a[i];
		__m256i inp = _mm256_load_si256( (__m256i*)(temp) );
		for(unsigned i=0;i<rem;i++) temp[i] = accu_c[i];
		__m256i out = _mm256_load_si256( (__m256i*)(temp) );

		__m256i r0 = _mm256_shuffle_epi8(ml, inp&mask );
		__m256i r1 = _mm256_shuffle_epi8(mh, _mm256_srli_epi16(_mm256_andnot_si256(mask,inp),4) );
		r0 ^= r1^out;
		_mm256_store_si256( (__m256i*)(temp) , r0 );

		for(unsigned i=0;i<rem;i++) accu_c[i] = temp[i];
	}
#endif
}



/////////////////////////////////////////////////////////////////////////////////////



static inline
uint8_t gf16v_dot_avx2( const uint8_t * a , const uint8_t * b , unsigned n_byte )
{
	uint8_t v1[32] __attribute__((aligned(32)));
	uint8_t v2[32] __attribute__((aligned(32)));
	uint8_t v3[32] __attribute__((aligned(32)));

	unsigned n_xmm = n_byte>>4;
	unsigned n_rem = n_byte&15;
	__m256i r = _mm256_setzero_si256();
	for(unsigned i=0;i<n_xmm;i++) {
		__m128i inp1 = _mm_loadu_si128(  (__m128i*)(a+i*16) );
		__m128i inp2 = _mm_loadu_si128(  (__m128i*)(b+i*16) );
		gf16v_split_16to32_sse( (__m128i *)v1 , inp1 );
		gf16v_split_16to32_sse( (__m128i *)v2 , inp2 );
		r ^= tbl32_gf16_mul( _mm256_load_si256( (__m256i*)(v1) ) , _mm256_load_si256( (__m256i*)(v2) ) );
	}
	if( n_rem ) {
		_mm_store_si128( (__m128i*)(v3) , _mm_setzero_si128() );
		for(unsigned i=0;i<n_rem;i++) v3[i] = a[n_xmm*16+i];
		__m128i inp1 = _mm_load_si128(  (__m128i*)(v3) );
		for(unsigned i=0;i<n_rem;i++) v3[i] = b[n_xmm*16+i];
		__m128i inp2 = _mm_load_si128(  (__m128i*)(v3) );
		gf16v_split_16to32_sse( (__m128i *)v1 , inp1 );
		gf16v_split_16to32_sse( (__m128i *)v2 , inp2 );
		r ^= tbl32_gf16_mul( _mm256_load_si256( (__m256i*)(v1) ) , _mm256_load_si256( (__m256i*)(v2) ) );
	}
	__m128i rr = _mm256_extracti128_si256(r, 1)^_mm256_castsi256_si128(r);
	rr ^= _mm_srli_si128(rr,8);
	rr ^= _mm_srli_si128(rr,4);
	rr ^= _mm_srli_si128(rr,2);
	rr ^= _mm_srli_si128(rr,1);
	rr ^= _mm_srli_epi16(rr,4);
	return _mm_extract_epi16(rr,0)&0xf;
}



//////////////////////////  matrix multiplications  /////////////////////////////////////////////////



void gf16mat_prod_multab_avx2( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * multab );

void gf16mat_prod_avx2( uint8_t * c , const uint8_t * mat_a , unsigned a_h_byte , unsigned a_w , const uint8_t * b );

void gf256mat_prod_multab_avx2( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * multab );

void gf256mat_prod_avx2( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b );



/////////////////////////  gaussian elimination   /////////////////////////////////////////////


unsigned gf16mat_solve_linear_eq_avx2( uint8_t * sol , const uint8_t * inp_mat , const uint8_t * c_terms , unsigned n );




#ifdef  __cplusplus
}
#endif



#endif
