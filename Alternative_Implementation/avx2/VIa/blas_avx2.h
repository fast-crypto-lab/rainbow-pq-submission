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

/*
static inline
void gf256v_predicated_add_avx2( uint8_t * accu_b, uint8_t predicate , const uint8_t * a , unsigned _num_byte ) {
	//uint8_t temp[32] __attribute__((aligned(32)));
	uint32_t pr_u32 = ((uint32_t)0)-((uint32_t)predicate);
	__m256i pr_u256 = _mm256_set1_epi32( pr_u32 );

	unsigned n_ymm = (_num_byte)>>5;
	unsigned i=0;
	for(;i<n_ymm;i++) {
		__m256i inp = _mm256_loadu_si256( (__m256i*) (a+i*32) );
		__m256i out = _mm256_loadu_si256( (__m256i*) (accu_b+i*32) );
		out ^= (inp&pr_u256);
		_mm256_storeu_si256( (__m256i*) (accu_b+i*32) , out );
	}
	if( 0 != (_num_byte&0x1f) ) {
		n_ymm <<= 5;
		gf256v_predicated_add_sse( accu_b + n_ymm , predicate , a + n_ymm , _num_byte&0x1f );
	}
}
*/


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










/////////////////  GF( 16 ) /////////////////////////////////////





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





static inline
void gf16mat_prod_multab_avx2( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * multab ) {
	assert( n_A_width <= 256 );
	assert( n_A_vec_byte <= 128 );
	if( 16 >= n_A_vec_byte ) { return gf16mat_prod_multab_sse(c,matA,n_A_vec_byte,n_A_width,multab); }

	__m256i mask_f = _mm256_load_si256( (__m256i*)__mask_low);

	__m256i r0[4];
	__m256i r1[4];
	unsigned n_ymm = ((n_A_vec_byte + 31)>>5);
	for(unsigned i=0;i<n_ymm;i++) r0[i] = _mm256_setzero_si256();
	for(unsigned i=0;i<n_ymm;i++) r1[i] = _mm256_setzero_si256();

	for(unsigned i=0;i<n_A_width;i++) {
		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*16) );
		__m256i mt = _mm256_inserti128_si256(_mm256_castsi128_si256(ml),ml,1);
		for(unsigned j=0;j<n_ymm;j++) {
			__m256i inp = _mm256_loadu_si256( (__m256i*)(matA+j*32) );
			r0[j] ^= _mm256_shuffle_epi8( mt , inp&mask_f );
			r1[j] ^= _mm256_shuffle_epi8( mt , _mm256_srli_epi16(inp,4)&mask_f );
		}
		matA += n_A_vec_byte;
	}
	uint8_t temp[128] __attribute__((aligned(32)));
	for(unsigned i=0;i<n_ymm;i++) _mm256_store_si256( (__m256i*)(temp + i*32) , r0[i]^_mm256_slli_epi16(r1[i],4) );
	for(unsigned i=0;i<n_A_vec_byte;i++) c[i] = temp[i];
}

#if 0
static inline
void gf16mat_prod_avx2( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	assert( n_A_width <= 128 );
	assert( n_A_vec_byte <= 64 );

	uint8_t multab[128*16] __attribute__((aligned(32)));
	gf16v_generate_multab_sse( multab , b , n_A_width );

	gf16mat_prod_multab_avx2( c , matA , n_A_vec_byte , n_A_width , multab );
}
#else
static inline
void gf16mat_prod_avx2( uint8_t * c , const uint8_t * mat_a , unsigned a_h_byte , unsigned a_w , const uint8_t * b ) {
	assert( a_w <= 256 );
	assert( a_h_byte <= 128 );
	if( 16 >= a_h_byte ) { return gf16mat_prod_sse(c,mat_a,a_h_byte,a_w,b); }

	__m256i mask_f = _mm256_load_si256( (__m256i*)__mask_low);

	__m256i r0[4];
	__m256i r1[4];
	unsigned n_ymm = ((a_h_byte+31)>>5);
	for(unsigned i=0;i<n_ymm;i++) r0[i] = _mm256_setzero_si256();
	for(unsigned i=0;i<n_ymm;i++) r1[i] = _mm256_setzero_si256();

	uint8_t _x[256] __attribute__((aligned(32)));
	gf16v_split_sse( _x , b , a_w );
	for(unsigned i=0;i < ((a_w+31)>>5); i++) {
		__m256i lxi = tbl32_gf16_log( _mm256_load_si256( (__m256i*)(_x+i*32) ) );
		_mm256_store_si256( (__m256i*)(_x+i*32) , lxi );
	}

	for(unsigned i=0;i< a_w -1;i++) {
		_x[0] = _x[i];
		__m256i ml = _mm256_broadcastb_epi8( _mm_load_si128((__m128i*)_x) );
		for(unsigned j=0;j<n_ymm;j++) {
			__m256i inp = _mm256_loadu_si256( (__m256i*)(mat_a+j*32) );
			r0[j] ^= tbl32_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[j] ^= tbl32_gf16_mul_log( _mm256_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
		mat_a += a_h_byte;
	}
	unsigned n_32 = (a_h_byte>>5);
	unsigned n_32_rem = a_h_byte&0x1f;
	{
		/// last column
		unsigned i=a_w-1;
		_x[0] = _x[i];
		__m256i ml = _mm256_broadcastb_epi8( _mm_load_si128((__m128i*)_x) );
		for(unsigned j=0;j<n_32;j++) {
			__m256i inp = _mm256_loadu_si256( (__m256i*)(mat_a+j*32) );
			r0[j] ^= tbl32_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[j] ^= tbl32_gf16_mul_log( _mm256_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
		if( n_32_rem ) {
			unsigned j = n_32;
			__m256i inp = _load_ymm( mat_a+j*32 , n_32_rem );
			r0[j] ^= tbl32_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[j] ^= tbl32_gf16_mul_log( _mm256_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
	}

	for(unsigned i=0;i<n_32;i++) _mm256_storeu_si256( (__m256i*)(c + i*32) , r0[i]^_mm256_slli_epi16(r1[i],4) );
	if( n_32_rem ) _store_ymm( c + n_32*32 , n_32_rem , r0[n_32]^_mm256_slli_epi16(r1[n_32],4) );
}
#endif






static inline
uint8_t _if_zero_then_0xf(uint8_t p ) {
	return (p-1)>>4;
}

static inline
unsigned _linear_solver_32x32_avx2( uint8_t * r , const uint8_t * mat_32x32 , const uint8_t * cc )
{

	uint8_t mat[32*32] __attribute__((aligned(32)));
	for(unsigned i=0;i<32;i++) gf16v_split_sse( mat + i*32 , mat_32x32 + i*16 , 32 );

	__m256i mask_f = _mm256_load_si256((__m256i const *) __mask_low);

	uint8_t temp[32] __attribute__((aligned(32)));
	uint8_t pivots[32] __attribute__((aligned(32)));

	uint8_t rr8 = 1;
	for(unsigned i=0;i<32;i++) {
		for(unsigned j=0;j<32;j++) pivots[j] = mat[j*32+i];
			if( 0 == i ) {
				gf16v_split_sse( temp , cc , 32 );
				for(unsigned j=0;j<32;j++) mat[j*32] = temp[j];
			}
		__m256i rowi = _mm256_load_si256( (__m256i*)(mat+i*32) );
		for(unsigned j=i+1;j<32;j++) {
			temp[0] = _if_zero_then_0xf( pivots[i] );
			__m256i mask_zero = _mm256_broadcastb_epi8(_mm_load_si128((__m128i*)temp));

			__m256i rowj = _mm256_load_si256( (__m256i*)(mat+j*32) );
			rowi ^= mask_zero&rowj;
			//rowi ^= predicate_zero&(*(__m256i*)(mat+j*32));
			pivots[i] ^= temp[0]&pivots[j];
		}
		uint8_t is_pi_nz = _if_zero_then_0xf(pivots[i]);
		is_pi_nz = ~is_pi_nz;
		rr8 &= is_pi_nz;

		temp[0] = pivots[i];
		__m128i inv_rowi = tbl_gf16_inv( _mm_load_si128((__m128i*)temp) );
		pivots[i] = _mm_extract_epi8( inv_rowi , 0 );

		__m256i log_pivots = tbl32_gf16_log( _mm256_load_si256( (__m256i*)pivots ) );
		_mm256_store_si256( (__m256i*)pivots , log_pivots );

		temp[0] = pivots[i];
		__m256i logpi = _mm256_broadcastb_epi8( _mm_load_si128((__m128i*)temp) );
		rowi = tbl32_gf16_mul_log( rowi , logpi , mask_f );
		__m256i log_rowi = tbl32_gf16_log( rowi );
		for(unsigned j=0;j<32;j++) {
			if(i==j) {
				_mm256_store_si256( (__m256i*)(mat+j*32) , rowi );
				continue;
			}
			__m256i rowj = _mm256_load_si256( (__m256i*)(mat+j*32) );
			temp[0] = pivots[j];
			__m256i logpj = _mm256_broadcastb_epi8( _mm_load_si128((__m128i*)temp) );
			rowj ^= tbl32_gf16_mul_log_log( log_rowi , logpj , mask_f );
			_mm256_store_si256( (__m256i*)(mat+j*32) , rowj );
		}
	}

	for(unsigned i=0;i<32;i+=2) {
		//gf16v_set_ele( r , i , mat[i*32] );
		r[i>>1] = mat[i*32]| (mat[(i+1)*32]<<4);
	}
	return rr8;
}

static inline
unsigned gf16mat_solve_linear_eq_avx2( uint8_t * sol , const uint8_t * inp_mat , const uint8_t * c_terms , unsigned n )
{
	if( 32 == n ) {
		return _linear_solver_32x32_avx2( sol , inp_mat , c_terms );
	} else  {
		return gf16mat_solve_linear_eq_sse( sol , inp_mat , c_terms , n );
	}
}





///////////////////////////////  GF( 256 ) ////////////////////////////////////////////////////





static inline
void gf256mat_prod_multab_avx2( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * multab ) {
	assert( n_A_width <= 256 );
	assert( n_A_vec_byte <= 256 );
	if( 16 >= n_A_vec_byte ) { return gf256mat_prod_multab_sse(c,matA,n_A_vec_byte,n_A_width,multab); }

	__m256i mask_f = _mm256_load_si256((__m256i const *) __mask_low);

	__m256i r[8];
	unsigned n_ymm = ((n_A_vec_byte + 31)>>5);
	for(unsigned i=0;i<n_ymm;i++) r[i] = _mm256_setzero_si256();

	for(unsigned i=0;i<n_A_width;i++) {
		__m256i mt = _mm256_load_si256( (__m256i*)( multab + i*32) );
		__m256i ml = _mm256_permute2x128_si256(mt,mt,0x00 );
		__m256i mh = _mm256_permute2x128_si256(mt,mt,0x11 );
		for(unsigned j=0;j<n_ymm;j++) {
			__m256i inp = _mm256_loadu_si256( (__m256i*)(matA+j*32) );
			r[j] ^= _mm256_shuffle_epi8( ml , inp&mask_f );
			r[j] ^= _mm256_shuffle_epi8( mh , _mm256_srli_epi16(inp,4)&mask_f );
		}
		matA += n_A_vec_byte;
	}
	uint8_t r8[256] __attribute__((aligned(32)));
	for(unsigned i=0;i<n_ymm;i++) _mm256_store_si256( (__m256i*)(r8 + i*32) , r[i] );
	for(unsigned i=0;i<n_A_vec_byte;i++) c[i] = r8[i];
}


#if 0
// slower
static inline
void gf256mat_prod_secure_avx2( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	assert( n_A_width <= 128 );
	assert( n_A_vec_byte <= 80 );

	uint8_t multab[256*16] __attribute__((aligned(32)));
	gf256v_generate_multab_sse( multab , b , n_A_width );

	gf256mat_prod_multab_avx2( c , matA , n_A_vec_byte , n_A_width , multab );
}
#else
static inline
void gf256mat_prod_avx2( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	assert( n_A_width <= 256 );
	assert( n_A_vec_byte <= 256 );
	if( 16 >= n_A_vec_byte ) { return gf256mat_prod_sse(c,matA,n_A_vec_byte,n_A_width,b); }

	__m256i mask_f = _mm256_load_si256( (__m256i*)__mask_low);

	__m256i r[8];
	unsigned n_ymm = ((n_A_vec_byte + 31)>>5);
	for(unsigned i=0;i<n_ymm;i++) r[i] = _mm256_setzero_si256();

	uint8_t x0[256] __attribute__((aligned(32)));
	uint8_t x1[256] __attribute__((aligned(32)));
	for(unsigned i=0;i<n_A_width;i++) x0[i] = b[i];
	for(unsigned i=0;i< ((n_A_width+31)>>5);i++) {
		__m256i inp = _mm256_load_si256((__m256i*)(x0+i*32));
		__m256i i0 = inp&mask_f;
		__m256i i1 = _mm256_srli_epi16(inp,4)&mask_f;
		_mm256_store_si256((__m256i*)(x0+i*32),tbl32_gf16_log(i0));
		_mm256_store_si256((__m256i*)(x1+i*32),tbl32_gf16_log(i1));
	}

	for(unsigned i=0;i<n_A_width;i++) {
		x0[0] = x0[i]; __m256i m0 = _mm256_broadcastb_epi8( _mm_load_si128((__m128i*)x0) );
		x1[0] = x1[i]; __m256i m1 = _mm256_broadcastb_epi8( _mm_load_si128((__m128i*)x1) );
		//__m128i ml = _mm_set1_epi8(x[i]);
		for(unsigned j=0;j<n_ymm;j++) {
			__m256i inp = _mm256_loadu_si256( (__m256i*)(matA+j*32) );
			__m256i l_i0 = tbl32_gf16_log(inp&mask_f);
			__m256i l_i1 = tbl32_gf16_log(_mm256_srli_epi16(inp,4)&mask_f);

			__m256i ab0 = tbl32_gf16_mul_log_log( l_i0 , m0 , mask_f );
			__m256i ab1 = tbl32_gf16_mul_log_log( l_i1 , m0 , mask_f )^tbl32_gf16_mul_log_log( l_i0 , m1 , mask_f );
			__m256i ab2 = tbl32_gf16_mul_log_log( l_i1 , m1 , mask_f );
			__m256i ab2x8 = tbl32_gf16_mul_0x8( ab2 );

			r[j] ^= ab0 ^ ab2x8 ^ _mm256_slli_epi16( ab1^ab2 , 4 );
		}
		matA += n_A_vec_byte;
	}
	for(unsigned i=0;i<n_ymm;i++) _mm256_store_si256( (__m256i*)(x0 + i*32) , r[i] );
	for(unsigned i=0;i<n_A_vec_byte;i++) c[i] = x0[i];
}
#endif







#ifdef  __cplusplus
}
#endif



#endif
