#ifndef _BLAS_SSE_H_
#define _BLAS_SSE_H_

#include "gf16.h"


#include "emmintrin.h"
#include "tmmintrin.h"

#include "blas_config.h"
#include "assert.h"

#include "gf16_sse.h"



#ifdef  __cplusplus
extern  "C" {
#endif






static inline
__m128i _load_xmm( const uint8_t *a , unsigned _num_byte ) {
	uint8_t temp[32] __attribute__((aligned(32)));
	assert( 16 >= _num_byte );
	assert( 0 < _num_byte );
	for(unsigned i=0;i<_num_byte;i++) temp[i] = a[i];
	return _mm_load_si128((__m128i*)temp);
}

static inline
void _store_xmm( uint8_t *a , unsigned _num_byte , __m128i data ) {
	uint8_t temp[32] __attribute__((aligned(32)));
	assert( 16 >= _num_byte );
	assert( 0 < _num_byte );
	_mm_store_si128((__m128i*)temp,data);
	for(unsigned i=0;i<_num_byte;i++) a[i] = temp[i];
}


static inline
void linearmap_8x8_sse( uint8_t * a , __m128i ml , __m128i mh , __m128i mask , unsigned _num_byte ) {
	unsigned n_16 = _num_byte>>4;
	for(unsigned i=0;i<n_16;i++) {
		__m128i inp = _mm_loadu_si128( (__m128i*)(a+i*16) );
		__m128i r0 = linear_transform_8x8_128b( ml , mh , inp , mask );
		_mm_storeu_si128( (__m128i*)(a+i*16) , r0 );
	}

	unsigned rem = _num_byte&15;
	if( rem ) {
		__m128i inp = _load_xmm( a + n_16*16 , rem );
		__m128i r0 = linear_transform_8x8_128b( ml , mh , inp , mask );
		_store_xmm( a + n_16*16 , rem , r0 );
	}
}


static inline
void linearmap_8x8_accu_sse( uint8_t * accu_c, const uint8_t * a , __m128i ml , __m128i mh , __m128i mask , unsigned _num_byte ) {
	unsigned n_16 = _num_byte>>4;
	for(unsigned i=0;i<n_16;i++) {
		__m128i inp = _mm_loadu_si128( (__m128i*)(a+i*16) );
		__m128i out = _mm_loadu_si128( (__m128i*)(accu_c+i*16) );
		__m128i r0 = linear_transform_8x8_128b( ml , mh , inp , mask );
		r0 ^= out;
		_mm_storeu_si128( (__m128i*)(accu_c+i*16) , r0 );
	}
	unsigned rem = _num_byte&15;
	if( rem ) {
		__m128i inp = _load_xmm( a + n_16*16 , rem );
		__m128i out = _load_xmm( accu_c + n_16*16 , rem );
		__m128i r0 = linear_transform_8x8_128b( ml , mh , inp , mask );
		r0 ^= out;
		_store_xmm( accu_c + n_16*16 , rem , r0 );
	}
}





//////////////////////   basic functions  ///////////////////////////////////////////////




static inline
void gf256v_add_sse( uint8_t * accu_b, const uint8_t * a , unsigned _num_byte ) {
	//uint8_t temp[32] __attribute__((aligned(32)));
	unsigned n_xmm = (_num_byte)>>4;
	for(unsigned i=0;i<n_xmm;i++) {
		__m128i inp = _mm_loadu_si128( (__m128i*) (a+i*16) );
		__m128i out = _mm_loadu_si128( (__m128i*) (accu_b+i*16) );
		out ^= inp;
		_mm_storeu_si128( (__m128i*) (accu_b+i*16) , out );
	}
	if( 0 == (_num_byte&0xf) ) return;
	for(unsigned j=0;j<(_num_byte&0xf);j++) {
		accu_b[n_xmm*16+j] ^= a[n_xmm*16+j];
	}
}




///////////////////////////////



extern const unsigned char __mask_low[];
extern const unsigned char * __gf16_mul;
extern const unsigned char __gf256_mul[];



static inline
void gf16v_mul_scalar_sse( uint8_t * a, uint8_t gf16_b , unsigned _num_byte ) {
	unsigned b = gf16_b&0xf;
	__m128i ml = _mm_load_si128( (__m128i*) (__gf16_mul + 32*b) );
	__m128i mh = _mm_load_si128( (__m128i*) (__gf16_mul + 32*b + 16) );
	__m128i mask = _mm_set1_epi8(0xf);

	linearmap_8x8_sse( a, ml , mh , mask , _num_byte );
}



static inline
void gf16v_madd_sse( uint8_t * accu_c, const uint8_t * a , uint8_t gf16_b, unsigned _num_byte ) {
	unsigned b = gf16_b&0xf;
	__m128i ml = _mm_load_si128( (__m128i*) (__gf16_mul + 32*b) );
	__m128i mh = _mm_load_si128( (__m128i*) (__gf16_mul + 32*b + 16) );
	__m128i mask = _mm_set1_epi8(0xf);

	linearmap_8x8_accu_sse( accu_c , a , ml , mh , mask , _num_byte );
}


static inline
void gf256v_mul_scalar_sse( uint8_t * a, uint8_t _b , unsigned _num_byte ) {
	unsigned b = _b;
	__m128i ml = _mm_load_si128( (__m128i*) (__gf256_mul + 32*b) );
	__m128i mh = _mm_load_si128( (__m128i*) (__gf256_mul + 32*b + 16) );
	__m128i mask = _mm_set1_epi8(0xf);

	linearmap_8x8_sse( a, ml , mh , mask , _num_byte );
}

static inline
void gf256v_madd_sse( uint8_t * accu_c, const uint8_t * a , uint8_t _b, unsigned _num_byte ) {
	unsigned b = _b;
	__m128i ml = _mm_load_si128( (__m128i*) (__gf256_mul + 32*b) );
	__m128i mh = _mm_load_si128( (__m128i*) (__gf256_mul + 32*b + 16) );
	__m128i mask = _mm_set1_epi8(0xf);

	linearmap_8x8_accu_sse( accu_c , a , ml , mh , mask , _num_byte );
}












///////////////// transpose 16x16 //////////////////////////////


static inline
void transpose_16x16_sse( uint8_t * r , const uint8_t * a ) {
//	for(unsigned j=0;j<16;j++)
//		for(unsigned k=0;k<16;k++) r[j*16+k] = a[k*16+j];

	__m128i a0 = _mm_load_si128( (__m128i*) a );
	__m128i a1 = _mm_load_si128( (__m128i*) (a+16) );
	__m128i b0 = _mm_unpacklo_epi8( a0 , a1 );
	__m128i b1 = _mm_unpackhi_epi8( a0 , a1 );

	__m128i a2 = _mm_load_si128( (__m128i*) (a+16*2) );
	__m128i a3 = _mm_load_si128( (__m128i*) (a+16*3) );
	__m128i b2 = _mm_unpacklo_epi8( a2 , a3 );
	__m128i b3 = _mm_unpackhi_epi8( a2 , a3 );

	__m128i c0 = _mm_unpacklo_epi16( b0 , b2 );
	__m128i c1 = _mm_unpacklo_epi16( b1 , b3 );
	__m128i c2 = _mm_unpackhi_epi16( b0 , b2 );
	__m128i c3 = _mm_unpackhi_epi16( b1 , b3 );

	__m128i a4 = _mm_load_si128( (__m128i*) (a+16*4) );
	__m128i a5 = _mm_load_si128( (__m128i*) (a+16*5) );
	__m128i b4 = _mm_unpacklo_epi8( a4 , a5 );
	__m128i b5 = _mm_unpackhi_epi8( a4 , a5 );

	__m128i a6 = _mm_load_si128( (__m128i*) (a+16*6) );
	__m128i a7 = _mm_load_si128( (__m128i*) (a+16*7) );
	__m128i b6 = _mm_unpacklo_epi8( a6 , a7 );
	__m128i b7 = _mm_unpackhi_epi8( a6 , a7 );

	__m128i c4 = _mm_unpacklo_epi16( b4 , b6 );
	__m128i c5 = _mm_unpacklo_epi16( b5 , b7 );
	__m128i c6 = _mm_unpackhi_epi16( b4 , b6 );
	__m128i c7 = _mm_unpackhi_epi16( b5 , b7 );

	__m128i d0 = _mm_unpacklo_epi32( c0 , c4 );
	__m128i d1 = _mm_unpacklo_epi32( c1 , c5 );
	__m128i d2 = _mm_unpacklo_epi32( c2 , c6 );
	__m128i d3 = _mm_unpacklo_epi32( c3 , c7 );
	__m128i d4 = _mm_unpackhi_epi32( c0 , c4 );
	__m128i d5 = _mm_unpackhi_epi32( c1 , c5 );
	__m128i d6 = _mm_unpackhi_epi32( c2 , c6 );
	__m128i d7 = _mm_unpackhi_epi32( c3 , c7 );
/////
	__m128i a8 = _mm_load_si128( (__m128i*) (a+16*8) );
	__m128i a9 = _mm_load_si128( (__m128i*) (a+16*9) );
	__m128i b8 = _mm_unpacklo_epi8( a8 , a9 );
	__m128i b9 = _mm_unpackhi_epi8( a8 , a9 );

	__m128i a10 = _mm_load_si128( (__m128i*) (a+16*10) );
	__m128i a11 = _mm_load_si128( (__m128i*) (a+16*11) );
	__m128i b10 = _mm_unpacklo_epi8( a10 , a11 );
	__m128i b11 = _mm_unpackhi_epi8( a10 , a11 );

	__m128i c8 = _mm_unpacklo_epi16( b8 , b10 );
	__m128i c9 = _mm_unpacklo_epi16( b9 , b11 );
	__m128i c10 = _mm_unpackhi_epi16( b8 , b10 );
	__m128i c11 = _mm_unpackhi_epi16( b9 , b11 );

	__m128i a12 = _mm_load_si128( (__m128i*) (a+16*12) );
	__m128i a13 = _mm_load_si128( (__m128i*) (a+16*13) );
	__m128i b12 = _mm_unpacklo_epi8( a12 , a13 );
	__m128i b13 = _mm_unpackhi_epi8( a12 , a13 );

	__m128i a14 = _mm_load_si128( (__m128i*) (a+16*14) );
	__m128i a15 = _mm_load_si128( (__m128i*) (a+16*15) );
	__m128i b14 = _mm_unpacklo_epi8( a14 , a15 );
	__m128i b15 = _mm_unpackhi_epi8( a14 , a15 );

	__m128i c12 = _mm_unpacklo_epi16( b12 , b14 );
	__m128i c13 = _mm_unpacklo_epi16( b13 , b15 );
	__m128i c14 = _mm_unpackhi_epi16( b12 , b14 );
	__m128i c15 = _mm_unpackhi_epi16( b13 , b15 );

	__m128i d8 = _mm_unpacklo_epi32( c8 , c12 );
	__m128i d9 = _mm_unpacklo_epi32( c9 , c13 );
	__m128i d10 = _mm_unpacklo_epi32( c10 , c14 );
	__m128i d11 = _mm_unpacklo_epi32( c11 , c15 );
	__m128i d12 = _mm_unpackhi_epi32( c8 , c12 );
	__m128i d13 = _mm_unpackhi_epi32( c9 , c13 );
	__m128i d14 = _mm_unpackhi_epi32( c10 , c14 );
	__m128i d15 = _mm_unpackhi_epi32( c11 , c15 );
//////
	__m128i e0 = _mm_unpacklo_epi64( d0 , d8 );
	__m128i e8 = _mm_unpackhi_epi64( d0 , d8 );
	_mm_store_si128( (__m128i*)( r + 16*0 ) , e0 );
	_mm_store_si128( (__m128i*)( r + 16*1 ) , e8 );

	__m128i e1 = _mm_unpacklo_epi64( d1 , d9 );
	__m128i e9 = _mm_unpackhi_epi64( d1 , d9 );
	_mm_store_si128( (__m128i*)( r + 16*8 ) , e1 );
	_mm_store_si128( (__m128i*)( r + 16*9 ) , e9 );

	__m128i e2 = _mm_unpacklo_epi64( d2 , d10 );
	__m128i e10 = _mm_unpackhi_epi64( d2 , d10 );
	_mm_store_si128( (__m128i*)( r + 16*4 ) , e2 );
	_mm_store_si128( (__m128i*)( r + 16*5 ) , e10 );

	__m128i e3 = _mm_unpacklo_epi64( d3 , d11 );
	__m128i e11 = _mm_unpackhi_epi64( d3 , d11 );
	_mm_store_si128( (__m128i*)( r + 16*0xc ) , e3 );
	_mm_store_si128( (__m128i*)( r + 16*0xd ) , e11 );

	__m128i e4 = _mm_unpacklo_epi64( d4 , d12 );
	__m128i e12 = _mm_unpackhi_epi64( d4 , d12 );
	_mm_store_si128( (__m128i*)( r + 16*2 ) , e4 );
	_mm_store_si128( (__m128i*)( r + 16*3 ) , e12 );

	__m128i e5 = _mm_unpacklo_epi64( d5 , d13 );
	__m128i e13 = _mm_unpackhi_epi64( d5 , d13 );
	_mm_store_si128( (__m128i*)( r + 16*0xa ) , e5 );
	_mm_store_si128( (__m128i*)( r + 16*0xb ) , e13 );

	__m128i e6 = _mm_unpacklo_epi64( d6 , d14 );
	__m128i e14 = _mm_unpackhi_epi64( d6 , d14 );
	_mm_store_si128( (__m128i*)( r + 16*6 ) , e6 );
	_mm_store_si128( (__m128i*)( r + 16*7 ) , e14 );

	__m128i e7 = _mm_unpacklo_epi64( d7 , d15 );
	__m128i e15 = _mm_unpackhi_epi64( d7 , d15 );
	_mm_store_si128( (__m128i*)( r + 16*0xe ) , e7 );
	_mm_store_si128( (__m128i*)( r + 16*15 ) , e15 );
}





/////////////////  GF( 16 ) /////////////////////////////////////




static inline
void gf16v_generate_multab_16_sse( uint8_t * _multab_byte , const uint8_t * _x0 )
{
	uint8_t multab[16*16] __attribute__((aligned(32)));
	__m128i cc = _mm_load_si128( (__m128i*) (_x0) );
	for(unsigned j=0;j<16;j++) {
		__m128i mt = _mm_load_si128( (__m128i*) (__gf16_mulx2 + 32*j) );
		_mm_store_si128( (__m128i*)(multab + j*16) , _mm_shuffle_epi8( mt, cc ) );
	}
	transpose_16x16_sse( _multab_byte , multab );
}

static inline
void gf16v_split_16to32_sse( __m128i * x_align , __m128i a )
{
	__m128i mask_f = _mm_set1_epi8(0xf);
	__m128i al = a&mask_f;
	__m128i ah = _mm_srli_epi16( a,4 )&mask_f;

	__m128i a0 = _mm_unpacklo_epi8( al , ah );
	__m128i a1 = _mm_unpackhi_epi8( al , ah );

	_mm_store_si128( x_align , a0 );
	_mm_store_si128( x_align + 1 , a1 );
}

static inline
void gf16v_generate_multab_sse( uint8_t * _multabs , const uint8_t * x , unsigned n )
{
	uint8_t _x[32] __attribute__((aligned(32)));

	unsigned n_32 = n>>5;
	unsigned n_16 = n>>4;
	unsigned n_16_rem = n&0xf;

	for(unsigned i=0;i<n_32;i++) {
		//for(unsigned j=0;j<16;j++) _x[j] = x[i*16+j];
		//__m128i x32 = _mm_load_si128( (__m128i*) _x );
		__m128i x32 = _mm_loadu_si128( (__m128i*) (x+i*16) );
		gf16v_split_16to32_sse( (__m128i*)_x , x32 );

		gf16v_generate_multab_16_sse( _multabs +  i*2*16*16 , _x );
		gf16v_generate_multab_16_sse( _multabs +  i*2*16*16 + 16*16 , _x + 16 );
	}
	if( n_16&1 ) {  /// n_16 is odd
		unsigned idx = n_16-1;

		for(unsigned j=0;j<8;j++) _x[j] = x[ idx*8 + j];

		__m128i x32 = _mm_load_si128( (__m128i*) _x );
		gf16v_split_16to32_sse( (__m128i*)_x , x32 );
		gf16v_generate_multab_16_sse( _multabs +  idx*16*16 , _x );
	}

	uint8_t multab[16*16] __attribute__((aligned(32)));
	if( n_16_rem ) {
		unsigned rem_byte = (n_16_rem + 1)/2;
		for(unsigned j=0;j<rem_byte;j++) _x[j] = x[n_16*8 + j];

		__m128i x32 = _mm_load_si128( (__m128i*) _x );
		gf16v_split_16to32_sse( (__m128i*)_x , x32 );
		gf16v_generate_multab_16_sse( multab , _x );

		for(unsigned j=0;j<n_16_rem;j++) {
			__m128i temp = _mm_load_si128( (__m128i*) ( multab + 16*j) );
			_mm_store_si128( (__m128i *) (_multabs + n_16*16*16 + 16*j) , temp );
		}
	}
}



static inline
void gf16v_split_sse( uint8_t * x_align32 , const uint8_t * _x , unsigned n_gf16 )
{
	assert( n <= 512 ); /// for spliting gf256v
	uint8_t * x = x_align32;
	unsigned n_byte = (n_gf16+1)/2;
	unsigned n_16 = n_byte>>4;
	unsigned n_16_rem = n_byte&0xf;
	__m128i mask_f = _mm_set1_epi8(0xf);
	for(unsigned i=0;i<n_16;i++) {
		__m128i inp = _mm_loadu_si128( (__m128i*)_x ); _x += 16;
		__m128i il = inp&mask_f;
		__m128i ih = _mm_srli_epi16(inp,4)&mask_f;
		_mm_store_si128( (__m128i*)( x+ 32*i ) , _mm_unpacklo_epi8(il,ih) );
		_mm_store_si128( (__m128i*)( x+ 32*i + 16 ) , _mm_unpackhi_epi8(il,ih) );
	}
	if( n_16_rem ) {
		unsigned i = n_16;
		__m128i inp = _load_xmm( _x , n_16_rem );
		__m128i il = inp&mask_f;
		__m128i ih = _mm_srli_epi16(inp,4)&mask_f;
		_mm_store_si128( (__m128i*)( x+ 32*i ) , _mm_unpacklo_epi8(il,ih) );
		_mm_store_si128( (__m128i*)( x+ 32*i + 16 ) , _mm_unpackhi_epi8(il,ih) );
	}
}


static inline
uint8_t gf16v_dot_sse( const uint8_t * a , const uint8_t * b , unsigned n_byte )
{
	uint8_t v1[32] __attribute__((aligned(32)));
	uint8_t v2[32] __attribute__((aligned(32)));
	uint8_t v3[32] __attribute__((aligned(32)));

	unsigned n_xmm = n_byte>>4;
	unsigned n_rem = n_byte&15;
	__m128i r = _mm_setzero_si128();
	for(unsigned i=0;i<n_xmm;i++) {
		__m128i inp1 = _mm_loadu_si128(  (__m128i*)(a+i*16) );
		__m128i inp2 = _mm_loadu_si128(  (__m128i*)(b+i*16) );
		gf16v_split_16to32_sse( (__m128i *)v1 , inp1 );
		gf16v_split_16to32_sse( (__m128i *)v2 , inp2 );
		r ^= tbl_gf16_mul( _mm_load_si128( (__m128i*)(v1) ) , _mm_load_si128( (__m128i*)(v2) ) );
		r ^= tbl_gf16_mul( _mm_load_si128( (__m128i*)(v1+16) ) , _mm_load_si128( (__m128i*)(v2+16) ) );
	}
	if( n_rem ) {
		_mm_store_si128( (__m128i*)(v3) , _mm_setzero_si128() );
		for(unsigned i=0;i<n_rem;i++) v3[i] = a[n_xmm*16+i];
		__m128i inp1 = _mm_load_si128(  (__m128i*)(v3) );
		for(unsigned i=0;i<n_rem;i++) v3[i] = b[n_xmm*16+i];
		__m128i inp2 = _mm_load_si128(  (__m128i*)(v3) );
		gf16v_split_16to32_sse( (__m128i *)v1 , inp1 );
		gf16v_split_16to32_sse( (__m128i *)v2 , inp2 );
		r ^= tbl_gf16_mul( _mm_load_si128( (__m128i*)(v1) ) , _mm_load_si128( (__m128i*)(v2) ) );
		r ^= tbl_gf16_mul( _mm_load_si128( (__m128i*)(v1+16) ) , _mm_load_si128( (__m128i*)(v2+16) ) );
	}
	r ^= _mm_srli_si128(r,8);
	r ^= _mm_srli_si128(r,4);
	r ^= _mm_srli_si128(r,2);
	r ^= _mm_srli_si128(r,1);
	r ^= _mm_srli_epi16(r,4);
	return _mm_extract_epi16(r,0)&0xf;
}



static inline
void gf16mat_prod_multab_sse( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * multab ) {
	assert( n_A_width <= 224 );
	assert( n_A_width > 0 );
	assert( n_A_vec_byte <= 112 );

	__m128i mask_f = _mm_set1_epi8(0xf);

	__m128i r0[7];
	__m128i r1[7];

	unsigned n_xmm = ((n_A_vec_byte + 15)>>4);
	for(unsigned i=0;i<n_xmm;i++) r0[i] = _mm_setzero_si128();
	for(unsigned i=0;i<n_xmm;i++) r1[i] = _mm_setzero_si128();

	for(unsigned i=0;i<n_A_width-1;i++) {
		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*16) );
		//__m128i mh = _mm_slli_epi16( ml , 4 );
		for(unsigned j=0;j<n_xmm;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(matA+j*16) );
			r0[j] ^= _mm_shuffle_epi8( ml , inp&mask_f );
			r1[j] ^= _mm_shuffle_epi8( ml , _mm_srli_epi16(inp,4)&mask_f );
		}
		matA += n_A_vec_byte;
	}
	unsigned n_16 = (n_A_vec_byte>>4);
	unsigned n_16_rem = n_A_vec_byte&0xf;
	{
		/// last column
		unsigned i=n_A_width-1;

		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*16) );
		for(unsigned j=0;j<n_16;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(matA+j*16) );
			r0[j] ^= _mm_shuffle_epi8( ml , inp&mask_f );
			r1[j] ^= _mm_shuffle_epi8( ml , _mm_srli_epi16(inp,4)&mask_f );
		}
		if( n_16_rem ) {
			__m128i inp = _load_xmm( matA + n_16*16 , n_16_rem );
			r0[n_16] ^= _mm_shuffle_epi8( ml , inp&mask_f );
			r1[n_16] ^= _mm_shuffle_epi8( ml , _mm_srli_epi16(inp,4)&mask_f );
		}
	}

	for(unsigned i=0;i<n_16;i++) _mm_storeu_si128( (__m128i*)(c + i*16) , r0[i]^_mm_slli_epi16(r1[i],4) );
	if( n_16_rem ) _store_xmm( c + n_16*16 , n_16_rem , r0[n_16]^_mm_slli_epi16(r1[n_16],4) );
}



static inline
uint8_t _gf16v_get_ele( const uint8_t *a , unsigned i ) {
	uint8_t r = a[i>>1];
	if( 0 == (i&1) ) return r&0xf;
	else return r>>4;
}


#if 0
/// slower
static inline
void gf16mat_prod_sse( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	assert( n_A_width <= 128 );
	assert( n_A_vec_byte <= 64 );

	uint8_t multab[128*16] __attribute__((aligned(32)));
	gf16v_generate_multab_sse( multab , b , n_A_width );

	gf16mat_prod_multab_sse( c , matA , n_A_vec_byte , n_A_width , multab );
}
#else
/// faster
static inline
void gf16mat_prod_sse( uint8_t * c , const uint8_t * mat_a , unsigned a_h_byte , unsigned a_w , const uint8_t * b ) {
	assert( a_w <= 224 );
	assert( a_h_byte <= 112 );

	__m128i mask_f = _mm_set1_epi8(0xf);

	__m128i r0[7];
	__m128i r1[7];
	unsigned n_xmm = ((a_h_byte+15)>>4);
	for(unsigned i=0;i<n_xmm;i++) r0[i] = _mm_setzero_si128();
	for(unsigned i=0;i<n_xmm;i++) r1[i] = _mm_setzero_si128();

	uint8_t _x[224] __attribute__((aligned(32)));
	gf16v_split_sse( _x , b , a_w );
	for(unsigned i=0;i < ((a_w+15)>>4); i++) {
		__m128i xi = _mm_load_si128( (__m128i*)(_x+i*16) );
		_mm_store_si128( (__m128i*)(_x+i*16) , tbl_gf16_log( xi ) );
	}

	for(unsigned i=0;i< a_w -1;i++) {
		__m128i ml = _mm_set1_epi8( _x[i] );
		for(unsigned j=0;j<n_xmm;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(mat_a+j*16) );
			r0[j] ^= tbl_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[j] ^= tbl_gf16_mul_log( _mm_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
		mat_a += a_h_byte;
	}
	unsigned n_16 = (a_h_byte>>4);
	unsigned n_16_rem = a_h_byte&0xf;
	{
		/// last column
		unsigned i=a_w-1;
		__m128i ml = _mm_set1_epi8( _x[i] );
		for(unsigned j=0;j<n_16;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(mat_a+j*16) );
			r0[j] ^= tbl_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[j] ^= tbl_gf16_mul_log( _mm_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
		if( n_16_rem ) {
			__m128i inp = _load_xmm( mat_a + n_16*16 , n_16_rem );
			r0[n_16] ^= tbl_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[n_16] ^= tbl_gf16_mul_log( _mm_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
	}

	for(unsigned i=0;i<n_16;i++) _mm_storeu_si128( (__m128i*)(c + i*16) , r0[i]^_mm_slli_epi16(r1[i],4) );
	if( n_16_rem ) _store_xmm( c + n_16*16 , n_16_rem , r0[n_16]^_mm_slli_epi16(r1[n_16],4) );
}
#endif




/// access aligned memory.
static inline
unsigned _gf16mat_gauss_elim_sse( uint8_t * mat , unsigned h , unsigned w_byte )
{
	assert( 0==(w_byte&31) ); /// w_byte is a multiple of 32.
	assert( 224 >= h );

	uint8_t pivot_column[224] __attribute__((aligned(32)));
	uint8_t _multab[224*16] __attribute__((aligned(32)));
	__m128i mask_f = _mm_set1_epi8( 0xf );
	unsigned w_2 = w_byte;
	unsigned w_n_16 = w_byte>>4;

	uint8_t r8 = 1;
	for(unsigned i=0;i<h;i++) {
		unsigned offset_16 = i>>5;
		uint8_t * ai = mat + w_2*i;
		for(unsigned j=i+1;j<h;j++) {
			uint8_t * aj = mat + w_2*j;

			uint8_t predicate = !gf16_is_nonzero( _gf16v_get_ele(ai,i) );
			uint32_t pr_u32 = ((uint32_t)0)-((uint32_t)predicate);
			__m128i pr_u128 = _mm_set1_epi32( pr_u32 );
			for(unsigned k=offset_16;k<w_n_16;k++) {
				__m128i inp0 = _mm_load_si128( (__m128i*)(ai+k*16) );
				__m128i inp1 = _mm_load_si128( (__m128i*)(aj+k*16) );
				__m128i r = inp0^(inp1&pr_u128);
				_mm_store_si128( (__m128i*)(ai+k*16) , r );
			}
		}
		pivot_column[0] = _gf16v_get_ele( ai , i );
		r8 &= gf16_is_nonzero( pivot_column[0] );
		__m128i p128 = _mm_load_si128( (__m128i*) pivot_column );
		__m128i inv_p = tbl_gf16_inv( p128 );
		_mm_store_si128( (__m128i*) pivot_column , inv_p );
		pivot_column[i] = pivot_column[0];
		for(unsigned j=0;j<h;j++) {
			if( i==j) continue;
			uint8_t * aj = mat + w_2*j;
			pivot_column[j] = _gf16v_get_ele( aj , i );
		}
		unsigned h_16 = (h+15)>>4;
		for(unsigned j=0;j<h_16;j++) { gf16v_generate_multab_16_sse( _multab + j*16*16 , pivot_column + j*16 ); }

		{
			/// pivot row
			unsigned j=i;
			uint8_t * aj = mat + w_2*j;
			__m128i ml = _mm_load_si128( (__m128i*)(_multab + 16*j) );
			__m128i mh = _mm_slli_epi16( ml , 4 );
			for(unsigned k=offset_16;k<w_n_16;k++) {
				__m128i inp = _mm_load_si128( (__m128i*)(aj+k*16) );
				__m128i r = linear_transform_8x8_128b( ml , mh , inp , mask_f );
				_mm_store_si128( (__m128i*)(aj+k*16) , r );
			}
		}
		for(unsigned j=0;j<h;j++) {
			if( i == j ) continue;
			uint8_t * aj = mat + w_2*j;
			__m128i ml = _mm_load_si128( (__m128i*)(_multab + 16*j) );
			__m128i mh = _mm_slli_epi16( ml , 4 );
			for(unsigned k=offset_16;k<w_n_16;k++) {
				__m128i inp0 = _mm_load_si128( (__m128i*)(ai+k*16) );
				__m128i inp = _mm_load_si128( (__m128i*)(aj+k*16) );
				__m128i r = inp ^ linear_transform_8x8_128b( ml , mh , inp0 , mask_f );
				_mm_store_si128( (__m128i*)(aj+k*16) , r );
			}
		}
	}
	return r8;
}


static inline
unsigned gf16mat_gauss_elim_sse( uint8_t * mat , unsigned h , unsigned w )
{
	assert( 0 == (w&1) ); // w is even.
	assert( 448 >= w );
	assert( 224 >= h );

	uint8_t _mat[224*224] __attribute__((aligned(32)));
	unsigned w_2 = (w+1)>>1;
	unsigned w_byte_32 = ((w_2+31)>>5) <<5;

	for(unsigned i=0;i<h;i++) for(unsigned j=0;j<w_2;j++) _mat[i*w_byte_32+j] = mat[i*w_2+j];
	unsigned r = _gf16mat_gauss_elim_sse( _mat , h , w_byte_32 );
	for(unsigned i=0;i<h;i++) for(unsigned j=0;j<w_2;j++) mat[i*w_2+j] = _mat[i*w_byte_32+j];
	return r;
}




static inline
unsigned gf16mat_solve_linear_eq_sse( uint8_t * sol , const uint8_t * inp_mat , const uint8_t * c_terms , unsigned n )
{
        assert( 64 >= n );

	uint8_t mat[ 64*64 ] __attribute__((aligned(32)));
	uint8_t _c[64] __attribute__((aligned(32)));
	gf16v_split_sse( _c , c_terms , n );

	unsigned n_byte = (n+1)>>1;
	unsigned n_byte_32 = ((n_byte+1+31)>>5 ) <<5;

	for(unsigned i=0;i<n;i++) {
		for( unsigned j=0;j<n_byte;j++) mat[ i*n_byte_32+j ] = inp_mat[i*n_byte+j];
		mat[i*n_byte_32+n_byte] = _c[i];
	}
	unsigned r8 = _gf16mat_gauss_elim_sse( mat , n , n_byte_32 );
	for(unsigned i=0;i<n;i+=2) {
		sol[i>>1] = mat[i*n_byte_32+n_byte]|(mat[(i+1)*n_byte_32+n_byte]<<4);
	}
	return r8;
}





///////////////////////////////  GF( 256 ) ////////////////////////////////////////////////////




static inline
void gf256v_generate_multab_sse( uint8_t * _multabs , const uint8_t * _x , unsigned n )
{
	gf16v_generate_multab_sse( _multabs , _x , 2*n );

	__m128i mul_8 = _mm_load_si128( (__m128i*)(__gf16_mulx2 + 32*8) );
	for(unsigned i=0;i<n;i++) {
		__m128i ml = _mm_load_si128( (__m128i*) (_multabs+32*i) );
		__m128i mh = _mm_load_si128( (__m128i*) (_multabs+32*i+16) );
		__m128i ml256 = _mm_slli_epi16( mh,4) | ml;
		__m128i mh256 = _mm_slli_epi16(ml^mh,4)|_mm_shuffle_epi8(mul_8,mh);
		_mm_store_si128( (__m128i*) (_multabs+32*i) , ml256 );
		_mm_store_si128( (__m128i*) (_multabs+32*i+16) , mh256 );
	}
}


/// XXX: un-tested.
static inline
void gf256mat_prod_multab_sse( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * multab ) {
	assert( n_A_width <= 192 );
	assert( n_A_width > 0 );
	assert( n_A_vec_byte <= 192 );

	__m128i mask_f = _mm_set1_epi8(0xf);
	__m128i r[12];
	unsigned n_xmm = ((n_A_vec_byte + 15)>>4);
	for(unsigned i=0;i<n_xmm;i++) r[i] = _mm_setzero_si128();

	for(unsigned i=0;i<n_A_width-1;i++) {
		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*32) );
		__m128i mh = _mm_load_si128( (__m128i*)( multab + i*32+16) );
		for(unsigned j=0;j<n_xmm;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(matA+j*16) );
			r[j] ^= linear_transform_8x8_128b( ml , mh , inp , mask_f );
		}
		matA += n_A_vec_byte;
	}
	unsigned n_16 = (n_A_vec_byte>>4);
	unsigned n_16_rem = n_A_vec_byte&0xf;
	{
		/// last column
		unsigned i=n_A_width-1;
		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*32) );
		__m128i mh = _mm_load_si128( (__m128i*)( multab + i*32+16) );
		for(unsigned j=0;j<n_16;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(matA+j*16) );
			r[j] ^= linear_transform_8x8_128b( ml , mh , inp , mask_f );
		}
		if( n_16_rem ) {
			__m128i inp = _load_xmm( matA + n_16*16 , n_16_rem );
			r[n_16] ^= linear_transform_8x8_128b( ml , mh , inp , mask_f );
		}
	}
	for(unsigned i=0;i<n_16;i++) _mm_storeu_si128( (__m128i*)(c + i*16) , r[i] );
	if( n_16_rem ) _store_xmm( c + n_16*16 , n_16_rem , r[n_16] );
}



static inline
void gf256mat_prod_sse( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	assert( n_A_width <= 192 );
	assert( n_A_vec_byte <= 192 );

	uint8_t multab[192*16*2] __attribute__((aligned(32)));
	gf256v_generate_multab_sse( multab , b , n_A_width );

	gf256mat_prod_multab_sse( c , matA , n_A_vec_byte , n_A_width , multab );
}





///////////////////////////////////////////////////////////////////////////




static inline
unsigned _gf256mat_gauss_elim_sse( uint8_t * mat , unsigned h , unsigned w )
{
	assert( 0 == (w&15) );
	unsigned n_xmm = w>>4;

	__m128i mask_0 = _mm_setzero_si128();

	uint8_t rr8 = 1;
	for(unsigned i=0;i<h;i++) {
		unsigned char i_r = i&0xf;
		unsigned i_d = i>>4;

		uint8_t * mi = mat+i*w;

		for(unsigned j=i+1;j<h;j++) {
			__m128i piv_i = _mm_load_si128( (__m128i*)( mi + i_d*16 ) );
			uint8_t * mj = mat+j*w;
			__m128i piv_j = _mm_load_si128( (__m128i*)( mj + i_d*16 ) );

			__m128i is_madd = _mm_cmpeq_epi8( piv_i , mask_0 ) ^ _mm_cmpeq_epi8( piv_j , mask_0 );
			__m128i madd_mask = _mm_shuffle_epi8( is_madd , _mm_set1_epi8(i_r) );

			piv_i ^= madd_mask&piv_j;
			_mm_store_si128( (__m128i*)( mi+ i_d*16 ) , piv_i );
			for(unsigned k=i_d+1;k<n_xmm;k++) {
				piv_i = _mm_load_si128( (__m128i*)( mi + k*16 ) );
				piv_j = _mm_load_si128( (__m128i*)( mj + k*16 ) );

				piv_i ^= madd_mask&piv_j;
				_mm_store_si128( (__m128i*)( mi+ k*16 ) , piv_i );
			}
		}
		rr8 &= gf256_is_nonzero( mi[i] );

		__m128i _pivot = _mm_set1_epi8( mi[i] );
		__m128i _ip = tbl_gf256_inv( _pivot );
		for(unsigned k=i_d;k<n_xmm;k++) {
			__m128i rowi = _mm_load_si128( (__m128i*)(mi+k*16) );
			rowi = tbl_gf256_mul( rowi , _ip );
			_mm_store_si128( (__m128i*)(mi+k*16) , rowi );
		}

		for(unsigned j=0;j<h;j++) {
			if(i==j) continue;

			uint8_t * mj = mat+j*w;
			__m128i mm = _mm_set1_epi8( mj[i] );

			for(unsigned k=i_d;k<n_xmm;k++) {
				__m128i rowi = _mm_load_si128( (__m128i*)(mi+k*16) );
				rowi = tbl_gf256_mul( rowi , mm );
				rowi ^= _mm_load_si128( (__m128i*)(mj+k*16) );
				_mm_store_si128( (__m128i*)(mj+k*16) , rowi );
			}
		}
	}
	return rr8;
}


static inline
unsigned gf256mat_gauss_elim_sse( uint8_t * mat , unsigned h , unsigned w )
{
	assert( 512 >= w );
	assert( 256 >= h );

	uint8_t _mat[512*256] __attribute__((aligned(32)));
	unsigned w_16 = ((w+15)>>4) <<4;

	for(unsigned i=0;i<h;i++) for(unsigned j=0;j<w;j++) _mat[i*w_16+j] = mat[i*w+j];
	unsigned r = _gf256mat_gauss_elim_sse( _mat , h , w_16 );
	for(unsigned i=0;i<h;i++) for(unsigned j=0;j<w;j++) mat[i*w+j] = _mat[i*w_16+j];
	return r;
}


static inline
unsigned gf256mat_solve_linear_eq_sse( uint8_t * sol , const uint8_t * inp_mat , const uint8_t * c_terms, unsigned n )
{
	assert( 48 >= n );

	uint8_t mat[48*64] __attribute__((aligned(32)));
	unsigned mat_width = (((n+1)+15)>>4)<<4;

	for(unsigned i=0;i<n;i++) {
		for(unsigned j=0;j<n;j++) mat[i*mat_width+j] = inp_mat[i*n+j];
		mat[i*mat_width+n] = c_terms[i];
	}
	unsigned r = _gf256mat_gauss_elim_sse( mat , n , mat_width );
	for(unsigned i=0;i<n;i++) sol[i] = mat[i*mat_width+n];
	return r;
}






#ifdef  __cplusplus
}
#endif



#endif
