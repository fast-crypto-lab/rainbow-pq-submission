

#include "blas.h"

#include "blas_sse.h"

#include "mpkc.h"

#include "mpkc_sse.h"

#include "string.h" /// for memcpy

#include "assert.h"




static void gf16mpkc_mq_multab_n_m32_sse( uint8_t * z , const uint8_t * pk_mat , const uint8_t * multab , unsigned n, unsigned m )
{
	assert( 32 >= m );
	__m128i r_l = _mm_setzero_si128();
	__m128i r_h = _mm_setzero_si128();
	__m128i tmp_l;
	__m128i tmp_h;

	unsigned n_byte = (m+1)>>1;
	__m128i mask_f = _mm_set1_epi8( 0xf );

	for(unsigned i=0;i<n;i++){
		__m128i mt = _mm_load_si128( (const __m128i*) (multab+i*16) );
		__m128i inp = _mm_loadu_si128( (const __m128i*)(pk_mat) );
		r_l ^= _mm_shuffle_epi8( mt , inp&mask_f );
		r_h ^= _mm_shuffle_epi8( mt , _mm_srli_epi16(inp,4)&mask_f );
		pk_mat += n_byte;
	}

	for(unsigned k=0;k<n;k++) {
		tmp_l = _mm_setzero_si128();
		tmp_h = _mm_setzero_si128();
		__m128i mt0 = _mm_load_si128( (const __m128i*) (multab+k*16) );
		for(unsigned i=0;i<=k;i++) {
			__m128i mt = _mm_load_si128( (const __m128i*) (multab+i*16) );
			__m128i inp = _mm_loadu_si128( (const __m128i*)(pk_mat) );
			tmp_l ^= _mm_shuffle_epi8( mt , inp&mask_f );
			tmp_h ^= _mm_shuffle_epi8( mt , _mm_srli_epi16(inp,4)&mask_f );
			pk_mat += n_byte;
		}
		r_l ^= _mm_shuffle_epi8( mt0 , tmp_l );
		r_h ^= _mm_shuffle_epi8( mt0 , tmp_h );
	}

	uint8_t r[16]  __attribute__((aligned(32)));
	_mm_store_si128( (__m128i*)(r) , r_l^_mm_slli_epi16( r_h , 4 ) );
	/// linear terms
	gf256v_add( r , pk_mat , n_byte );

	memcpy( z , r , n_byte );
}




void gf16mpkc_mq_multab_n_m_sse( uint8_t * z , const uint8_t * pk_mat , const uint8_t * multab , unsigned n, unsigned m )
{
	if( 32 >= m ) { return gf16mpkc_mq_multab_n_m32_sse(z,pk_mat,multab,n,m); }
	assert( 128 >= m );
	__m128i r_l[4];
	__m128i r_h[4];
	__m128i tmp_l[4];
	__m128i tmp_h[4];

	unsigned n_xmm = (m+31)>>5;
	unsigned n_byte = (m+1)>>1;
	__m128i mask_f = _mm_set1_epi8( 0xf );

	for(unsigned i=0;i<n_xmm;i++) r_l[i] = _mm_setzero_si128();
	for(unsigned i=0;i<n_xmm;i++) r_h[i] = _mm_setzero_si128();

	for(unsigned i=0;i<n;i++){
		__m128i mt = _mm_load_si128( (const __m128i*) (multab+i*16) );
		for(unsigned j=0;j<n_xmm;j++) {
			__m128i inp = _mm_loadu_si128( (const __m128i*)(pk_mat + j*16) );
			r_l[j] ^= _mm_shuffle_epi8( mt , inp&mask_f );
			r_h[j] ^= _mm_shuffle_epi8( mt , _mm_srli_epi16(inp,4)&mask_f );
		}
		pk_mat += n_byte;
	}

	for(unsigned k=0;k<n;k++) {
		for(unsigned i=0;i<n_xmm;i++) tmp_l[i] = _mm_setzero_si128();
		for(unsigned i=0;i<n_xmm;i++) tmp_h[i] = _mm_setzero_si128();
		__m128i mt0 = _mm_load_si128( (const __m128i*) (multab+k*16) );
		for(unsigned i=0;i<=k;i++) {
			__m128i mt = _mm_load_si128( (const __m128i*) (multab+i*16) );
			for(unsigned j=0;j<n_xmm;j++) {
				__m128i inp = _mm_loadu_si128( (const __m128i*)(pk_mat + j*16) );
				tmp_l[j] ^= _mm_shuffle_epi8( mt , inp&mask_f );
				tmp_h[j] ^= _mm_shuffle_epi8( mt , _mm_srli_epi16(inp,4)&mask_f );
			}
			pk_mat += n_byte;
		}
		for(unsigned j=0;j<n_xmm;j++) {
			r_l[j] ^= _mm_shuffle_epi8( mt0 , tmp_l[j] );
			r_h[j] ^= _mm_shuffle_epi8( mt0 , tmp_h[j] );
		}
	}

	uint8_t r[64]  __attribute__((aligned(32)));
	for(unsigned i=0;i<n_xmm;i++) {
		_mm_store_si128( (__m128i*)(r+i*16) , r_l[i]^_mm_slli_epi16( r_h[i], 4 ) );
	}
	/// linear terms
	gf256v_add( r , pk_mat , n_byte );

	memcpy( z , r , n_byte );
}





void gf16mpkc_mq_eval_n_m_sse( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m )
{
	assert( 204 >= n );

	uint8_t multab[204*16] __attribute__((aligned(32)));
#if 1
	/// faster !!!
	gf16v_generate_multab_sse( multab , w , n );
#else
	for(unsigned i=0;i<n;i++) {
		unsigned xi = gf16v_get_ele( w , i );
		__m128i mi = _mm_load_si128( (const __m128i*)(__gf16_mul + xi*32 ) );
		_mm_store_si128( (__m128i*)(multab + i*16) , mi );
	}
#endif
	gf16mpkc_mq_multab_n_m_sse( z , pk_mat , multab , n , m );
}



