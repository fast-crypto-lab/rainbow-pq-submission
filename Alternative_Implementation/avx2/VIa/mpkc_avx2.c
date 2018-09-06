

#include "blas.h"

#include "blas_avx2.h"

#include "mpkc.h"

#include "mpkc_avx2.h"

#include "string.h"



static
void mq_gf16_n96_m64_vartime_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w )
{
	uint8_t _x0[96] __attribute__((aligned(32)));
	__m128i mask16 = _mm_set1_epi8( 0xf );
	__m128i w0,w1;
	w0 = _mm_loadu_si128( (__m128i*) w );
	w1 = _mm_srli_epi16( w0 , 4 ) & mask16;
	w0 &= mask16;
	_mm_store_si128( (__m128i*) _x0 , _mm_unpacklo_epi8(w0,w1) );
	_mm_store_si128( (__m128i*) (_x0+16) , _mm_unpackhi_epi8(w0,w1) );
	w0 = _mm_loadu_si128( (__m128i*) (w+16) );
	w1 = _mm_srli_epi16( w0 , 4 ) & mask16;
	w0 &= mask16;
	_mm_store_si128( (__m128i*) (_x0+32) , _mm_unpacklo_epi8(w0,w1) );
	_mm_store_si128( (__m128i*) (_x0+48) , _mm_unpackhi_epi8(w0,w1) );
	w0 = _mm_loadu_si128( (__m128i*) (w+32) );
	w1 = _mm_srli_epi16( w0 , 4 ) & mask16;
	w0 &= mask16;
	_mm_store_si128( (__m128i*) (_x0+64) , _mm_unpacklo_epi8(w0,w1) );
	_mm_store_si128( (__m128i*) (_x0+80) , _mm_unpackhi_epi8(w0,w1) );

        __m256i mask = _mm256_load_si256( (__m256i*) __mask_low );

	__m256i r0 = _mm256_setzero_si256();
	__m256i r1 = _mm256_setzero_si256();
	for(unsigned i=0;i<96;i++) {
		unsigned b = _x0[i];
		__m256i ml = _mm256_load_si256( (__m256i*) (__gf16_mulx2 + 32*b) );

		__m256i inp = _mm256_load_si256( (__m256i*)pk_mat ); pk_mat += 32;
		r0 ^= _mm256_shuffle_epi8( ml , inp&mask );
		r1 ^= _mm256_shuffle_epi8( ml , _mm256_srli_epi16(inp,4)&mask );
	}

	for(unsigned i=0;i<96;i++) {
		if( 0 == _x0[i] ) {
			pk_mat += 32*(i+1);
			continue;
		}
		__m256i temp0 = _mm256_setzero_si256();
		__m256i temp1 = _mm256_setzero_si256();
		__m256i ml;
		for(unsigned j=0;j<=i;j++) {
			unsigned b = _x0[j];
			ml = _mm256_load_si256( (__m256i*) (__gf16_mulx2 + 32*b) );
			__m256i inp = _mm256_load_si256( (__m256i*)pk_mat ); pk_mat += 32;

			temp0 ^= _mm256_shuffle_epi8( ml , inp&mask );
			temp1 ^= _mm256_shuffle_epi8( ml , _mm256_srli_epi16(inp,4)&mask );
		}
		r0 ^= _mm256_shuffle_epi8( ml , temp0 );
		r1 ^= _mm256_shuffle_epi8( ml , temp1 );
	}
	__m256i rr = r0^_mm256_slli_epi16(r1,4)^ _mm256_load_si256( (__m256i*)pk_mat );
	_mm256_storeu_si256( (__m256i*)z , rr );
}


static
void mq_gf16_n96_m64_vartime_avx2_unalign( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w )
{
	uint8_t _x0[96] __attribute__((aligned(32)));
	__m128i mask16 = _mm_set1_epi8( 0xf );
	__m128i w0,w1;
	w0 = _mm_loadu_si128( (__m128i*) w );
	w1 = _mm_srli_epi16( w0 , 4 ) & mask16;
	w0 &= mask16;
	_mm_store_si128( (__m128i*) _x0 , _mm_unpacklo_epi8(w0,w1) );
	_mm_store_si128( (__m128i*) (_x0+16) , _mm_unpackhi_epi8(w0,w1) );
	w0 = _mm_loadu_si128( (__m128i*) (w+16) );
	w1 = _mm_srli_epi16( w0 , 4 ) & mask16;
	w0 &= mask16;
	_mm_store_si128( (__m128i*) (_x0+32) , _mm_unpacklo_epi8(w0,w1) );
	_mm_store_si128( (__m128i*) (_x0+48) , _mm_unpackhi_epi8(w0,w1) );
	w0 = _mm_loadu_si128( (__m128i*) (w+32) );
	w1 = _mm_srli_epi16( w0 , 4 ) & mask16;
	w0 &= mask16;
	_mm_store_si128( (__m128i*) (_x0+64) , _mm_unpacklo_epi8(w0,w1) );
	_mm_store_si128( (__m128i*) (_x0+80) , _mm_unpackhi_epi8(w0,w1) );

        __m256i mask = _mm256_load_si256( (__m256i*) __mask_low );

	__m256i r0 = _mm256_setzero_si256();
	__m256i r1 = _mm256_setzero_si256();
	for(unsigned i=0;i<96;i++) {
		unsigned b = _x0[i];
		__m256i ml = _mm256_load_si256( (__m256i*) (__gf16_mulx2 + 32*b) );

		__m256i inp = _mm256_loadu_si256( (__m256i*)pk_mat ); pk_mat += 32;
		r0 ^= _mm256_shuffle_epi8( ml , inp&mask );
		r1 ^= _mm256_shuffle_epi8( ml , _mm256_srli_epi16(inp,4)&mask );
	}

	for(unsigned i=0;i<96;i++) {
		if( 0 == _x0[i] ) {
			pk_mat += 32*(i+1);
			continue;
		}
		__m256i temp0 = _mm256_setzero_si256();
		__m256i temp1 = _mm256_setzero_si256();
		__m256i ml;
		for(unsigned j=0;j<=i;j++) {
			unsigned b = _x0[j];
			ml = _mm256_load_si256( (__m256i*) (__gf16_mulx2 + 32*b) );
			__m256i inp = _mm256_loadu_si256( (__m256i*)pk_mat ); pk_mat += 32;

			temp0 ^= _mm256_shuffle_epi8( ml , inp&mask );
			temp1 ^= _mm256_shuffle_epi8( ml , _mm256_srli_epi16(inp,4)&mask );
		}
		r0 ^= _mm256_shuffle_epi8( ml , temp0 );
		r1 ^= _mm256_shuffle_epi8( ml , temp1 );
	}
	__m256i rr = r0^_mm256_slli_epi16(r1,4)^ _mm256_loadu_si256( (__m256i*)pk_mat );
	_mm256_storeu_si256( (__m256i*)z , rr );
}






//void mpkc_pub_map_gf16_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m)
void gf16mpkc_pubmap_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m)
{
	if( (96== n ) && (64== m ) ) {
		if( 0==(((uint64_t)pk_mat)&0x1f) ) mq_gf16_n96_m64_vartime_avx2(z,pk_mat,w);
		else mq_gf16_n96_m64_vartime_avx2_unalign(z,pk_mat,w);
		return;
	}

        assert( n <= 256 );
        assert( m <= 256 );
        uint8_t tmp[128] __attribute__((aligned(32)));
        unsigned m_byte = (m+1)/2;
        uint8_t *r = z;
        //memset(r,0,m_byte);

        gf16mat_prod( r , pk_mat , m_byte , n , w );
        pk_mat += n*m_byte;

        uint8_t _x[256] __attribute__((aligned(32)));
        gf16v_split( _x , w , n );

        for(unsigned i=0;i<n;i++) {
                memset( tmp , 0 , m_byte );
                for(unsigned j=0;j<=i;j++) {
                        gf16v_madd( tmp , pk_mat , _x[j] , m_byte );
                        pk_mat += m_byte;
                }
                gf16v_madd( r , tmp , _x[i] , m_byte );
        }
        gf256v_add( r , pk_mat , m_byte );

}




#include "mpkc_sse.h"

#include "assert.h"



static void gf16mpkc_mq_multab_n_m64_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * multab , unsigned n, unsigned m )
{
	assert( 64 >= m );

	__m256i r_l;
	__m256i r_h;
	__m256i tmp_l;
	__m256i tmp_h;

	unsigned n_byte = (m+1)>>1;
	__m256i mask_f = _mm256_load_si256( (const __m256i*) __mask_low );

	r_l = _mm256_setzero_si256();
	r_h = _mm256_setzero_si256();

	for(unsigned i=0;i<n;i++){
		__m256i mt = _mm256_broadcastsi128_si256( _mm_load_si128( (const __m128i*) (multab+i*16) ) );
		__m256i inp = _mm256_loadu_si256( (const __m256i*) pk_mat );
		r_l ^= _mm256_shuffle_epi8( mt , inp&mask_f );
		r_h ^= _mm256_shuffle_epi8( mt , _mm256_srli_epi16(inp,4)&mask_f );
		pk_mat += n_byte;
	}

	for(unsigned k=0;k<n;k++) {
		tmp_l = _mm256_setzero_si256();
		tmp_h = _mm256_setzero_si256();
		__m256i mt0 = _mm256_broadcastsi128_si256( _mm_load_si128( (const __m128i*) (multab+k*16) ) );
		for(unsigned i=0;i<=k;i++) {
			__m256i mt = _mm256_broadcastsi128_si256( _mm_load_si128( (const __m128i*) (multab+i*16) ) );
			__m256i inp = _mm256_loadu_si256( (const __m256i*) pk_mat );
			tmp_l ^= _mm256_shuffle_epi8( mt , inp&mask_f );
			tmp_h ^= _mm256_shuffle_epi8( mt , _mm256_srli_epi16(inp,4)&mask_f );
			pk_mat += n_byte;
		}
		r_l ^= _mm256_shuffle_epi8( mt0 , tmp_l );
		r_h ^= _mm256_shuffle_epi8( mt0 , tmp_h );
	}

	uint8_t r[32]  __attribute__((aligned(32)));
	_mm256_store_si256( (__m256i*) r , r_l^_mm256_slli_epi16( r_h , 4 ) );
	/// linear terms
	gf256v_add( r , pk_mat , n_byte );
	memcpy( z , r , n_byte );
}



void gf16mpkc_mq_multab_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * multab , unsigned n, unsigned m )
{
	if( m <= 32 ) { return gf16mpkc_mq_multab_n_m_sse(z,pk_mat,multab,n,m); }
	if( m <= 64 ) { return gf16mpkc_mq_multab_n_m64_avx2(z,pk_mat,multab,n,m); }

	assert( 128 >= m );

	__m256i r_l[2];
	__m256i r_h[2];
	__m256i tmp_l[2];
	__m256i tmp_h[2];

	unsigned n_ymm = (m+63)>>6;
	unsigned n_byte = (m+1)>>1;
	__m256i mask_f = _mm256_load_si256( (const __m256i*) __mask_low );

	for(unsigned i=0;i<n_ymm;i++) r_l[i] = _mm256_setzero_si256();
	for(unsigned i=0;i<n_ymm;i++) r_h[i] = _mm256_setzero_si256();

	for(unsigned i=0;i<n;i++){
		__m256i mt = _mm256_broadcastsi128_si256( _mm_load_si128( (const __m128i*) (multab+i*16) ) );
		for(unsigned j=0;j<n_ymm;j++) {
			__m256i inp = _mm256_loadu_si256( (const __m256i*)(pk_mat + j*32) );
			r_l[j] ^= _mm256_shuffle_epi8( mt , inp&mask_f );
			r_h[j] ^= _mm256_shuffle_epi8( mt , _mm256_srli_epi16(inp,4)&mask_f );
		}
		pk_mat += n_byte;
	}

	for(unsigned k=0;k<n;k++) {
		for(unsigned i=0;i<n_ymm;i++) tmp_l[i] = _mm256_setzero_si256();
		for(unsigned i=0;i<n_ymm;i++) tmp_h[i] = _mm256_setzero_si256();
		__m256i mt0 = _mm256_broadcastsi128_si256( _mm_load_si128( (const __m128i*) (multab+k*16) ) );
		for(unsigned i=0;i<=k;i++) {
			__m256i mt = _mm256_broadcastsi128_si256( _mm_load_si128( (const __m128i*) (multab+i*16) ) );
			for(unsigned j=0;j<n_ymm;j++) {
				__m256i inp = _mm256_loadu_si256( (const __m256i*)(pk_mat + j*32) );
				tmp_l[j] ^= _mm256_shuffle_epi8( mt , inp&mask_f );
				tmp_h[j] ^= _mm256_shuffle_epi8( mt , _mm256_srli_epi16(inp,4)&mask_f );
			}
			pk_mat += n_byte;
		}
		for(unsigned j=0;j<n_ymm;j++) {
			r_l[j] ^= _mm256_shuffle_epi8( mt0 , tmp_l[j] );
			r_h[j] ^= _mm256_shuffle_epi8( mt0 , tmp_h[j] );
		}
	}

	uint8_t r[64]  __attribute__((aligned(32)));
	for(unsigned i=0;i<n_ymm;i++) {
		_mm256_store_si256( (__m256i*)(r+i*32) , r_l[i]^_mm256_slli_epi16( r_h[i], 4 ) );
	}
	/// linear terms
	gf256v_add( r , pk_mat , n_byte );
	memcpy( z , r , n_byte );
}





void gf16mpkc_mq_eval_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m )
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
	gf16mpkc_mq_multab_n_m_avx2( z , pk_mat , multab , n , m );
}



