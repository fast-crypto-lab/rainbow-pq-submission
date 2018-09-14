
#include "blas.h"

#include <immintrin.h>

#include "gf31_sse.h"

#include "blas_avx2.h"


#include "assert.h"




void gf31mat_prod_avx2( uint8_t * c , const uint8_t * mat , unsigned n_mat_h , unsigned n_mat_w , const uint8_t * b ) {
#ifdef _TWO_COL_MAT_
	assert( 256 >= n_mat_h );
	assert( 256 >= n_mat_w );
	assert( 0 == (n_mat_w&1) );

	__m256i r[16];
	unsigned n_ymm = (n_mat_h+15)>>4;
	for(unsigned i=0;i<n_ymm;i++) r[i] = _mm256_setzero_si256();

	const uint16_t * b_u16 = (const uint16_t *)b;

	for(unsigned i=0;i<n_mat_w/2;i++) {
		__m256i bi = _mm256_set1_epi16( b_u16[i] );

		for(unsigned j=0;j<n_ymm;j++) {
			__m256i tmp = _mm256_loadu_si256( (__m256i*)(mat+j*32) );
			r[j] = _mm256_add_epi16( r[j] , _mm256_maddubs_epi16( tmp , bi ) );
		}
		mat += n_mat_h*2;
	}


	uint8_t temp[256] __attribute__((aligned(32)));
	for(unsigned j=0;j<n_ymm;j++) {
		r[j] = _gf31v_reduce_u16_avx2( r[j] );
		r[j] = _gf31v_reduce_u16_avx2( r[j] );
	}
	unsigned n_ymm_2 = (n_ymm+1)/2;
	for(unsigned j=0;j<n_ymm_2;j++){
		r[j] = _mm256_packs_epi16( r[j*2] , r[j*2+1] );
		r[j] = _mm256_permute4x64_epi64( r[j] , 0xd8 ); //     3,1,2,0
		r[j] = _gf31v_reduce_u8_avx2( r[j] );
		_mm256_store_si256( (__m256i*)(&temp[32*j]) , r[j] );
	}

	for(unsigned i=0;i<n_mat_h;i++) c[i] = temp[i];
#else
	assert( 256 >= n_mat_h );
	assert( 256 >= n_mat_w );

	__m256i r[16];
	unsigned n_ymm = (n_mat_h+15)>>4;
	for(unsigned i=0;i<n_ymm;i++) r[i] = _mm256_setzero_si256();

	//__m256i tmp_col[16];
	__m256i zero = _mm256_setzero_si256();

	for(unsigned i=0;i<n_mat_w;i++) {
		__m256i bi = _mm256_set1_epi16( b[i] );

		for(unsigned j=0;j<n_ymm;j++) {
			__m256i tmp = _mm256_loadu_si256( (__m256i*)(mat+j*32) );
			__m256i t0 = _mm256_unpacklo_epi8( tmp , zero );
			__m256i t1 = _mm256_unpackhi_epi8( tmp , zero );
			r[j*2] = _mm256_add_epi16( r[j*2] , _mm256_mullo_epi16( t0 , bi ) );
			r[j*2+1] = _mm256_add_epi16( r[j*2+1] , _mm256_mullo_epi16( t1 , bi ) );
		}
		mat += n_mat_h;
	}
	uint8_t temp[256] __attribute__((aligned(32)));
	for(unsigned j=0;j<n_ymm;j++) {
		r[j*2] = _gf31v_reduce_u16_avx2( r[j*2] );
		r[j*2+1] = _gf31v_reduce_u16_avx2( r[j*2+1] );
		r[j*2] = _gf31v_reduce_u16_avx2( r[j*2] );
		r[j*2+1] = _gf31v_reduce_u16_avx2( r[j*2+1] );

		r[j*2] = _mm256_packs_epi16( r[j*2] , r[j*2+1] );
		r[j*2] = _gf31v_reduce_u8_avx2( r[j*2] );
		_mm256_store_si256( (__m256i*)(&temp[32*j]) , r[j*2] );
	}

	for(unsigned i=0;i<n_mat_h;i++) c[i] = temp[i];
#endif
}



static
unsigned _gf31mat_gauss_elim_avx2_core( uint16_t * mat , unsigned h , unsigned w )
{
	assert( 400 >= w );
	assert( 0 == (w&15) );
	__m256i ai_ymm[25];

	unsigned char r8 = 1;
	unsigned n_ymm = w>>4;

	for(unsigned i=0;i<h;i++) {
		uint16_t * ai = mat + w*i;
		unsigned st_ymm = i>>4;
		for(unsigned j=i+1;j<h;j++) {
			uint16_t * aj = mat + w*j;
			short mm = gf31_is_nonzero(ai[i])^gf31_is_nonzero(aj[i]);
			__m256i mask = _mm256_set1_epi16( 0-mm );

			for(unsigned k=st_ymm;k<n_ymm;k++) {
				__m256i ai_k = _mm256_add_epi16( _mm256_load_si256( (__m256i*)(ai+k*16) ) , _mm256_load_si256( (__m256i*)(aj+k*16) )&mask );
				_mm256_store_si256( (__m256i*)(ai+k*16) , ai_k );
			}
		}
		r8 &= gf31_is_nonzero(ai[i]);
		uint8_t pivot = ai[i];
		//uint16_t inv_p = gf31_inv( pivot ); /// XXX:
		uint16_t inv_p = gf31_inv_sse( pivot ); /// XXX:

		__m256i mul_p = _mm256_set1_epi16( inv_p );
		for(unsigned k=st_ymm;k<n_ymm;k++) {
			ai_ymm[k] = _mm256_load_si256( (__m256i*)(ai+k*16) );
			ai_ymm[k] = _mm256_mullo_epi16( mul_p , ai_ymm[k] );
			ai_ymm[k] = _gf31v_reduce_u16_avx2( ai_ymm[k] );
			ai_ymm[k] = _gf31v_reduce_u8_avx2( ai_ymm[k] );
			_mm256_store_si256( (__m256i*)(ai+k*16) , ai_ymm[k] );
		}

		__m256i mask_62 = _mm256_set1_epi16(62);
		for(unsigned j=0;j<h;j++) {
			if(i==j) continue;
			uint16_t * aj = mat + w*j;
#if 1
			__m256i aj_i = _mm256_set1_epi16( aj[i] );

			for(unsigned k=st_ymm;k<n_ymm;k++) {
				__m256i aixaj_i = _mm256_mullo_epi16( ai_ymm[k] , aj_i );
				__m256i tmp = _gf31v_reduce_u16_avx2( aixaj_i );
				tmp = _gf31v_reduce_u16_avx2( tmp );

				__m256i aj_ymm = _mm256_add_epi16( mask_62 , _mm256_load_si256( (__m256i*)(aj+k*16) ) );

				tmp = _mm256_sub_epi16( aj_ymm , tmp );
				tmp = _gf31v_reduce_u16_avx2( tmp );

				aj_ymm = _gf31v_reduce_u8_avx2( tmp );
				_mm256_store_si256( (__m256i*)(aj+k*16) , aj_ymm );
			}
#else
			uint16_t aji = aj[i];
			for(unsigned k=0;k<w;k++) {
				aj[k] = aj[k]+31- ((ai[k]*aji)%31);
				aj[k] %= 31;
			}
#endif
		}
	}
	return r8;
}


unsigned _gf31mat_gauss_elim_avx2( uint8_t * mat , unsigned h , unsigned w )
{
	assert( 200 >= h );
	assert( 400 >= w );
	uint16_t mat_16[200*400] __attribute__((aligned(32)));

	unsigned w_16 = ((w+15)>>4)<<4;
	for(unsigned i=0;i<h;i++) {
		gf31v_u8_to_u16( (uint8_t*)(mat_16+i*w_16) , mat + i*w , w );
	}

	unsigned r = _gf31mat_gauss_elim_avx2_core( mat_16 , h , w_16 );
	for(unsigned i=0;i<h;i++) {
		for(unsigned j=0;j<w;j++) mat[i*w+j] = mat_16[i*w_16+j];
	}
	return r;
}

