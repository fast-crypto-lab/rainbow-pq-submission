
#include "blas.h"

#include "blas_avx2.h"

#include "mpkc_avx2.h"

#include <string.h>  /// for memcpy

#include <assert.h>


void gf256mpkc_pub_map_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n , unsigned m )
{
	assert( 192 >= n );
	assert( 128 >= m );
	assert( 32 <= m );

	uint8_t r[128]   __attribute__((aligned(32))) = {0};
	uint8_t tmp[128] __attribute__((aligned(32)));

	unsigned align_len = ((m+31)>>5) <<5;
	for(unsigned i=0;i<n;i++) {
		gf256v_madd( r , pk_mat , w[i] , align_len );
		pk_mat += m;
	}
	for(unsigned i=0;i<n;i++) {
		gf256v_set_zero( tmp , 128 );
		for(unsigned j=0;j<=i;j++) {
			gf256v_madd( tmp , pk_mat , w[j] , align_len );
			pk_mat += m;
		}
		gf256v_madd( r , tmp , w[i] , align_len );
	}
	gf256v_add( r , pk_mat , m );
	memcpy( z , r , m );
}



void gf256mpkc_mq_multab_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * multab , unsigned n, unsigned m)
{
	assert( 128 >= m );

	uint8_t tmp[128] __attribute__((aligned(32)));
	uint8_t tmp2[128] __attribute__((aligned(32)));
	uint8_t r[128] __attribute__((aligned(32))) = {0};

	const uint8_t * linear_mat = pk_mat;
	gf256mat_prod_multab_avx2( r , linear_mat , m , n , multab );

	const uint8_t * quad_mat = pk_mat + n*m;
	for(unsigned i=0;i<n;i++) {
		gf256mat_prod_multab_avx2( tmp , quad_mat , m , i+1 , multab );
		quad_mat += (i+1)*m;

		gf256mat_prod_multab_avx2( tmp2 , tmp , m , 1 , multab+i*32 );
		gf256v_add( r , tmp2 , m );
	}
	gf256v_add( r , quad_mat , m );

	memcpy( z , r , m );
}



void gf256mpkc_mq_eval_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m)
{
	assert( 192 >= n );

	uint8_t multab[192*32] __attribute__((aligned(32)));
	gf256v_generate_multab_sse( multab , w , n );

	gf256mpkc_mq_multab_n_m_avx2( z , pk_mat , w , n , m );
}


