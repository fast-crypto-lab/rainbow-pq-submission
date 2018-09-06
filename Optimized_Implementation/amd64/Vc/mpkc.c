
#include "mpkc.h"

#include "string.h"

#include "assert.h"

#define IDX_XSQ(i,n_var) (((2*(n_var)+1-i)*(i)/2)+n_var)

/// xi <= xj
#define IDX_QTERMS_REVLEX(xi,xj) ((xj)*(xj+1)/2 + (xi))



void _gf256mpkc_mq_eval_n_m( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m )
{
	assert( 128 >= m );
	uint8_t tmp[128];
	uint8_t *r = z;
	memset(r,0,m);

	const uint8_t * linear_mat = pk_mat;
	for(unsigned i=0;i<n;i++) {
		gf256v_madd( r , linear_mat , w[i] , m );
		linear_mat += m;
	}

	const uint8_t * quad_mat = pk_mat + n*m;
	for(unsigned i=0;i<n;i++) {
		memset( tmp , 0 , m );
		for(unsigned j=0;j<=i;j++) {
			gf256v_madd( tmp , quad_mat , w[j] , m );
			quad_mat += m;
		}
		gf256v_madd( r , tmp , w[i] , m );
	}
	gf256v_add( r , quad_mat , m );
}



void _gf256mpkc_interpolate_n_m( uint8_t * poly , void (*quad_poly)(void *,const void *,const void *) , const void * key , unsigned n , unsigned m )
{
	assert( 192 >= n );
	assert( 128 >= m );
	uint8_t tmp[192] = {0};
	uint8_t tmp_r0[128] = {0};
	uint8_t tmp_r1[128] = {0};
	uint8_t tmp_r2[128] = {0};

	const unsigned n_var = n;

	uint8_t * const_terms = poly + (TERMS_QUAD_POLY(n)-1)*(m);
	gf256v_set_zero(tmp, n );
	quad_poly( const_terms , key , tmp );

	uint8_t v_2x2 = gf16_mul(2,2);
	for(unsigned i=0;i<n_var;i++) {
		gf256v_set_zero(tmp, n );
		tmp[i] = 1;
		quad_poly( tmp_r0 , key , tmp ); /// v + v^2
		gf256v_add( tmp_r0 , const_terms , m );

		memcpy( tmp_r2 , tmp_r0 , m );
		gf256v_mul_scalar( tmp_r0 , v_2x2 , m ); /// 3v + 3v^2
		tmp[i] = 2;
		quad_poly( tmp_r1 , key , tmp ); /// 2v + 3v^2
		gf256v_add(tmp_r1 , const_terms , m );

		gf256v_add( tmp_r0 , tmp_r1 , m );     /// v
		gf256v_add( tmp_r2 , tmp_r0 , m);   /// v^2
		memcpy( poly + m*i , tmp_r0 , m );
		memcpy( poly + m*(n_var+IDX_QTERMS_REVLEX(i,i)) , tmp_r2 , m );
	}

	for(unsigned i=0;i<n_var;i++) {
		unsigned base_idx = n_var+IDX_QTERMS_REVLEX(0,i);
		memcpy( tmp_r1 , poly + m*i , m );
		memcpy( tmp_r2 , poly + m*(n_var+IDX_QTERMS_REVLEX(i,i)) , m );

		for(unsigned j=0;j<i;j++) {
			gf256v_set_zero(tmp, n );
			tmp[i] = 1;
			tmp[j] = 1;

			quad_poly( tmp_r0 , key , tmp ); /// v1 + v1^2 + v2 + v2^2 + v1v2
			gf256v_add( tmp_r0 , const_terms , m );

			gf256v_add( tmp_r0 , tmp_r1 , m );
			gf256v_add( tmp_r0 , tmp_r2 , m );
			gf256v_add( tmp_r0 , poly+m*j , m );
			gf256v_add( tmp_r0 , poly+m*(n_var+IDX_QTERMS_REVLEX(j,j)) , m );

			memcpy( poly + m*(base_idx+j), tmp_r0 , m );
		}
	}
}



