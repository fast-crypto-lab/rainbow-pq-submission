
#include "mpkc.h"

#include "blas.h"

#include "string.h"  /// for memcpy

#include "assert.h"



#define IDX_XSQ(i,n_var) (((2*(n_var)+1-i)*(i)/2)+n_var)

/// presuming: xi <= xj
#define IDX_QTERMS_REVLEX(xi,xj) ((xj)*(xj+1)/2 + (xi))



void _gf16mpkc_mq_eval_n_m( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m )
{
	assert( n <= 256 );
	assert( m <= 256 );
	uint8_t tmp[128] ;
	unsigned m_byte = (m+1)>>1;

	uint8_t _x[256];
	for(unsigned i=0;i<n;i++) _x[i] = gf16v_get_ele( w , i );

	uint8_t *r = z;
	gf256v_set_zero( r , m_byte );
	for(unsigned i=0;i<n;i++) {
		gf16v_madd( r , pk_mat , _x[i] , m_byte );
		pk_mat += m_byte;
	}
	for(unsigned i=0;i<n;i++) {
		gf256v_set_zero( tmp , m_byte );
		for(unsigned j=0;j<=i;j++) {
			gf16v_madd( tmp , pk_mat , _x[j] , m_byte );
			pk_mat += m_byte;
		}
		gf16v_madd( r , tmp , _x[i] , m_byte );
	}
	gf256v_add( r , pk_mat , m_byte );
}




#define _MAX_N 256
#define _MAX_M_BYTE 128

void _gf16mpkc_interpolate_n_m( uint8_t * poly , void (*quad_poly)(void *,const void *,const void *) , const void * key , unsigned n , unsigned m )
{
	assert( n <= 256 );
	assert( m <= 256 );

	uint8_t tmp[_MAX_N] = {0};
	uint8_t tmp_r0[_MAX_M_BYTE] = {0};
	uint8_t tmp_r1[_MAX_M_BYTE] = {0};
	uint8_t tmp_r2[_MAX_M_BYTE] = {0};
	const unsigned n_var = n;
	const unsigned m_byte = (m+1)>>1;

	uint8_t * const_terms = poly + (TERMS_QUAD_POLY(n_var)-1)*(m_byte);
	gf256v_set_zero(tmp,n_var);
	quad_poly( const_terms , key , tmp );

	uint8_t v_2x2 = gf16_mul(2,2);
	for(unsigned i=0;i<n_var;i++) {
		gf256v_set_zero(tmp,n_var);
		gf16v_set_ele(tmp,i,1);
		quad_poly( tmp_r0 , key , tmp ); /// v + v^2
		gf256v_add( tmp_r0 , const_terms , m_byte );

		memcpy( tmp_r2 , tmp_r0 , m_byte );
		gf16v_mul_scalar( tmp_r0 , v_2x2 , m_byte ); /// 3v + 3v^2
		gf16v_set_ele(tmp,i,2);
		quad_poly( tmp_r1 , key , tmp ); /// 2v + 3v^2
		gf256v_add( tmp_r1 , const_terms , m_byte );

		gf256v_add( tmp_r0 , tmp_r1 , m_byte );     /// v
		gf256v_add( tmp_r2 , tmp_r0 , m_byte );   /// v^2
		memcpy( poly + m_byte*i , tmp_r0 , m_byte );
		memcpy( poly + m_byte*(n_var+IDX_QTERMS_REVLEX(i,i)) , tmp_r2 , m_byte );
	}

	for(unsigned i=0;i<n_var;i++) {
		unsigned base_idx = n_var+IDX_QTERMS_REVLEX(0,i);
		memcpy( tmp_r1 , poly + m_byte*i , m_byte );
		memcpy( tmp_r2 , poly + m_byte*(n_var+IDX_QTERMS_REVLEX(i,i)) , m_byte );

		for(unsigned j=0;j<i;j++) {
			gf256v_set_zero(tmp,n_var);
			gf16v_set_ele(tmp,i,1);
			gf16v_set_ele(tmp,j,1);

			quad_poly( tmp_r0 , key , tmp ); /// v1 + v1^2 + v2 + v2^2 + v1v2
			gf256v_add( tmp_r0 , const_terms , m_byte );

			gf256v_add( tmp_r0 , tmp_r1 , m_byte );
			gf256v_add( tmp_r0 , tmp_r2 , m_byte );
			gf256v_add( tmp_r0 , poly+ m_byte*j , m_byte );
			gf256v_add( tmp_r0 , poly+ m_byte*(n_var+IDX_QTERMS_REVLEX(j,j)) , m_byte );

			memcpy( poly + m_byte*(base_idx+j), tmp_r0 , m_byte );
		}
	}
}



