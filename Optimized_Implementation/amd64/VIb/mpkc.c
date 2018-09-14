
#include "blas.h"

#include "mpkc.h"

#include "string.h"

#include "assert.h"


#define IDX_XSQ(i,n_var) (((2*(n_var)+1-i)*(i)/2)+n_var)

/// xi <= xj
#define IDX_QTERMS_REVLEX(xi,xj) ((xj)*(xj+1)/2 + (xi))




#define _MAX_N  (196)
#define _MAX_M  (128)



void to_maddusb_format_mq( uint8_t * z , const uint8_t * x , unsigned _n, unsigned m )
{
	assert( _MAX_M >= m );
	unsigned n = (_n+1)*(_n+2)/2;
	uint8_t temp[256];
	while( n > 1 ) {
		for(unsigned i=0;i<m;i++) temp[i*2] = x[i];
		for(unsigned i=0;i<m;i++) temp[i*2+1] = x[m+i];
		for(unsigned i=0;i<m*2;i++) z[i] = temp[i];

		n -= 2;
		x += 2*m;
		z += 2*m;
	}
	if( 1 == n ) {
		for(unsigned i=0;i<m;i++) z[i] = x[i];
	}
}


void maddusb_to_normal_mq( uint8_t * z , const uint8_t * x , unsigned _n, unsigned m )
{
	assert( _MAX_M >= m );
	unsigned n = (_n+1)*(_n+2)/2;
	uint8_t temp[256];
	while( n > 1 ) {
		for(unsigned i=0;i<m;i++) temp[i] = x[i*2];
		for(unsigned i=0;i<m;i++) temp[m+i] = x[i*2+1];
		for(unsigned i=0;i<m*2;i++) z[i] = temp[i];

		n -= 2;
		x += 2*m;
		z += 2*m;
	}
	if( 1 == n ) {
		for(unsigned i=0;i<m;i++) z[i] = x[i];
	}
}







#ifdef _TWO_COL_MAT_

static inline
void _generate_quadratic_terms_revlex_gf31_n( uint8_t * r , const uint8_t * a , unsigned n )
{
	for(unsigned i=0;i<n;i++) {
		memcpy( r , a , i+1 );
		gf31v_mul_scalar( r , a[i] , i+1 );
		r += (i+1);
	}
}

void _gf31mpkc_mq_eval_n_m( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m)
{
	assert( 0 == (n %2 ) );
	assert( _MAX_M >= m );
	assert( _MAX_N >= n );

	uint8_t quad_terms[TERMS_QUAD_POLY(_MAX_N)];
	uint8_t tmp[_MAX_M] ;
	uint8_t *r = z;
	memset(r,0,m);

	const uint8_t * mat_linear = pk_mat;
	gf31mat_prod( tmp , mat_linear , m , n , w );
	gf31v_add( r , tmp , m );

	const uint8_t * mat_quadratic = pk_mat + (n)*(m);
	_generate_quadratic_terms_revlex_gf31_n( quad_terms , w , n );
	gf31mat_prod( tmp , mat_quadratic , m , TERMS_QUAD_POLY(n)-n-1 , quad_terms );
	gf31v_add( r , tmp , m );

	const uint8_t * vec_constant  = pk_mat + (TERMS_QUAD_POLY(n)-1)*(m);
	gf31v_add( r , vec_constant , m );
}

#else

void _gf31mpkc_mq_eval_n_m( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m)
{
	assert( _MAX_M >= m );

	uint8_t tmp[_MAX_M];
	uint8_t *r = z;
	memset(r,0,m);

	const uint8_t * linear_mat = pk_mat;
	for(unsigned i=0;i<n;i++) {
		gf31v_madd( r , linear_mat , w[i] , m );
		linear_mat += m;
	}

	const uint8_t * quad_mat = pk_mat + n*m;
	for(unsigned i=0;i<n;i++) {
		memset( tmp , 0 , m );
		for(unsigned j=0;j<=i;j++) {
			gf31v_madd( tmp , quad_mat , w[j] , m );
			quad_mat += m;
		}
		gf31v_madd( r , tmp , w[i] , m );
	}
	gf31v_add( r , quad_mat , m );
}

#endif



void gf31mpkc_interpolate_n_m( uint8_t * poly , void (*quad_poly)(void *,const void *,const void *) , const void * key , unsigned n, unsigned m)
{
	assert( _MAX_N >= n );
	assert( _MAX_M >= m );

	uint8_t tmp_r0[_MAX_M] = {0};
	uint8_t tmp_r1[_MAX_M] = {0};
	uint8_t tmp_r2[_MAX_M] = {0};
	uint8_t tmp[_MAX_N] = {0};

	const unsigned n_var = n;
	uint8_t * const_terms = poly + (TERMS_QUAD_POLY(n)-1)*(m);
	gf31v_set_zero(tmp,n);
	quad_poly( const_terms , key , tmp );

	for(unsigned i=0;i<n_var;i++) {
		gf31v_set_zero(tmp,n);
		tmp[i] = 1;
		quad_poly( tmp_r0 , key , tmp ); /// v + v^2
		gf31v_sub( tmp_r0 , const_terms , m );
		tmp[i] = 30; /// 30 = -1
		quad_poly( tmp_r1 , key , tmp ); /// -v + v^2
		gf31v_sub( tmp_r1 , const_terms , m );
		//memcpy( tmp_r2 , tmp_r1 , m );

		gf31v_sub( tmp_r0 , tmp_r1 , m );    /// 2*v
		gf31v_mul_scalar( tmp_r0 , 16 , m ); /// 2*v * 16 = 32*v mod 31 = v
		gf31v_add( tmp_r1 , tmp_r0 , m);    ///  v^2
		memcpy( poly + m*i , tmp_r0 , m );
		memcpy( poly + m*(n_var+IDX_QTERMS_REVLEX(i,i)) , tmp_r1 , m );
	}

	for(unsigned i=0;i<n_var;i++) {
		unsigned base_idx = n_var+IDX_QTERMS_REVLEX(0,i);
		memcpy( tmp_r1 , poly + m*i , m );
		memcpy( tmp_r2 , poly + m*(n_var+IDX_QTERMS_REVLEX(i,i)) , m );

		for(unsigned j=0;j<i;j++) {
			gf31v_set_zero(tmp,n);
			tmp[i] = 1;
			tmp[j] = 1;

			quad_poly( tmp_r0 , key , tmp ); /// v1 + v1^2 + v2 + v2^2 + v1v2
			gf31v_sub( tmp_r0 , const_terms , m );

			gf31v_sub( tmp_r0 , tmp_r1 , m );
			gf31v_sub( tmp_r0 , tmp_r2 , m );
			gf31v_sub( tmp_r0 , poly+m*j , m );
			gf31v_sub( tmp_r0 , poly+m*(n_var+IDX_QTERMS_REVLEX(j,j)) , m );

			memcpy( poly + m*(base_idx+j), tmp_r0 , m );
		}
	}
#ifdef _TWO_COL_MAT_
	to_maddusb_format_mq( poly , poly , n , m);
#endif
}


