
#include "rainbow_config.h"

#include "rainbow.h"

#include "gf16.h"

#include "blas.h"

#include "stdint.h"

#include "stdlib.h"

#include "string.h"



#ifdef _RAINBOW_256


#include "mpkc.h"

void rainbow_pubmap( uint8_t * z , const uint8_t * poly , const uint8_t * w )
{
	gf256mpkc_mq_eval_n_m( z , poly , w , _PUB_N , _PUB_M );
}

static void mpkc_interpolate_gf256( uint8_t * poly , void (*quad_poly)(void *,const void *,const void *) , const void * key )
{
	_gf256mpkc_interpolate_n_m( poly , quad_poly , key , _PUB_N , _PUB_M );
}




#ifndef _DEBUG_RAINBOW_
static unsigned rainbow_ivs_central_map( uint8_t * r , const rainbow_ckey * k , const uint8_t * a );
static void rainbow_central_map( uint8_t * r , const rainbow_ckey * k , const uint8_t * a );
#endif




#ifndef _DEBUG_RAINBOW_
static
#endif
void rainbow_pubmap_seckey( uint8_t * z , const rainbow_key * sk , const uint8_t * w ) {

	uint8_t tt[_PUB_N_BYTE]  = {0};
	uint8_t tt2[_PUB_N_BYTE]  = {0};

	gf256mat_prod( tt , sk->mat_t , _PUB_N_BYTE , _SEC_N , w );
	gf256v_add( tt , sk->vec_t , _PUB_N_BYTE );

	rainbow_central_map( tt2 , & sk->ckey , tt );

	gf256mat_prod( z , sk->mat_s , _PUB_M_BYTE , _PUB_M , tt2 );
	gf256v_add( z , sk->vec_s , _PUB_M_BYTE );

}



static void gen_mat_inv( uint8_t * s , uint8_t * inv_s , unsigned height ,  uint8_t * buffer )
{
	/// max trails: 128
	for(unsigned i=0;i<128;i++) {
		uint8_t r = gf256mat_rand_inv( s , inv_s , height , buffer );
		if( 0 != r ) break;
	}
}

#ifndef _DEBUG_RAINBOW_
static
#endif
void rainbow_genkey_debug( rainbow_key * pk , rainbow_key * sk )
{
#ifdef _CONSISTENT_WITH_ALGO_6_
	gen_mat_inv( pk->mat_s , sk->mat_s , _PUB_M , (uint8_t *)&sk->ckey );
	gf256v_rand( pk->vec_s , _PUB_M );
	memcpy( sk->vec_s , pk->vec_s , _PUB_M_BYTE );

	gen_mat_inv( pk->mat_t , sk->mat_t , _PUB_N , (uint8_t *)&sk->ckey );
	gf256v_rand( pk->vec_t , _PUB_N );
	memcpy( sk->vec_t , pk->vec_t , _PUB_N_BYTE );

	gf256v_rand( (uint8_t *)&sk->ckey , sizeof(rainbow_ckey) );
	memcpy( (void *)&pk->ckey , (void *)&sk->ckey , sizeof(rainbow_ckey) );
#else
	gf256v_rand( (uint8_t *)&sk->ckey , sizeof(rainbow_ckey) );

	gen_mat_inv( pk->mat_s , sk->mat_s , _PUB_M , (uint8_t *)&pk->ckey );
	gen_mat_inv( pk->mat_t , sk->mat_t , _PUB_N , (uint8_t *)&pk->ckey );

	gf256v_rand( pk->vec_t , _PUB_N );
	memcpy( sk->vec_t , pk->vec_t , _PUB_N_BYTE );

	gf256v_rand( pk->vec_s , _PUB_M );
	memcpy( sk->vec_s , pk->vec_s , _PUB_M_BYTE );

	memcpy( (void *)&pk->ckey , (void *)&sk->ckey , sizeof(rainbow_ckey) );
#endif
}



#include "mpkc.h"

static inline
void rainbow_pubmap_wrapper( void * z, const void* pk_key, const void * w) {
	rainbow_pubmap_seckey( (uint8_t *)z , (const rainbow_key *)pk_key, (const uint8_t *)w );
}


void rainbow_genkey( uint8_t * pk , uint8_t * sk )
{

	rainbow_key _pk;
	rainbow_genkey_debug( &_pk , (rainbow_key *)sk );

	mpkc_interpolate_gf256( pk , rainbow_pubmap_wrapper , (const void*) &_pk );

	pk[_PUB_KEY_LEN-1] = _SALT_BYTE;
	sk[_SEC_KEY_LEN-1] = _SALT_BYTE;
}




unsigned rainbow_secmap( uint8_t * w , const rainbow_key * sk , const uint8_t * z )
{
	uint8_t _z[_PUB_N_BYTE] ;
	uint8_t y[_PUB_N_BYTE] ;
	uint8_t x[_PUB_N_BYTE] ;

	memcpy(_z,z,_PUB_M_BYTE);
	gf256v_add(_z,sk->vec_s,_PUB_M_BYTE);
	gf256mat_prod(y,sk->mat_s,_PUB_M_BYTE,_PUB_M,_z);

	unsigned succ = 0;
	unsigned time = 0;
	while( !succ ) {
		if( 256 == time ) break;

		gf256v_rand( x , _V1 );

		succ = rainbow_ivs_central_map( x , & sk->ckey , y );
		time ++;
	};

	gf256v_add(x,sk->vec_t,_PUB_N_BYTE);
	gf256mat_prod(w,sk->mat_t,_PUB_N_BYTE,_PUB_N,x);

	// return time;
	if( 256 <= time ) return -1;
	return 0;
}





/////////////////////////////

static inline
void matrix_transpose( uint8_t * r , const uint8_t * a , unsigned n )
{
	for(unsigned i=0;i<n;i++) {
		for(unsigned j=0;j<n;j++) {
			r[i*n+j] = a[j*n+i];
		}
	}
}

static inline
void gen_l1_mat( uint8_t * mat , const rainbow_ckey * k , const uint8_t * v ) {
	for(unsigned i=0;i<_O1;i++) {
		gf256mat_prod( mat + i*_O1 , k->l1_vo[i] , _O1 , _V1 , v );
		gf256v_add( mat + i*_O1 , k->l1_o + i*_O1 , _O1 );
	}
}

static inline
void gen_l2_mat( uint8_t * mat , const rainbow_ckey * k , const uint8_t * v ) {
	for(unsigned i=0;i<_O2;i++) {
		gf256mat_prod( mat + i*_O2 , k->l2_vo[i] , _O2 , _V2 , v );
		gf256v_add( mat + i*_O2 , k->l2_o + i*_O2 , _O2 );
	}
}



#ifndef _DEBUG_RAINBOW_
static
#endif
void rainbow_central_map( uint8_t * r , const rainbow_ckey * k , const uint8_t * a )
{
	/// warning: presume: _O2 > _O1
	uint8_t mat1[_O2*_O2] ;
	uint8_t mat2[_O2*_O2] ;
	uint8_t temp[_O2] ;

	gen_l1_mat( mat1 , k , a );
	matrix_transpose( mat2 , mat1 , _O1 );
	gf256mat_prod( r , mat2 , _O1 , _O1 , a+_V1 );
	gf256mpkc_mq_eval_n_m( temp , k->l1_vv , a , _V1 , _O1 );
	gf256v_add( r , temp , _O1 );

	gen_l2_mat( mat1 , k , a );
	matrix_transpose( mat2 , mat1 , _O2 );
	gf256mat_prod( r+_O1 , mat2 , _O2 , _O2 , a+_V2 );
	gf256mpkc_mq_eval_n_m( temp , k->l2_vv , a , _V2 , _O2 );
	gf256v_add( r+_O1 , temp , _O2 );

}



#ifndef _DEBUG_RAINBOW_
static
#endif
unsigned rainbow_ivs_central_map( uint8_t * r , const rainbow_ckey * k , const uint8_t * a )
{
	uint8_t mat1[_O1*_O1] ;
	uint8_t mat2[_O2*_O2] ;
	uint8_t temp[_O2] ;
	gf256mpkc_mq_eval_n_m( temp , k->l1_vv , r , _V1 , _O1 );
	gf256v_add( temp  , a , _O1 );
	gen_l1_mat( mat1 , k , r );
	unsigned r1 = gf256mat_solve_linear_eq( r+_V1 , mat1 , temp , _O1 );

	gen_l2_mat( mat2 , k , r );
	gf256mpkc_mq_eval_n_m( temp , k->l2_vv , r , _V2 , _O2 );
	gf256v_add( temp  , a+_O1 , _O2 );
	unsigned r2 = gf256mat_solve_linear_eq( r+_V2 , mat2 , temp , _O2 );

	return r1&r2;
}



#include "hash_utils.h"


/// algorithm 7
int rainbow_sign( uint8_t * signature , const uint8_t * _sk , const uint8_t * _digest )
{
	const rainbow_key * sk = (const rainbow_key *)_sk;
	const rainbow_ckey * k = &( sk->ckey);
//// line 1 - 5
	uint8_t mat_l1[_O1*_O1] ;
	uint8_t mat_l2[_O2*_O2] ;
	uint8_t temp_o1[_O1]  = {0};
	uint8_t temp_o2[_O2] ;
	uint8_t vinegar[_V1] ;
	unsigned l1_succ = 0;
	unsigned time = 0;
	while( !l1_succ ) {
		if( 512 == time ) break;
		gf256v_rand( vinegar , _V1 );
		gen_l1_mat( mat_l1 , k , vinegar );

		// presuming: _O2*_O2 >= _O1*_O1
		// check if full-ranked mat_l1
		memcpy( mat_l2 , mat_l1 , _O1*_O1 );
		l1_succ = gf256mat_gauss_elim( mat_l2 , _O1 , _O1 );
		time ++;
	}
	uint8_t temp_vv1[_O1] ;
	gf256mpkc_mq_eval_n_m( temp_vv1 , k->l1_vv , vinegar , _V1 , _O1 );

//// line 7 - 14
	uint8_t _z[_PUB_N_BYTE] ;
	uint8_t y[_PUB_N_BYTE] ;
	uint8_t x[_PUB_N_BYTE] ;
	uint8_t w[_PUB_N_BYTE] ;

	uint8_t digest_salt[_HASH_LEN + _SALT_BYTE] = {0};
	uint8_t * salt = digest_salt + _HASH_LEN;
	memcpy( digest_salt , _digest , _HASH_LEN );

	memcpy( x , vinegar , _V1 );

	unsigned succ = 0;
	while( !succ ) {
		if( 512 == time ) break;

		gf256v_rand( salt , _SALT_BYTE );  /// line 8
		sha2_chain_msg( _z , _PUB_M_BYTE , digest_salt , _HASH_LEN+_SALT_BYTE ); /// line 9

		gf256v_add(_z,sk->vec_s,_PUB_M_BYTE);
		gf256mat_prod(y,sk->mat_s,_PUB_M_BYTE,_PUB_M,_z); /// line 10

		memcpy( temp_o1 , temp_vv1 , _O1 );
		gf256v_add( temp_o1 , y , _O1 );
		gf256mat_solve_linear_eq( x + _V1 , mat_l1 , temp_o1 , _O1 );

		gen_l2_mat( mat_l2 , k , x );
		gf256mpkc_mq_eval_n_m( temp_o2 , k->l2_vv , x , _V2 , _O2 );
		gf256v_add( temp_o2 , y+_O1 , _O2 );
		succ = gf256mat_solve_linear_eq( x + _V2 , mat_l2 , temp_o2 , _O2 );  /// line 13

		time ++;
	};

	gf256v_add(x,sk->vec_t,_PUB_N_BYTE);
	gf256mat_prod(w,sk->mat_t,_PUB_N_BYTE,_PUB_N,x);

	memset( signature , 0 , _SIGNATURE_BYTE );
	// return time;
	if( 256 <= time ) return -1;
	gf256v_add( signature , w , _PUB_N_BYTE );
	gf256v_add( signature + _PUB_N_BYTE , salt , _SALT_BYTE );
	return 0;
}

/// algorithm 8
int rainbow_verify( const uint8_t * digest , const uint8_t * signature , const uint8_t * pk )
{
	unsigned char digest_ck[_PUB_M_BYTE];
	rainbow_pubmap( digest_ck , pk , signature );

	unsigned char correct[_PUB_M_BYTE];
	unsigned char digest_salt[_HASH_LEN + _SALT_BYTE];
	memcpy( digest_salt , digest , _HASH_LEN );
	memcpy( digest_salt+_HASH_LEN , signature+_PUB_N_BYTE , _SALT_BYTE );
	sha2_chain_msg( correct , _PUB_M_BYTE , digest_salt , _HASH_LEN+_SALT_BYTE );

	unsigned char cc = 0;
	for(unsigned i=0;i<_PUB_M_BYTE;i++) {
		cc |= (digest_ck[i]^correct[i]);
	}

	return (0==cc)? 0: -1;
}






#endif  /// _RAINBOW_256_
