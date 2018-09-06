
#include "rainbow_config.h"

#include "rainbow_16.h"

#include "blas.h"

#include "string.h"

#include "assert.h"

#include "prng_utils.h"


#ifdef _RAINBOW_16


#include "mpkc.h"

void rainbow_pubmap( uint8_t * z , const uint8_t * poly , const uint8_t * w )
{
	gf16mpkc_pubmap_n_m_avx2( z , poly , w , _PUB_N , _PUB_M );
}

static void mpkc_interpolate_gf16( uint8_t * poly , void (*quad_poly)(void *,const void *,const void *) , const void * key )
{
	_gf16mpkc_interpolate_n_m( poly , quad_poly , key , _PUB_N , _PUB_M );
}






#ifndef _DEBUG_RAINBOW_

static unsigned rainbow_ivs_central_map( uint8_t * r , const rainbow_ckey * k , const uint8_t * a );

static void rainbow_central_map( uint8_t * r , const rainbow_ckey * k , const uint8_t * a );

static void rainbow_pubmap_seckey( uint8_t * z , const rainbow_key * sk , const uint8_t * w );

static void rainbow_genkey_debug( rainbow_key * pk , rainbow_key * sk );

#endif


#ifndef _DEBUG_RAINBOW_
static
#endif
void rainbow_pubmap_seckey( uint8_t * z , const rainbow_key * sk , const uint8_t * w ) {

	uint8_t tt[_PUB_N_BYTE] __attribute__((aligned(32))) = {0};
	uint8_t tt2[_PUB_N_BYTE] __attribute__((aligned(32))) = {0};

	gf16mat_prod( tt , sk->mat_t , _PUB_N_BYTE , _PUB_N , w );
	gf256v_add( tt , sk->vec_t , _PUB_N_BYTE );

	rainbow_central_map( tt2 , & sk->ckey , tt );

	gf16mat_prod( z , sk->mat_s , _PUB_M_BYTE , _PUB_M , tt2 );
	gf256v_add( z , sk->vec_s , _PUB_M_BYTE );

}




static void gen_mat_inv( uint8_t * s , uint8_t * inv_s , unsigned h , uint8_t * buffer )
{
	/// max trials: 128
	for(unsigned i=0;i<128;i++) {
		uint8_t r = gf16mat_rand_inv( s , inv_s , h , buffer );
		if( r ) break;
	}
}



/// algorithm 6

#ifndef _DEBUG_RAINBOW_
static
#endif
void rainbow_genkey_debug( rainbow_key * pk , rainbow_key * sk )
{
#ifdef _CONSISTENT_WITH_ALGO_6_
	gen_mat_inv( pk->mat_s , sk->mat_s , _PUB_M , (uint8_t *)&sk->ckey );
	gf256v_rand( pk->vec_s , _PUB_M_BYTE );
	memcpy( sk->vec_s , pk->vec_s , _PUB_M_BYTE );

	gen_mat_inv( pk->mat_t , sk->mat_t , _PUB_N , (uint8_t *)&sk->ckey );
	gf256v_rand( pk->vec_t , _PUB_N_BYTE );
	memcpy( sk->vec_t , pk->vec_t , _PUB_N_BYTE );

	gf256v_rand( (uint8_t *)&sk->ckey , sizeof(rainbow_ckey) );
	memcpy( (uint8_t *)&pk->ckey , (uint8_t *)&sk->ckey , sizeof(rainbow_ckey) );
#else
	gf256v_rand( (uint8_t *)&sk->ckey , sizeof(rainbow_ckey) );

	gen_mat_inv( pk->mat_t , sk->mat_t , _PUB_N , (uint8_t *)&pk->ckey );
	gen_mat_inv( pk->mat_s , sk->mat_s , _PUB_M , (uint8_t *)&pk->ckey );

	gf256v_rand( pk->vec_t , _PUB_N_BYTE );
	memcpy( sk->vec_t , pk->vec_t , _PUB_N_BYTE );

	gf256v_rand( pk->vec_s , _PUB_M_BYTE );
	memcpy( sk->vec_s , pk->vec_s , _PUB_M_BYTE );

	memcpy( (uint8_t *)&pk->ckey , (uint8_t *)&sk->ckey , sizeof(rainbow_ckey) );
#endif
}



static inline
void rainbow_pubmap_wrapper( void * z, const void* pk_key, const void * w) {
	rainbow_pubmap_seckey( (uint8_t *)z , (const rainbow_key *)pk_key, (const uint8_t *)w );
}


void rainbow_genkey( uint8_t * pk , uint8_t * sk )
{

	rainbow_key _pk;
	rainbow_genkey_debug( &_pk , (rainbow_key *)sk );

	mpkc_interpolate_gf16( pk , rainbow_pubmap_wrapper , (const void*) &_pk );

	pk[_PUB_KEY_LEN-1] = _SALT_BYTE;
	sk[_SEC_KEY_LEN-1] = _SALT_BYTE;
}


unsigned rainbow_secmap( uint8_t * w , const rainbow_key * sk , const uint8_t * z )
{
	//if( gf256v_is_zero(z,_PUB_M_BYTE) ) { memset(w,0,_PUB_N_BYTE); return 0; }

	uint8_t _z[_PUB_M_BYTE] __attribute__((aligned(32)));
	uint8_t y[_PUB_N_BYTE] __attribute__((aligned(32)));
	uint8_t x[_PUB_N_BYTE] __attribute__((aligned(32)));

	memcpy( _z , z , _PUB_M_BYTE );
	gf256v_add(_z,sk->vec_s,_PUB_M_BYTE);
	gf16mat_prod(y,sk->mat_s,_PUB_M_BYTE,_PUB_M,_z);

	unsigned succ = 0;
	unsigned time = 0;
	while( !succ ) {
		if( 256 == time ) break;

		gf256v_rand( x , _PUB_N_BYTE - _PUB_M_BYTE );

		succ = rainbow_ivs_central_map( x , & sk->ckey , y );
		time ++;
	};

	gf256v_add(x,sk->vec_t,_PUB_N_BYTE);
	gf16mat_prod(w,sk->mat_t,_PUB_N_BYTE,_PUB_N,x);

	if( succ ) return 0;
	return time;
}





/////////////////////////////


static inline
void gen_l1_mat( uint8_t * mat , const rainbow_ckey * k , const uint8_t * multab ) {
	for(unsigned i=0;i<_O1;i++) {
		gf16mat_prod_multab_sse( mat + i*_O1_BYTE , k->l1_vo[i] , _O1_BYTE , _V1 , multab );
	}
	gf256v_add( mat , k->l1_o , _O1_BYTE*_O1 );
}

static inline
void gen_l2_mat( uint8_t * mat , const rainbow_ckey * k , const uint8_t * multab ) {
	for(unsigned i=0;i<_O2;i++) {
		gf16mat_prod_multab_sse( mat + i*_O2_BYTE , k->l2_vo[i] , _O2_BYTE , _V2 , multab );
	}
	gf256v_add( mat , k->l2_o , _O2_BYTE*_O2 );
}



#ifndef _DEBUG_RAINBOW_
static
#endif
void rainbow_central_map( uint8_t * r , const rainbow_ckey * k , const uint8_t * a ) {
#ifdef _DEBUG_MPKC_
memcpy( r , a+_V1_BYTE , _PUB_M_BYTE );
return;
#endif
	uint8_t mat1[_O2*_O2_BYTE] __attribute__((aligned(32)));
	uint8_t temp[_O2_BYTE] __attribute__((aligned(32)));

	uint8_t multab[(_V2)*16] __attribute__((aligned(32)));
	gf16v_generate_multab_sse( multab , a , _V2 );

	gen_l1_mat( mat1 , k , multab );

	gf16rowmat_prod( r , mat1 , _O1 , _O1_BYTE , a+_V1_BYTE );
	gf16mpkc_mq_multab_n_m( temp , k->l1_vv , multab , _V1 , _O1 );
	gf256v_add( r , temp , _O1_BYTE );

	gen_l2_mat( mat1 , k , multab );

	gf16rowmat_prod( r+_O1_BYTE , mat1 , _O2 , _O2_BYTE , a+_V2_BYTE );
	gf16mpkc_mq_multab_n_m( temp , k->l2_vv , multab , _V2 , _O2 );
	gf256v_add( r+_O1_BYTE , temp , _O2_BYTE );
}





#ifndef _DEBUG_RAINBOW_
static
#endif
unsigned rainbow_ivs_central_map( uint8_t * r , const rainbow_ckey * k , const uint8_t * a )
{
	uint8_t mat1[_O1*_O1] __attribute__((aligned(32)));
	uint8_t temp[_O1_BYTE] __attribute__((aligned(32)));

	uint8_t multab[(_V2)*16] __attribute__((aligned(32)));
	gf16v_generate_multab_sse( multab , r , _V1 );

	gf16mpkc_mq_multab_n_m( temp , k->l1_vv , multab , _V1 , _O1 );
	gf256v_add( temp  , a , _O1_BYTE );
	gen_l1_mat( mat1 , k , multab );
	//unsigned r1 = linear_solver_l1( r+_V1_BYTE , mat1 , temp );
	unsigned r1 = gf16mat_solve_linear_eq( r+_V1_BYTE , mat1 , temp , _O1 );

	gf16v_generate_multab_sse( multab + (_V1)*16 , r+_V1_BYTE , _O1 );

	uint8_t mat2[_O2*_O2] __attribute__((aligned(32)));
	uint8_t temp2[_O2_BYTE] __attribute__((aligned(32)));
	gen_l2_mat( mat2 , k , multab );
	gf16mpkc_mq_multab_n_m( temp2 , k->l2_vv , multab , _V2 , _O2 );
	gf256v_add( temp2  , a+_O1_BYTE , _O2_BYTE );
	//unsigned r2 = linear_solver_l2( r+_V2_BYTE , mat2 , temp2 );
	unsigned r2 = gf16mat_solve_linear_eq( r+_V2_BYTE , mat2 , temp2 , _O2 );

	return r1&r2;
}


#include "hash_utils.h"


/// algorithm 7
int rainbow_sign( uint8_t * signature , const uint8_t * _sk , const uint8_t * _digest )
{
	const rainbow_key * sk = (const rainbow_key *)_sk;
	const rainbow_ckey * k = &( sk->ckey);
//// line 1 - 5
	uint8_t mat_l1[_O1*_O1] __attribute__((aligned(32)));
	uint8_t mat_l2[_O2*_O2] __attribute__((aligned(32)));
	uint8_t temp_o1[_O1_BYTE] __attribute__((aligned(32))) = {0};
	uint8_t temp_o2[_O2_BYTE] __attribute__((aligned(32)));
	uint8_t vinegar[_V1_BYTE] __attribute__((aligned(32)));

	uint8_t multab[(_V2)*16] __attribute__((aligned(32)));

	unsigned l1_succ = 0;
	unsigned time = 0;
	while( !l1_succ ) {
		if( 512 == time ) break;
		gf256v_rand( vinegar , _V1_BYTE );
		gf16v_generate_multab_sse( multab , vinegar , _V1 );
		gen_l1_mat( mat_l1 , k , multab );

		// presuming: _O2*_O2 >= _O1*_O1
		// check if full-ranked mat_l1
		memcpy( mat_l2 , mat_l1 , _O1*_O1_BYTE );
		l1_succ = gf16mat_gauss_elim( mat_l2 , _O1 , _O1 );
		//l1_succ = linear_solver_l1( temp_o1 , mat_l1 , temp_o1 );
		time ++;
	}
	uint8_t temp_vv1[_O1_BYTE] __attribute__((aligned(32)));
	gf16mpkc_mq_multab_n_m( temp_vv1 , k->l1_vv , multab , _V1 , _O1 );

	//// line 7 - 14
	uint8_t _z[_PUB_M_BYTE] __attribute__((aligned(32)));
	uint8_t y[_PUB_M_BYTE] __attribute__((aligned(32)));
	uint8_t x[_PUB_N_BYTE] __attribute__((aligned(32)));
	uint8_t w[_PUB_N_BYTE] __attribute__((aligned(32)));
	uint8_t digest_salt[_HASH_LEN + _SALT_BYTE] = {0};
	uint8_t * salt = digest_salt + _HASH_LEN;
	memcpy( digest_salt , _digest , _HASH_LEN );

	memcpy( x , vinegar , _V1_BYTE );
	unsigned succ = 0;
	while( !succ ) {
		if( 512 == time ) break;

		gf256v_rand( salt , _SALT_BYTE );  /// line 8
		sha2_chain_msg( _z , _PUB_M_BYTE , digest_salt , _HASH_LEN+_SALT_BYTE ); /// line 9

		gf256v_add(_z,sk->vec_s,_PUB_M_BYTE);
		gf16mat_prod(y,sk->mat_s,_PUB_M_BYTE,_PUB_M,_z); /// line 10

		memcpy( temp_o1 , temp_vv1 , _O1_BYTE );
		gf256v_add( temp_o1 , y , _O1_BYTE );
		//linear_solver_l1( x + _V1_BYTE , mat_l1 , temp_o1 );
		gf16mat_solve_linear_eq( x + _V1_BYTE , mat_l1 , temp_o1 , _O1 );

		gf16v_generate_multab_sse( multab+(_V1)*16 , x+_V1_BYTE , _O1 );

		gen_l2_mat( mat_l2 , k , multab );
		gf16mpkc_mq_multab_n_m( temp_o2 , k->l2_vv , multab , _V2 , _O2 );
		gf256v_add( temp_o2 , y+_O1_BYTE , _O2_BYTE );
		//succ = linear_solver_l2( x + _V2_BYTE , mat_l2 , temp_o2 );  /// line 13
		succ = gf16mat_solve_linear_eq( x + _V2_BYTE , mat_l2 , temp_o2 , _O2 );  /// line 13

		time ++;
	};
	gf256v_add(x,sk->vec_t,_PUB_N_BYTE);
	gf16mat_prod(w,sk->mat_t,_PUB_N_BYTE,_PUB_N,x);

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



#endif  /// _RAINBOW_16
