#ifndef _BLAS_COMM_H_
#define _BLAS_COMM_H_

#include <stdint.h>

#include <stdio.h>

#include "blas.h"



#ifdef  __cplusplus
extern  "C" {
#endif



///////////////////  prng //////////////////////////////////


#include "prng_utils.h"


static inline
void gf16v_rand( uint8_t * a , unsigned _num_ele ) {
	if( 0 == _num_ele ) return;
	unsigned _num_byte = (_num_ele+1)/2;
	prng_bytes( a , _num_byte );
	if( _num_ele & 1 ) a[_num_byte-1] &= 0xf;
}


static inline
void gf256v_rand( uint8_t * a , unsigned _num_byte ) {
	prng_bytes( a , _num_byte );
}




/////////////////  input/output   /////////////////////////////////////


static inline
void gf256v_fdump(FILE * fp, const uint8_t *v, unsigned _num_byte) {
	fprintf(fp,"[%2d][",_num_byte);
	for(unsigned i=0;i<_num_byte;i++) { fprintf(fp,"0x%02x,",v[i]); if(7==(i%8)) fprintf(fp," ");}
	fprintf(fp,"]");
}



static inline
void gf256v_set_zero( uint8_t * b, unsigned _num_byte ) { gf256v_add( b , b , _num_byte ); }



static inline
unsigned gf256v_is_zero( const uint8_t * a, unsigned _num_byte ) {
	unsigned char r = 0;
	for(unsigned i=0;i<_num_byte;i++) r |= a[i];
	return (0==r);
}




static inline
unsigned char gf16v_get_ele( const uint8_t * a , unsigned i ) {
	unsigned char r = a[i>>1];
	r = ( i&1 )? (r>>4):(r&0xf);
	return r;
}


static inline
unsigned char gf16v_set_ele( uint8_t * a , unsigned i , uint8_t v ) {
	unsigned char m = (i&1)? 0xf : 0xf0;
	a[i>>1] &= m;
	m = ( i&1 )? v<<4 : v&0xf;
	a[i>>1] |= m;
	return v;
}



static inline
void gf16v_split( uint8_t * z , const uint8_t * x , unsigned n )
{
	for(unsigned i=0;i<n;i++) z[i] = gf16v_get_ele( x , i );
}




/////////////////////////////////////


static inline
void gf16mat_fdump(FILE * fp, const uint8_t *v, unsigned n_vec_byte , unsigned n_vec ) {
	for(unsigned i=0;i<n_vec;i++) {
		fprintf(fp,"[%d]",i);
		gf256v_fdump(fp,v,n_vec_byte);
		fprintf(fp,"\n");
		v += n_vec_byte;
	}
}

static inline
void gf256mat_fdump(FILE * fp, const uint8_t *v, unsigned n_vec_byte , unsigned n_vec ) {
	gf16mat_fdump(fp,v,n_vec_byte,n_vec);
}



///////////////// multiplications  ////////////////////////////////

/// polynomial multplication
/// School boook
static inline
void gf256v_polymul( uint8_t * c, const uint8_t * a , const uint8_t * b , unsigned _num ) {
	for(unsigned i=0;i<_num*2-1;i++) c[i] = 0;
	for(unsigned i=0;i<_num;i++) gf256v_madd( c+i , a , b[i] , _num );
}


///////////  matrix-vector


static inline
void gf16rowmat_prod( uint8_t * c , const uint8_t * matA , unsigned n_A_height , unsigned n_A_vec_byte , const uint8_t * b ) {
	for(unsigned i=0;i<n_A_height;i++) {
		gf16v_set_ele( c , i , gf16v_dot( matA , b , n_A_vec_byte ) );
		matA += n_A_vec_byte;
	}
}

static inline
void gf16mat_prod( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	gf256v_set_zero(c,n_A_vec_byte);
	for(unsigned i=0;i<n_A_width;i++) {
		uint8_t bb = gf16v_get_ele( b , i );
		gf16v_madd( c , matA , bb , n_A_vec_byte );
		matA += n_A_vec_byte;
	}
}



static inline
void gf256mat_prod( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	gf256v_set_zero(c,n_A_vec_byte);
	for(unsigned i=0;i<n_A_width;i++) {
		gf256v_madd( c , matA , b[i] , n_A_vec_byte );
		matA += n_A_vec_byte;
	}
}


/////////// matrix-matrix


static inline
void gf16mat_mul( uint8_t * c , const uint8_t * a , const uint8_t * b , unsigned len_vec ) {
	unsigned n_vec_byte = (len_vec+1)/2;
	for(unsigned k=0;k<len_vec;k++){
		gf256v_set_zero( c , n_vec_byte );
		const uint8_t * bk = b + n_vec_byte * k;
		for(unsigned i=0;i<len_vec;i++) {
			uint8_t bb = gf16v_get_ele( bk , i );
			gf16v_madd( c , a + n_vec_byte * i , bb , n_vec_byte  );
		}
		c += n_vec_byte;
	}
}


static inline
void gf256mat_mul( uint8_t * c , const uint8_t * a , const uint8_t * b , unsigned len_vec ) {
	unsigned n_vec_byte = len_vec;
	for(unsigned k=0;k<len_vec;k++){
		gf256v_set_zero( c , n_vec_byte );
		const uint8_t * bk = b + n_vec_byte * k;
		for(unsigned i=0;i<len_vec;i++) {
			gf256v_madd( c , a + n_vec_byte * i , bk[i] , n_vec_byte  );
		}
		c += n_vec_byte;
	}
}




/////////////////   algorithms:  gaussian elim  //////////////////


static inline
unsigned gf16mat_gauss_elim( uint8_t * mat , unsigned h , unsigned w )
{
	/// assert( 0==(w&1) );  w must be even !!!
	unsigned n_w_byte = (w+1)/2;
	unsigned r8 = 1;
	for(unsigned i=0;i<h;i++) {
		unsigned offset_byte = i>>1;
		uint8_t * ai = mat + n_w_byte*i;
		for(unsigned j=i+1;j<h;j++) {
			uint8_t * aj = mat + n_w_byte*j;
			gf256v_predicated_add( ai+offset_byte , !gf16_is_nonzero( gf16v_get_ele(ai,i) ) , aj+offset_byte , n_w_byte-offset_byte );
		}
		uint8_t pivot = gf16v_get_ele( ai , i );
		r8 &= gf16_is_nonzero( pivot );
		pivot = gf16_inv( pivot );
		offset_byte = (i+1)>>1;
		gf16v_mul_scalar( ai+offset_byte , pivot , n_w_byte-offset_byte );
		for(unsigned j=0;j<h;j++) {
			if(i==j) continue;
			uint8_t * aj = mat + n_w_byte*j;
			gf16v_madd( aj+offset_byte , ai+offset_byte , gf16v_get_ele( aj , i ) , n_w_byte-offset_byte );
		}
	}
	return r8;
}


static inline
void gf16mat_submat( uint8_t * mat2 , unsigned w2 , unsigned st , const uint8_t * mat , unsigned w , unsigned h )
{
	unsigned n_byte_w1 = (w+1)/2;
	unsigned n_byte_w2 = (w2+1)/2;
	unsigned st_2 = st/2;
	for(unsigned i=0;i<h;i++) {
		for(unsigned j=0;j<n_byte_w2;j++) mat2[i*n_byte_w2+j] = mat[i*n_byte_w1+st_2+j];
	}
}


static inline
unsigned gf16mat_inv( uint8_t * inv_a , const uint8_t * a , unsigned H , uint8_t * buffer )
{
	unsigned n_w_byte = (H+1)/2;

	uint8_t * aa = buffer;
	for(unsigned i=0;i<H;i++) {
		uint8_t * ai = aa + i*2*n_w_byte;
		gf256v_set_zero( ai , 2*n_w_byte );
		gf256v_add( ai , a + i*n_w_byte , n_w_byte );
		gf16v_set_ele( ai + n_w_byte , i , 1 );
	}
	unsigned char r8 = gf16mat_gauss_elim( aa , H , 2*H );  /// XXX: will 2*H fail if H is odd ???
	gf16mat_submat( inv_a , H , H , aa , 2*H , H );
	return r8;
}



static inline
unsigned gf16mat_rand_inv( uint8_t * a , uint8_t * b , unsigned H , uint8_t * buffer )
{
	unsigned n_w_byte = (H+1)/2;

	for(unsigned i=0;i<H;i++) {
		gf16v_rand( a + i*n_w_byte , H );
	}
	return gf16mat_inv( b , a , H , buffer );
}



static inline
void gf16mat_subcolumn( uint8_t * col , unsigned idx , const uint8_t * mat , unsigned w , unsigned h )
{
	unsigned n_byte_w = (w+1)/2;
	for(unsigned i=0;i<h;i++) {
		const uint8_t * mat_i = mat + n_byte_w*i;
		uint8_t qq = gf16v_get_ele( mat_i , idx );
		gf16v_set_ele( col , i , qq );
	}
}






/////////////////////////////////////////////////



static inline
unsigned gf256mat_gauss_elim( uint8_t * mat , unsigned h , unsigned w )
{
	unsigned r8 = 1;
	for(unsigned i=0;i<h;i++) {
		uint8_t * ai = mat + w*i;
		for(unsigned j=i+1;j<h;j++) {
			uint8_t * aj = mat + w*j;
			gf256v_predicated_add( ai + i , !gf256_is_nonzero(ai[i]) , aj + i , w-i );
		}
		r8 &= gf256_is_nonzero(ai[i]);
		uint8_t pivot = ai[i];
		pivot = gf256_inv( pivot );
		gf256v_mul_scalar( ai + (i+1) , pivot , w - (i+1) );
		for(unsigned j=0;j<h;j++) {
			if(i==j) continue;
			uint8_t * aj = mat + w*j;
			gf256v_madd( aj + (i+1) , ai + (i+1) , aj[i] , w - (i+1) );
		}
	}
	return r8;
}

static inline
void gf256mat_submat( uint8_t * mat2 , unsigned w2 , unsigned st , const uint8_t * mat , unsigned w , unsigned h )
{
	for(unsigned i=0;i<h;i++) {
		for(unsigned j=0;j<w2;j++) mat2[i*w2+j] = mat[i*w+st+j];
	}
}

static inline
unsigned gf256mat_inv( uint8_t * inv_a , const uint8_t * a , unsigned H , uint8_t * buffer )
{
	uint8_t * aa = buffer;
	for(unsigned i=0;i<H;i++) {
		uint8_t * ai = aa + i*2*H;
		gf256v_set_zero( ai , 2*H );
		gf256v_add( ai , a + i*H , H );
		ai[H+i] = 1;
	}
	unsigned char r8 = gf256mat_gauss_elim( aa , H , 2*H );
	gf256mat_submat( inv_a , H , H , aa , 2*H , H );
	return r8;
}



static inline
unsigned gf256mat_rand_inv( uint8_t * a , uint8_t * b , unsigned H , uint8_t * buffer )
{
	for(unsigned i=0;i<H;i++) {
		gf256v_rand( a + i*H , H );
	}
	return gf256mat_inv( b , a , H , buffer );
}






#ifdef  __cplusplus
}
#endif



#endif

