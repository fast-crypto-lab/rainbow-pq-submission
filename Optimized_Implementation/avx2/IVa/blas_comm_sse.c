
#include "gf16.h"


#include "emmintrin.h"
#include "tmmintrin.h"

#include "blas_config.h"

#include "gf16_sse.h"

#include "blas_sse.h"

#include "assert.h"






void gf16mat_prod_multab_sse( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * multab ) {
	assert( n_A_width <= 224 );
	assert( n_A_width > 0 );
	assert( n_A_vec_byte <= 112 );

	__m128i mask_f = _mm_set1_epi8(0xf);

	__m128i r0[7];
	__m128i r1[7];

	unsigned n_xmm = ((n_A_vec_byte + 15)>>4);
	for(unsigned i=0;i<n_xmm;i++) r0[i] = _mm_setzero_si128();
	for(unsigned i=0;i<n_xmm;i++) r1[i] = _mm_setzero_si128();

	for(unsigned i=0;i<n_A_width-1;i++) {
		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*16) );
		//__m128i mh = _mm_slli_epi16( ml , 4 );
		for(unsigned j=0;j<n_xmm;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(matA+j*16) );
			r0[j] ^= _mm_shuffle_epi8( ml , inp&mask_f );
			r1[j] ^= _mm_shuffle_epi8( ml , _mm_srli_epi16(inp,4)&mask_f );
		}
		matA += n_A_vec_byte;
	}
	unsigned n_16 = (n_A_vec_byte>>4);
	unsigned n_16_rem = n_A_vec_byte&0xf;
	{
		/// last column
		unsigned i=n_A_width-1;

		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*16) );
		for(unsigned j=0;j<n_16;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(matA+j*16) );
			r0[j] ^= _mm_shuffle_epi8( ml , inp&mask_f );
			r1[j] ^= _mm_shuffle_epi8( ml , _mm_srli_epi16(inp,4)&mask_f );
		}
		if( n_16_rem ) {
			__m128i inp = _load_xmm( matA + n_16*16 , n_16_rem );
			r0[n_16] ^= _mm_shuffle_epi8( ml , inp&mask_f );
			r1[n_16] ^= _mm_shuffle_epi8( ml , _mm_srli_epi16(inp,4)&mask_f );
		}
	}

	for(unsigned i=0;i<n_16;i++) _mm_storeu_si128( (__m128i*)(c + i*16) , r0[i]^_mm_slli_epi16(r1[i],4) );
	if( n_16_rem ) _store_xmm( c + n_16*16 , n_16_rem , r0[n_16]^_mm_slli_epi16(r1[n_16],4) );
}



static inline
uint8_t _gf16v_get_ele( const uint8_t *a , unsigned i ) {
	uint8_t r = a[i>>1];
	if( 0 == (i&1) ) return r&0xf;
	else return r>>4;
}


#if 0
/// slower
void gf16mat_prod_sse( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	assert( n_A_width <= 128 );
	assert( n_A_vec_byte <= 64 );

	uint8_t multab[128*16] __attribute__((aligned(32)));
	gf16v_generate_multab_sse( multab , b , n_A_width );

	gf16mat_prod_multab_sse( c , matA , n_A_vec_byte , n_A_width , multab );
}
#else
/// faster
void gf16mat_prod_sse( uint8_t * c , const uint8_t * mat_a , unsigned a_h_byte , unsigned a_w , const uint8_t * b ) {
	assert( a_w <= 224 );
	assert( a_h_byte <= 112 );

	__m128i mask_f = _mm_set1_epi8(0xf);

	__m128i r0[7];
	__m128i r1[7];
	unsigned n_xmm = ((a_h_byte+15)>>4);
	for(unsigned i=0;i<n_xmm;i++) r0[i] = _mm_setzero_si128();
	for(unsigned i=0;i<n_xmm;i++) r1[i] = _mm_setzero_si128();

	uint8_t _x[224] __attribute__((aligned(32)));
	gf16v_split_sse( _x , b , a_w );
	for(unsigned i=0;i < ((a_w+15)>>4); i++) {
		__m128i xi = _mm_load_si128( (__m128i*)(_x+i*16) );
		_mm_store_si128( (__m128i*)(_x+i*16) , tbl_gf16_log( xi ) );
	}

	for(unsigned i=0;i< a_w -1;i++) {
		__m128i ml = _mm_set1_epi8( _x[i] );
		for(unsigned j=0;j<n_xmm;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(mat_a+j*16) );
			r0[j] ^= tbl_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[j] ^= tbl_gf16_mul_log( _mm_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
		mat_a += a_h_byte;
	}
	unsigned n_16 = (a_h_byte>>4);
	unsigned n_16_rem = a_h_byte&0xf;
	{
		/// last column
		unsigned i=a_w-1;
		__m128i ml = _mm_set1_epi8( _x[i] );
		for(unsigned j=0;j<n_16;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(mat_a+j*16) );
			r0[j] ^= tbl_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[j] ^= tbl_gf16_mul_log( _mm_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
		if( n_16_rem ) {
			__m128i inp = _load_xmm( mat_a + n_16*16 , n_16_rem );
			r0[n_16] ^= tbl_gf16_mul_log( inp&mask_f , ml , mask_f );
			r1[n_16] ^= tbl_gf16_mul_log( _mm_srli_epi16(inp,4)&mask_f , ml , mask_f );
		}
	}

	for(unsigned i=0;i<n_16;i++) _mm_storeu_si128( (__m128i*)(c + i*16) , r0[i]^_mm_slli_epi16(r1[i],4) );
	if( n_16_rem ) _store_xmm( c + n_16*16 , n_16_rem , r0[n_16]^_mm_slli_epi16(r1[n_16],4) );
}
#endif




/// access aligned memory.
static
unsigned _gf16mat_gauss_elim_sse( uint8_t * mat , unsigned h , unsigned w_byte )
{
	assert( 0==(w_byte&31) ); /// w_byte is a multiple of 32.
	assert( 224 >= h );

	uint8_t pivot_column[224] __attribute__((aligned(32)));
	uint8_t _multab[224*16] __attribute__((aligned(32)));
	__m128i mask_f = _mm_set1_epi8( 0xf );
	unsigned w_2 = w_byte;
	unsigned w_n_16 = w_byte>>4;

	uint8_t r8 = 1;
	for(unsigned i=0;i<h;i++) {
		unsigned offset_16 = i>>5;
		uint8_t * ai = mat + w_2*i;
		for(unsigned j=i+1;j<h;j++) {
			uint8_t * aj = mat + w_2*j;

			uint8_t predicate = !gf16_is_nonzero( _gf16v_get_ele(ai,i) );
			uint32_t pr_u32 = ((uint32_t)0)-((uint32_t)predicate);
			__m128i pr_u128 = _mm_set1_epi32( pr_u32 );
			for(unsigned k=offset_16;k<w_n_16;k++) {
				__m128i inp0 = _mm_load_si128( (__m128i*)(ai+k*16) );
				__m128i inp1 = _mm_load_si128( (__m128i*)(aj+k*16) );
				__m128i r = inp0^(inp1&pr_u128);
				_mm_store_si128( (__m128i*)(ai+k*16) , r );
			}
		}
		pivot_column[0] = _gf16v_get_ele( ai , i );
		r8 &= gf16_is_nonzero( pivot_column[0] );
		__m128i p128 = _mm_load_si128( (__m128i*) pivot_column );
		__m128i inv_p = tbl_gf16_inv( p128 );
		_mm_store_si128( (__m128i*) pivot_column , inv_p );
		pivot_column[i] = pivot_column[0];
		for(unsigned j=0;j<h;j++) {
			if( i==j) continue;
			uint8_t * aj = mat + w_2*j;
			pivot_column[j] = _gf16v_get_ele( aj , i );
		}
		unsigned h_16 = (h+15)>>4;
		for(unsigned j=0;j<h_16;j++) { gf16v_generate_multab_16_sse( _multab + j*16*16 , pivot_column + j*16 ); }

		{
			/// pivot row
			unsigned j=i;
			uint8_t * aj = mat + w_2*j;
			__m128i ml = _mm_load_si128( (__m128i*)(_multab + 16*j) );
			__m128i mh = _mm_slli_epi16( ml , 4 );
			for(unsigned k=offset_16;k<w_n_16;k++) {
				__m128i inp = _mm_load_si128( (__m128i*)(aj+k*16) );
				__m128i r = linear_transform_8x8_128b( ml , mh , inp , mask_f );
				_mm_store_si128( (__m128i*)(aj+k*16) , r );
			}
		}
		for(unsigned j=0;j<h;j++) {
			if( i == j ) continue;
			uint8_t * aj = mat + w_2*j;
			__m128i ml = _mm_load_si128( (__m128i*)(_multab + 16*j) );
			__m128i mh = _mm_slli_epi16( ml , 4 );
			for(unsigned k=offset_16;k<w_n_16;k++) {
				__m128i inp0 = _mm_load_si128( (__m128i*)(ai+k*16) );
				__m128i inp = _mm_load_si128( (__m128i*)(aj+k*16) );
				__m128i r = inp ^ linear_transform_8x8_128b( ml , mh , inp0 , mask_f );
				_mm_store_si128( (__m128i*)(aj+k*16) , r );
			}
		}
	}
	return r8;
}


unsigned gf16mat_gauss_elim_sse( uint8_t * mat , unsigned h , unsigned w )
{
	assert( 0 == (w&1) ); // w is even.
	assert( 448 >= w );
	assert( 224 >= h );

	uint8_t _mat[224*224] __attribute__((aligned(32)));
	unsigned w_2 = (w+1)>>1;
	unsigned w_byte_32 = ((w_2+31)>>5) <<5;

	for(unsigned i=0;i<h;i++) for(unsigned j=0;j<w_2;j++) _mat[i*w_byte_32+j] = mat[i*w_2+j];
	unsigned r = _gf16mat_gauss_elim_sse( _mat , h , w_byte_32 );
	for(unsigned i=0;i<h;i++) for(unsigned j=0;j<w_2;j++) mat[i*w_2+j] = _mat[i*w_byte_32+j];
	return r;
}




unsigned gf16mat_solve_linear_eq_sse( uint8_t * sol , const uint8_t * inp_mat , const uint8_t * c_terms , unsigned n )
{
        assert( 64 >= n );

	uint8_t mat[ 64*64 ] __attribute__((aligned(32)));
	uint8_t _c[64] __attribute__((aligned(32)));
	gf16v_split_sse( _c , c_terms , n );

	unsigned n_byte = (n+1)>>1;
	unsigned n_byte_32 = ((n_byte+1+31)>>5 ) <<5;

	for(unsigned i=0;i<n;i++) {
		for( unsigned j=0;j<n_byte;j++) mat[ i*n_byte_32+j ] = inp_mat[i*n_byte+j];
		mat[i*n_byte_32+n_byte] = _c[i];
	}
	unsigned r8 = _gf16mat_gauss_elim_sse( mat , n , n_byte_32 );
	for(unsigned i=0;i<n;i+=2) {
		sol[i>>1] = mat[i*n_byte_32+n_byte]|(mat[(i+1)*n_byte_32+n_byte]<<4);
	}
	return r8;
}





///////////////////////////////  GF( 256 ) ////////////////////////////////////////////////////






/// XXX: un-tested.
void gf256mat_prod_multab_sse( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * multab ) {
	assert( n_A_width <= 192 );
	assert( n_A_width > 0 );
	assert( n_A_vec_byte <= 192 );

	__m128i mask_f = _mm_set1_epi8(0xf);
	__m128i r[12];
	unsigned n_xmm = ((n_A_vec_byte + 15)>>4);
	for(unsigned i=0;i<n_xmm;i++) r[i] = _mm_setzero_si128();

	for(unsigned i=0;i<n_A_width-1;i++) {
		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*32) );
		__m128i mh = _mm_load_si128( (__m128i*)( multab + i*32+16) );
		for(unsigned j=0;j<n_xmm;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(matA+j*16) );
			r[j] ^= linear_transform_8x8_128b( ml , mh , inp , mask_f );
		}
		matA += n_A_vec_byte;
	}
	unsigned n_16 = (n_A_vec_byte>>4);
	unsigned n_16_rem = n_A_vec_byte&0xf;
	{
		/// last column
		unsigned i=n_A_width-1;
		__m128i ml = _mm_load_si128( (__m128i*)( multab + i*32) );
		__m128i mh = _mm_load_si128( (__m128i*)( multab + i*32+16) );
		for(unsigned j=0;j<n_16;j++) {
			__m128i inp = _mm_loadu_si128( (__m128i*)(matA+j*16) );
			r[j] ^= linear_transform_8x8_128b( ml , mh , inp , mask_f );
		}
		if( n_16_rem ) {
			__m128i inp = _load_xmm( matA + n_16*16 , n_16_rem );
			r[n_16] ^= linear_transform_8x8_128b( ml , mh , inp , mask_f );
		}
	}
	for(unsigned i=0;i<n_16;i++) _mm_storeu_si128( (__m128i*)(c + i*16) , r[i] );
	if( n_16_rem ) _store_xmm( c + n_16*16 , n_16_rem , r[n_16] );
}



void gf256mat_prod_sse( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
	assert( n_A_width <= 192 );
	assert( n_A_vec_byte <= 192 );

	uint8_t multab[192*16*2] __attribute__((aligned(32)));
	gf256v_generate_multab_sse( multab , b , n_A_width );

	gf256mat_prod_multab_sse( c , matA , n_A_vec_byte , n_A_width , multab );
}





///////////////////////////////////////////////////////////////////////////




static
unsigned _gf256mat_gauss_elim_sse( uint8_t * mat , unsigned h , unsigned w )
{
	assert( 0 == (w&15) );
	unsigned n_xmm = w>>4;

	__m128i mask_0 = _mm_setzero_si128();

	uint8_t rr8 = 1;
	for(unsigned i=0;i<h;i++) {
		unsigned char i_r = i&0xf;
		unsigned i_d = i>>4;

		uint8_t * mi = mat+i*w;

		for(unsigned j=i+1;j<h;j++) {
			__m128i piv_i = _mm_load_si128( (__m128i*)( mi + i_d*16 ) );
			uint8_t * mj = mat+j*w;
			__m128i piv_j = _mm_load_si128( (__m128i*)( mj + i_d*16 ) );

			__m128i is_madd = _mm_cmpeq_epi8( piv_i , mask_0 ) ^ _mm_cmpeq_epi8( piv_j , mask_0 );
			__m128i madd_mask = _mm_shuffle_epi8( is_madd , _mm_set1_epi8(i_r) );

			piv_i ^= madd_mask&piv_j;
			_mm_store_si128( (__m128i*)( mi+ i_d*16 ) , piv_i );
			for(unsigned k=i_d+1;k<n_xmm;k++) {
				piv_i = _mm_load_si128( (__m128i*)( mi + k*16 ) );
				piv_j = _mm_load_si128( (__m128i*)( mj + k*16 ) );

				piv_i ^= madd_mask&piv_j;
				_mm_store_si128( (__m128i*)( mi+ k*16 ) , piv_i );
			}
		}
		rr8 &= gf256_is_nonzero( mi[i] );

		__m128i _pivot = _mm_set1_epi8( mi[i] );
		__m128i _ip = tbl_gf256_inv( _pivot );
		for(unsigned k=i_d;k<n_xmm;k++) {
			__m128i rowi = _mm_load_si128( (__m128i*)(mi+k*16) );
			rowi = tbl_gf256_mul( rowi , _ip );
			_mm_store_si128( (__m128i*)(mi+k*16) , rowi );
		}

		for(unsigned j=0;j<h;j++) {
			if(i==j) continue;

			uint8_t * mj = mat+j*w;
			__m128i mm = _mm_set1_epi8( mj[i] );

			for(unsigned k=i_d;k<n_xmm;k++) {
				__m128i rowi = _mm_load_si128( (__m128i*)(mi+k*16) );
				rowi = tbl_gf256_mul( rowi , mm );
				rowi ^= _mm_load_si128( (__m128i*)(mj+k*16) );
				_mm_store_si128( (__m128i*)(mj+k*16) , rowi );
			}
		}
	}
	return rr8;
}



unsigned gf256mat_gauss_elim_sse( uint8_t * mat , unsigned h , unsigned w )
{
	assert( 512 >= w );
	assert( 256 >= h );

	uint8_t _mat[512*256] __attribute__((aligned(32)));
	unsigned w_16 = ((w+15)>>4) <<4;

	for(unsigned i=0;i<h;i++) for(unsigned j=0;j<w;j++) _mat[i*w_16+j] = mat[i*w+j];
	unsigned r = _gf256mat_gauss_elim_sse( _mat , h , w_16 );
	for(unsigned i=0;i<h;i++) for(unsigned j=0;j<w;j++) mat[i*w+j] = _mat[i*w_16+j];
	return r;
}



unsigned gf256mat_solve_linear_eq_sse( uint8_t * sol , const uint8_t * inp_mat , const uint8_t * c_terms, unsigned n )
{
	assert( 48 >= n );

	uint8_t mat[48*64] __attribute__((aligned(32)));
	unsigned mat_width = (((n+1)+15)>>4)<<4;

	for(unsigned i=0;i<n;i++) {
		for(unsigned j=0;j<n;j++) mat[i*mat_width+j] = inp_mat[i*n+j];
		mat[i*mat_width+n] = c_terms[i];
	}
	unsigned r = _gf256mat_gauss_elim_sse( mat , n , mat_width );
	for(unsigned i=0;i<n;i++) sol[i] = mat[i*mat_width+n];
	return r;
}




