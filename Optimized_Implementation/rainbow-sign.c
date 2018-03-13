
#include <stdio.h>

#include "rainbow_config.h"

#include "utils.h"
#include "hash_utils.h"
#include "prng_utils.h"

#include "api.h"



int main( int argc , char ** argv )
{
	printf( "%s\n", CRYPTO_ALGNAME );

        printf("sk size: %lu\n", CRYPTO_SECRETKEYBYTES );
        printf("pk size: %d\n",  CRYPTO_PUBLICKEYBYTES );
        printf("hash size: %d\n", _HASH_LEN );
        printf("signature size: %d\n\n", CRYPTO_BYTES );

	if( !((3 == argc) || (4 == argc)) ) {
		printf("Usage:\n\n\trainbow-sign sk_file_name file_to_be_signed [random_seed_file]\n\n");
		return -1;
	}

	unsigned char rnd_seed[48] = {0};
	if( 4 == argc ) {
		int rr = byte_from_file( rnd_seed , 48 , argv[3] );
		if( 0 != rr ) printf("read seed file fail.\n");
		prng_seed( rnd_seed , 48 );
		prng_dump_set( 1 );
	} else {
		prng_seed_file( "dev/random" );
	}

	uint8_t *_sk = (uint8_t*)malloc( CRYPTO_SECRETKEYBYTES );

	FILE * fp;
	int r = 0;

	fp = fopen( argv[1] , "r");
	if( NULL == fp ) {
		printf("fail to open secret key file.\n");
		return -1;
	}
	r = byte_fget( fp ,  _sk , CRYPTO_SECRETKEYBYTES );
	fclose( fp );
	if( CRYPTO_SECRETKEYBYTES != r ) {
		printf("fail to load key file.\n");
		return -1;
	}

	unsigned char * msg = NULL;
	unsigned long long mlen = 0;
	r = byte_read_file( &msg , &mlen , argv[2] );
	if( 0 != r ) {
		printf("fail to read message file.\n");
		return -1;
	}

	unsigned char * signature = malloc( mlen + CRYPTO_BYTES );
	if( NULL == signature ) {
		printf("alloc memory for signature buffer fail.\n");
		return -1;
	}

	unsigned long long smlen = 0;
	r = crypto_sign( signature, &smlen, msg , mlen , _sk );
	if( 0 != r ) {
		printf("sign() fail.\n");
		return -1;
	}

	byte_fdump( stdout , CRYPTO_ALGNAME " signature"  , signature + mlen , CRYPTO_BYTES );
	printf("\n");

	if( 4 == argc ) {
		printf("\n");
		byte_fdump( stdout , "random seed[48] " , rnd_seed , 48 );
		unsigned char * randomness;
		unsigned n_rnd = prng_dump( &randomness );
		printf("\nused randomness[%d] ", n_rnd);
		byte_fdump( stdout , "" , randomness , n_rnd );
		printf("\n");
	}

	free( msg );
	free( signature );
	free( _sk );

	return 0;
}

