
#include <stdio.h>

#include "rainbow_config.h"

#include "utils.h"
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
		printf("Usage:\n\n\trainbow-genkey pk_file_name sk_file_name [random_seed_file]\n\n");
		return -1;
	}

	unsigned char rnd_seed[48] = {0};
	if( 4 == argc ) {
		int rr = byte_from_file( rnd_seed , 48 , argv[3] );
		if( 0 != rr ) printf("read seed file fail.\n");
		prng_seed( rnd_seed , 48 );
		prng_dump_set( 1 );
	} else {
		prng_seed_file( "/dev/random" );
	}

	uint8_t *_sk = (uint8_t*)malloc( CRYPTO_SECRETKEYBYTES );
	uint8_t * qp_pk = (uint8_t*)malloc( CRYPTO_PUBLICKEYBYTES );
	FILE * fp;

	int r = crypto_sign_keypair( qp_pk, _sk );
	if( 0 != r ) {
		printf("%s genkey fails.\n", CRYPTO_ALGNAME );
		return -1;
	}

	fp = fopen( argv[1] , "w+");
	if( NULL == fp ) {
		printf("fail to open public key file.\n");
		return -1;
	}
	byte_fdump( fp , CRYPTO_ALGNAME " public key" , qp_pk , CRYPTO_PUBLICKEYBYTES );
	fclose( fp );

	fp = fopen( argv[2] , "w+");
	if( NULL == fp ) {
		printf("fail to open secret key file.\n");
		return -1;
	}
	//ptr = (unsigned char *)&sk;
	//sprintf(msg,"%s secret key", name);
	byte_fdump( fp ,  CRYPTO_ALGNAME " secret key" , _sk , CRYPTO_SECRETKEYBYTES );
	fclose( fp );

	printf("generate %s pk/sk success.\n" , CRYPTO_ALGNAME );

	if( 4 == argc ) {
		printf("\n");
		byte_fdump( stdout , "random seed[48] " , rnd_seed , 48 );
		unsigned char * randomness;
		unsigned n_rnd = prng_dump( &randomness );
		printf("\nused randomness[%d] ", n_rnd);
		byte_fdump( stdout , "" , randomness , n_rnd );
		printf("\n");
	}

	free( _sk );
	free( qp_pk );

	return 0;
}

