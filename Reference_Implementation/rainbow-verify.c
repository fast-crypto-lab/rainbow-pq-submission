
#include <stdio.h>
#include <string.h>

#include "rainbow_config.h"
//#include "rainbow.h"

#include "utils.h"
#include "hash_utils.h"

#include "api.h"


int main( int argc , char ** argv )
{
	printf( "%s\n", CRYPTO_ALGNAME );

        printf("sk size: %lu\n", CRYPTO_SECRETKEYBYTES );
        printf("pk size: %d\n",  CRYPTO_PUBLICKEYBYTES );
        printf("hash size: %d\n", _HASH_LEN );
        printf("signature size: %d\n\n", CRYPTO_BYTES );

	if( 4 != argc ) {
                printf("Usage:\n\n\trainbow-verify pk_file_name signature_file_name message_file_name\n\n");
                return -1;
        }

	//uint8_t pk[_PUB_KEY_LEN];
	uint8_t * pk = (uint8_t *) malloc( CRYPTO_PUBLICKEYBYTES );

	FILE * fp;
	int r;

	fp = fopen( argv[1] , "r");
	if( NULL == fp ) {
		printf("fail to open public key file.\n");
		return -1;
	}
	r = byte_fget( fp ,  pk , CRYPTO_PUBLICKEYBYTES );
	fclose( fp );
	if( CRYPTO_PUBLICKEYBYTES != r ) {
		printf("fail to load key file.\n");
		return -1;
	}

	unsigned char * msg = NULL;
	unsigned long long mlen = 0;
	r = byte_read_file( &msg , &mlen , argv[3] );
	if( 0 != r ) {
		printf("fail to read message file.\n");
		return -1;
	}

	unsigned char * signature = malloc( mlen + CRYPTO_BYTES );
	if( NULL == signature ) {
		printf("alloc memory for signature buffer fail.\n");
		return -1;
	}
	memcpy( signature , msg , mlen );
	fp = fopen( argv[2] , "r");
	if( NULL == fp ) {
		printf("fail to open signature file.\n");
		return -1;
	}
	r = byte_fget( fp ,  signature + mlen , CRYPTO_BYTES );
	fclose( fp );
	if( CRYPTO_BYTES != r ) {
		printf("fail to load signature file.\n");
		return -1;
	}

	r = crypto_sign_open( msg , &mlen , signature , mlen + CRYPTO_BYTES , pk );

	free( msg );
	free( signature );
	free( pk );

	if( 0 == r ) {
		printf("%s verification success.\n" , CRYPTO_ALGNAME );
		return 0;
	} else {
		printf("Verification fail.\n" );
		return -1;
	}
}

