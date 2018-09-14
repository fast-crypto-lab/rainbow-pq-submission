#ifndef _SHA256_UTILS_H_
#define _SHA256_UTILS_H_

#include <stdio.h>
#include <stdlib.h>


/// for the definition of _HASH_LEN.
#include "hash_len_config.h"


#ifdef  __cplusplus
extern  "C" {
#endif


int sha2_chain_msg( unsigned char * digest , unsigned n_digest , const unsigned char * m , unsigned long long mlen );



#ifdef  __cplusplus
}
#endif



#endif

