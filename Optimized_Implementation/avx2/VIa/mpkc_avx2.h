
#ifndef _MPKC_AVX2_H_
#define _MPKC_AVX2_H_




#ifdef  __cplusplus
extern  "C" {
#endif


void gf16mpkc_pubmap_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m );

void gf16mpkc_mq_eval_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m );

void gf16mpkc_mq_multab_n_m_avx2( uint8_t * z , const uint8_t * pk_mat , const uint8_t * multab , unsigned n, unsigned m );


#ifdef  __cplusplus
}
#endif


#endif
