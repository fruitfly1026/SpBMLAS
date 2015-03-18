/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */



#pragma once

#include "config.h"
#include "sparse_io.h"
#include "sparse_operations.h"
#include "test_spmv.h"
#include "benchmark_spmv.h"
#include "spmv_host/dia_host.h"
#include "spmv_host/ell_host.h"
#include "spmv_host/csr_host.h"
#include "spmv_host/coo_host.h"


template <typename IndexType, typename ValueType>
int test_dia_matrix_kernels(const csr_matrix<IndexType,ValueType>& csr, int kernel_tag, double *gflops, FILE *fp_feature)  
{
    printf("\n####  Testing DIA Kernels  ####\n");
    double max_fill = 20;
    IndexType max_diags = static_cast<IndexType>( (max_fill * csr.num_nonzeros) / csr.num_rows + 1 );

    // CREATE DIA MATRIX
    printf("\tcreating dia_matrix:");
    dia_matrix<IndexType,ValueType> dia = csr_to_dia<IndexType,ValueType>(csr, max_diags, fp_feature);
    printf("\n");

    if (dia.num_nonzeros == 0 && csr.num_nonzeros != 0)
    {     
	    printf("\tNumber of diagonals (%d) excedes limit (%d)\n", dia.complete_ndiags, max_diags);
	    return 0;
    }

    printf("\tFound %d diagonals\n", dia.complete_ndiags);
    double occupy_ratio = (double)dia.num_nonzeros / (dia.complete_ndiags*dia.num_rows);
    printf("\tTotal DIA occupy ratio: %.1lf %% \n", occupy_ratio*100);

    // TEST FORMAT
    if ( kernel_tag == 1)
    {
      test_spmv_kernel(csr, spmv_csr_serial_host_simple<IndexType,ValueType>, 
                       dia, spmv_dia_serial_host_simple<IndexType,ValueType>, 
                       "dia_serial_simple");

      benchmark_spmv_on_host(dia, spmv_dia_serial_host_simple<IndexType, ValueType>,   "dia_serial_simple" );
    }
#if 0
    else if ( kernel_tag == 2 )
    {
      test_spmv_kernel(csr, spmv_csr_serial_host_simple<IndexType,ValueType>, 
                       dia, spmv_dia_serial_host_sse<IndexType,ValueType>, 
                       "dia_serial_sse");

      benchmark_spmv_on_host(dia, spmv_dia_serial_host_sse<IndexType, ValueType>,   "dia_serial_sse" );
    }
#endif

    *gflops = dia.gflops;
    delete_host_matrix(dia);
    return 0;
}

template <typename IndexType, typename ValueType>
int test_ell_matrix_kernels(const csr_matrix<IndexType,ValueType>& csr, int kernel_tag, double *gflops, FILE *fp_feature)  
{
    printf("\n####  Testing ELL Kernels  ####\n");
    double max_fill = 20;
    IndexType max_cols_per_row = static_cast<IndexType>( (max_fill * csr.num_nonzeros) / csr.num_rows + 1 );

    // CREATE ELL MATRIX
    printf("\tcreating ell_matrix:");
    ell_matrix<IndexType,ValueType> ell = csr_to_ell<IndexType,ValueType>(csr, max_cols_per_row, fp_feature);
    printf("\n");
    if (ell.num_nonzeros == 0 && csr.num_nonzeros != 0)
    {      
	    printf("\tmax_RD (%d) excedes limit (%d)\n", ell.max_RD, max_cols_per_row);
	    return 0;
    }

    printf("\tELL has %d columns per row\n", ell.max_RD);
    double nzs_ratio = (double)csr.num_nonzeros/(ell.max_RD *ell.num_rows);
    printf("\tNonzeros occupy ratio: %.1lf %%\n", nzs_ratio*100);
    printf("\tMin nzs per row: %d, Max nzs per row: %d\n", ell.min_RD, ell.max_RD); 

    // TEST FORMAT
    if ( kernel_tag == 1)
    {
      test_spmv_kernel(csr, spmv_csr_serial_host_simple<IndexType,ValueType>, 
                       ell, spmv_ell_serial_host_simple<IndexType,ValueType>, 
                       "ell_serial_simple");

      benchmark_spmv_on_host(ell, spmv_ell_serial_host_simple<IndexType, ValueType>,     "ell_serial_simple" );
    }
#if 0
    else if ( kernel_tag == 2)
    {
      test_spmv_kernel(csr, spmv_csr_serial_host_simple<IndexType,ValueType>, 
                       ell, spmv_ell_serial_host_sse<IndexType,ValueType>, 
                       "ell_serial_sse");

      benchmark_spmv_on_host(ell, spmv_ell_serial_host_sse<IndexType, ValueType>,     "ell_serial_sse" );
    }
#endif

    *gflops = ell.gflops;
    delete_host_matrix(ell);
    return 0;
}

template <typename IndexType, typename ValueType>
int test_csr_matrix_kernels(csr_matrix<IndexType,ValueType>& csr, int kernel_tag, double *gflops, FILE *fp_feature)
{
    printf("\n####  Testing CSR Kernels  ####\n");

    // TEST KERNELS
    if ( kernel_tag == 1)
    {
      test_spmv_kernel(csr, spmv_csr_serial_host_simple<IndexType,ValueType>,   
                       csr, spmv_csr_serial_host_simple<IndexType,ValueType>, 
                       "csr_serial_simple");

      benchmark_spmv_on_host(csr,   spmv_csr_serial_host_simple<IndexType, ValueType>,       "csr_serial_simple" );
    }
#if 0
    else if ( kernel_tag == 2)
    {
      test_spmv_kernel(csr, spmv_csr_serial_host_simple<IndexType,ValueType>,   
                       csr, spmv_csr_serial_host_sse<IndexType,ValueType>, 
                       "csr_serial_sse");

      benchmark_spmv_on_host(csr,   spmv_csr_serial_host_sse<IndexType, ValueType>,       "csr_serial_sse" );
    }
#endif

    *gflops = csr.gflops;
    return 0;
}


template <typename IndexType, typename ValueType>
int test_coo_matrix_kernels(const csr_matrix<IndexType,ValueType>& csr, int kernel_tag, double *gflops, FILE *fp_feature)
{
    printf("\n####  Testing COO Kernels  ####\n");

    // CREATE COO MATRIX
    printf("\tcreating coo_matrix:");
    coo_matrix<IndexType,ValueType> coo = csr_to_coo<IndexType,ValueType>(csr, fp_feature);  
		//TODO change CSR to COO again, but COO is the input format from MM matrices.
    printf("\n");

    // TEST FORMAT
    if ( kernel_tag == 1 )
    {
      test_spmv_kernel(csr, spmv_csr_serial_host_simple<IndexType,ValueType>, 
                       coo, spmv_coo_serial_host_simple<IndexType,ValueType>, 
                       "coo_serial_simple");

      benchmark_spmv_on_host(coo, spmv_coo_serial_host_simple<IndexType, ValueType>,     "coo_serial_simple");
    }
#if 0
    else if ( kernel_tag == 2 )
    {
      test_spmv_kernel(csr, spmv_csr_serial_host_simple<IndexType,ValueType>, 
                       coo, spmv_coo_serial_host_sse<IndexType,ValueType>, 
                       "coo_serial_sse");

      benchmark_spmv_on_host(coo, spmv_coo_serial_host_sse<IndexType, ValueType>,     "coo_serial_sse");
    }
#endif
 
    *gflops = coo.gflops;
    delete_host_matrix(coo);
    return 0;
}

