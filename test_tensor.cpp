#include <iostream>
#include <stdio.h>
#include "cmdline.h"
#include "gallery.h"
#include "tests.h"
#include "config.h"
#include "timer.h"

using namespace std;

int main()
{

    sptensor<int,int> coo;

    coo = read_coo_tensor<int,int>("examples/tensor_1.mtx");

	cout<<coo.num_dims<<endl;
	cout<<coo.num_nonzeros<<endl;
	for (int i=0;i<coo.num_dims;i++)
		cout<<coo.dims[i]<<" ";
	cout<<endl;
	for (int i=0;i<coo.num_nonzeros;i++)
	{
		cout<<coo.vals[i]<<" ";
	}
	cout<<endl;
	sptenmat <int,int> coo_mat;

	static const int r[] = {0,1};
	vector<int> row (r, r + sizeof(r) / sizeof(r[0]) );	

	static const int c[] = {2,3};
	vector<int> col (c, c + sizeof(c) / sizeof(c[0]) );	

	coo_mat=sptensor_to_sptenmat(coo, row, col);
	
	cout<<endl<<":::SPTENMAT:::: "<<std::endl;
	
	for(int i = 0; i < coo_mat.num_nonzeros;i++)
		cout<<coo_mat.subs[i][0]<<" "<<coo_mat.subs[i][1]<<" "<<coo_mat.vals[i]<<std::endl;

	csr_matrix<int,int> cs;
	cs=sptenmat_to_csr(coo_mat);

	cout<<endl<<"::::CSR:::: "<<endl;

	cout<<"Values: ";
	for(int i=0;i<cs.num_nonzeros;i++)
		cout<<cs.Ax[i]<<" ";
	cout<<std::endl;

	cout<<"Column Indices: ";
	for(int i=0;i<cs.num_nonzeros;i++)
		cout<<cs.Aj[i]<<" ";	
	cout<<std::endl;

	cout<<"Row Pointers: ";
	for (int i=0;i<cs.num_rows+1;i++)
		cout<<cs.Ap[i]<<" ";
	cout<<std::endl;
		
}
