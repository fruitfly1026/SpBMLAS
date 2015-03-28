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

    coo = read_coo_tensor<int,int>("examples/tensor_2.mtx");

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
	
	cout<<"SPTENMAT "<<std::endl;
	
	for(int i = 0; i < coo_mat.num_nonzeros;i++)
		cout<<coo_mat.subs[i][0]<<" "<<coo_mat.subs[i][1]<<" "<<coo_mat.vals[i]<<std::endl;
	
		
}
