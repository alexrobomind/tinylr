#include <tinylr/tinylr.hpp>
#include <tinylr/test.hpp>

#include <iostream>

int main(int argc, char** argv) {
	auto mat = tinylr::make_matrix<
		double,
		tinylr::pivot::none
	>(
		tinylr::dim::fixed<4>()
		//tinylr::dim::dynamic(4)
	);
	
	mat.at(0, 0) = 0;
	mat.at(1, 0) = 1;
	//mat.at(2, 0) = 0;
	
	mat.at(0, 1) = 1;
	mat.at(1, 1) = 1;
	//mat.at(2, 1) = 0;
	
	//mat.at(0, 2) = 0;
	//mat.at(1, 2) = 0;
	//mat.at(2, 2) = -1;
	
	tinylr::print(mat);
	
	mat.lr_inplace();
	tinylr::print(mat);
	
	tinylr::print(tinylr::expand_lr(mat));
	
	std::cout << sizeof(mat) << std::endl;
	std::cout << sizeof(mat.data) << std::endl;
	std::cout << sizeof(mat.dimm) << std::endl;
	std::cout << sizeof(mat.pivots) << std::endl;
}