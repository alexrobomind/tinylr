#include <tinylr/tinylr.hpp>
#include <tinylr/test.hpp>

#include <iostream>

int main(int argc, char** argv) {
	auto mat = tinylr::make_matrix<
		double,
		tinylr::pivot::absmax,
		true
	>(
		tinylr::dim::fixed<3>()
		//tinylr::dim::dynamic(4)
	);
	
	std::cout << sizeof(mat) << std::endl;
	std::cout << sizeof(mat.data) << std::endl;
	std::cout << sizeof(mat.dimm) << std::endl;
	std::cout << sizeof(mat.pivots) << std::endl;
	
	mat.at(0, 0) = 0;
	mat.at(1, 0) = 1;
	mat.at(2, 0) = 0;
	
	mat.at(0, 1) = 1;
	mat.at(1, 1) = 1;
	mat.at(2, 1) = 0;
	
	mat.at(0, 2) = 0;
	mat.at(1, 2) = 5;
	mat.at(2, 2) = -1;
	
	tinylr::print(mat);
	
	mat.lr_inplace();
	tinylr::print(mat);
	
	tinylr::print(tinylr::expand_lr(mat));
	
	std::vector<double> left(3);
	std::vector<double> right(3);
	left[0] = 1;
	left[1] = 2;
	left[2] = 16;
	
	std::cout << "Input:" << std::endl;
	for(auto e : left)
		std::cout << "  " << e;
	std::cout << std::endl;
	
	mat.vmult(left, right);
	std::cout << "Output:" << std::endl;
	for(auto e : right)
		std::cout << "  " << e;
	std::cout << std::endl;
	
	mat.vmult_inv(right, left);
	std::cout << "Inverse:" << std::endl;
	for(auto e : left)
		std::cout << "  " << e;
	std::cout << std::endl;
}