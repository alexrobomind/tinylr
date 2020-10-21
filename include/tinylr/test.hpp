#ifndef TINYLR_TEST_HPP
#define TINYLR_TEST_HPP

#include <iostream>

namespace tinylr {
	template<typename Mat>
	void print(const Mat& mat) {	
		std::cout << "Pivots:";
		
		for(size_t i = 0; i < mat.dim(); ++i)
			std::cout << "  " << mat.pivot(i);
		std::cout << std::endl;
		
		std::cout << "LR" << std::endl;
		for(size_t i = 0; i < mat.dim(); ++i) {
			for(size_t j = 0; j < mat.dim(); ++j) {
				std::cout << "  " << mat.at(i, j);
			}
			std::cout << std::endl;
		}
	}
	
	template<typename Mat>
	Mat expand_lr(const Mat& mat) {
		Mat out(mat.dimm);
		
		for(size_t i = 0; i < mat.dim(); ++i) {
			for(size_t j = 0; j < mat.dim(); ++j) {
				using Num = typename Mat::Number;
				Num accum = 0;
				
				for(size_t k = 0; k <= i && k <= j; ++k) {
					const Num l_element = (k == i && Mat::invert_diagonal) ? 1.0 / mat.at(k, k) : mat.at(i, k);
					const Num r_element = k == j ? 1.0 : mat.at(k, j);
					
					accum += l_element * r_element;
				}
				
				out.at(mat.pivots.get(i), j) = accum;
			}
		}
		
		return out;
	}
}

#endif // TINYLR_TEST_HPP