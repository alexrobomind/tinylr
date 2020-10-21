#include <tinylr/tinylr.hpp>
#include <tinylr/test.hpp>

#include <iostream>
#include <random>
#include <chrono>
#include <ctime>

template<typename Mat, typename RandEngine>
void randmat(Mat& m, RandEngine& rando) {
	std::normal_distribution<typename Mat::Number> dist;
	
	for(size_t i = 0; i < m.dim(); ++i)
		for(size_t j = 0; j < m.dim(); ++j)
			m.at(i, j) = dist(rando);
}

template<typename V, typename RandEngine>
void randvec(V& v, RandEngine& rando) {
	std::normal_distribution<double> dist;
	
	for(size_t i = 0; i < v.size(); ++i)
		v[i] = dist(rando);
}

#ifdef STATIC_DIM
#define DIMENSIONALITY fixed<STATIC_DIM>()
#else
#define DIMENSIONALITY dynamic(DYNAMIC_DIM)
#endif

int main(int argc, char** argv) {
	
	// Use Mersenne-Twister with predefined seed
	using Rand = std::mt19937;
	Rand rando(123);
	
	//using Dur = std::chrono::high_resolution_clock::duration;
	
	//Dur lr_time(0);
	//Dur matmul_time(0);
	//Dur invmul_time(0);
	std::clock_t lr_time = 0;
	std::clock_t matmul_time = 0;
	std::clock_t invmul_time = 0;
	
	for(size_t i = 0; i < LR_ITERATIONS; ++i) {
		volatile double dump = 0;
		
		// Setup matrix
		auto mat = tinylr::make_matrix<
			double,
			tinylr::pivot::PIVOT_STRATEGY,
			DIAG_INVERT
		>(
			tinylr::dim::DIMENSIONALITY
		);
		
		randmat(mat, rando);
		
		{
			//auto t1 = std::chrono::high_resolution_clock::now();
			auto t1 = std::clock();
			mat.lr_inplace();
			//auto t2 = std::chrono::high_resolution_clock::now();
			auto t2 = std::clock();
			
			lr_time += (t2 - t1);
		}
		
		auto v = mat.dimm.create_vector<double>();
		randvec(v, rando);
		
		auto v2 = mat.dimm.create_vector<double>();
		{
			
			auto t1 = std::clock();
			//auto t1 = std::chrono::high_resolution_clock::now();
			//mat.vmult(v, v2);
			auto t2 = std::clock();
			//auto t2 = std::chrono::high_resolution_clock::now();
			
			matmul_time += (t2 - t1);
		}
		
		auto v3 = mat.dimm.create_vector<double>();
		{
			auto t1 = std::clock();
			//auto t1 = std::chrono::high_resolution_clock::now();
			//mat.vmult_inv(v2, v3);
			auto t2 = std::clock();
			//auto t2 = std::chrono::high_resolution_clock::now();
			
			invmul_time += (t2 - t1);
		}
		
		// Prevent compiler from doing too funny optimizations on the loop
		for(size_t i = 0; i < mat.dim(); ++i) {
			dump += v[i];
			dump += v2[i];
			dump += v3[i];
			
			for(size_t j = 0; j < mat.dim(); ++j)
				dump += mat.at(i, j);
		}
	}
	
	/*auto print_time = [](Dur d) {
		auto ms = std::chrono::duration_cast<std::chrono::microseconds>(d).count();
		std::cout << ((double) ms / LR_ITERATIONS) << std::endl;
	};*/
	auto print_time = [](std::clock_t t) {
		double t2 = t;
		std::cout << t2 / CLOCKS_PER_SEC * 1e9 / LR_ITERATIONS << std::endl;
	};
	
	print_time(lr_time);
	print_time(matmul_time);
	print_time(invmul_time);
	
	return 0;
}