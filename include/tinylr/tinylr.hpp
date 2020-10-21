#ifndef TINYLR_TINYLR_HPP
#define TINYLR_TINYLR_HPP

#include <array>
#include <vector>

namespace tinylr {
	namespace pivot {
		enum strategy {
			none = 0,
			absmax = 1
		};
	}
	
	namespace internal {
		/** This class is responsible for defining and initializing
		 *  dimension-dependent data structures. This allows very fast
		 *  data structures if the dimensionality is known at compile-time
		 */
		template<size_t d>
		struct StaticDimension {
			constexpr size_t dim() const { return d; } //< Returns the no. of space dimensions
			
			template<typename T>
			using Matrix = std::array<T, d * d>; //< Type for matrix data
			
			template<typename T>
			using Vector = std::array<T, d>; //< Type for vector data
			
			template<typename T>
			Matrix<T> create_matrix() const { return Matrix<T>(); }
			
			template<typename T>
			Vector<T> create_vector() const { return Vector<T>(); }
		};
		
		struct DynamicDimension {
			size_t d;
			DynamicDimension(size_t d) : d(d) {}
			
			size_t dim() const { return d; }
			
			template<typename T>
			using Matrix = std::vector<T>;
			
			template<typename T>
			using Vector = std::vector<T>;
			
			template<typename T>
			Matrix<T> create_matrix() const { return Matrix<T>(dim() * dim()); }
			
			template<typename T>
			Vector<T> create_vector() const { return Vector<T>(dim()); }
		};
		
		/** This class manages the pivoting permutation. This allows enable
		 *  and disable pivoting at compile time
		 */
		template<typename T, typename Dim, pivot::strategy strat>
		struct PivotEngine {
			typename Dim::template Vector<size_t> store;
			
			PivotEngine(const Dim& d) :
				store(d.template create_vector<size_t>())
			{
				for(size_t i = 0; i < d.dim(); ++i)
					store[i] = i;
			}
			
			/** Update the pivot information for column 'col'. This is
			 *  done by looking at all the lower rows with idx >= col, and checking
			 *  which one has the highest absolute value. Then, this row is permuted
			 *  into the 'col'th row
			 */
			template<typename Mat>
			void pivot(size_t col, Mat& mat) {				
				T num = std::abs(mat.pat(col, col));
				size_t choice = col;
				
				for(size_t i = col + 1; i < store.size(); ++i) {
					T num2 = mat.pat(i, col);
					
					if(std::abs(num2) > num) {
						num = num2;
						choice = i;
					}
				}
				
				std::swap(store[col], store[choice]);
			}
			
			/** Reads the stored pivot permutation */
			size_t get(size_t n) const { return store[n]; }
		};
		
		/** Implementation of pivot engine that performs
		 *  no pivoting.
		 */
		template<typename T, typename Dim>
		struct PivotEngine<T, Dim, pivot::none> {
			PivotEngine(const Dim& d) {}
			
			template<typename Mat>
			void pivot(size_t row, Mat)
			{}
			
			size_t get(size_t n) const { return n; }
		};
	}
	
	namespace dim {
		template<size_t dim>
		auto fixed() { return internal::StaticDimension<dim>(); }
		
		auto dynamic(size_t dim) { return internal::DynamicDimension(dim); }
	}
	
	template<typename Num, typename Dim, pivot::strategy pstrat>
	struct Matrix {
		using Number = Num;
		using Storage = typename Dim::template Matrix<Num>;
		
		Storage data;
		
		Dim dimm;
		internal::PivotEngine<Num, Dim, pstrat> pivots;
		
		Matrix(const Dim& d) :
			data(d.template create_matrix<Num>()),
			dimm(d),
			pivots(d)
		{}
		
		size_t dim() const { return dimm.dim(); }
		size_t pivot(size_t i) const { return pivots.get(i); }
		
		Num& pat(size_t i, size_t j) {
			return at(pivots.get(i), j);
		}
		
		Num& at(size_t i, size_t j) {
			return data[dimm.dim() * i + j];
		}
		
		const Num& pat(size_t i, size_t j) const {
			return at(pivots.get(i), j);
		}
		
		const Num& at(size_t i, size_t j) const {
			return data[dimm.dim() * i + j];
		}
		
		void lr_inplace() {
			for(size_t i = 0; i < dimm.dim(); ++i)
				process_step<true>(i);
		}
		
		void lr_inplace_noinvert() {
			for(size_t i = 0; i < dimm.dim(); ++i)
				process_step<false>(i);
		}
		
		template<typename Tin, typename Tout>
		void substitution(const Tin& in, Tout& out) const {
			auto temp = dimm.template create_linear<Num>();
			for(size_t i = 0; i < dimm.dim(); ++i)
				temp[i] = in[pivots.get(i)];
			
			forward_substitution(temp);
			backward_substitution(temp);
			
			for(size_t i = 0; i < dimm.dim(); ++i)
				out[i] = temp[i];
		}
		
	private:
		template<bool invert>
		void process_step(size_t step) {
			pivots.pivot(step, *this);
			
			// Normalize the row of the U matrix so that diagonal is 1
			{
				const Num lead = pat(step, step);
				const Num invlead = 1 / lead;
				
				for(size_t col = step + 1; col < dimm.dim(); ++col)
					pat(step, col) *= invlead;
				
				// Store the inverse diagonal element of L matrix in diagonal of store
				//  (We don't need the diagonal itself for forward substitution)
				if(invert)
					pat(step, step) = invlead;
			}
			
			// Subtract for all lower rows multiple of this row so that their column element
			// goes to 0. This row is already normalized, so no division neccessary.
			for(size_t row = step + 1; row < dimm.dim(); ++row) {
				const Num lead = pat(row, step);
				
				for(size_t col = step + 1; col < dimm.dim(); ++col) {
					pat(row, col) -= lead * pat(step, col);
				}
			}
		}
		
		/** Forward substitution along L matrix */
		void forward_substitution(typename Dim::template Vector<Num>& temp) {
			for(size_t i = 0; i < dimm.dim(); ++i) {
				// Divide by diagonal of L matrix
				// Since we store the the inverse diagonal, we have
				// to multiply by the stored value.
				temp[i] *= pat(i, i);
				
				for(size_t j = i + 1; j < dimm.dim(); ++j) {
					temp[j] -= temp[i] * pat(j, i);
				}
			}
		}
		
		/** Forward substitution along U matrix */
		void backward_substitution(typename Dim::template Vector<Num>& temp) {
			// U matrix has diagonal 1, so no scaling required
			
			for(size_t i = 0; i < dimm.dim(); ++i) {
				const size_t revi = dimm.dim() - i;
				
				for(size_t j = 0; j < revi; ++j) {
					temp[j] -= temp[i] * pat(j, i);
				}
			}
		}
	};
	
	template<typename Num, pivot::strategy pstrat = pivot::absmax, typename Dim>
	Matrix<Num, Dim, pstrat> make_matrix(const Dim& d) {
		return Matrix<Num, Dim, pstrat>(d);
	}
}

#endif // TINYLR_TINYLR_HPP