#ifndef TINYLR_TINYLR_HPP
#define TINYLR_TINYLR_HPP

#include <array>
#include <vector>
#include <cmath>

#include <iostream>

namespace tinylr {
	namespace pivot {
		enum strategy {
			none = 0,
			absmax = 1,
			absmax_swap = 2
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
			static_assert(sizeof(T) == 0, "Unknown pivot strategy");
		};
		
		template<typename T, typename Dim>
		struct PivotEngine<T, Dim, pivot::absmax> {
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
				T num = std::abs(mat.at(col, col));
				size_t choice = col;
				
				for(size_t i = col + 1; i < store.size(); ++i) {
					T num2 = mat.at(i, col);
					
					if(std::abs(num2) > num) {
						num = num2;
						choice = i;
					}
				}
				
				std::swap(store[col], store[choice]);
			}
			
			/** Reads the stored pivot permutation */
			size_t get(size_t n) const { return store[n]; }
			size_t get_at(size_t n) const { return store[n]; }
		};
		
		/** Pivot engine variant that implements the same pivoting strategy as above,
		 *  but does not install an indirection when accessing the data buffer. Instead
		 *  swaps the rows in the data buffer */
		template<typename T, typename Dim>
		struct PivotEngine<T, Dim, pivot::absmax_swap> {
			typename Dim::template Vector<size_t> store;
			
			PivotEngine(const Dim& d) :
				store(d.template create_vector<size_t>())
			{
				for(size_t i = 0; i < d.dim(); ++i)
					store[i] = i;
			}
			template<typename Mat>
			void pivot(size_t col, Mat& mat) {				
				T num = std::abs(mat.at(col, col));
				size_t choice = col;
				
				for(size_t i = col + 1; i < store.size(); ++i) {
					T num2 = mat.at(i, col);
					
					if(std::abs(num2) > num) {
						num = num2;
						choice = i;
					}
				}
				
				std::swap(store[col], store[choice]);
				
				for(size_t j = 0; j < store.size(); ++j) {
					std::swap(mat.at_raw(col, j), mat.at_raw(choice, j));
				}
			}
			
			/** Reads the stored pivot permutation */
			size_t get(size_t n) const { return store[n]; }
			size_t get_at(size_t n) const { return n; }
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
			size_t get_at(size_t n) const { return n; }
		};
	}
	
	namespace dim {
		template<size_t dim>
		auto fixed() { return internal::StaticDimension<dim>(); }
		
		auto dynamic(size_t dim) { return internal::DynamicDimension(dim); }
	}
	
	template<typename Num, typename Dim, pivot::strategy pstrat, bool inv_diag>
	struct Matrix {
		using Number = Num;
		using Storage = typename Dim::template Matrix<Num>;
		using Vector = typename Dim::template Vector<Num>;
		
		static constexpr bool invert_diagonal = inv_diag;
		
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
		
		/** Access to the data view provided by the pivot engine */
		Num& at(size_t i, size_t j) {
			return at_raw(pivots.get_at(i), j);
		}
		
		/** Raw data buffer access. Safe to use before calling lr_inplace. DO
		 *  NOT USE AFTERWARDS. The storage layout is pivot engine dependent.
		 *  Use at(...) instead.
		 */
		Num& at_raw(size_t i, size_t j) {
			return data[dimm.dim() * i + j];
		}
		
		const Num det() {
			Num result = 1;
			for(size_t i = 0; i < dimm.dim(); ++i)
				result *= at(i, i);
			return invert_diagonal ? 1.0 / result : result;
		}
		
		const Num inv_det() {
			Num result = 1;
			for(size_t i = 0; i < dimm.dim(); ++i)
				result *= at(i, i);
			return invert_diagonal ? result : 1.0 / result;
		}
		
		const Num& at(size_t i, size_t j) const {
			return at_raw(pivots.get_at(i), j);
		}
		
		const Num& at_raw(size_t i, size_t j) const {
			return data[dimm.dim() * i + j];
		}
		
		void lr_inplace() {
			for(size_t i = 0; i < dimm.dim(); ++i)
				process_step(i);
		}
		
		template<typename Tin, typename Tout>
		void vmult_inv(const Tin& in, Tout& out) const {
			auto temp = dimm.template create_vector<Num>();
			for(size_t i = 0; i < dimm.dim(); ++i)
				temp[i] = in[pivots.get(i)];
			
			forward_substitution(temp);
			backward_substitution(temp);
			
			for(size_t i = 0; i < dimm.dim(); ++i)
				out[i] = temp[i];
		}
		
		template<typename Tin, typename Tout>
		void vmult(const Tin& in, Tout& out) const {
			auto temp = dimm.template create_vector<Num>();
			
			// R matrix
			for(size_t i = 0; i < dimm.dim(); ++i) {
				Num buf = in[i];
				for(size_t j = i + 1; j < dimm.dim(); ++j)
					buf += at(i, j) * in[j];
				temp[i] = buf;
			}
			
			// L matrix
			for(size_t i = 0; i < dimm.dim(); ++i) {
				Num buf = invert_diagonal ? temp[i] / at(i, i) : temp[i] * at(i, i);
				for(size_t j = 0; j < i; ++j) {
					buf += at(i, j) * temp[i];					
				}
				out[pivots.get(i)] = buf;
			}
		}
		
		
	private:
		void process_step(size_t step) {
			pivots.pivot(step, *this);
			
			// Normalize the row of the U matrix so that diagonal is 1
			{
				const Num lead = at(step, step);
				const Num invlead = 1 / lead;
				
				for(size_t col = step + 1; col < dimm.dim(); ++col)
					at(step, col) *= invlead;
				
				// Store the inverse diagonal element of L matrix in diagonal of store
				//  (We don't need the diagonal itself for forward substitution)
				if(invert_diagonal)
					at(step, step) = invlead;
			}
			
			// Subtract for all lower rows multiple of this row so that their column element
			// goes to 0. This row is already normalized, so no division neccessary.
			for(size_t row = step + 1; row < dimm.dim(); ++row) {
				const Num lead = at(row, step);
				
				for(size_t col = step + 1; col < dimm.dim(); ++col) {
					at(row, col) -= lead * at(step, col);
				}
			}
		}
		
		/** Forward substitution along L matrix */
		void forward_substitution(typename Dim::template Vector<Num>& temp) const {
			for(size_t i = 0; i < dimm.dim(); ++i) {
				// Divide by diagonal of L matrix
				// Since we store the the inverse diagonal, we have
				// to multiply by the stored value.
				if(invert_diagonal)
					temp[i] *= at(i, i);
				else
					temp[i] /= at(i, i);
				
				for(size_t j = i + 1; j < dimm.dim(); ++j) {
					temp[j] -= temp[i] * at(j, i);
				}
			}
		}
		
		/** Forward substitution along U matrix */
		void backward_substitution(typename Dim::template Vector<Num>& temp) const {
			// U matrix has diagonal 1, so no scaling required
			
			for(size_t revi = 0; revi < dimm.dim(); ++revi) {
				const size_t i = dimm.dim() - revi - 1;
				
				for(size_t j = 0; j < i; ++j) {
					temp[j] -= temp[i] * at(j, i);
				}
			}
		}
	};
	
	template<typename Num, pivot::strategy pstrat = pivot::absmax, bool invert_diagonal = true, typename Dim>
	Matrix<Num, Dim, pstrat, invert_diagonal> make_matrix(const Dim& d) {
		return Matrix<Num, Dim, pstrat, invert_diagonal>(d);
	}
}

#endif // TINYLR_TINYLR_HPP