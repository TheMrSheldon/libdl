#pragma once

#include <iterator>
#include <memory>

namespace dl::utils {

	/**
	 * @brief Represents a generic iterator that can be initialized to hold any other class instance that satisfies the
	 * std::input_or_output_iterator concept.
	 * 
	 * @details Represents a generic iterator that can be initialized to hold any other class instance that satisfies
	 * the std::input_or_output_iterator concept. For example consider the following code:
	 * ```cpp
	 * std::vector<int> myvec = {1, 2, 3, 4};
	 * GenericIterator<int> begin = myvec.begin();
	 * GenericIterator<int> end = myvec.end();
	 * ```
	 * `begin` and `end` again are valid iterators but they are not specific to the container they iterate through.
	 * 
	 * @tparam T the type of the elements being iterated.
	 */
	template <typename T>
	class GenericIterator {
	private:
		/**
		 * @brief The "abstract iterator" (AIter) provides a class interface for forward iterators (i.e. increment,
		 * dereference and equality operations) together with the option to make a copy and get a unique typeid to check
		 * if two iterators are of the same type.
		 * 
		 * @see mogli::utils::Iter
		 */
		class AIter {
		public:
			/** @brief Virtual default destructor. **/
			virtual ~AIter() noexcept = default;
			/**
			 * @brief "Dereferences" the iterator, i.e., returns the value the iterator is currently pointing to.
			 * 
			 * @return T the value the iterator is currently pointing to.
			 */
			virtual T deref() const noexcept = 0;
			/**
			 * @brief Advances the iterator to the next element (if it exists). If this iterator is the end-iterator,
			 * the behavior is undefined.
			 */
			virtual void advance() noexcept = 0;
			/**
			 * @brief Checks if this operator is equal to `other`. Generally this should check if they have the same
			 * type (see AIter::type()) and then check if the underlying iterators are the same.
			 * 
			 * @param other The iterator to check equality against.
			 * @return true iff this iterator is equal to the other one. 
			 */
			virtual bool equals(const AIter& other) const noexcept = 0;
			/**
			 * @brief Returns the typeid of this. This can be used to check if to implementations of AIter are of the
			 * same type.
			 * @details Returns the typeid of this. This can be used to check if to implementations of AIter are of the
			 * same type.
			 * ```cpp
			 * class A : AIter { ... };
			 * class B : AIter { ... };
			 * A a1, a2;
			 * B b;
			 * static_assert(a1.type() == a2.type());
			 * static_assert(a1.type() != b.type());
			 * ```
			 * 
			 * @return The typeinfo representing the iterator implementation's type.
			 */
			virtual const std::type_info& type() const = 0;
			/**
			 * @brief Clones this iterator. A shallow clone suffices. This only acts as a virtual copy constructor
			 * (https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Virtual_Constructor).
			 * 
			 * @return A clone of the current iterator.
			 */
			virtual std::unique_ptr<AIter> clone() const = 0;
		};

		/**
		 * @brief The Iter class wraps objects that suffice the std::input_or_output_iterator concept to implement the
		 * AIter interface.
		 * 
		 * @tparam TIt The datatype of the underlying iterator.
		 */
		template <std::input_or_output_iterator TIt>
		class Iter final : public AIter {
		private:
			/**
			 * @brief The underlying iterator that is used to implement the AIter interface.
			 */
			TIt iterator;
			/**
			 * @brief A shared_ptr to data that should be deleted when the iterator is deleted.
			 * @details This is intended to store the underlying data into that the iterator iterates over. As such,
			 * this allows returning the begin-end-iterator-pair from a function and keeping them point to valid memory
			 * until the last of the iterators is destructed.
			 * ```cpp
			 * auto getData() {
			 *     auto data = std::make_shared<std::vector>({ 1, 2, 3, 4 });
			 *     return std::make_tuple(Iter(data->begin(), data), Iter(data->end(), data));
			 * }
			 * ```
			 */
			std::shared_ptr<void> userdata;

		public:
			/**
			 * @brief Constructs a new Iter instance wrapping the provided iterator.
			 * 
			 * @param iterator The iterator using which to implement the AIter interface.
			 * @see Iter(TIt&, std::shared_ptr<auto>)
			 */
			explicit Iter(TIt& iterator) noexcept : iterator(iterator), userdata(nullptr) {}

			/**
			 * @brief Constructs a new Iter instance wrapping the provided iterator.
			 * 
			 * @param iterator The iterator using which to implement the AIter interface.
			 * @param userdata (Optional) additional userdata holding the data the iterator iterates over.
			 * @see Iter(Iter&)
			 */
			Iter(TIt& iterator, std::shared_ptr<auto> userdata) : iterator(iterator), userdata(userdata) {}

			T deref() const noexcept override { return *iterator; }
			void advance() noexcept override { std::ranges::advance(iterator, 1); }
			bool equals(const AIter& other) const noexcept override {
				return other.type() == type() && static_cast<const Iter<TIt>&>(other).iterator == iterator;
			}
			const std::type_info& type() const override { return typeid(this); }
			std::unique_ptr<AIter> clone() const override { return std::make_unique<Iter<TIt>>(*this); }
		};

		/**
		 * @brief The AIter implementation used to implement the generic iterator.
		 */
		std::unique_ptr<AIter> impl;

	public:
		using iterator_category = std::input_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = T;
		using pointer = T*;
		using reference = T&;

		/**
		 * @brief Default constructor that creates a nullptr-iterator.
		 */
		GenericIterator() : impl(nullptr) {}

		/**
		 * @brief Copy-constructs a new generic iterator.
		 * 
		 * @param other The iterator to copy.
		 */
		GenericIterator(const GenericIterator<T>& other) : impl(other.impl->clone()) {}

		/**
		 * @brief Move-constructs a new generic iterator.
		 * 
		 * @param other The iterator to move.
		 */
		GenericIterator(GenericIterator<T>&& other) : impl(std::move(other.impl)) {}

		/**
		 * @brief Constructs a new GenericIterator from the provided arbitrary object that satisfies the
		 * std::input_or_output_iterator concept.
		 * 
		 * @tparam TIt The datatype of the underlying iterator.
		 * @param iterator The underlying iterator.
		 * @see GenericIterator(TIt, std::shared_ptr<auto>)
		 */
		template <std::input_or_output_iterator TIt>
		explicit GenericIterator(TIt iterator) noexcept : impl(std::make_unique<Iter<TIt>>(iterator)) {}

		/**
		 * @brief Constructs a new GenericIterator from the provided arbitrary object that satisfies the
		 * std::input_or_output_iterator concept. The additional userdata parameter can keep the underlying data that
		 * the iterator iterates over such that it is not deleted as long as at least one iterator is still alive.
		 * 
		 * @tparam TIt The datatype of the underlying iterator.
		 * @param iterator The underlying iterator.
		 * @param userdata (Optional) additional userdata holding the data the iterator iterates over.
		 * @see GenericIterator(TIt)
		 */
		template <std::input_or_output_iterator TIt>
		GenericIterator(TIt iterator, std::shared_ptr<auto> userdata) noexcept
				: impl(std::make_unique<Iter<TIt>>(iterator, userdata)) {}

		/**
		 * @brief Assignment opterator.
		 * 
		 * @param other 
		 * @return A reference to this.
		 */
		GenericIterator<T>& operator=(const GenericIterator<T>& other) noexcept {
			impl = other.impl;
			return *this;
		}

		/**
		 * @brief Move-assignment operator.
		 * 
		 * @param other 
		 * @return A reference to this.
		 */
		GenericIterator<T>& operator=(GenericIterator<T>&& other) noexcept {
			impl = std::move(other.impl);
			return *this;
		}

		/**
		 * @brief Dereferences the iterator. That is, returns the data it currently points at.
		 * 
		 * @return The data the iterator currently points at.
		 */
		T operator*() const { return impl->deref(); }
		/**
		 * @brief Increments the iterator. That is, advancing it forward by one.
		 * 
		 * @return A reference to this iterator.
		 */
		GenericIterator<T>& operator++() noexcept {
			impl->advance();
			return *this;
		}
		/**
		 * @brief Late increments the iterator. That is, advances the iterator by one but returns a copy of the iterator
		 * prior to advancing.
		 * 
		 * @return The iterator before it got advanced.
		 */
		GenericIterator<T> operator++(int) noexcept {
			auto copy = *this;
			(*this)++;
			return copy;
		}
		/**
		 * @brief Checks if this iterator is equal to the provided one.
		 * 
		 * @param other The iterator to check equality against.
		 * @return true if and only if this iterator is equal to `other`.
		 */
		bool operator==(const GenericIterator<T>& other) const { return impl->equals(*other.impl); }
	};

	static_assert(std::input_iterator<GenericIterator<int>>);
} // namespace dl::utils