\page basicIntro Getting Started
\tableofcontents

\attention This introduction guide is aimed at people with no prior experience in C++ or CMake.

# Setting up the Project
\todo WIP

# A Simple Training Example
\todo Example training, validating and evaluating on MNIST Digits
Create a new file called `main.cpp` in your `src` directory with the following contents:
```{cpp}
// todo
```

# Common Pitfalls for Newcommers to C++
\note Please do not let these pitfalls make you think that C++ may be too daunting... after all everything is daunting when one first starts out. Instead, let it be a lesson to use linters to look over your shoulder and teach your -- as should be done in any language.

1. If a class should be inherited from, it **must** declare a public virtual destructor:
    ```{cpp}
    #include <iostream>

    class WrongBase {};
    class RightBase {
    public:
        virtual ~RightBase() = default;
    };
    class WrongDerived : public WrongBase {
    public:
        ~WrongDerived() {
            std::cout << "WrongDerived Destroyed" << std::endl;
        }
    };
    class RightDerived : public RightBase {
    public:
        virtual ~RightDerived() {
            std::cout << "RightDerived Destroyed" << std::endl;
        }
    };

    int main(int argc, char* argv[]) {
        WrongBase* ptrA = new WrongDerived;
        RightBase* ptrB = new RightDerived;
        delete ptrA;
        delete ptrB;
    }
    ```
    will only print out `RightDerived Destroyed` and **not** `WrongDerived Destroyed`.
    \note This problem is not exclusive to raw pointers and can also occur if you don't use them. In this case, raw pointers are the simplest form of showing it crash and burn.
2. Passing child-types by value. Consider the following example:
    ```{cpp}
    #include <iostream>

    class Base {
    public:
        virtual ~Base() = default;
        virtual void doSomething() {
            std::cout << "Base" << std::endl;
        }
    };
    class Derived : public Base {
    public:
        virtual void doSomething() override {
            std::cout << "Derived" << std::endl;
        }
    };

    void wrong(Base base) { // Wrong; Arguments are passed *by value*!
        base.doSomething();
    }
    void right(Base& base) { // Correct; Arguments are passed *by reference* :)
        base.doSomething();
    }

    int main(int argc, char* argv[]) {
        Derived derived;
        wrong(derived);
        right(derived);
    }
    ```
    If you run this, you will see that it prints out `Base` and then `Derived`, which may be surprising. But consider this: `wrong(derived)` actually creates a **copy** of the object to pass by copying `derived`. As such, the code is equivalent to
    ```{cpp}
    Base base = Base(derived);
    base.doSomething();
    ```
3. Do not forget include guards in your headers
4. [Rule of 3, Rule of 5, Rule of 0](https://en.cppreference.com/w/cpp/language/rule_of_three)