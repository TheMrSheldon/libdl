<h1 align="center">libdl</h1>
<p align="center">
    <img src="thirdparty/twemoji-brain.svg" height=100pt/>
    <h3 align="center">Simple yet powerful deep learning</h3>
</p>
<p align="center">
    <a><img alt="GPL 2.0 License" src="https://img.shields.io/github/license/TheMrSheldon/libdl.svg"/></a>
    <a><img alt="Current Release" src="https://img.shields.io/github/release/TheMrSheldon/libdl.svg"/></a>
    <br>
    <a href="https://themrsheldon.github.io/libdl">Documentation</a> &nbsp;|&nbsp;
    <a href="https://themrsheldon.github.io/libdl/namespaces.html">API</a> &nbsp;|&nbsp;
    <a href="https://github.com/TheMrSheldon/libdl/tree/main/examples">Examples</a> &nbsp;|&nbsp;
    <a href="#citation">Citation</a>
</p>

# Usage
```cpp
int main(int argc, char* argv[]) {
	MyModel model;

	auto conf = dl::TrainerConfBuilder<MyModel>()
                .setDataset<MyDataset>()
                .setOptimizer<dl::optim::GradientDescent>(model.parameters())
                .addObserver(dl::observers::limitEpochs(10))
                .addObserver(dl::observers::earlyStopping(3))
                .addObserver(dl::observers::consoleUI())
                .build();
	auto trainer = dl::Trainer(std::move(conf));
	trainer.fit(model, dl::lossAdapter(dl::loss::mse));
	trainer.test(model);
	return 0;
}
```

# Installation
## CMake
```cmake
FetchContent_Declare(libdl GIT_REPOSITORY https://github.com/TheMrSheldon/libdl.git)
FetchContent_MakeAvailable(libdl)
target_link_libraries(<mytarget> PUBLIC libdl)
```