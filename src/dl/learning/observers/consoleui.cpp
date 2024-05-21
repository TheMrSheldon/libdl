#include <dl/learning/trainer.hpp>

#include <dl/model/model.hpp>

#include <ftxui/component/captured_mouse.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <cassert>

/** \todo implement me **/

using dl::ModelBase;
using dl::TrainerObserver;
using dl::TrainStage;

static size_t getRAMUsage_KB();
static size_t getTotalRAM_KB();

class ConsoleUI final : public TrainerObserver {
private:
	ftxui::ScreenInteractive* screen;
	ftxui::Component container;
	std::thread renderThread;

	ConsoleUI(const ConsoleUI& other) = delete;
	ConsoleUI(ConsoleUI&& other) = delete;
	ConsoleUI& operator=(const ConsoleUI& other) = delete;
	ConsoleUI& operator=(ConsoleUI&& other) = delete;

	// Progress Information
	size_t epoch;
	float progress = 0;

	const ModelBase* model;

	static constexpr auto WindowBorderColor = ftxui::Color::GrayDark;

public:
	ConsoleUI() {
		renderThread = std::thread([this] {
			auto tmp = ftxui::ScreenInteractive::Fullscreen();
			screen = &tmp;

			auto canvas = ftxui::Canvas(20, 20);
			canvas.DrawPointLine(0, 0, 6, 5, ftxui::Color::GrayLight);
			canvas.DrawPointLine(6, 5, 12, 5, ftxui::Color::GrayLight);
			canvas.DrawPointLine(12, 5, 18, 10, ftxui::Color::GrayLight);
			canvas.DrawPointLine(18, 10, 20, 12, ftxui::Color::GrayLight);

			std::vector<std::string> options{" Loss ", " Console "};
			int choice;
			auto tabtoggle = ftxui::Toggle(&options, &choice);

			auto utilAndLoss = ftxui::Renderer([this, canvas, tabtoggle] {
				std::map<std::string, float> bars = {
						{"CPU: ", .50f},
						{"RAM: ", getRAMUsage_KB() / (float)getTotalRAM_KB()},
						{"GPU: ", .25f},
						{"VRAM:", .1f},
				};
				ftxui::Elements elements;
				for (auto&& [text, value] : bars) {
					elements.push_back(
							ftxui::hbox(
									{ftxui::text(text + " |"),
									 ftxui::gaugeRight(value
									 ) | ftxui::color(ftxui::LinearGradient(ftxui::Color::Green, ftxui::Color::Red)),
									 ftxui::text("|")}
							) |
							ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 30)
					);
				}
				return ftxui::vbox(
						{ftxui::window(
								 ftxui::hbox({ftxui::text(" Utilization ") | ftxui::inverted}),
								 ftxui::flexbox(
										 {elements},
										 ftxui::FlexboxConfig{
												 .direction = ftxui::FlexboxConfig::Direction::Row,
												 .wrap = ftxui::FlexboxConfig::Wrap::Wrap,
												 .justify_content = ftxui::FlexboxConfig::JustifyContent::Stretch,
												 .gap_x = 1
										 }
								 )
						 ) | ftxui::color(WindowBorderColor),
						 ftxui::window(
								 ftxui::hbox({ftxui::text(" Loss ") | ftxui::inverted, ftxui::text(" Console ")}),
								 ftxui::vbox({ftxui::canvas(canvas)})
						 ) | ftxui::color(WindowBorderColor) |
								 ftxui::flex}
				);
			});
			auto info = ftxui::Renderer([this] {
				ftxui::Elements elements;
				std::map<std::string, std::string> tmp{
						{"Seed", "XXXXXX"},
						{"DSet Size (Train / Val. / Test):", "XXXXX / XXX / XXX"},
				};
				if (model) {
					tmp["Model"] = "XXX";
					tmp["Version"] = "vXXX";
					tmp["#Parameters"] = std::format("{}", model->numParameters());
					tmp["#Trainable"] = std::format("{}", model->numTrainableParams());
					tmp["Memory for Model"] = "XXX MB";
				}
				for (auto&& [key, value] : tmp) {
					elements.push_back(ftxui::hbox({
							ftxui::text(key) | ftxui::color(ftxui::Color::Blue),
							ftxui::filler(),
							ftxui::text(value) | ftxui::color(ftxui::Color::GrayDark),
					}));
				}
				return ftxui::vbox(
						{ftxui::window(
								 ftxui::hbox({ftxui::text(" General ") | ftxui::inverted}), ftxui::vbox(elements)
						 ) |
						 ftxui::flex | ftxui::color(WindowBorderColor)}
				);
			});
			auto progressView = ftxui::Renderer([this] {
				auto prefix = std::format(" Epoch: {:3} |", epoch);
				auto suffix = std::format("|{: 5.2} % ", 100 * progress);
				return ftxui::window(
							   ftxui::hbox({ftxui::text(" Progesss ") | ftxui::inverted}),
							   ftxui::hbox({ftxui::text(prefix), ftxui::gaugeRight(progress), ftxui::text(suffix)}) |
									   ftxui::color(ftxui::Color::White)
					   ) |
					   ftxui::color(WindowBorderColor);
			});
			int infoSize = 40;
			auto container = utilAndLoss;
			container = ftxui::ResizableSplitRight(info, container, &infoSize);

			auto main = ftxui::Container::Vertical({container | ftxui::flex, progressView});
			screen->Loop(main);
		});
	}
	virtual ~ConsoleUI() {
		screen->Exit();
		renderThread.join();
	}
	virtual void onTrainingBegun(const ModelBase& model) override {
		this->model = &model;
		screen->PostEvent(ftxui::Event::Custom);
	}
	virtual void onTrainingEnded(const ModelBase& model) override {}
	virtual void enterTrainingStage(TrainStage stage) override {}
	virtual void exitTrainingStage() override {}
	virtual void progressChanged(size_t epoch, size_t total, size_t step) override {
		this->epoch = epoch;
		this->progress = step / (float)total;
		screen->PostEvent(ftxui::Event::Custom);
	}
};

std::unique_ptr<TrainerObserver> dl::observers::consoleUI() noexcept { return std::make_unique<ConsoleUI>(); }

#include <sys/resource.h>
#include <sys/sysinfo.h>

size_t getRAMUsage_KB() {
	struct rusage usage;
	int ret;
	ret = getrusage(RUSAGE_SELF, &usage);
	return usage.ru_maxrss; // in KB
}

size_t getTotalRAM_KB() {
	struct sysinfo info;
	assert(sysinfo(&info) == 0);
	return info.totalram * (size_t)info.mem_unit / 1024;
}