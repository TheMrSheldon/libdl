#include <dl/learning/trainer.hpp>

#include <dl/model/model.hpp>

#include <ftxui/component/captured_mouse.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ncurses.h>

#include <cassert>

/** \todo implement me **/

using dl::ModelBase;
using dl::TrainerObserver;
using dl::TrainStage;

static size_t getRAMUsage_KB();
static size_t getTotalRAM_KB();

/*
class NCursesUI final : public TrainerObserver {
private:
	WINDOW *utilWin, *infoWin, *progWin, *lossWin;
	static constexpr size_t ProgWinHeight = 3;
	static constexpr size_t InfoWinWidth = 40;
	static constexpr size_t UtilWinHeight = 4;

public:
	NCursesUI() {
		initscr();
		utilWin = newwin(UtilWinHeight, COLS - InfoWinWidth, 0, 0);
		refresh();
		box(utilWin, 0, 0);
		drawUtilizationWindow();

		infoWin = newwin(LINES - ProgWinHeight, 0, 0, COLS - InfoWinWidth);
		refresh();
		box(infoWin, 0, 0);
		mvwprintw(infoWin, 0, 1, "General");

		progWin = newwin(0, 0, LINES - ProgWinHeight, 0);
		refresh();
		box(progWin, 0, 0);
		mvwprintw(progWin, 0, 1, "Progress");
		drawProgressWindow(0, 0, 0);

		lossWin = newwin(LINES - (UtilWinHeight + ProgWinHeight), COLS - InfoWinWidth, 4, 0);
		refresh();
		box(lossWin, 0, 0);
		mvwprintw(lossWin, 0, 1, "Loss");
		wrefresh(lossWin);
	}
	virtual ~NCursesUI() { endwin(); }
	virtual void onBeginTraining(const ModelBase& model) override { drawInfoWindow(model); }
	virtual void enterTrainingStage(TrainStage stage) override {}
	virtual void exitTrainingStage() override {}
	virtual void progressChanged(size_t epoch, size_t total, size_t step) override {
		drawProgressWindow(epoch, step, total);
	}

	void drawUtilizationWindow() const noexcept {
		auto ram = getRAMUsage_KB() >> 10;
		auto ramtot = getTotalRAM_KB() >> 10;
		float utilization = ram / (float)ramtot;
		const unsigned barSize = 30;
		std::string barStr((size_t)(utilization * barSize), '|');
		mvwprintw(utilWin, 0, 1, "Utilization");
		mvwprintw(utilWin, 1, 1, "CPU:  [|||||||||||||||||||||         XX %%]");
		mvwprintw(utilWin, 2, 1, "GPU:  [|||||||||||||||||||||         XX %%]");
		mvwprintw(
				utilWin, 1, (COLS - InfoWinWidth) / 2, "RAM:  [%-*s] % 5i / % 5i MiB", barSize, barStr.c_str(),
				(int)ram, (int)ramtot
		);
		mvwprintw(utilWin, 2, (COLS - InfoWinWidth) / 2, "VRAM: [|||||||||||||||||||||            XX %%]");
		wrefresh(utilWin);
	}

	void drawProgressWindow(size_t epoch, size_t step, size_t maxsteps) const noexcept {
		if (maxsteps == 0)
			maxsteps = 1;
		const float percentage = step / (float)maxsteps;
		const unsigned barSize = COLS - 25;
		std::string barStr((size_t)(percentage * barSize), '#');
		mvwprintw(
				progWin, 1, 2, "Epoch: % 3x [%-*s] % 5.1f %%", (unsigned)epoch, barSize, barStr.c_str(),
				percentage * 100.0f
		);
		wrefresh(progWin);
	}

	void drawInfoWindow(const dl::ModelBase& model) const noexcept {
		const auto numParams = model.numParameters();
		const auto numTrainable = model.numTrainableParams();

		mvwprintw(infoWin, 1, 1, "Model: XXXX");
		mvwprintw(infoWin, 2, 1, "Version: vXX");
		mvwprintw(infoWin, 3, 1, "#Parameters: % 7i", numParams);
		mvwprintw(infoWin, 4, 1, "#Trainable:  % 7i (% 5.2f %%)", numTrainable, numTrainable * (100.0f) / numParams);
		mvwprintw(infoWin, 5, 1, "Memory for Model: XXX MB");
		mvwprintw(infoWin, 6, 1, "Seed: XXXXXXXX");
		mvwprintw(infoWin, 7, 1, "DSet Size (Train / Val. / Test):");
		mvwprintw(infoWin, 8, 1, "    XXXXX / XXX / XXX");
		mvwprintw(infoWin, 9, 1, "Last Epoch took XX min");
		wrefresh(infoWin);
	}
};*/

class NCursesUI final : public TrainerObserver {
private:
	ftxui::ScreenInteractive* screen;
	ftxui::Component container;
	std::thread renderThread;

	NCursesUI(const NCursesUI& other) = delete;
	NCursesUI(NCursesUI&& other) = delete;
	NCursesUI& operator=(const NCursesUI& other) = delete;
	NCursesUI& operator=(NCursesUI&& other) = delete;

	size_t epoch;
	float progress = 0;

public:
	NCursesUI() {
		renderThread = std::thread([this] {
			auto tmp = ftxui::ScreenInteractive::Fullscreen();
			screen = &tmp;

			auto canvas = ftxui::Canvas(10, 10);

			auto utilAndLoss = ftxui::Renderer([this, canvas] {
				return ftxui::vbox(
						{ftxui::window(ftxui::text("Utilization"), ftxui::hbox({})),
						 ftxui::window(ftxui::text("Loss"), ftxui::vbox({ftxui::canvas(canvas)}))}
				);
			});
			auto info = ftxui::Renderer([this] {
				return ftxui::vbox({ftxui::window(ftxui::text("General"), ftxui::vbox({ftxui::filler()}))});
			});
			auto progressView = ftxui::Renderer([this] {
				auto prefix = std::format("Epoch: {:3} [", epoch);
				auto suffix = std::format("]{: 5.2} %", 100 * progress);
				return ftxui::window(
						ftxui::text("Progress"),
						ftxui::hbox({ftxui::text(prefix), ftxui::gaugeRight(progress), ftxui::text(suffix)})
				);
			});
			int infoSize = 40;
			auto container = utilAndLoss;
			container = ftxui::ResizableSplitRight(info, container, &infoSize);

			auto main = ftxui::Container::Vertical({container, progressView});
			screen->Loop(main);
		});
	}
	virtual ~NCursesUI() {
		screen->Exit();
		renderThread.join();
	}
	virtual void onBeginTraining(const ModelBase& model) override { drawInfoWindow(model); }
	virtual void enterTrainingStage(TrainStage stage) override {}
	virtual void exitTrainingStage() override {}
	virtual void progressChanged(size_t epoch, size_t total, size_t step) override {
		this->epoch = epoch;
		this->progress = step / (float)total;
		// drawProgressWindow(epoch, step, total);
		screen->PostEvent(ftxui::Event::Custom);
	}

	void drawUtilizationWindow() const noexcept {}

	void drawProgressWindow(size_t epoch, size_t step, size_t maxsteps) const noexcept {}

	void drawInfoWindow(const dl::ModelBase& model) const noexcept {}
};

std::unique_ptr<TrainerObserver> dl::observers::ncursesUI() noexcept { return std::make_unique<NCursesUI>(); }

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