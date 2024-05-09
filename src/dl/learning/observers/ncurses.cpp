#include <dl/learning/trainer.hpp>

#include <ncurses.h>

/** \todo implement me **/

using dl::TrainerObserver;
using dl::TrainStage;

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
		mvwprintw(infoWin, 1, 1, "Model: XXXX");
		mvwprintw(infoWin, 2, 1, "Version: vXX");
		mvwprintw(infoWin, 3, 1, "#Parameters: XXX");
		mvwprintw(infoWin, 4, 1, "#Trainable: XXX (XX %%)");
		mvwprintw(infoWin, 5, 1, "Memory for Model: XXX MB");
		mvwprintw(infoWin, 6, 1, "Seed: XXXXXXXX");
		mvwprintw(infoWin, 7, 1, "DSet Size (Train / Val. / Test):");
		mvwprintw(infoWin, 8, 1, "    XXXXX / XXX / XXX");
		mvwprintw(infoWin, 9, 1, "Last Epoch took XX min");
		wrefresh(infoWin);

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
	virtual void enterTrainingStage(TrainStage stage) override {}
	virtual void exitTrainingStage() override {}
	virtual void progressChanged(size_t epoch, size_t total, size_t step) override {
		drawProgressWindow(epoch, step, total);
	}

	void drawUtilizationWindow() const noexcept {
		mvwprintw(utilWin, 0, 1, "Utilization");
		mvwprintw(utilWin, 1, 1, "CPU:  [|||||||||||||||||||||         XX %%]");
		mvwprintw(utilWin, 2, 1, "GPU:  [|||||||||||||||||||||         XX %%]");
		mvwprintw(utilWin, 1, (COLS - InfoWinWidth) / 2, "RAM:  [|||||||||||||||||||||            XX %%]");
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
};

std::unique_ptr<TrainerObserver> dl::observers::ncursesUI() noexcept { return std::make_unique<NCursesUI>(); }