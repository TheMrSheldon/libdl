#include <dl/learning/dataset.hpp>
#include <dl/logging.hpp>
#include <dl/tensor/tensorptr.hpp>
#include <dl/utils/urlstream.hpp>

#include <yaml-cpp/yaml.h>

#include <cassert>
#include <map>
#include <string>
#include <string_view>
#include <vector>

struct HFDatasetTaskConfig final {
	std::string name;
	std::map<std::string, std::string> datafiles;
};

struct HFAutoTrainTaskConfig final {
	std::string name;
	std::string task;
	std::string taskid;
	std::map<std::string, std::string> splits;
	std::map<std::string, std::string> colmapping;
};

struct HFAutoTrainConfig final {
	std::map<std::string, HFAutoTrainTaskConfig> tasks;
};

struct HFDatasetConfig final {
	std::optional<std::string> prettyName;
	std::map<std::string, HFDatasetTaskConfig> configs;
	HFAutoTrainConfig autotrain;
};

static bool fromYAML(HFDatasetTaskConfig& config, const YAML::Node& yaml) {
	config.name = yaml["config_name"].as<std::string>();
	for (const auto& node : yaml["data_files"])
		config.datafiles[node["split"].as<std::string>()] = node["path"].as<std::string>();
	return true;
}

static bool fromYAML(HFAutoTrainTaskConfig& config, const YAML::Node& yaml) {
	config.name = yaml["config"].as<std::string>();
	config.task = yaml["task"].as<std::string>();
	config.taskid = yaml["task_id"].as<std::string>();
	for (const auto& node : yaml["splits"])
		config.splits[node.first.as<std::string>()] = node.second.as<std::string>();
	for (const auto& node : yaml["col_mapping"])
		config.colmapping[node.first.as<std::string>()] = node.second.as<std::string>();
	return true;
}

static bool fromYAML(HFAutoTrainConfig& config, const YAML::Node& yaml) {
	for (const auto& node : yaml["train-eval-index"]) {
		HFAutoTrainTaskConfig cfg;
		fromYAML(cfg, node);
		config.tasks[cfg.name] = std::move(cfg);
	}
	return true;
}

static bool fromYAML(HFDatasetConfig& config, const YAML::Node& yaml) {
	static auto logger = dl::logging::getLogger("hfdataset");
	if (yaml["pretty_name"].IsDefined())
		config.prettyName = yaml["pretty_name"].as<std::string>();
	logger->debug("Loading Dataset {}", config.prettyName);
	if (yaml["configs"].IsDefined()) {
		logger->debug("Reading task configurations");
		for (const auto& node : yaml["configs"]) {
			HFDatasetTaskConfig cfg;
			fromYAML(cfg, node);
			logger->debug("Configuration: {}", cfg.name);
			for (const auto& [key, value] : cfg.datafiles)
				logger->debug("\t{}: {}", key, value);
			config.configs[cfg.name] = std::move(cfg);
		}
	}
	if (yaml["train-eval-index"].IsDefined()) {
		logger->debug("Reading autotrain configuration");
		fromYAML(config.autotrain, yaml["train-eval-index"]);
	}
	return true;
}

#include <iostream>

template <typename R, typename... Args>
class ParquetDataLoader : public dl::Dataloader<R(Args...)> {
public:
	using Instance = typename std::tuple<R, std::remove_reference_t<Args>...>;
	using Iterator = typename dl::utils::GenericIterator<Instance>;

	ParquetDataLoader(std::string_view url, std::vector<std::string> inputcols, std::vector<std::string> outcols);

	virtual dl::utils::GenericIterator<Instance> begin() override { throw std::runtime_error("Not yet implemented"); }
	virtual dl::utils::GenericIterator<Instance> end() override { throw std::runtime_error("Not yet implemented"); }
};

template <typename>
class HFDataset;

template <typename R, typename... Args>
class HFDataset<R(Args...)> : public dl::Dataset<R(Args...)> {
private:
	dl::logging::LoggerPtr logger;
	std::string baseurl;
	HFDatasetConfig config;
	std::string confname;

	HFDataset(const HFDataset&) = delete;
	HFDataset& operator=(const HFDataset&) = delete;

	HFDataset(std::string_view baseurl, const HFDatasetConfig& config, std::string_view confname)
			: logger(dl::logging::getLogger("hfdataset")), baseurl(baseurl), config(config), confname(confname) {
		logger->debug("Created {} from {} for config '{}'", config.prettyName, baseurl, confname);
	}

public:
	virtual std::unique_ptr<dl::Dataloader<R(Args...)>> trainingData() override {
		throw std::runtime_error("Not yet implemented");
	}
	virtual std::unique_ptr<dl::Dataloader<R(Args...)>> validationData() override {
		throw std::runtime_error("Not yet implemented");
	}
	virtual std::unique_ptr<dl::Dataloader<R(Args...)>> testData() override {
		throw std::runtime_error("Not yet implemented");
	}

	static HFDataset load(std::string_view url, std::string_view confname = "default") {
		/** \todo make this cleaner and less hardcoded **/
		assert(url.starts_with("hf:"));
		dl::utils::URLStream stream(
				std::format("https://huggingface.co/datasets/{}/raw/main/README.md", url.substr(3).data()).c_str()
		);
		char tmp[3];
		stream.read(tmp, sizeof(tmp)); // Skip the first 3 characters (assuming they are "---")
		HFDatasetConfig config;
		fromYAML(config, YAML::Load(stream));
		const auto& c = config.configs[confname.data()];
		return HFDataset(url, config, confname);
	}
};

int main(int argc, char* argv[]) {
	dl::logging::setVerbosity(dl::logging::Verbosity::Debug);
	auto logger = dl::logging::getLogger("main");

	auto dataset = HFDataset<dl::TensorPtr(dl::TensorPtr)>::load("hf:thagen/SCITE", "causality detection");
	return 0;
}