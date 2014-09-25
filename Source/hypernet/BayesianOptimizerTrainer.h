/*
AI Lib
Copyright (C) 2014 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#pragma once

#include <hypernet/BayesianOptimizer.h>
#include <hypernet/Experiment.h>

namespace hn {
	class BayesianOptimizerTrainer {
	private:
		std::vector<std::shared_ptr<Experiment>> _experiments;

		HyperNet _hyperNet;
		float _fitness;

	public:
		BayesianOptimizer _optimizer;

		size_t _runsPerExperiment;

		BayesianOptimizerTrainer();

		void create(const Config &config, float minWeight, float maxWeight, std::mt19937 &generator, float activationMultiplier);

		void evaluate(const Config &config, std::mt19937 &generator);

		void update(const Config &config, std::mt19937 &generator);
	
		void writeBestToStream(std::ostream &os) const;

		void addExperiment(const std::shared_ptr<Experiment> &experiment) {
			_experiments.push_back(experiment);
		}

		void removeExperiment(size_t index) {
			_experiments.erase(_experiments.begin() + index);
		}

		size_t getNumExperiments() const {
			return _experiments.size();
		}

		std::shared_ptr<Experiment> getExperiment(size_t index) {
			return _experiments[index];
		}

		float getCurrentFitness() const {
			return _fitness;
		}
	};
}