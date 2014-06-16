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

#include <hypernet/EvolutionaryAlgorithm.h>
#include <hypernet/Experiment.h>

namespace hn {
	class EvolutionaryTrainer {
	private:
		std::vector<std::shared_ptr<Experiment>> _experiments;

	public:
		EvolutionaryAlgorithm _evolutionaryAlgorithm;

		size_t _runsPerExperiment;

		EvolutionaryTrainer();

		void create(const Config &config, size_t populationSize, std::mt19937 &generator, float activationMultiplier);

		void evaluate(const Config &config, std::mt19937 &generator);
		void reproduce(const Config &config, std::mt19937 &generator);

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
	};
}