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

#include <hypernet/BayesianOptimizerTrainer.h>

#include <algorithm>

using namespace hn;

BayesianOptimizerTrainer::BayesianOptimizerTrainer()
: _runsPerExperiment(4), _fitness(0.0f)
{}

void BayesianOptimizerTrainer::create(const Config &config, float minWeight, float maxWeight, std::mt19937 &generator, float activationMultiplier) {
	_hyperNet.createRandom(config, 10000, 0.02f, -2.0f, 2.0f, generator, activationMultiplier);

	std::vector<float> weights;
	_hyperNet.getWeightsVector(weights);

	std::vector<float> minBounds(weights.size(), minWeight);
	std::vector<float> maxBounds(weights.size(), maxWeight);

	_optimizer.create(weights.size(), minBounds, maxBounds);

	_optimizer.generateNewVariables(generator);

	_hyperNet.createFromWeightsVector(config, _optimizer.getCurrentVariables());
}

void BayesianOptimizerTrainer::evaluate(const Config &config, std::mt19937 &generator) {
	_fitness = 0.0f;

	for (size_t i = 0; i < _experiments.size(); i++)
	for (size_t j = 0; j < _runsPerExperiment; j++)
		_fitness += _experiments[i]->getExperimentWeight() * _experiments[i]->evaluate(_hyperNet, config, generator);

	_fitness /= _experiments.size() * _runsPerExperiment;

	_optimizer.update(_fitness);
}

void BayesianOptimizerTrainer::update(const Config &config, std::mt19937 &generator) {
	_optimizer.generateNewVariables(generator);

	_hyperNet.createFromWeightsVector(config, _optimizer.getCurrentVariables());
}

void BayesianOptimizerTrainer::writeBestToStream(std::ostream &os) const {
	_hyperNet.writeToStream(os);
}