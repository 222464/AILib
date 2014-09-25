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

#include <hypernet/BayesianOptimizer.h>

#include <algorithm>
#include <iostream>

#include <assert.h>

using namespace hn;

BayesianOptimizer::BayesianOptimizer()
: _defaultStdDev(100.0f)
{}

void BayesianOptimizer::create(size_t numVariables, const std::vector<float> &minBounds, const std::vector<float> &maxBounds) {
	_sampleField.create(numVariables, 1);

	_minBounds = minBounds;
	_maxBounds = maxBounds;

	assert(numVariables == _minBounds.size() && _minBounds.size() == _maxBounds.size());
}

void BayesianOptimizer::generateNewVariables(std::mt19937 &generator) {
	_currentVariables.resize(_sampleField.getXSize());

	// Totally random values if below 2 samples
	if (_sampleField.getNumSamples() < 2) {
		for (size_t xi = 0; xi < _currentVariables.size(); xi++) {
			std::uniform_real_distribution<float> distBounds(_minBounds[xi], _maxBounds[xi]);

			_currentVariables[xi] = distBounds(generator);
		}
	}
	else {
		// Find highest valued data sample
		SampleField::Sample highest = _sampleField.getSample(0);

		for (size_t s = 1; s < _sampleField.getNumSamples(); s++)
		if (_sampleField.getSample(s)._y[0] > highest._y[0])
			highest = _sampleField.getSample(s);

		float stdDev = _defaultStdDev / (1.0f + std::sqrt(_sampleField.getVarianceAtX(highest._x)));
		
		std::normal_distribution<float> distNormal(0.0f, stdDev);

		for (size_t xi = 0; xi < _currentVariables.size(); xi++)
			_currentVariables[xi] = std::max(_minBounds[xi], std::min(_maxBounds[xi], highest._x[xi] + distNormal(generator)));
	}
}

void BayesianOptimizer::update(float fitness) {
	SampleField::Sample s;
	s._x = _currentVariables;
	s._y = std::vector<float>(1, fitness);

	_sampleField.addSample(s);
}