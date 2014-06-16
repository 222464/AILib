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

#include <hypernet/SampleField.h>
#include <random>

namespace hn {
	class BayesianOptimizer {
	private:
		std::vector<float> _currentVariables;

		std::vector<float> _minBounds, _maxBounds;

	public:
		float _defaultStdDev;

		SampleField _sampleField;

		BayesianOptimizer();

		void create(size_t numVariables, const std::vector<float> &minBounds, const std::vector<float> &maxBounds);

		void generateNewVariables(std::mt19937 &generator);

		void update(float fitness);

		const std::vector<float> &getCurrentVariables() const {
			return _currentVariables;
		}

		const std::vector<float> &getMinBounds() const {
			return _minBounds;
		}

		const std::vector<float> &getMaxBounds() const {
			return _maxBounds;
		}
	};
}