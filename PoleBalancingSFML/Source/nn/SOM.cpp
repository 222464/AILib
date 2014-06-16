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

#include <nn/SOM.h>

#include <algorithm>
#include <random>

using namespace nn;

SOM::SOM()
: _neighborhoodRadius(0.125f), _alpha(0.1f)
{}

void SOM::createRandom(size_t numInputs, size_t dimensions, size_t dimensionSize, float minWeight, float maxWeight, unsigned long seed) {
	_numInputs = numInputs;
	_dimensions = dimensions;
	_dimensionSize = dimensionSize;

	_nodes.resize(std::pow(_dimensionSize, _dimensions));

	std::mt19937 generator(seed);

	std::uniform_real_distribution<float> distribution(minWeight, maxWeight);

	for (size_t n = 0; n < _nodes.size(); n++) {
		_nodes[n]._weights.resize(_numInputs);

		for (size_t i = 0; i < _numInputs; i++)
			_nodes[n]._weights[i] = distribution(generator);
	}
}

SOM::SOMNode &SOM::getNode(const SOMCoords &coords) {
	SOMCoords wrapped = coords;

	// Wrap coordinates
	for (size_t d = 0; d < coords._coords.size(); d++) {
		wrapped._coords[d] = wrapped._coords[d] % _dimensionSize;

		if (wrapped._coords[d] < 0)
			wrapped._coords[d] += _dimensionSize;
	}

	size_t linearCoord = 0;

	for (size_t d = 0, coordOffset = 1; d < _dimensions; d++, coordOffset *= _dimensionSize)
		linearCoord += wrapped._coords[d] * coordOffset;

	return _nodes[linearCoord];
}

SOM::SOMCoords SOM::getBestMatchingUnit(const std::vector<float> &input) {
	float minDifference = _nodes[0].differenceSquared(input);

	size_t minIndex = 0;

	for (size_t n = 1; n < _nodes.size(); n++) {
		float difference = _nodes[n].differenceSquared(input);

		if (difference < minDifference) {
			minDifference = difference;
			minIndex = n;
		}
	}

	// Convert from linear to dimensional coordinates
	SOMCoords coords;
	coords._coords.resize(_dimensions);

	for (size_t d = 0, coordOffset = 1; d < _dimensions; d++, coordOffset *= _dimensionSize)
		coords._coords[d] = (minIndex / coordOffset) % _dimensionSize;

	return coords;
}

void SOM::updateNeighborhood(const SOMCoords &centerCoords, const std::vector<float> &target) {
	int radius = static_cast<int>(std::ceil(static_cast<float>(_dimensionSize) * _neighborhoodRadius));
	int dimensionSizei = static_cast<int>(_dimensionSize);
	float radiusSquaredf = static_cast<float>(radius * radius);

	SOMCoords min, max;
	min._coords.resize(_dimensions);
	max._coords.resize(_dimensions);

	for (size_t d = 0; d < _dimensions; d++) {
		min._coords[d] = centerCoords._coords[d] - radius;
		max._coords[d] = centerCoords._coords[d] + radius;
	}

	SOMCoords parseCoords;
	parseCoords._coords = min._coords;

	while (parseCoords._coords != max._coords) {
		// Update current unit
		SOMNode &node = getNode(parseCoords);

		float distSquared = 0.0f;

		for (size_t d = 0; d < _dimensions; d++) {
			float delta = static_cast<float>(centerCoords._coords[d]) - static_cast<float>(parseCoords._coords[d]);
			distSquared += delta * delta;
		}

		float influence = std::expf(-distSquared / (2.0f * radiusSquaredf));

		for (size_t i = 0; i < node._weights.size(); i++)
			node._weights[i] += influence * _alpha * (target[i] - node._weights[i]);

		// Increment coordinates
		parseCoords._coords[0]++;

		for (size_t d = 0; d < _dimensions - 1; d++)
		if (parseCoords._coords[d] >= max._coords[d]) {
			parseCoords._coords[d] = min._coords[d];
			parseCoords._coords[d + 1]++;
		}
	}
}