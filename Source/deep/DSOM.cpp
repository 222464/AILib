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

#include <deep/DSOM.h>

#include <algorithm>
#include <random>

using namespace deep;

DSOM::DSOM()
: _neighborhoodRadius(0.0625f), _alpha(0.01f), _radiusDecay(1.0f),
_gaussianScalar(1.0f)
{}

void DSOM::createRandom(size_t numInputs, size_t dimensions, size_t dimensionSize, float minWeight, float maxWeight, std::mt19937 &generator) {
	_numInputs = numInputs;
	_dimensions = dimensions;
	_dimensionSize = dimensionSize;

	_nodes.resize(std::pow(_dimensionSize, _dimensions));

	std::uniform_real_distribution<float> distribution(minWeight, maxWeight);

	for (size_t n = 0; n < _nodes.size(); n++) {
		_nodes[n]._weights.resize(_numInputs);

		for (size_t i = 0; i < _numInputs; i++)
			_nodes[n]._weights[i] = distribution(generator);
	}
}

DSOM::SOMNode &DSOM::getNode(const SOMCoords &coords) {
	SOMCoords wrapped = coords;

	// Wrap coordinates
	for (size_t d = 0; d < coords._coords.size(); d++) {
		wrapped._coords[d] = wrapped._coords[d] % static_cast<int>(_dimensionSize);

		if (wrapped._coords[d] < 0)
			wrapped._coords[d] += _dimensionSize;
	}

	size_t linearCoord = 0;

	for (size_t d = 0, coordOffset = 1; d < _dimensions; d++, coordOffset *= _dimensionSize)
		linearCoord += wrapped._coords[d] * coordOffset;

	return _nodes[linearCoord];
}

DSOM::SOMCoords DSOM::getBestMatchingUnit(const std::vector<float> &input) {
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

DSOM::SOMCoordsReal DSOM::getBestMatchingUnitReal(const std::vector<float> &input, float closenessFactor) {
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
	SOMCoords centerCoords;
	centerCoords._coords.resize(_dimensions);

	for (size_t d = 0, coordOffset = 1; d < _dimensions; d++, coordOffset *= _dimensionSize)
		centerCoords._coords[d] = (minIndex / coordOffset) % _dimensionSize;

	// Check surrounding coordinates and interpolate
	SOMCoords minimum, maximum;
	minimum._coords.resize(_dimensions);
	maximum._coords.resize(_dimensions);

	for (size_t d = 0; d < _dimensions; d++) {
		minimum._coords[d] = centerCoords._coords[d] - 1;
		maximum._coords[d] = centerCoords._coords[d] + 1;
	}

	SOMCoordsReal result;
	result._coords.assign(_dimensions, 0.0f);

	SOMCoords parseCoords;
	parseCoords._coords = minimum._coords;

	float totalInfluence = 0.0f;

	while (parseCoords._coords != maximum._coords) {
		// Update current unit
		SOMNode &node = getNode(parseCoords);

		// Interpolate into direction of this node based on it's similarity to the target vs the similarity of the chosen node to the target
		float neighborMinDifference = node.differenceSquared(input);

		float influence = 2.0f - 2.0f * sigmoid(closenessFactor * (neighborMinDifference - minDifference));

		totalInfluence += influence;

		for (size_t d = 0; d < _dimensions; d++)
			result._coords[d] += (static_cast<float>(centerCoords._coords[d]) + 0.5f * static_cast<float>(parseCoords._coords[d] - centerCoords._coords[d])) * influence;

		// Increment coordinates
		parseCoords._coords[0]++;

		for (size_t d = 0; d < _dimensions - 1; d++)
		if (parseCoords._coords[d] > maximum._coords[d]) {
			parseCoords._coords[d] = minimum._coords[d];
			parseCoords._coords[d + 1]++;
		}
	}

	float invTotalInfluence = 1.0f / totalInfluence;

	for (size_t d = 0; d < _dimensions; d++)
		result._coords[d] *= invTotalInfluence;

	return result;
}

void DSOM::updateNeighborhood(const SOMCoords &centerCoords, const std::vector<float> &target) {
	int radius = static_cast<int>(std::ceil(static_cast<float>(_dimensionSize) * _neighborhoodRadius));
	int dimensionSizei = static_cast<int>(_dimensionSize);
	float radiusSquaredf = static_cast<float>(radius * radius);

	SOMCoords minimum, maximum;
	minimum._coords.resize(_dimensions);
	maximum._coords.resize(_dimensions);

	for (size_t d = 0; d < _dimensions; d++) {
		minimum._coords[d] = centerCoords._coords[d] - radius;
		maximum._coords[d] = centerCoords._coords[d] + radius;
	}

	SOMCoords parseCoords;
	parseCoords._coords = minimum._coords;

	while (parseCoords._coords != maximum._coords) {
		// Update current unit
		SOMNode &node = getNode(parseCoords);

		float distSquared = 0.0f;

		for (size_t d = 0; d < _dimensions; d++) {
			float delta = static_cast<float>(centerCoords._coords[d]) - static_cast<float>(parseCoords._coords[d]);
			distSquared += delta * delta;
		}

		float dist = std::sqrt(distSquared);

		float influence = (_neighborhoodRadius - dist) / _neighborhoodRadius * std::exp(-_gaussianScalar * distSquared / (2.0f * radiusSquaredf));

		for (size_t i = 0; i < node._weights.size(); i++)
			node._weights[i] += influence * _alpha * (target[i] - node._weights[i]);

		// Increment coordinates
		parseCoords._coords[0]++;

		for (size_t d = 0; d < _dimensions - 1; d++)
		if (parseCoords._coords[d] > maximum._coords[d]) {
			parseCoords._coords[d] = minimum._coords[d];
			parseCoords._coords[d + 1]++;
		}
	}

	_neighborhoodRadius *= _radiusDecay;
}