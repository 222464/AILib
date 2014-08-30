#include <apssc/APSSC.h>

#include <iostream>

const float _pi = 3.141592f;
const float _2pi = 2.0f * _pi;

using namespace apssc;

void APSSC::create(size_t numLabels, size_t dimensions, size_t numAnts) {
	_dimensions = dimensions;
	_numLabels = numLabels;

	_allAnts.resize(numAnts);

	_colonies.resize(_numLabels);

	for (size_t i = 0; i < _numLabels; i++)
		_colonies[i]._colonyCenter = Eigen::VectorXf::Zero(_dimensions);

	for (size_t i = 0; i < _allAnts.size(); i++) {
		_allAnts[i]._position = Eigen::VectorXf::Zero(_dimensions);
		_allAnts[i]._pheromones = Eigen::VectorXf::Zero(_numLabels);
		//_allAnts[i]._colonyCovariances.assign(_numLabels, Eigen::VectorXf::Zero(_dimensions));
	}
}

void APSSC::prepare() {
	for (size_t i = 0; i < _allAnts.size(); i++)
	if (_allAnts[i]._label != -1) {
		_colonies[_allAnts[i]._label]._antIndices.push_back(i);
		_labeledAntIndices.push_back(i);
	}
	else
		_unlabeledAntIndices.push_back(i);
}

void APSSC::trainingCycle(float averageDecay, float pheromoneDecay) {
	float gaussianCoeff = 1.0f;// / (std::pow(_2pi, static_cast<float>(_dimensions)* 0.5f));

	// Update colony matrix data
	/*for (size_t i = 0; i < _colonies.size(); i++) {
		// Get average position
		Eigen::VectorXf averagePosition(_dimensions);

		for (size_t j = 0; j < averagePosition.size(); j++)
			averagePosition(j) = 0.0f;

		for (size_t j = 0; j < _colonies[i]._antIndices.size(); j++)
		for (size_t k = 0; k < _dimensions; k++)
			averagePosition(k) += _allAnts[_colonies[i]._antIndices[j]]._position(k);

		float colonySizeInv = 1.0f / _colonies[i]._antIndices.size();

		averagePosition *= colonySizeInv;

		//float averageValue = averagePosition.mean();

		if (_t == 0) {
			//_colonies[i]._expectedPosition = averagePosition;

			for (size_t j = 0; j < _colonies[i]._expectedPosition.size(); j++)
			for (size_t k = 0; k < _colonies[i]._expectedPosition.size(); k++)
				_colonies[i]._covarianceMatrix(j, k) = (averagePosition(j) - _colonies[i]._expectedPosition(j)) * (averagePosition(k) - _colonies[i]._expectedPosition(k));
		}
		else {
			_colonies[i]._expectedPosition = (1.0f - averageDecay) * _colonies[i]._expectedPosition + averageDecay * averagePosition;

			for (size_t j = 0; j < _colonies[i]._expectedPosition.size(); j++)
			for (size_t k = 0; k < _colonies[i]._expectedPosition.size(); k++)
				_colonies[i]._covarianceMatrix(j, k) = (1.0f - averageDecay) * _colonies[i]._covarianceMatrix(j, k) + averageDecay * (averagePosition(j) - _colonies[i]._expectedPosition(j)) * (averagePosition(k) - _colonies[i]._expectedPosition(k));
		}

		std::cout << _colonies[i]._covarianceMatrix << std::endl;

		_colonies[i]._covarianceDet = _colonies[i]._covarianceMatrix.fullPivLu().determinant();
		_colonies[i]._covarianceMatrixInv = _colonies[i]._covarianceMatrix.fullPivLu().inverse();
	}*/

	float numLabelsInv = 1.0f / _numLabels;

	std::vector<std::tuple<size_t, size_t>> antsToColonize;

	for (size_t i = 0; i < _unlabeledAntIndices.size(); i++) {
		Ant &uAnt = _allAnts[_unlabeledAntIndices[i]];
		
		for (size_t j = 0; j < _colonies.size(); j++) {
			float averageAggregation = 0.0f;

			for (size_t k = 0; k < _colonies[j]._antIndices.size(); k++) {
				Eigen::VectorXf difference = uAnt._position - _allAnts[_colonies[j]._antIndices[k]]._position;

				//averageAggregation += gaussianCoeff / std::sqrt(_colonies[j]._covarianceDet) * std::exp(-0.5f * (difference.transpose() * _colonies[j]._covarianceMatrixInv * difference)(0, 0));
				averageAggregation += gaussianCoeff * std::exp(-0.5f * (difference.transpose() * difference)(0, 0));
			}

			averageAggregation /= _colonies[j]._antIndices.size();

			uAnt._pheromones(j) = (1.0f - pheromoneDecay) * uAnt._pheromones(j) + pheromoneDecay * averageAggregation;
		}

		Eigen::VectorXf npd = uAnt._pheromones.normalized();

		float maxNpdElement = npd(0);

		for (size_t j = 1; j < npd.size(); j++)
			maxNpdElement = std::max(maxNpdElement, npd(j));

		float minR = npd(0) / maxNpdElement;
		size_t minRIndex = 0;

		for (size_t j = 1; j < _colonies.size(); j++) {
			float r = npd(j) / maxNpdElement;

			if (r < minR) {
				minR = r;
				minRIndex = j;
			}
		}

		if (minR <= numLabelsInv)
			antsToColonize.push_back(std::make_tuple(minRIndex, i));
	}

	// Add new ants to colonies
	for (size_t i = 0; i < antsToColonize.size(); i++) {
		_allAnts[std::get<1>(antsToColonize[i])]._label = std::get<0>(antsToColonize[i]);

		_colonies[std::get<0>(antsToColonize[i])]._antIndices.push_back(std::get<1>(antsToColonize[i]));
	}

	// Update colony centers
	for (size_t i = 0; i < _colonies.size(); i++) {
		_colonies[i]._colonyCenter = Eigen::VectorXf::Zero(_dimensions);

		for (size_t j = 0; j < _colonies[i]._antIndices.size(); j++)
			_colonies[i]._colonyCenter += _allAnts[_colonies[i]._antIndices[j]]._position;

		_colonies[i]._colonyCenter /= _colonies[i]._antIndices.size();
	}

	// Update ant covariances with center of all colonies
	//for (size_t i = 0; i < _allAnts.size(); i++) {
	//	_allAnts[i]._colonyCovariances = (1.0f - averageDecay) * _allAnts[i]._colonyCovariances + averageDecay * ()
	//}

	_t++;
}

int APSSC::classify(const std::vector<float> &position) {
	Eigen::VectorXf positionV(position.size());
	Eigen::VectorXf pheromones = Eigen::VectorXf::Zero(_numLabels);

	for (size_t i = 0; i < position.size(); i++)
		positionV(i) = position[i];

	float gaussianCoeff = 1.0f;// 1.0f / (std::pow(_2pi, static_cast<float>(_dimensions)* 0.5f));

	for (size_t j = 0; j < _colonies.size(); j++) {
		float averageAggregation = -999999.0f;

		for (size_t k = 0; k < _colonies[j]._antIndices.size(); k++) {
			Eigen::VectorXf difference = positionV - _allAnts[_colonies[j]._antIndices[k]]._position;

			averageAggregation = std::max(averageAggregation, gaussianCoeff * std::exp(-0.5f * (difference.transpose() * difference)(0, 0)));
		}

		//averageAggregation /= _colonies[j]._antIndices.size();

		pheromones(j) = averageAggregation;
	}

	float maxP = pheromones(0);
	size_t maxPIndex = 0;

	for (size_t i = 1; i < _numLabels; i++) {
		float p = pheromones(i);

		if (p > maxP) {
			maxP = p;

			maxPIndex = i;
		}
	}

	return maxPIndex;
}