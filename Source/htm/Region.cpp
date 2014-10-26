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

#include <htm/Region.h>

#include <algorithm>
#include <list>
#include <assert.h>
#include <unordered_set>

using namespace htm;

void Region::createRandom(int inputWidth, int inputHeight, int connectionRadius, float initInhibitionRadius, int initNumSegments,
	int regionWidth, int regionHeight, int columnSize, float permanenceDistanceBias, float permanenceDistanceFalloff, float permanenceBiasFloor,
	float connectionPermanenceTarget, float connectionPermanenceStdDev, std::mt19937 &generator)
{
	std::normal_distribution<float> permanenceDist(connectionPermanenceTarget, connectionPermanenceStdDev);

	_regionWidth = regionWidth;
	_regionHeight = regionHeight;
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;
	_connectionRadius = connectionRadius;

	_columns.resize(_regionWidth * _regionHeight);

	size_t connectionSize = 4 * _connectionRadius * _connectionRadius;
	float regionWidthInv = 1.0f / _regionWidth;
	float regionHeightInv = 1.0f / _regionHeight;

	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		column._cells.resize(columnSize);

		for (int j = 0; j < columnSize; j++)
			column._cells[j]._segments.resize(initNumSegments);

		column._inhibitionRadius = initInhibitionRadius;

		float columnXf = static_cast<float>(i % _regionWidth) * regionWidthInv;
		float columnYf = static_cast<float>(i / _regionWidth) * regionHeightInv;

		int inputX = static_cast<int>(columnXf * _inputWidth);
		int inputY = static_cast<int>(columnYf * _inputHeight);

		for (int dx = -_connectionRadius; dx <= _connectionRadius; dx++)
		for (int dy = -_connectionRadius; dy <= _connectionRadius; dy++) {
			int connectionX = inputX + dx;
			int connectionY = inputY + dy;

			// If exists
			if (connectionX >= 0 && connectionY >= 0 && connectionX < _inputWidth && connectionY < _inputHeight) {
				float distSquared = dx * dx + dy * dy;

				// Create a possible connection
				Connection connection;
				connection._permanence = permanenceDist(generator) + permanenceDistanceBias * std::exp(-distSquared * permanenceDistanceFalloff) - permanenceBiasFloor;

				column._inputConnections.push_back(connection);
			}
		}
	}
}

bool Region::getOutput(int i) const {
	const Column &column = _columns[i];

	for (int j = 0; j < column._cells.size(); j++)
	if (column._cells[j]._activeState || column._cells[j]._predictiveState)
		return true;

	return false;
}

bool Region::getOutput(int x, int y) const {
	return getOutput(x + y * _regionWidth);
}

bool Region::getPrediction(int i, int t) const {
	const Column &column = _columns[i];

	for (int j = 0; j < column._cells.size(); j++)
	if (column._cells[j]._numPredictionSteps == t)
		return true;
	
	return false;
}

bool Region::getPrediction(int x, int y, int t) const {
	return getPrediction(x + y * _regionWidth, t);
}

void Region::setColumnsToOutput() {
	_activeColumnIndices.clear();

	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		bool setActive = false;

		for (int j = 0; j < column._cells.size(); j++)
		if (column._cells[j]._activeState || column._cells[j]._predictiveState) {
			setActive = true;
			break;
		}

		column._active = setActive;

		if (column._active)
			_activeColumnIndices.push_back(i);
	}
}

bool Region::hasLearningCell(int x, int y) const {
	const Column &column = _columns[x + y * _regionWidth];

	for (int j = 0; j < column._cells.size(); j++)
	if (column._cells[j]._prevLearnState)
		return true;

	return false;
}

bool Region::hasSegments(int x, int y) const {
	const Column &column = _columns[x + y * _regionWidth];

	for (int j = 0; j < column._cells.size(); j++)
	if (!column._cells[j]._segments.empty())
		return true;

	return false;
}

bool Region::hasConnections(int x, int y) const {
	const Column &column = _columns[x + y * _regionWidth];

	for (int j = 0; j < column._cells.size(); j++)
	for (int k = 0; k < column._cells[j]._segments.size(); k++)
	if (!column._cells[j]._segments[k]._connections.empty())
		return true;

	return false;
}

void Region::getReconstruction(std::vector<bool> &output, float minOverlap, float minPermanence, bool fromPrediction) const {
	if (output.size() != _inputWidth * _inputHeight)
		output.resize(_inputWidth * _inputHeight);

	std::vector<float> accum;
	accum.assign(output.size(), 0.0f);

	float regionWidthInv = 1.0f / _regionWidth;
	float regionHeightInv = 1.0f / _regionHeight;

	for (int i = 0; i < _columns.size(); i++) {
		const Column &column = _columns[i];

		int columnX = i % _regionWidth;
		int columnY = i / _regionWidth;

		float columnXf = static_cast<float>(columnX) * regionWidthInv;
		float columnYf = static_cast<float>(columnY) * regionHeightInv;

		int inputX = static_cast<int>(columnXf * _inputWidth);
		int inputY = static_cast<int>(columnYf * _inputHeight);

		int connectionIndex = 0;

		int receptiveFieldSize = 0;

		// Calculate overlap
		float overlap = 0.0f;

		bool outputHere = fromPrediction ? getOutput(i) : column._active;

		for (int dx = -_connectionRadius; dx <= _connectionRadius; dx++)
		for (int dy = -_connectionRadius; dy <= _connectionRadius; dy++) {
			int connectionX = inputX + dx;
			int connectionY = inputY + dy;

			// If exists
			if (connectionX >= 0 && connectionY >= 0 && connectionX < _inputWidth && connectionY < _inputHeight) {
				if (outputHere && column._inputConnections[connectionIndex]._permanence > minPermanence)
					accum[connectionX + connectionY * _inputWidth]++;

				connectionIndex++;
			}
		}
	}

	float maximumAccum = 0.0f;

	for (int i = 0; i < accum.size(); i++)
		maximumAccum = std::max(maximumAccum, accum[i]);

	if (maximumAccum == 0.0f)
		return;

	float maximumAccumInv = 1.0f / maximumAccum;
	float minOverlapInv = 1.0f / minOverlap;

	for (int i = 0; i < accum.size(); i++) {
		if (accum[i] * maximumAccumInv > minOverlapInv)
			output[i] = true;
		else
			output[i] = false;
	}
}

void Region::getReconstructionAtTime(std::vector<bool> &output, float minOverlap, float minPermanence, int t) const {
	if (output.size() != _inputWidth * _inputHeight)
		output.resize(_inputWidth * _inputHeight);

	std::vector<float> accum;
	accum.assign(output.size(), 0.0f);

	float regionWidthInv = 1.0f / _regionWidth;
	float regionHeightInv = 1.0f / _regionHeight;

	for (int i = 0; i < _columns.size(); i++) {
		const Column &column = _columns[i];

		int columnX = i % _regionWidth;
		int columnY = i / _regionWidth;

		float columnXf = static_cast<float>(columnX) * regionWidthInv;
		float columnYf = static_cast<float>(columnY) * regionHeightInv;

		int inputX = static_cast<int>(columnXf * _inputWidth);
		int inputY = static_cast<int>(columnYf * _inputHeight);

		int connectionIndex = 0;

		int receptiveFieldSize = 0;

		// Calculate overlap
		float overlap = 0.0f;

		bool outputHere = getPrediction(i, t);

		for (int dx = -_connectionRadius; dx <= _connectionRadius; dx++)
		for (int dy = -_connectionRadius; dy <= _connectionRadius; dy++) {
			int connectionX = inputX + dx;
			int connectionY = inputY + dy;

			// If exists
			if (connectionX >= 0 && connectionY >= 0 && connectionX < _inputWidth && connectionY < _inputHeight) {
				if (outputHere && column._inputConnections[connectionIndex]._permanence > minPermanence)
					accum[connectionX + connectionY * _inputWidth]++;

				connectionIndex++;
			}
		}
	}

	float maximumAccum = 0.0f;

	for (int i = 0; i < accum.size(); i++)
		maximumAccum = std::max(maximumAccum, accum[i]);

	if (maximumAccum == 0.0f)
		return;

	float maximumAccumInv = 1.0f / maximumAccum;
	float minOverlapInv = 1.0f / minOverlap;

	for (int i = 0; i < accum.size(); i++) {
		if (accum[i] * maximumAccumInv > minOverlapInv)
			output[i] = true;
		else
			output[i] = false;
	}
}

void Region::spatialPooling(const std::vector<bool> &inputs, float minPermanence, float minOverlap, int desiredLocalActivity,
	float permanenceIncrease, float permanenceDecrease, float minDutyCycleRatio, float activeDutyCycleDecay,
	float overlapDutyCycleDecay, float subOverlapPermanenceIncrease,
	std::function<float(float, float)> &boostFunction)
{
	_activeColumnIndices.clear();

	size_t connectionSize = 4 * _connectionRadius * _connectionRadius;

	float regionWidthInv = 1.0f / _regionWidth;
	float regionHeightInv = 1.0f / _regionHeight;

	int totalReceptiveFieldSize = 0;

	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		int columnX = i % _regionWidth;
		int columnY = i / _regionWidth;

		float columnXf = static_cast<float>(columnX) * regionWidthInv;
		float columnYf = static_cast<float>(columnY) * regionHeightInv;

		int inputX = static_cast<int>(columnXf * _inputWidth);
		int inputY = static_cast<int>(columnYf * _inputHeight);

		int connectionIndex = 0;

		int receptiveFieldSize = 0;
		
		// Calculate overlap
		float overlap = 0.0f;

		for (int dx = -_connectionRadius; dx <= _connectionRadius; dx++)
		for (int dy = -_connectionRadius; dy <= _connectionRadius; dy++) {
			int connectionX = inputX + dx;
			int connectionY = inputY + dy;

			// If exists
			if (connectionX >= 0 && connectionY >= 0 && connectionX < _inputWidth && connectionY < _inputHeight) {
				column._inputConnections[connectionIndex]._active = false;

				if (column._inputConnections[connectionIndex]._permanence > minPermanence) {
					if (inputs[connectionX + connectionY * _inputWidth]) {
						column._inputConnections[connectionIndex]._active = true;
						overlap++;
					}

					receptiveFieldSize = std::max(receptiveFieldSize, std::max(std::abs(dx), std::abs(dy)));
				}

				connectionIndex++;
			}
		}

		totalReceptiveFieldSize += receptiveFieldSize;

		if (overlap < minOverlap) {
			overlap = 0.0f;
			column._overlapDutyCycle = (1.0f - overlapDutyCycleDecay) * column._overlapDutyCycle;
		}
		else {
			overlap *= column._boost;

			column._overlapDutyCycle = (1.0f - overlapDutyCycleDecay) * column._overlapDutyCycle + overlapDutyCycleDecay;
		}

		column._overlap = overlap;
	}

	float averageReceptiveFieldSize = static_cast<float>(totalReceptiveFieldSize) / _columns.size();

	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		column._active = false;

		if (column._overlap > 0.0f) {
			int columnX = i % _regionWidth;
			int columnY = i / _regionWidth;

			int numHigherThanThis = 0;

			int inhibitionRadius = std::ceil(column._inhibitionRadius);

			// Gather columns in inhibition radius
			for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
			for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
				int inhibitionX = columnX + dx;
				int inhibitionY = columnY + dy;

				// If exists
				if (inhibitionX >= 0 && inhibitionY >= 0 && inhibitionX < _regionWidth && inhibitionY < _regionHeight)
				if (_columns[inhibitionX + inhibitionY * _regionWidth]._overlap > column._overlap)
					numHigherThanThis++;
			}

			if (numHigherThanThis < desiredLocalActivity) {
				column._active = true;

				_activeColumnIndices.push_back(i);
				
				// Update synapses
				for (int j = 0; j < column._inputConnections.size(); j++)
				if (column._inputConnections[j]._active)
					column._inputConnections[j]._permanence = std::min(1.0f, column._inputConnections[j]._permanence + permanenceIncrease);
				else
					column._inputConnections[j]._permanence = std::max(0.0f, column._inputConnections[j]._permanence - permanenceDecrease);
			}
		}
	}

	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		int columnX = i % _regionWidth;
		int columnY = i / _regionWidth;

		float maxNeighborhoodDutyCycle = -999999.0f;

		// Columns in inhibition radius
		int inhibitionRadius = std::ceil(column._inhibitionRadius);

		for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int inhibitionX = columnX + dx;
			int inhibitionY = columnY + dy;

			// If exists
			if (inhibitionX >= 0 && inhibitionY >= 0 && inhibitionX < _regionWidth && inhibitionY < _regionHeight)
				maxNeighborhoodDutyCycle = std::max(maxNeighborhoodDutyCycle, _columns[inhibitionX + inhibitionY * _regionWidth]._activeDutyCycle);
		}

		column._minDutyCycle = minDutyCycleRatio * maxNeighborhoodDutyCycle;

		column._activeDutyCycle = (1.0f - activeDutyCycleDecay) * column._activeDutyCycle + activeDutyCycleDecay * (column._active ? 1.0f : 0.0f);

		column._boost = boostFunction(column._activeDutyCycle, column._minDutyCycle);

		if (column._overlapDutyCycle < column._minDutyCycle) {
			// Increase all permanences
			for (int j = 0; j < column._inputConnections.size(); j++)
				column._inputConnections[j]._permanence += subOverlapPermanenceIncrease * minPermanence;
		}

		column._inhibitionRadius = averageReceptiveFieldSize;
	}
}

void Region::stepBegin() {
	// Update prevs
	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		for (int j = 0; j < column._cells.size(); j++) {
			Cell &cell = column._cells[j];

			cell._prevActiveState = cell._activeState;
			cell._prevLearnState = cell._learnState;
			cell._prevPredictiveState = cell._predictiveState;
			cell._prevNumPredictionSteps = cell._numPredictionSteps;

			for (int k = 0; k < cell._segments.size(); k++) {
				Segment &segment = cell._segments[k];

				segment._prevActiveActivity = segment._activeActivity;
				segment._prevLearnActivity = segment._learnActivity;

				for (std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.begin(); it != segment._connections.end(); it++)
					it->second._prevActive = it->second._active;
			}
		}
	}

	// Clear states and update segment activities
	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		for (int j = 0; j < column._cells.size(); j++) {
			Cell &cell = column._cells[j];

			cell._activeState = false;
			cell._learnState = false;
			cell._predictiveState = false;
			cell._numPredictionSteps = 0;

			for (int k = 0; k < cell._segments.size(); k++) {
				Segment &segment = cell._segments[k];

				segment._activeActivity = 0;
				segment._learnActivity = 0;

				for (std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.begin(); it != segment._connections.end(); it++)
					it->second._active = false;
			}
		}
	}
}

void Region::temporalPoolingNoLearn(float minPermanence, int activationThreshold) {
	for (int a = 0; a < _activeColumnIndices.size(); a++) {
		int i = _activeColumnIndices[a];

		Column &column = _columns[i];

		int columnX = i % _regionWidth;
		int columnY = i / _regionWidth;

		bool bottomUpPredicted = false;
		bool learningCellChosen = false;

		for (int j = 0; j < column._cells.size(); j++) {
			Cell &cell = column._cells[j];

			if (cell._prevPredictiveState) {
				Segment* pActiveSegment = nullptr;

				// Find active segment (this is an OR operation, so can stop as soon as find one, except when it is not a sequence segment)
				for (int k = 0; k < cell._segments.size(); k++) {
					Segment &segment = cell._segments[k];

					if (segment._numPredictionSteps != 1)
						continue;

					if (segment._prevActiveActivity > activationThreshold) {
						if (pActiveSegment == nullptr)
							pActiveSegment = &segment;
						else if (pActiveSegment->_sequenceSegment) {
							if (segment._sequenceSegment && pActiveSegment->_prevActiveActivity < segment._prevActiveActivity)
								pActiveSegment = &segment;
						}
						else {
							if (pActiveSegment->_prevActiveActivity < segment._prevActiveActivity)
								pActiveSegment = &segment;
						}
					}
				}

				if (pActiveSegment != nullptr && pActiveSegment->_sequenceSegment) {
					bottomUpPredicted = true;

					cell._activeState = true;
				}
			}
		}

		if (!bottomUpPredicted) {
			for (int j = 0; j < column._cells.size(); j++) {
				Cell &cell = column._cells[j];

				cell._activeState = true;
			}
		}
	}

	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		for (int j = 0; j < column._cells.size(); j++) {
			Cell &cell = column._cells[j];

			for (int k = 0; k < cell._segments.size(); k++) {
				Segment &segment = cell._segments[k];

				segment._activeActivity = 0;
				segment._learnActivity = 0;

				for (std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.begin(); it != segment._connections.end(); it++) {
					it->second._active = false;

					if (it->second._permanence > minPermanence) {
						if (_columns[it->first._columnIndex]._cells[it->first._cellIndex]._activeState) {
							it->second._active = true;

							segment._activeActivity++;
						}
					}

					if (_columns[it->first._columnIndex]._cells[it->first._cellIndex]._activeState && _columns[it->first._columnIndex]._cells[it->first._cellIndex]._learnState) {
						//it->second._active = true;

						segment._learnActivity++;
					}
				}

				if (segment._activeActivity > activationThreshold) {
					if (!cell._predictiveState)
						cell._numPredictionSteps = segment._numPredictionSteps;
					else
						cell._numPredictionSteps = std::min(cell._numPredictionSteps, segment._numPredictionSteps);

					cell._predictiveState = true;
				}
			}
		}
	}
}

void Region::getBestMatchingCell(int columnIndex, int &cellIndex, int &segmentIndex, int predictionSteps, bool usePrevious, std::mt19937 &generator) {
	Column &column = _columns[columnIndex];

	// Go through cells and see which one has the best matching segment
	int maxActiveConnections = 0;
	cellIndex = -1;
	segmentIndex = -1;

	for (int j = 0; j < column._cells.size(); j++) {
		Cell &cell = column._cells[j];

		int maxSegmentIndex;

		getBestMatchingSegment(columnIndex, j, maxSegmentIndex, predictionSteps, usePrevious);

		if (maxSegmentIndex != -1) {
			int activeCount = usePrevious ? cell._segments[maxSegmentIndex]._prevActiveActivity : cell._segments[maxSegmentIndex]._activeActivity;

			if (activeCount > maxActiveConnections) {
				cellIndex = j;
				segmentIndex = maxSegmentIndex;
				maxActiveConnections = activeCount;
			}
		}
	}

	if (cellIndex == -1) {
		int minSegmentsCellIndex = 0;

		int numSame = 0;

		for (int j = 1; j < column._cells.size(); j++) {
			Cell &cell = column._cells[j];

			if (cell._segments.size() < column._cells[minSegmentsCellIndex]._segments.size()) {
				numSame = 1;
				minSegmentsCellIndex = j;
			}
			else if (cell._segments.size() == column._cells[minSegmentsCellIndex]._segments.size()) {
				numSame++;

				std::uniform_int_distribution<int> sameDist(0, numSame - 1);

				if (sameDist(generator) == 0)
					minSegmentsCellIndex = j;
			}
		}

		cellIndex = minSegmentsCellIndex;

		segmentIndex = -1;
	}
}

void Region::getBestMatchingSegment(int columnIndex, int cellIndex, int &segmentIndex, int predictionSteps, bool usePrevious) {
	Cell &cell = _columns[columnIndex]._cells[cellIndex];

	segmentIndex = -1;

	int maxActivity = 0;

	for (int k = 0; k < cell._segments.size(); k++) {
		Segment &segment = cell._segments[k];

		if (segment._numPredictionSteps != predictionSteps)
			continue;

		if (usePrevious) {
			if (segment._prevActiveActivity >= maxActivity) {
				segmentIndex = k;

				maxActivity = segment._prevActiveActivity;
			}
		}
		else {
			if (segment._activeActivity >= maxActivity) {
				segmentIndex = k;

				maxActivity = segment._prevActiveActivity;
			}
		}
	}
}

void Region::updateSegmentActiveSynapses(int columnIndex, int cellIndex, int segmentIndex, bool usePrevious, int numConnections, int learningRadius, SegmentUpdateType updateType, SegmentUpdate &segmentUpdate, std::mt19937 &generator) {
	Column &column = _columns[columnIndex];

	int columnX = columnIndex % _regionWidth;
	int columnY = columnIndex / _regionWidth;

	segmentUpdate._columnIndex = columnIndex;
	segmentUpdate._cellIndex = cellIndex;
	segmentUpdate._segmentIndex = segmentIndex;
	segmentUpdate._updateType = updateType;
	segmentUpdate._numPredictionSteps = 1; // Means is sequence segment

	if (segmentUpdate._segmentIndex != -1) {
		Segment &segment = column._cells[cellIndex]._segments[segmentIndex];

		for (std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.begin(); it != segment._connections.end(); it++)
		if (usePrevious) {
			if (it->second._prevActive)
				segmentUpdate._activeConnectionIndices.push_back(it->first);
			else
				segmentUpdate._inactiveConnectionIndices.push_back(it->first);
		}
		else {
			if (it->second._active)
				segmentUpdate._activeConnectionIndices.push_back(it->first);
			else
				segmentUpdate._inactiveConnectionIndices.push_back(it->first);
		}

		int numConnectionsAdd = numConnections - static_cast<int>(segment._connections.size());

		if (numConnectionsAdd > 0) {
			std::vector<ColumnAndCellIndices> availableCells;

			for (int dx = -learningRadius; dx <= learningRadius; dx++)
			for (int dy = -learningRadius; dy <= learningRadius; dy++) {
				int learningX = columnX + dx;
				int learningY = columnY + dy;

				// If exists
				if (learningX >= 0 && learningY >= 0 && learningX < _regionWidth && learningY < _regionHeight) {
					int neighborColumnIndex = learningX + learningY * _regionWidth;
					Column &neighborColumn = _columns[neighborColumnIndex];

					for (int neighborCellIndex = 0; neighborCellIndex < neighborColumn._cells.size(); neighborCellIndex++) {
						if (neighborColumnIndex == columnIndex && neighborCellIndex == cellIndex)
							continue;

						if (segment._connections.find(ColumnAndCellIndices(neighborColumnIndex, neighborCellIndex)) != segment._connections.end())
							continue;

						if (neighborColumn._cells[neighborCellIndex]._prevLearnState)
							availableCells.push_back(ColumnAndCellIndices(neighborColumnIndex, neighborCellIndex));
					}
				}
			}

			std::shuffle(availableCells.begin(), availableCells.end(), generator);

			int selectIndex = 0;

			while (numConnectionsAdd > 0 && selectIndex < availableCells.size()) {
				// Select random cell to connect to from available list
				segmentUpdate._activeConnectionIndices.push_back(availableCells[selectIndex]);

				numConnectionsAdd--;
				selectIndex++;
			}
		}
	}
	else {
		int numConnectionsAdd = numConnections;

		std::vector<ColumnAndCellIndices> availableCells;

		for (int dx = -learningRadius; dx <= learningRadius; dx++)
		for (int dy = -learningRadius; dy <= learningRadius; dy++) {
			int learningX = columnX + dx;
			int learningY = columnY + dy;

			// If exists
			if (learningX >= 0 && learningY >= 0 && learningX < _regionWidth && learningY < _regionHeight) {
				int neighborColumnIndex = learningX + learningY * _regionWidth;
				Column &neighborColumn = _columns[neighborColumnIndex];

				for (int neighborCellIndex = 0; neighborCellIndex < neighborColumn._cells.size(); neighborCellIndex++) {
					if (neighborColumnIndex == columnIndex && neighborCellIndex == cellIndex)
						continue;

					if (neighborColumn._cells[neighborCellIndex]._prevLearnState)
						availableCells.push_back(ColumnAndCellIndices(neighborColumnIndex, neighborCellIndex));
				}
			}
		}

		std::shuffle(availableCells.begin(), availableCells.end(), generator);

		int selectIndex = 0;

		while (numConnectionsAdd > 0 && selectIndex < availableCells.size()) {
			// Select random cell to connect to from available list
			segmentUpdate._activeConnectionIndices.push_back(availableCells[selectIndex]);

			numConnectionsAdd--;
			selectIndex++;
		}
	}
}

void Region::temporalPoolingLearn(float minPermanence, int learningRadius, int minLearningThreshold, int activationThreshold, int newNumConnections, float permanenceIncrease, float permanenceDecrease, float newConnectionPermanence, int maxSteps, std::mt19937 &generator) {
	// Phase 1
	for (int a = 0; a < _activeColumnIndices.size(); a++) {
		int i = _activeColumnIndices[a];

		Column &column = _columns[i];

		int columnX = i % _regionWidth;
		int columnY = i / _regionWidth;

		bool bottomUpPredicted = false;
		bool learningCellChosen = false;

		for (int j = 0; j < column._cells.size(); j++) {
			Cell &cell = column._cells[j];

			if (cell._prevPredictiveState) {
				Segment* pActiveSegment = nullptr;

				// Find active segment (this is an OR operation, so can stop as soon as find one, except when it is not a sequence segment)
				for (int k = 0; k < cell._segments.size(); k++) {
					Segment &segment = cell._segments[k];

					if (segment._prevActiveActivity > activationThreshold) {
						if (pActiveSegment == nullptr)
							pActiveSegment = &segment;
						else if (pActiveSegment->_sequenceSegment) {
							if (segment._sequenceSegment && pActiveSegment->_prevActiveActivity < segment._prevActiveActivity)
								pActiveSegment = &segment;
						}
						else {
							if (pActiveSegment->_prevActiveActivity < segment._prevActiveActivity)
								pActiveSegment = &segment;
						}
					}
				}

				if (pActiveSegment != nullptr && pActiveSegment->_sequenceSegment) {
					bottomUpPredicted = true;

					cell._activeState = true;

					if (pActiveSegment->_prevLearnActivity > activationThreshold) {
						learningCellChosen = true;

						cell._learnState = true;
					}
				}
			}
		}

		if (!bottomUpPredicted) {
			for (int j = 0; j < column._cells.size(); j++) {
				Cell &cell = column._cells[j];

				cell._activeState = true;
			}
		}

		if (!learningCellChosen) {
			int cellIndex;
			int segmentIndex;

			getBestMatchingCell(i, cellIndex, segmentIndex, 1, true, generator);

			column._cells[cellIndex]._learnState = true;

			SegmentUpdate segmentUpdate;

			updateSegmentActiveSynapses(i, cellIndex, segmentIndex, true, newNumConnections, learningRadius, _dueToActive, segmentUpdate, generator);

			segmentUpdate._numPredictionSteps = 1;

			column._cells[segmentUpdate._cellIndex]._segmentUpdates.push_back(segmentUpdate);
		}
	}

	// Phase 2
	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		int columnX = i % _regionWidth;
		int columnY = i / _regionWidth;

		for (int j = 0; j < column._cells.size(); j++) {
			Cell &cell = column._cells[j];

			for (int k = 0; k < cell._segments.size(); k++) {
				Segment &segment = cell._segments[k];

				segment._activeActivity = 0;
				segment._learnActivity = 0;

				for (std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.begin(); it != segment._connections.end(); it++) {
					//it->second._active = false;

					if (it->second._permanence > minPermanence) {
						if (_columns[it->first._columnIndex]._cells[it->first._cellIndex]._activeState) {
							it->second._active = true;

							segment._activeActivity++;
						}
					}

					if (_columns[it->first._columnIndex]._cells[it->first._cellIndex]._activeState && _columns[it->first._columnIndex]._cells[it->first._cellIndex]._learnState) {
						//it->second._active = true;

						segment._learnActivity++;
					}
				}

				if (segment._activeActivity > activationThreshold) {
					if (!cell._predictiveState)
						cell._numPredictionSteps = segment._numPredictionSteps;
					else
						cell._numPredictionSteps = std::min(cell._numPredictionSteps, segment._numPredictionSteps);

					cell._predictiveState = true;

					SegmentUpdate segmentUpdate;

					updateSegmentActiveSynapses(i, j, k, false, newNumConnections, learningRadius, _dueToPredictive, segmentUpdate, generator);

					cell._segmentUpdates.push_back(segmentUpdate);
				}
			}

			if (cell._predictiveState && cell._numPredictionSteps != maxSteps) {
				int segmentIndex;

				getBestMatchingSegment(i, j, segmentIndex, cell._numPredictionSteps + 1, true);

				SegmentUpdate segmentUpdate;

				updateSegmentActiveSynapses(i, j, segmentIndex, true, newNumConnections, learningRadius, _dueToPredictive, segmentUpdate, generator);

				if (segmentIndex == -1)
					segmentUpdate._numPredictionSteps = cell._numPredictionSteps + 1;

				cell._segmentUpdates.push_back(segmentUpdate);
			}
		}
	}

	// Phase 3
	for (int i = 0; i < _columns.size(); i++) {
		Column &column = _columns[i];

		int columnX = i % _regionWidth;
		int columnY = i / _regionWidth;

		for (int j = 0; j < column._cells.size(); j++) {
			Cell &cell = column._cells[j];

			std::vector<int> modifiedSegmentIndices;
			std::unordered_set<int> modifiedSegmentIndicesSet;

			std::vector<SegmentUpdate> keepUpdates;

			if (cell._learnState) {
				for (int s = 0; s < cell._segmentUpdates.size(); s++) {
					SegmentUpdate &segmentUpdate = cell._segmentUpdates[s];

					if (segmentUpdate._isNew && segmentUpdate._updateType == _dueToPredictive) {
						segmentUpdate._isNew = false;
						keepUpdates.push_back(segmentUpdate);
						continue;
					}

					segmentUpdate._isNew = false;

					if (segmentUpdate._segmentIndex == -1) {
						if (segmentUpdate._activeConnectionIndices.size() > activationThreshold) {
							cell._segments.push_back(Segment());

							Segment &segment = cell._segments.back();

							for (int c = 0; c < segmentUpdate._activeConnectionIndices.size(); c++) {
								// Create new connection
								Connection &connection = segment._connections[segmentUpdate._activeConnectionIndices[c]];

								connection._permanence = newConnectionPermanence;
							}

							segment._numPredictionSteps = std::max(1, segmentUpdate._numPredictionSteps);

							if (segment._numPredictionSteps == 1)
								segment._sequenceSegment = true;

							int modIndex = cell._segments.size() - 1;

							if (modifiedSegmentIndicesSet.find(modIndex) == modifiedSegmentIndicesSet.end()) {
								modifiedSegmentIndices.push_back(modIndex);
								modifiedSegmentIndicesSet.insert(modIndex);
							}
						}
						else
							keepUpdates.push_back(segmentUpdate);
					}
					else {
						Segment &segment = cell._segments[segmentUpdate._segmentIndex];

						for (int c = 0; c < segmentUpdate._activeConnectionIndices.size(); c++) {
							std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.find(segmentUpdate._activeConnectionIndices[c]);

							if (it == segment._connections.end()) {
								// Create new connection
								Connection &connection = segment._connections[segmentUpdate._activeConnectionIndices[c]];

								connection._permanence = newConnectionPermanence;
							}
							else {
								Connection &connection = it->second;

								connection._permanence = std::min(1.0f, connection._permanence + permanenceIncrease);
							}
						}

						for (int c = 0; c < segmentUpdate._inactiveConnectionIndices.size(); c++) {
							Connection &connection = segment._connections[segmentUpdate._inactiveConnectionIndices[c]];

							connection._permanence = std::max(0.0f, connection._permanence - permanenceDecrease);
						}

						int modIndex = segmentUpdate._segmentIndex;

						if (modifiedSegmentIndicesSet.find(modIndex) == modifiedSegmentIndicesSet.end()) {
							modifiedSegmentIndices.push_back(modIndex);
							modifiedSegmentIndicesSet.insert(modIndex);
						}
					}
				}
			}
			else if (!cell._predictiveState && cell._prevPredictiveState) {
				for (int s = 0; s < cell._segmentUpdates.size(); s++) {
					SegmentUpdate &segmentUpdate = cell._segmentUpdates[s];

					if (segmentUpdate._isNew && segmentUpdate._updateType == _dueToPredictive) {
						segmentUpdate._isNew = false;
						keepUpdates.push_back(segmentUpdate);
						continue;
					}

					segmentUpdate._isNew = false;

					if (segmentUpdate._segmentIndex != -1) {
						Segment &segment = cell._segments[segmentUpdate._segmentIndex];

						for (int c = 0; c < segmentUpdate._activeConnectionIndices.size(); c++) {
							std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.find(segmentUpdate._activeConnectionIndices[c]);

							if (it == segment._connections.end()) {
								// Create new connection
								Connection &connection = segment._connections[segmentUpdate._activeConnectionIndices[c]];

								connection._permanence = minPermanence;
							}
							else {
								Connection &connection = it->second;

								connection._permanence = std::max(0.0f, connection._permanence - permanenceDecrease);
							}
						}

						int modIndex = segmentUpdate._segmentIndex;

						if (modifiedSegmentIndicesSet.find(modIndex) == modifiedSegmentIndicesSet.end()) {
							modifiedSegmentIndices.push_back(modIndex);
							modifiedSegmentIndicesSet.insert(modIndex);
						}	
					}
					else
						keepUpdates.push_back(segmentUpdate);
				}
			}
			else if (cell._predictiveState && cell._prevPredictiveState && cell._numPredictionSteps > 1 && cell._prevNumPredictionSteps == 1) {
				for (int s = 0; s < cell._segmentUpdates.size(); s++) {
					SegmentUpdate &segmentUpdate = cell._segmentUpdates[s];

					if (segmentUpdate._isNew && segmentUpdate._updateType == _dueToPredictive) {
						segmentUpdate._isNew = false;
						keepUpdates.push_back(segmentUpdate);
						continue;
					}

					segmentUpdate._isNew = false;

					if (segmentUpdate._numPredictionSteps <= 1) {
						if (segmentUpdate._segmentIndex != -1) {
							Segment &segment = cell._segments[segmentUpdate._segmentIndex];

							for (int c = 0; c < segmentUpdate._activeConnectionIndices.size(); c++) {
								std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.find(segmentUpdate._activeConnectionIndices[c]);

								if (it == segment._connections.end()) {
									// Create new connection
									Connection &connection = segment._connections[segmentUpdate._activeConnectionIndices[c]];

									connection._permanence = newConnectionPermanence;
								}
								else {
									Connection &connection = it->second;

									connection._permanence = std::max(0.0f, connection._permanence - permanenceDecrease);
								}
							}

							int modIndex = segmentUpdate._segmentIndex;

							if (modifiedSegmentIndicesSet.find(modIndex) == modifiedSegmentIndicesSet.end()) {
								modifiedSegmentIndices.push_back(modIndex);
								modifiedSegmentIndicesSet.insert(modIndex);
							}	
						}
						else
							keepUpdates.push_back(segmentUpdate);
					}
					else
						keepUpdates.push_back(segmentUpdate);
				}
			}
			else {
				for (int s = 0; s < cell._segmentUpdates.size(); s++) {
					SegmentUpdate &segmentUpdate = cell._segmentUpdates[s];

					segmentUpdate._isNew = false;

					keepUpdates.push_back(segmentUpdate);
				}
			}

			cell._segmentUpdates = keepUpdates;

			if (cell._segmentUpdates.empty())
			for (int m = 0; m < modifiedSegmentIndices.size(); m++) {
				int s = modifiedSegmentIndices[m];

				Segment &segment = cell._segments[s];

				for (std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices>::iterator it = segment._connections.begin(); it != segment._connections.end();) {
					if (it->second._permanence == 0.0f)
						it = segment._connections.erase(it);
					else
						it++;
				}

				if (segment._connections.empty()) {
					cell._segments.erase(cell._segments.begin() + s);

					// Shift indices
					for (int n = m; n < modifiedSegmentIndices.size(); n++)
					if (modifiedSegmentIndices[n] > s)
						modifiedSegmentIndices[n]--;
				}
			}
		}
	}
}