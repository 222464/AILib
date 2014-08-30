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

#include <Eigen/Dense>
#include <Eigen/LU>
#include <vector>
#include <list>

namespace apssc {
	class APSSC {
	private:
		struct Ant {
			Eigen::VectorXf _position;

			Eigen::VectorXf _pheromones; // Size is number of labels

			int _label; // -1 means no label

			Ant()
				: _label(-1)
			{}
		};

		struct Colony {
			std::vector<int> _antIndices;

			Eigen::VectorXf _colonyCenter;
		};

		std::vector<Ant> _allAnts;
		std::vector<int> _labeledAntIndices;
		std::vector<int> _unlabeledAntIndices;
		std::vector<Colony> _colonies;

		size_t _dimensions;
		size_t _numLabels;

		size_t _t;

	public:
		APSSC()
			: _t(0)
		{}

		void create(size_t numLabels, size_t dimensions, size_t numAnts);

		size_t getNumAnts() const {
			return _allAnts.size();
		}

		size_t getNumLabeledAnts() const {
			return _labeledAntIndices.size();
		}

		size_t getNumUnlabeledAnts() const {
			return _unlabeledAntIndices.size();
		}

		void setAntPosition(size_t index, const std::vector<float> &position) {
			for (size_t i = 0; i < _dimensions; i++)
				_allAnts[index]._position(i) = position[i];
		}

		void setAntLabel(size_t index, int label) {
			_allAnts[index]._label = label;
		}

		int getAntLabel(size_t index) const {
			return _allAnts[index]._label;
		}

		void prepare();

		void trainingCycle(float averageDecay, float pheromoneDecay);
		int classify(const std::vector<float> &position);
	};
}