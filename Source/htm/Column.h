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

#include <htm/Cell.h>
#include <vector>

namespace htm {
	class Column {
	private:
		std::vector<Cell> _cells;
		std::vector<Connection> _inputConnections;

		float _boost;
		float _overlap;

		bool _active;

		float _activeDutyCycle;
		float _overlapDutyCycle;
		float _minDutyCycle;

		float _inhibitionRadius;

	public:
		Column()
			: _boost(1.0f), _overlap(0.0f), _active(false), _activeDutyCycle(0.0f),
			_minDutyCycle(0.0f), _overlapDutyCycle(0.0f), _inhibitionRadius(0.0f)
		{}

		bool isActive() const {
			return _active;
		}

		const Cell &getCell(int j) const {
			return _cells[j];
		}

		int getNumCells() const {
			return _cells.size();
		}

		friend class Region;
	};
}