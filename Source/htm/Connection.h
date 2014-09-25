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

namespace htm {
	struct ColumnAndCellIndices {
		int _columnIndex;
		int _cellIndex;

		ColumnAndCellIndices()
		{}

		ColumnAndCellIndices(int columnIndex, int cellIndex)
			: _columnIndex(columnIndex), _cellIndex(cellIndex)
		{}

		size_t operator()(const ColumnAndCellIndices &value) const {
			return static_cast<size_t>(_columnIndex ^ _cellIndex);
		}

		bool operator==(const ColumnAndCellIndices &other) const {
			return _columnIndex == other._columnIndex && _cellIndex == other._cellIndex;
		}
	};

	class Connection {
	private:
		float _permanence;

		bool _active;
		bool _prevActive;

	public:
		Connection()
			: _active(false), _prevActive(false)
		{}

		float getPermanence() const {
			return _permanence;
		}

		bool isActive() const {
			return _active;
		}

		friend class Cell;
		friend class Column;
		friend class Region;
	};
}