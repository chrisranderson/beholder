// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

function generateRandomMatrix(): number[] {
  const rows = getRandomArbitrary(10, 20);
  const columns = getRandomArbitrary(10, 20);
  let data = [];
  // Generate new data array.
  for (let row = 0; row < rows; row++) {
    let currentRow = [];
    for (let col = 0; col < columns; col++) {
      currentRow[col] = getRandomArbitrary(0, 20);
    }
    data[row] = currentRow;
  }
  return data;
}

// Returns a random number between min (inclusive) and max (exclusive)
function getRandomArbitrary(min, max: number): number {
  return Math.random() * (max - min) + min;
}

function getRandomColorRange(): string[] {
  return [getRandomColor(), getRandomColor()];
}

function getRandomColor(): string {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}
