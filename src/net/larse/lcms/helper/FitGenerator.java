/*
 * Copyright (c) 2015 Zhiqiang Yang.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

package net.larse.lcms.helper;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.interfaces.linsol.LinearSolver;

/**
 * A wrapper for OLS fitting.
 */
public class FitGenerator {
  private int numCols;
  private int numRows;

  private DenseMatrix64F matrixA;
  private DenseMatrix64F matrixB;

  public void init(int numCols, int numRows) {
    this.numCols = numCols;
    this.numRows = numRows;
    matrixA = new DenseMatrix64F(numRows, numCols);
    matrixB = new DenseMatrix64F(numRows, 1);
  }

  public void setObservation(int idx, int feature, double value) {
    matrixA.set(idx, feature, value);
  }

  public void setTarget(int idx, double target) {
    matrixB.set(idx, 0, target);
  }

  public boolean isLinear() { return true; }

  public double[] linearFit() {
    DenseMatrix64F matrixX = new DenseMatrix64F(numCols, 1);
    LinearSolver<DenseMatrix64F> solver =
        LinearSolverFactory.leastSquares(matrixA.getNumRows(), matrixA.getNumCols());
    solver.setA(matrixA);
    if (solver.quality() != 0) {
      solver.solve(matrixB, matrixX);
    }
    // TODO(gorelick): What should we return when the matrix isn't solvable?
    return matrixX.getData();
  }
}
