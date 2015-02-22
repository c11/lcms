/*
 * Copyright (c) 2015 Google, Inc.
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

package com.google.earthengine.lib.common;

import com.google.common.base.Preconditions;
import com.google.earthengine.api.task.SizeOf;

import org.ejml.alg.dense.linsol.LinearSolver;
import org.ejml.alg.dense.linsol.LinearSolverFactory;
import org.ejml.data.DenseMatrix64F;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Computes a multivariate linear regression via ordinary least squares.
 */
public class LinearLeastSquares implements Serializable, SizeOf.Measurable {
  private static final long serialVersionUID = 1;

  private static final long OBJ_SIZE =
      SizeOf.object(SizeOf.BOOLEAN + 6 * SizeOf.PTR + 3 * SizeOf.INT);

  // The following values were based on cursory examination of the EJML source.
  // They may not be exactly right (and may become less right after future
  // revisions of that code), but they should be close enough to make the
  // results useful.
  private static final long DENSE_MATRIX_OBJ_SIZE =
      SizeOf.object(2 * SizeOf.INT + SizeOf.PTR);
  private static final long CHOLESKY_DECOMPOSITION_OBJ_SIZE =
      SizeOf.object(2 * SizeOf.INT + 4 * SizeOf.PTR);
  private static final long LINEAR_SOLVER_OBJ_SIZE =
      SizeOf.object(3 * SizeOf.INT + 4 * SizeOf.PTR);

  private static long linearSolverHeapSize(int x) {
    return LINEAR_SOLVER_OBJ_SIZE
           + CHOLESKY_DECOMPOSITION_OBJ_SIZE
           + SizeOf.array(x * SizeOf.DOUBLE);
  }

  /**
   * Returns an (approximate) upper bound on the heap size of a
   * LinearLeastSquares instance with the given dimensions.  The
   * full memory consumption only occurs when getSolution() is
   * called; if you have only called addInput() the heap size
   * will be lower.
   */
  public static long heapSize(int numX, int numY) {
    // Compare with the heapSize() instance method.
    return OBJ_SIZE
           + SizeOf.array(SizeOf.DOUBLE * numX * (numX + 1) / 2)
           + SizeOf.array(SizeOf.DOUBLE * numY * numX)
           + SizeOf.array(SizeOf.DOUBLE * numY)
           + SizeOf.denseMatrix64F(numX, numX)
           + DENSE_MATRIX_OBJ_SIZE
           + linearSolverHeapSize(numX);
  }

  public final int numX;
  public final int numY;

  private int numInputs;
  // the upper-right elements of xMat (skipping (0, 0), which is numInputs)
  private final double[] xSums;
  // the elements of yMat
  private final double[] ySums;
  // the sums of y_i^2
  private final double[] y2Sums;

  // true if solver has been successfully initialized from xMat
  private transient boolean solved;

  private transient DenseMatrix64F xMat;
  private transient DenseMatrix64F yMat;
  private transient LinearSolver<DenseMatrix64F> solver;

  /**
   * Creates a solver to compute a linear least squares regression
   * with numX independent variables and numY dependent variables.
   *
   * <p>To use the solver, call addInput() at least numX times, and
   * then call getSolution() (and, optionally, getRmsResiduals()).
   *
   * <p>You may add additional inputs after calling getSolution() and get
   * an updated solution.  You may also add the the state of one solver
   * to another (assuming that they have the same values for numX and numY),
   * which is equivalent to adding each of the inputs of the second solver
   * to the first.
   */
  public LinearLeastSquares(int numX, int numY) {
    Preconditions.checkArgument(numX >= 1 && numY >= 1);
    this.numX = numX;
    this.numY = numY;
    this.xSums = new double[numX * (numX + 1) / 2];
    this.ySums = new double[numY * numX];
    this.y2Sums = new double[numY];
  }

  @Override
  public long heapSize() {
    return OBJ_SIZE
           + SizeOf.array(xSums) + SizeOf.array(ySums) + SizeOf.array(y2Sums)
           + (xMat == null ? 0 : SizeOf.denseMatrix64F(numX, numX))
           + (yMat == null ? 0 : DENSE_MATRIX_OBJ_SIZE)
           + (solver == null ? 0 : linearSolverHeapSize(numX));
  }

  // If the input is written as two matrices, with each
  // row corresponding to an observation:
  //   X = [ x0 x1 x2 ... ]     Y = [ y0 y1 y2 ... ]
  //       [ x0 x1 x2 ... ]         [ y0 y1 y2 ... ]
  //       [ x0 x1 x2 ... ]         [ y0 y1 y2 ... ]
  //            ...                      ...
  // then we define
  //    xMat = transpose(X) * X
  //    yMat = transpose(X) * Y
  //
  // Note that xMat has dimensions (numX, numX) and yMat has dimensions
  // (numX, numY).
  //
  // To compute the regression coefficients R we just solve xMat * R = yMat.
  //
  // We can build xMat and yMat incrementally, without storing X and Y:
  //   xMat[i, j] = sum(input[i] * input[j])
  //   yMat[i, j] = sum(input[i] * input[j + numX])
  // where the first numX elements of input are the X variables, and the final
  // numY elements of input are the Y values.
  //
  // For example, if numX = 3 and numY = 2 we get
  // xMat = [ sum(x0^2)   sum(x0*x1)  sum(x0*x2) ]
  //        [ sum(x0*x1)  sum(x1^2)   sum(x1*x2) ]
  //        [ sum(x0*x2)  sum(x1*x2)  sum(x2^2)  ]
  // and
  // yMat = [ sum(x0*y0)  sum(x0*y1) ]
  //        [ sum(x1*y0)  sum(x1*y1) ]
  //        [ sum(x2*y0)  sum(x2*y1) ]
  //
  // Since xMat is symmetric, we only need to store the upper right triangle
  // while we're accumulating inputs.
  //
  // We compute residual k with the formula
  //    Math.sqrt((sum(y_k^2) - dotProd(R[*, k], yMat[*, k])) / n)

  /**
   * Add one observation, using numX values from x starting with xStart and
   * numY values from y starting at yStart.
   */
  public void addInput(double[] x, int xStart, double[] y, int yStart) {
    ++numInputs;
    // update xSums
    int pos = 0;
    for (int i = 0; i < numX; ++i) {
      double xi = x[xStart + i];
      for (int i2 = 0; i2 <= i; ++i2) {
        xSums[pos++] += xi * x[xStart + i2];
      }
    }
    assert pos == xSums.length;
    // update y2Sums
    for (int j = 0; j < numY; ++j) {
      double yj = y[yStart + j];
      y2Sums[j] += yj * yj;
    }
    // update ySums
    pos = 0;
    for (int i = 0; i < numX; ++i) {
      double xi = x[xStart + i];
      for (int j = 0; j < numY; ++j) {
        double yj = y[yStart + j];
        ySums[pos++] += xi * yj;
      }
    }
    assert pos == ySums.length;
    solved = false;
  }

  /**
   * Compute results from the accumulated state.  Returns false if there were
   * not enough inputs.  Returns true if it was successful, and sets results
   * to have numX rows and numY columns, where each column contains the
   * coefficients for the corresponding dependent variable.
   */
  public boolean getSolution(DenseMatrix64F results) {
    if (numInputs < numX) {
      // not enough inputs, no point in trying
      return false;
    }
    if (xMat == null) {
      xMat = new DenseMatrix64F(numX, numX);
      // yMat can just point at the ySums array without copying
      yMat = DenseMatrix64F.wrap(numX, numY, ySums);
      solver = LinearSolverFactory.symmPosDef(numX);
    }
    // populate xMat from xSums (and numInputs)
    int pos = 0;
    for (int i = 0; i < numX; ++i) {
      for (int i2 = 0; i2 <= i; ++i2) {
        double sum = xSums[pos++];
        xMat.unsafe_set(i, i2, sum);
        if (i != i2) {
          xMat.unsafe_set(i2, i, sum);
        }
      }
    }
    assert pos == xSums.length;
    // setup the solver, and make sure the matrix is invertible
    if (solver.setA(xMat) && solver.quality() > 0) {
      results.reshape(numX, numY, false);
      // since yMat aliases the ySums array, it's ready to go
      solver.solve(yMat, results);
      solved = true;
    }
    return solved;
  }

  /**
   * Compute the square root of the mean squared residual for each dependent
   * variable.  May only be called after a successful call to getSolution(),
   * and must be given the (unmodified) results of that call and an array of
   * the correct size in which to store the residuals.
   */
  public void getRmsResiduals(DenseMatrix64F results, double[] residuals) {
    Preconditions.checkState(solved);
    Preconditions.checkArgument(residuals.length == numY
        && results.getNumRows() == numX
        && results.getNumCols() == numY);
    for (int i = 0; i < numY; ++i) {
      double sumSq = y2Sums[i];
      for (int j = 0; j < numX; ++j) {
        sumSq -= results.unsafe_get(j, i) * yMat.unsafe_get(j, i);
      }
      // due to roundoff, sumSq could end up slightly negative
      residuals[i] = (sumSq <= 0) ? 0 : Math.sqrt(sumSq / numInputs);
    }
  }

  /**
   * Reset the solver to its no-inputs state.
   */
  public void reset() {
    numInputs = 0;
    Arrays.fill(xSums, 0);
    Arrays.fill(ySums, 0);
    Arrays.fill(y2Sums, 0);
    solved = false;
  }

  /**
   * Add each element of src to the corresponding element of dest.
   */
  private static void add(double[] dest, double[] src) {
    for (int i = 0; i < src.length; ++i) {
      dest[i] += src[i];
    }
  }

  /**
   * Add all the inputs of another solver to this solver.  Does not change
   * the state of the other solver.
   */
  public void addInputsOf(LinearLeastSquares other) {
    Preconditions.checkArgument(other.numX == numX && other.numY == numY);
    numInputs += other.numInputs;
    add(xSums, other.xSums);
    add(ySums, other.ySums);
    add(y2Sums, other.y2Sums);
    solved = false;
  }
}
