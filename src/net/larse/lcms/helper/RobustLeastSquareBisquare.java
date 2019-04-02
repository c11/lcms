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

import java.util.Objects;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math.stat.descriptive.moment.StandardDeviation;
import org.ejml.alg.dense.decomposition.qr.QRColPivDecompositionHouseholderColumn_D64;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.interfaces.linsol.LinearSolver;
import org.ejml.ops.CommonOps;
import org.ejml.ops.MatrixFeatures;

import java.util.Arrays;

/**
 * Version of RobustLeastSquares that implements bisquare error function.
 * This is intended to exactly match the matlab statrobustfit_cor function.
 */
public class RobustLeastSquareBisquare {
  // The max number of iterations.
  private static final int MAX_ITERATIONS = 5;

  protected DenseMatrix64F a;
  protected DenseMatrix64F b;
  private double beta;

  public RobustLeastSquareBisquare(DenseMatrix64F a, DenseMatrix64F b, double beta) {
    this.a = a.copy();
    this.b = b.copy();
    this.beta = beta;
  }

  public void updateB(DenseMatrix64F b) {
    this.b = b;
  }

  public boolean getSolution(DenseMatrix64F results) {
    assert(a.numRows == b.numRows && a.numRows >= a.numCols);

    // Seed the search with the ordinary least squares solution.
    LinearSolver<DenseMatrix64F> solver = LinearSolverFactory.leastSquares(a.numRows, a.numCols);
    if (!solver.setA(a)) {
      return false;
    }

    DenseMatrix64F x = new DenseMatrix64F(a.numCols, b.numCols);
    solver.solve(b, x);

    /* Matlab code
    % Find the least squares solution.
    [Q,R,perm] = qr(X,0);
    tol = abs(R(1)) * max(n,p) * eps(class(R));
    xrank = sum(abs(diag(R)) > tol);
    if xrank==p
    b(perm,:) = R \ (Q'*y);
      else
    % Use only the non-degenerate parts of R and Q, but don't reduce
        % R because it is returned in stats and is expected to be of
    % full size.
      b(perm,:) = [R(1:xrank,1:xrank) \ (Q(:,1:xrank)'*y); zeros(p-xrank,1)];
      perm = perm(1:xrank);
    end
    b0 = zeros(size(b));
    */
    // Note: the above OLS could be removed to just use the following section
    // TODO: evaluate when non-degenerated part of R and Q is used to derive OLS
    QRColPivDecompositionHouseholderColumn_D64 decomp =
        new QRColPivDecompositionHouseholderColumn_D64();

    if (!decomp.decompose(a)) {
      return false;
    }

    //DenseMatrix64F Q = decomp.getQ(null, true);  // unused.
    DenseMatrix64F decompR = decomp.getR(null, true);
    int[] perm = decomp.getPivots();
    //    double tol = Math.abs(R.get(0,0)) * Math.max(a.getNumRows(), a.getNumCols()) *
    // Math.ulp(1.0);

    // Note: enable the following section if not using solver for OLS
    //    int xrank = 0;
    //    for (int i = 0; i < R.getNumRows(); i++) {
    //      if (Math.abs(R.get(i,i)) > tol) {
    //        xrank++;
    //      }
    //    }
    //    DenseMatrix64F coefs = new DenseMatrix64F(a.getNumCols(), b.getNumCols());
    //    double[] b = new double[a.getNumCols()]; //NOTE: assuming b is always a vector
    //
    //    Equation eq = new Equation();
    //    eq.alias(Q, "Q", R, "R", coefs, "coefs", b, "b");
    //    eq.process("coefs = R \\ (Q' * b)");
    //    for (int i = 0; i < coefs.getNumRows(); i++) {
    //      b[pivots[i]] = coefs.get(i, 0);
    //    }

    // Since we are not checking the xrank, perm and R always match in dimension
    DenseMatrix64F permutedA = new DenseMatrix64F(a.getNumRows(), perm.length);
    for (int i = 0; i < perm.length; i++) {
      for (int j = 0; j < a.getNumRows(); j++) {
        permutedA.set(j, i, a.get(j, perm[i]));
      }
    }

    DenseMatrix64F invR = decompR; // Don't need a copy; R is not used past here.
    if (!CommonOps.invert(invR)) {
      return false;
    }
    DenseMatrix64F E = new DenseMatrix64F(permutedA.getNumRows(), invR.getNumCols());
    CommonOps.mult(permutedA, invR, E);
    CommonOps.elementMult(E, E);
    DenseMatrix64F adjFactor = CommonOps.sumRows(E, null);

    // assuming b is always a vector
    for (int i = 0; i < adjFactor.getNumRows(); i++) {
      adjFactor.set(i, 0, 1.0 / Math.sqrt(1 - Math.min(0.9999, adjFactor.get(i, 0))));
    }

    /*
    % If we get a perfect or near perfect fit, the whole idea of finding
    % outliers by comparing them to the residual standard deviation becomes
    % difficult.  We'll deal with that by never allowing our estimate of the
    % standard deviation of the error term to get below a value that is a small
    % fraction of the standard deviation of the raw response values.
    */
    StandardDeviation sd = new StandardDeviation();
    double tinyS = 1e-6 * sd.evaluate(b.getData());
    if (tinyS == 0) {
      tinyS = 1.0;
    }

    // Initialize temporary arrays to be the right sizes.
    DenseMatrix64F r = b.copy();
    DenseMatrix64F radj = adjFactor.copy(); // new DenseMatrix64F(adjFactor.getNumRows(), 1);
    DenseMatrix64F bw = b.copy();
    DenseMatrix64F aw = a.copy();
    DenseMatrix64F x0 = x.copy();

    DenseMatrix64F ones = new DenseMatrix64F(1, a.getNumCols());
    CommonOps.fill(ones, 1);

    // NOTE: matlab code seems to have a logic error, where it is only run for 4 times at most.
    for (int iter = 1; iter <= MAX_ITERATIONS; iter++) {
      // Compute residual from previous fit then compute scale estimate
      // r = b - ax.
      r.set(b);
      CommonOps.multAdd(-1, a, x, r);
      CommonOps.elementMult(r, adjFactor, radj);

      int rank = MatrixFeatures.rank(aw);
      double madSigma = madsigma(radj.getData(), rank);
      DenseMatrix64F w = bisquare(radj, Math.max(madSigma, tinyS) * beta);

      // calculate weighted a and b
      CommonOps.elementMult(b, w, bw);
      CommonOps.mult(w, ones, aw);
      CommonOps.elementMult(aw, a);

      if (!solver.setA(aw)) {
        return false;
      }

      // Swap x and x0 so we can check for convergence after the solve.
      DenseMatrix64F tmp = x0;
      x0 = x;
      x = tmp;
      solver.solve(bw, x);
      if (MatrixFeatures.isEquals(x, x0, Math.sqrt(Math.ulp(1.0)))) {
        break;
      }
    }

    results.set(x);
    return true;
  }

  private DenseMatrix64F bisquare(DenseMatrix64F r, double s) {
    for (int i = 0; i < r.getNumRows(); i++) {
      for (int j = 0; j < r.getNumCols(); j++) {
        double v = Math.abs(r.get(i, j) / s);
        if (v > 1) {
          r.set(i, j, 0);
        } else {
          r.set(i, j, 1 - v * v);
        }
      }
    }
    return r;
  }

  /**
   * Compute MAD of adjusted residuals after dropping p closest to 0
   *
   * @param r
   * @param p
   * @return
   */
  private double madsigma(double[] r, int p) {
    double[] absRadj = new double[r.length];
    for (int i = 0; i < absRadj.length; i++) {
      absRadj[i] = Math.abs(r[i]);
    }
    Arrays.sort(absRadj, 0, absRadj.length);

    DescriptiveStatistics ds = new DescriptiveStatistics();
    for (int i = Math.max(0, p - 1); i < absRadj.length; i++) {
      ds.addValue(absRadj[i]);
    }

    return ds.getPercentile(50.0) / 0.6745;
  }
}
