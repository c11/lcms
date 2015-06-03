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
package net.larse.lcms.coneproj;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.apache.commons.math.stat.StatUtils;
import org.ejml.alg.dense.decomposition.CholeskyDecomposition;
import org.ejml.alg.dense.decomposition.DecompositionFactory;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.Collections;


/**
 * This class replicates the coneproj.cpp function from R package coneproj.
 *
 * Xiyue Liao, Mary C. Meyer (2014). coneproj: An R Package for the Primal or Dual Cone Projections
 * with Routines for Constrained Regression. Journal of Statistical Software, 61(12), 1-22.
 * URL http://www.jstatsoft.org/v61/i12/.
 *
 * @auther Zhiqiang Yang, 2/17/2015
 *
 * TODO: this is the first pass of implementing the algorithms, there are a lot of redundant code,
 * which can be solidated to simplify the implementation.
 *
 * The three projection codes share a lot of commond code, there should be a way to simplify the logic.
 *
 */
public class ConeProj {


  /**
   * This routine implements the hinge algorithm for cone projection to minimize
   * ||y - θ||^2 over the cone C of the form \{θ: Aθ ≥ 0\}.
   *
   * @param y A vector of length n
   * @param matrix A constraint matrix. The rows of amat must be irreducible.
   *               The column number of amat must equal the length of y.
   * @param weight An optional nonnegative vector of weights of length n.
   *               If w is not given, all weights are taken to equal 1.
   *               Otherwise, the minimization of (y - θ)'w(y - θ) over C is returned.
   * @return PolarConeProjectionResult
   */
  public static PolarConeProjectionResult coneA(double[] y, SimpleMatrix matrix, double[] weight) {
    //TODO: apply weight first

    return coneA(y, matrix);
  }

  /**
   * This routine implements the hinge algorithm for cone projection to minimize
   * ||y - θ||^2 over the cone C of the form \{θ: Aθ ≥ 0\}.
   *
   * @param y A vector of length n
   * @param matrix A constraint matrix. The rows of amat must be irreducible.
   *               The column number of amat must equal the length of y.
   * @return PolarConeProjectionResult
   */
  public static PolarConeProjectionResult coneA(double[] y, SimpleMatrix matrix) {
    int n = y.length;
    double[] ny = Arrays.copyOf(y, y.length);
    SimpleMatrix smNY = new SimpleMatrix(new DenseMatrix64F(y.length, 1, true, ny));
    return ConeProj.coneA(smNY, matrix);
  }
  /**
   * This routine implements the hinge algorithm for cone projection to minimize
   * ||y - θ||^2 over the cone C of the form \{θ: Aθ ≥ 0\}.
   *
   * @param y A vector of length n
   * @param matrix A constraint matrix. The rows of amat must be irreducible.
   *               The column number of amat must equal the length of y.
   * @return PolarConeProjectionResult
   */
  public static PolarConeProjectionResult coneA(SimpleMatrix y, SimpleMatrix matrix) {

    PolarConeProjectionResult result = null;

//    int n = y.length;
    int n = y.numRows();
    int m = matrix.numRows();

    SimpleMatrix namat = new SimpleMatrix(matrix);
//    double[] ny = Arrays.copyOf(y, y.length);
//    SimpleMatrix smNY = new SimpleMatrix(new DenseMatrix64F(y.length, 1, true, ny));

    double sm = 1e-8;

    int[] h = new int[m];
    int[] obs = new int[m];
    for (int i = 0; i < m; i++) {
      obs[i] = i;
    }

    boolean check = false;
    //TODO: watch whether this will make a deep copy or shallow copy?
    SimpleMatrix amat_in = new SimpleMatrix(namat);

    for (int i = 0; i < m; i++) {
      SimpleMatrix tmp = namat.extractVector(true, i);
      SimpleMatrix nnamat = tmp.mult(tmp.transpose());
      amat_in.setRow(i, 0, tmp.divide(Math.sqrt(nnamat.get(0,0))).getMatrix().getData());
    }

    SimpleMatrix delta = amat_in.negative();
//    SimpleMatrix b2 = delta.mult(smNY);
    SimpleMatrix b2 = delta.mult(y);
    SimpleMatrix theta = new SimpleMatrix(n, 1);
    double maxB2 = StatUtils.max(b2.getMatrix().getData());
    if (maxB2 > 2 * sm) {
      for (int i = 0; i < b2.getNumElements(); i++) {
        double tmp = b2.get(i);
        if (Math.abs(tmp -maxB2) < sm) {
          h[i] = 1;
          break;
        }
      }
    }
    else {
      check = true;
    }


    //TODO: continue here check values
    int nrep = 0;
    while (!check && nrep < (n * n)) {
      nrep++;

      //In the Rcpp code: indice = indice.elem(find(h == 1));
      // which seems always have only one item with h==1
      IntArrayList indices = new IntArrayList();
      for (int i = 0; i < h.length; i++) {
        if (h[i]==1) {
          indices.add(i);
        }
      }

      SimpleMatrix xmat = new SimpleMatrix(indices.size(), delta.numCols());
      for (int i = 0; i < indices.size(); i++) {
        xmat.setRow(i, 0, delta.extractVector(true, indices.get(i)).getMatrix().getData());
      }

//      SimpleMatrix a = xmat.mult(xmat.transpose()).solve(xmat.mult(smNY));
      SimpleMatrix a = xmat.mult(xmat.transpose()).solve(xmat.mult(y));
      double[] avec = new double[m];
      double minA = StatUtils.min(a.getMatrix().getData());
      if (minA < -sm) {
        for (int i = 0; i < indices.size(); i++) {
          avec[indices.get(i)] = a.get(i);
        }

        double minAvec = StatUtils.min(avec);
        IntArrayList minAvecIndex = new IntArrayList();
        for (int i = 0; i < avec.length; i++) {
          if (avec[i] == minAvec) {
            minAvecIndex.add(i);
          }
        }
        //TODO: check is this necessary
        Collections.sort(minAvecIndex);
        h[minAvecIndex.get(0)] = 0;
        check = false;
      }
      else {
        check = true;
        theta = xmat.transpose().mult(a);
//        b2 = delta.mult(smNY.minus(theta)).divide(n);
        b2 = delta.mult(y.minus(theta)).divide(n);

        maxB2 = StatUtils.max(b2.getMatrix().getData());
        if (maxB2 > 2 * sm) {
          for (int i = 0; i < b2.getNumElements(); i++) {
            double tmp = b2.get(i);
            if (Math.abs(tmp -maxB2) < sm) {
              h[i] = 1;
              check = false;
              break;
            }
          }
        }
      }
    }

//    SimpleMatrix thetahat = smNY.minus(theta);
    SimpleMatrix thetahat = y.minus(theta);

    int sum = 0;
    for (int i = 0; i < h.length; i++) {
      sum += h[i];
    }

    result = new PolarConeProjectionResult(nrep, thetahat, n - sum);
    if (nrep > (n*n-1)) {

      result = null;
    }

    return result;
  }

  /**
   * This routine implements the hinge algorithm for cone projection to minimize ||y - θ||^2
   * over the cone C of the form \{θ: θ = v + ∑ b_iδ_i, i = 1,…,m, b_1,…, b_m ≥ 0\}, v is in V.
   *
   * @param y A vector of length n.
   * @param delta A matrix whose rows are the constraint cone edges. The rows of delta must be
   *              irreducible. Its column number must equal the length of y. No row of delta is
   *              contained in the column space of vmat.
   * @param vmat A matrix whose columns are the basis of the linear space contained in the constraint
   *             cone. Its row number must equal the length of y. The columns of vmat must be linearly
   *             independent. The default is vmat = NULL
   * @param weight An optional nonnegative vector of weights of length n. If w is not given, all weights
   *               are taken to equal 1. Otherwise, the minimization of (y - θ)'w(y - θ) over C is returned.
   *               The default is w = NULL.
   * @return
   */
  public static ConstraintConeProjectionResult coneB(double[] y, SimpleMatrix delta, SimpleMatrix vmat, double[] weight) {
    //TODO: implement weight factor
    return coneB(y, delta, vmat);
  }

  /**
   * This routine implements the hinge algorithm for cone projection to minimize ||y - θ||^2
   * over the cone C of the form \{θ: θ = v + ∑ b_iδ_i, i = 1,…,m, b_1,…, b_m ≥ 0\}, v is in V.
   *
   * @param y A vector of length n.
   * @param delta A matrix whose rows are the constraint cone edges. The rows of delta must be
   *              irreducible. Its column number must equal the length of y. No row of delta is
   *              contained in the column space of vmat.
   * @param vmat A matrix whose columns are the basis of the linear space contained in the constraint
   *             cone. Its row number must equal the length of y. The columns of vmat must be linearly
   *             independent. The default is vmat = NULL
   * @return
   */
  public static ConstraintConeProjectionResult coneB(double[] y, SimpleMatrix delta, SimpleMatrix vmat) {
    Object result = null;

    SimpleMatrix nvmat = new SimpleMatrix(vmat);
    int n = y.length;
    int m = delta.numRows();
    int p = nvmat.numCols();

    double[] ny = Arrays.copyOf(y, y.length);
    SimpleMatrix smNY = new SimpleMatrix(new DenseMatrix64F(y.length, 1, true, ny));

    SimpleMatrix ndelta = new SimpleMatrix(delta);

    //This line is just default initialization, and should not be used.
    SimpleMatrix a = new SimpleMatrix(m, 1);
    SimpleMatrix sigma;
    int [] h;
    int [] obs;

    SimpleMatrix theta = new SimpleMatrix(n, 1);

    double sm = 1e-8;
    boolean check = false;

    double[] scalar = new double[m];
    SimpleMatrix delta_in = new SimpleMatrix(ndelta);

    for (int i = 0; i < m; i++) {
      SimpleMatrix row = ndelta.extractVector(true, i);
      SimpleMatrix nndelta = row.mult(row.transpose());
      scalar[i] = Math.sqrt(nndelta.get(0,0));
      delta_in.setRow(i, 0, row.divide(scalar[i]).getMatrix().getData());
    }

    if (nvmat.getNumElements()==0) {
      p--;
      sigma = new SimpleMatrix(delta_in);
      h = new int[m];
      obs = linspace(m);
    }
    else {
      sigma = new SimpleMatrix(m+p, n);

      for (int i = 0; i < p; i++) {
        sigma.setRow(i, 0, nvmat.extractVector(false, i).getMatrix().getData());
      }
      for (int i = p; i < m+p; i++) {
        sigma.setRow(i, 0, delta_in.extractVector(true, i - p).getMatrix().getData());
      }

      h = new int[m+p];
      for (int i = 0; i < p; i++) {
        h[i] = 1;
      }
      obs = linspace(m+p);

      SimpleMatrix tmp = nvmat.transpose().mult(nvmat).solve(nvmat.transpose().mult(smNY));
      theta = nvmat.mult(tmp);
    }

    //NOTE: this line above have been verified.

    SimpleMatrix b2 = sigma.mult(smNY.minus(theta)).divide(n);

    int nrep = 0;
    double maxB2 = StatUtils.max(b2.getMatrix().getData());
    if (maxB2 > 2 * sm) {
      check = false;
      for (int i = 0; i < b2.getNumElements(); i++) {
        if (b2.get(i) == maxB2) {
          h[i] = 1;
          break;
        }
      }
    }
    else {
      check = true;
      theta.set(0);

      if (nvmat.getNumElements()==0) {
        a = new SimpleMatrix(m, 1);
        a.set(0);
      }
      else {
        a = nvmat.transpose().mult(nvmat).solve(nvmat.transpose().mult(smNY));
        theta = nvmat.mult(a);
      }

      double[] avec = new double[m+p];
      if (nvmat.getNumElements()>0) {
        int tmp = -1;
        for (int i = 0; i < h.length; i++) {
          if (h[i] == 1) {
            avec[i] = a.get(++tmp);
          }
        }
      }

      int sum = 0;
      for (int i = 0; i < h.length; i++) {
        sum += h[i];
      }
      return new ConstraintConeProjectionResult(sum, theta.getMatrix().getData(), nrep, avec);
    }

    while (!check && nrep < n*n) {
      nrep++;

      IntArrayList indice = new IntArrayList();
      for (int i = 0; i < h.length; i++) {
        if (h[i] == 1) {
          indice.add(i);
        }
      }

      SimpleMatrix xmat = new SimpleMatrix(indice.size(), sigma.numCols());
      for (int i = 0; i < indice.size(); i++) {
        xmat.setRow(i, 0, sigma.extractVector(true, indice.get(i)).getMatrix().getData());
      }

      a = xmat.mult(xmat.transpose()).solve(xmat.mult(smNY));
      double[] a_sub = new double[a.getNumElements()-p];

      double minASub = Double.POSITIVE_INFINITY;
      for (int i = p; i < a.getNumElements(); i++) {
        minASub = minASub <= a.get(i) ? minASub : a.get(i);
      }

      if (minASub < (-sm)) {
        double[] avec = new double[m+p];
        int tmp = -1;
        for (int i = 0; i < h.length; i++) {
          if (h[i] == 1) {
            avec[i] = a.get(++tmp);
          }
        }

        double minAVecSub = Double.POSITIVE_INFINITY;
        for (int i = p; i < p+m; i++) {
          minAVecSub = minAVecSub <= avec[i] ? minAVecSub : avec[i];
        }

        //TODO: I think this logic can be simplified.
        IntArrayList minAvecIndex = new IntArrayList();
        for (int i = 0; i < avec.length; i++) {
          if (avec[i] == minAVecSub) {
            minAvecIndex.add(i);
          }
        }
        //TODO: check is this necessary
        Collections.sort(minAvecIndex);
        h[minAvecIndex.get(minAvecIndex.size()-1)] = 0;
        check = false;
      }
      else {
        check = true;
        theta = xmat.transpose().mult(a);

        b2 = sigma.mult(smNY.minus(theta)).divide(n);

        maxB2 = StatUtils.max(b2.getMatrix().getData());
        if (maxB2 > 2 * sm) {
          for (int i = 0; i < b2.getNumElements(); i++) {
            if (Math.abs(b2.get(i) -maxB2) < sm) {
              h[i] = 1;
              check = false;
              break;
            }
          }
        }
      }
    }

    double[] avec = new double[m+p];
    int tmp = -1;
    for (int i = 0; i < h.length; i++) {
      if (h[i] == 1) {
        avec[i] = a.get(++tmp);
      }
    }

    double[] avec_orig = new double[m+p];
    for (int i = 0; i < p; i++) {
      avec_orig[i] = avec[i];
    }

    for (int i = p; i < (m+p); i++) {
      avec_orig[i] = avec[i] / scalar[i-p];
    }

    int sum = 0;
    for (int i = 0; i < h.length; i++) {
      sum += h[i];
    }

//    if(nrep > (n * n - 1)){Rcpp::Rcout << "Fail to converge in coneproj!Too many steps! Number of steps:" << nrep << std::endl;}
//    return wrap(Rcpp::List::create(Named("yhat") = theta, Named("coefs") = avec_orig, Named("nrep") = nrep, Named("dim") = sum(h)));

    if (nrep > (n*n-1)) {
      return null;
    }
    else {
      return new ConstraintConeProjectionResult(sum, theta.getMatrix().getData(), nrep, avec_orig);
    }
  }


  /**
   * Given a positive definite n by n matrix Q and a constant vector c in R^n,
   * the object is to find θ in R^n to minimize θ'Qθ - 2c'θ subject to Aθ ≥ b,
   * for an irreducible constraint matrix A. This routine transforms into a
   * cone projection problem for the constrained solution.
   *
   * To get the constrained solution to θ'Qθ - 2c'θ subject to Aθ ≥ b, this routine
   * makes the Cholesky decomposition of Q. Let U'U = Q, and define φ = Uθ and z = U^{-1}c,
   * where U^{-1} is the inverse of U. Then we minimize ||z - φ||^2, subject to Bφ ≥ 0,
   * where B = AU^{-1}. It is now a cone projection problem with the constraint cone C of
   * the form \{φ: Bφ ≥ 0 \}. This routine gives the estimation of θ, which is U^{-1} times
   * the estimation of φ.
   *
   *
   * @param q A n by n positive definite matrix.
   * @param c A vector of length n.
   * @param amat A m by n constraint matrix. The rows of amat must be irreducible.
   * @param b A vector of length m. Its default value is 0.
   * @return PolarConeProjectionResult
   */
  public static PolarConeProjectionResult qprog(SimpleMatrix q, double[] c, SimpleMatrix amat, double[] b) {
    SimpleMatrix nc = new SimpleMatrix(c.length, 1, true, c);
    return ConeProj.qprog(q, nc, amat, b);
  }

  public static PolarConeProjectionResult qprog(SimpleMatrix q, SimpleMatrix c, SimpleMatrix amat, double[] b) {
      PolarConeProjectionResult result = null;

//    int n = c.length;
//    int m = amat.numRows();
//    SimpleMatrix nc = new SimpleMatrix(c.length, 1, true, c);

    int n = c.numRows();
    int m = amat.numRows();
    SimpleMatrix nc = c;

    SimpleMatrix namat = new SimpleMatrix(amat);
    SimpleMatrix nq = new SimpleMatrix(q);
    SimpleMatrix theta0 = new SimpleMatrix(n, 1);
    SimpleMatrix nnc = new SimpleMatrix(n, 1);

    boolean constraint = false;
    for (int i = 0; i < b.length; i++) {
      if (b[i] != 0) {
        constraint = true;
        break;
      }
    }

    if (constraint) {
      SimpleMatrix nb = new SimpleMatrix(10,1,true, b);
      theta0 = namat.solve(nb);
      nnc = nc.minus(nq.mult(theta0));
    }
    else {
      nnc = nc;
    }

    CholeskyDecomposition<DenseMatrix64F> chol = DecompositionFactory.chol(nq.numRows(), true);
    if (!chol.decompose(nq.getMatrix())) {
      throw new RuntimeException("Cholesky failed!");
    }
    SimpleMatrix preu = SimpleMatrix.wrap(chol.getT(null));

    SimpleMatrix u = ConeProj.trimatu(preu);
    SimpleMatrix z = u.invert().transpose().mult(nnc);
    SimpleMatrix atil = namat.mult(u.invert());

    double sm = 1e-8;
    int[] h = new int[m];
    int[] obs = linspace(m);
    boolean check = false;

    for (int i = 0; i < m; i++) {
      SimpleMatrix row = atil.extractVector(true, i);
      SimpleMatrix atilnorm = row.mult(row.transpose());
      atil.setRow(i, 0, row.divide(atilnorm.get(0,0)).getMatrix().getData());
    }

    SimpleMatrix delta = atil.negative();
    SimpleMatrix b2 = delta.mult(z);

    SimpleMatrix phi = new SimpleMatrix(n, 1);

    double maxB2 = StatUtils.max(b2.getMatrix().getData());
    if (maxB2 > 2 * sm) {
      for (int i = 0; i < b2.getNumElements(); i++) {
        double tmp = b2.get(i);
        if (b2.get(i) ==maxB2) {
          h[i] = 1;
          break;
        }
      }
    }
    else {
      check = true;
    }

    int nrep = 0;
    while (!check && nrep < (n*n)) {
      nrep++;

      IntArrayList indices = new IntArrayList();
      for (int i = 0; i < h.length; i++) {
        if (h[i]==1) {
          indices.add(i);
        }
      }

      SimpleMatrix xmat = new SimpleMatrix(indices.size(), delta.numCols());
      for (int i = 0; i < indices.size(); i++) {
        xmat.setRow(i, 0, delta.extractVector(true, indices.get(i)).getMatrix().getData());
      }

      SimpleMatrix a = xmat.mult(xmat.transpose()).solve(xmat.mult(z));
      double[] avec = new double[m];
      double minA = StatUtils.min(a.getMatrix().getData());
      if (minA < -sm) {
        for (int i = 0; i < indices.size(); i++) {
          avec[indices.get(i)] = a.get(i);
        }

        double minAvec = StatUtils.min(avec);
        IntArrayList minAvecIndex = new IntArrayList();
        for (int i = 0; i < avec.length; i++) {
          if (avec[i] == minAvec) {
            minAvecIndex.add(i);
          }
        }
        //TODO: check is this necessary
        Collections.sort(minAvecIndex);
        h[minAvecIndex.get(0)] = 0;
        check = false;
      }
      else {
        check = true;
        phi = xmat.transpose().mult(a);
        b2 = delta.mult(z.minus(phi)).divide(n);

        maxB2 = StatUtils.max(b2.getMatrix().getData());
        if (maxB2 > 2 * sm) {
          for (int i = 0; i < b2.getNumElements(); i++) {
            if (b2.get(i) == maxB2) {
              h[i] = 1;
              check = false;
              break;
            }
          }
        }
      }
    }

    SimpleMatrix thetahat = u.solve(z.minus(phi));
    if (constraint) {
      thetahat = thetahat.plus(theta0);
    }

    if (nrep > (n*n-1)) {
      result = null;
    }
    else {
      int sum = 0;
      for (int i = 0; i < h.length; i++) {
        sum += h[i];
      }
      result = new PolarConeProjectionResult(n-sum, thetahat, nrep);
    }
    return result;
  }

  public static SimpleMatrix trimatu(SimpleMatrix amat) {
    SimpleMatrix result = new SimpleMatrix(amat);
    for (int i = 0; i < result.numRows(); i++) {
      for (int j = 0; j < result.numCols(); j++) {
        if (j<i) {
          result.set(i, j, 0);
        }
      }
    }
    return result;
  }

  public static int[] linspace(int n) {
    int[] result = new int[n];
    for (int i = 0; i < n; i++) {
      result[i] = i;
    }
    return result;
  }

}
