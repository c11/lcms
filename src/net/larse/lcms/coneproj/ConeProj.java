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
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;
import sun.java2d.pipe.SpanShapeRenderer;

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
    PolarConeProjectionResult result = null;

    int n = y.length;
    int m = matrix.numRows();

    SimpleMatrix namat = new SimpleMatrix(matrix);
    double[] ny = Arrays.copyOf(y, y.length);
    SimpleMatrix smNY = new SimpleMatrix(new DenseMatrix64F(y.length, 1, true, ny));

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
    SimpleMatrix b2 = delta.mult(smNY);

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

      SimpleMatrix a = xmat.mult(xmat.transpose()).solve(xmat.mult(smNY));
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
        b2 = delta.mult(smNY.minus(theta)).divide(n);


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

    SimpleMatrix thetahat = smNY.minus(theta);

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
  public static Object coneB(double[] y, SimpleMatrix delta, SimpleMatrix vmat, double[] weight) {
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
  public static Object coneB(double[] y, SimpleMatrix delta, SimpleMatrix vmat) {
    Object result = null;

    SimpleMatrix nvmat = new SimpleMatrix(vmat);
    int n = y.length;
    int m = delta.numRows();
    int p = nvmat.numCols();

    double[] ny = Arrays.copyOf(y, y.length);
    SimpleMatrix smNY = new SimpleMatrix(new DenseMatrix64F(y.length, 1, true, ny));

    SimpleMatrix ndelta = new SimpleMatrix(delta);

    SimpleMatrix a, sigma;
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
            avec[i] = a[++tmp];
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

      double minASub = Double.NEGATIVE_INFINITY;
      for (int i = p; i < a.getNumElements(); i++) {
        minASub = minASub <= a[i] ? minASub : a[i];
      }

      if (minASub < (-sm)) {
        double[] avec = new double[m+p];
        int tmp = -1;
        for (int i = 0; i < h.length; i++) {
          if (h[i] == 1) {
            avec[i] = a[++tmp];
          }
        }

        double minAVecSub = Double.NEGATIVE_INFINITY;
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

        //TODO: continue here: line 208
//        check = 1;
//        theta = xmat.t() * a;
//        b2 = sigma * (ny - theta) / n;
//
//        if(max(b2) > 2 * sm){
//          int i = min(obs.elem(find(b2 == max(b2))));
//          check = 0;
//          h(i) = 1;
//        }





      }

    }


    int k = 0;



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
