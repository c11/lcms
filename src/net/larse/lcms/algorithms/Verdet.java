package net.larse.lcms.algorithms;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import net.larse.lcms.helper.AlgorithmBase;
import org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
/**
 * Implements the VeRDET change detection:
 * {citation is XXX}
 *
 * This implementation only implement the change analysis, no spatial segmentation
 * is implemented in this class. I am considering spatial segmentation as a preprocessing
 * step to achieve spatial cohesive pixel blocks.
 *
 * @auther Zhiqiang Yang, 04/1/2015
 *
 * In this implementation, the parameter names are kept the same as the original code to
 * maintain easy cross check. They should be change to more meaningful names afterwards.
 *
 */

public final class Verdet {
  static class Args extends AlgorithmBase.ArgsBase {
    @Doc(help = "convergence tolerance")
    @Optional
    double tolerance = 0.0001;

    @Doc(help = "End year of training period, exclusive.")
    @Optional
    double alpha = 1/30.0;


    @Doc(help = "Maximum number of runs for convergence.")
    @Optional
    int nRuns = 100;
  }

  private final Args args;

  public Verdet() {
    this.args = new Args();
  }

  public Verdet(Args args) {
    this.args = args;
  }

  /**
   * The main change method for verdet.
   *
   * @param a, filtered smoothed spectral array one value per year
   * @return
   */
  public double[] getResult(double[] a) {
    //Fixe negative scores
    for (int i = 0; i < a.length; i++) {
      a[i] = a[i] < 0.0 ? 1e-5 : a[i];
    }

    double[] X = fitPiecewiseLinear(a);

    double[] score = new double[X.length];
    for (int i = 1; i < score.length; i++) {
      score[i] = X[i] - X[i-1];
    }

    return score;
  }

  private double[] fitPiecewiseLinear(double[] B) {
    double[] G = tv1DMany(B);

    //TODO: should this be a parameter?
    double dx = 0.005;

    double[] Y = B.clone();

    //Orignal Matlab code use loop to iterate the multi-dimensional G
    //TODO: check to see if the multi-dimensional data are necessary

    int[] N = new int[B.length];
    for (int i = 0; i < B.length; i++) {
      N[i] = i;
    }

    double[] T = G.clone();

    double[] T1 = new double[B.length];

    double[] YY = Y.clone();

    while (!Arrays.equals(T1, T)) {
      T1 = T.clone();

      //TODO: the follow section can be optimized to remove unnecessary array
      //I think most of them can be simplied to use just local variables.
      double[] dT = new double[T.length-1];
      double[] Q = new double[T.length-1];
      double[] num = new double[T.length-2];
      double[] den = new double[T.length-2];
      double[] R = new double[T.length-2];

      int[] C = new int[T.length];
      C[0] = 1;
      C[T.length-1] = 1;
      int[] cc = new int[T.length];

      IntArrayList f = new IntArrayList();
      f.add(0); //add first element

      for (int i = 0; i < dT.length; i++) {
        dT[i] = T[i+1] - T[i];
        Q[i] = Math.sqrt(dx + dT[i]*dT[i]);
        if (i < dT.length-1) {
          num[i] = dT[i] * (T[i + 2] - T[i + 1]);
          den[i] = Q[i] * Math.sqrt(dx + (T[i + 2] - T[i + 1]) * (T[i + 2] - T[i + 1]));
          R[i] = 1 - (num[i] + dx) / den[i];
          if (R[i] > 0.025) {
            C[i + 1] = 1;
            f.add(i + 1);
          }
        }
        if (i > 0) {
          cc[i] = cc[i-1] + C[i];
        }
      }

      //Add the last elements
      f.add(T.length-1);
      f.add(T.length);

      cc[T.length-1] = cc[T.length-2] + C[T.length-1];

      double[] g = new double[B.length];
      for (int i = 0; i < g.length; i++) {
        g[i] = 1.0* (N[i] - f.get(cc[i])) / (f.get(cc[i]+1) - f.get(cc[i]));
      }

      int nA = cc[cc.length-1] + 1;
      double[][] A = new double[B.length][nA];

      for (int i = 0; i < B.length; i++) {
        int idx = cc[i] * B.length + N[i];
        int row = idx % B.length;
        int col = idx / B.length;

        A[row][col] = 1 - g[i];
        if (i < B.length-1) {
          A[row][col+1] = g[i];
        }
      }

      OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
      ols.setNoIntercept(true);
      ols.newSampleData(YY, A);
      double[] b = ols.estimateRegressionParameters();

      SimpleMatrix st = new SimpleMatrix(A).mult(new SimpleMatrix(2, 1, true, b));

      T = st.getMatrix().getData().clone();
    }

    return T;
  }


  /**
   * The original implemention uses a multi-dimension array for X,
   * which does not see to be necessary, as each time it is only run on
   * single index.
   *
   * //TODO: variable names followed original matlab code, change to meaningful names
   *
   * @param X
   */
  private double[] tv1DMany(double[] X) {

    SimpleMatrix A = new SimpleMatrix(X.length, X.length);
    for (int i = 0; i < X.length; i++) {
      for (int j = 0; j < X.length; j++) {
        A.set(j, i, i <= j ? 1.0 : 0);
      }
    }

    SimpleMatrix AtA = A.transpose().mult(A);

    double[] O = new double[X.length];

    //Orignal Matlab code use loop to iterate the multi-diemsional X
    double[] f = new double[X.length];
    for (int i = 0; i < X.length; i++) {
      f[i] = X[i] - X[0];
    }

    SimpleMatrix Atf = A.transpose().mult(new SimpleMatrix(f.length, 1, true, f));

    //initialize some initial variables
    double[] u = f.clone();
    double[] u1 = new double[X.length];
    for (int i=0; i < X.length; i++) {
      u1[i] = Double.POSITIVE_INFINITY;
    }

    for (int i=0; i < args.nRuns; i++) {
      System.out.println(i);

      double[] E = new double[X.length-1];
      SimpleMatrix t = new SimpleMatrix(X.length, X.length);
      SimpleMatrix L = new SimpleMatrix(X.length, X.length);

      for (int j = 0; j < E.length; j++) {


        E[j] = args.alpha / (1e-6 + Math.abs(u[j+1]-u[j]));
        t.set(j, j+1, E[j]);
        if (j == 0) {
          L.set(j, j, E[j]);
        }
        else {
          L.set(j, j, E[j-1]+E[j]);
        }
      }
      L.set(X.length-1, X.length-1, E[E.length-1]);

      L = L.minus(t).minus(t.transpose());

      u = AtA.plus(L).solve(Atf).getMatrix().getData();

      //have we reach convergence
      boolean success = true;
      for (int ui = 0; ui < u.length; ui++) {
        if (Math.abs(u1[ui] - u[ui]) > args.tolerance) {
          success = false;
          break;
        }
      }
      if (success) {
        break;
      }

      //Original Matlab code, which may not be necessary here
      //if any(isnan(u)); u=u1; break; end;
      u1 = u.clone();
    }

    //Integration U, adding f0 back in
    O = A.mult(new SimpleMatrix(u.length, 1, true, u)).getMatrix().getData();
    for (int i = 0; i < O.length; i++) {
      O[i]+= X[0];
    }

    return O;
  }
}
