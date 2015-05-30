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

/*
 * implementation details:
 *  need to convert two files: find_disturbance.m
 *
 * it's 'fit_piecewise_linear.m' and 'tv_1d_many.m'.
 * The 'fit_piecewise_linear' is called from 'find_disturbances.m' on line 97.
 * There is a bit of code on either side of that call that is in my port as well.
 * My port is at X:\matt\ensemble_python\ensemble\verdet with most of the relevant code in 'verdet.py'.
 *
 * May 24, 2015, Z. Yang
 * remove unused variables
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

  private int size;
  double[] T1;
  private int[] C;
  private int[] cc;
  private double[] g;
  private double[] O;
  private OLSMultipleLinearRegression ols;
  private SimpleMatrix A;
  private SimpleMatrix f;

  public Verdet() {
    this(new Args());
  }

  public Verdet(Args args) {
    this.args = args;
    ols = new OLSMultipleLinearRegression();
    ols.setNoIntercept(true);
  }

  /**
   * Compute the verdet scores.
   * @param a, scores calculated in verdet
   */
  public double[] getResult(double[] a) {
    init(a.length);

    //Fixe negative scores
    for (int i = 0; i < a.length; i++) {
      a[i] = a[i] < 0.0 ? 1e-5 : a[i];
    }

    double[] X = piecewiseLinear(a);

    double[] score = new double[a.length];
    for (int i = 1; i < score.length; i++) {
      score[i] = X[i] - X[i-1];
    }

    return score;
  }

  public void init(int size) {
    if (this.size == size) {
      return;
    }

    this.size = size;
    T1 = new double[size];
    C = new int[size];
    cc = new int[size];
    g = new double[size];
    O = new double[size];

    A = new SimpleMatrix(size, size);
    f = new SimpleMatrix(size, 1);
  }

  public double[] piecewiseLinear(double[] B) {
    //TODO: should this be a parameter?
    double dx = 0.005;

    //Orignal Matlab code use loop to iterate the multi-dimensional G
    //TODO: check to see if the multi-dimensional data are necessary
    Arrays.fill(T1, 0);
    Arrays.fill(cc, 0);

    // G was unused except to initialize T.
    double[] T = tv1DMany(B);

    //    double[] YY = B.clone();

    while (!Arrays.equals(T1, T)) {
      // Dont use clone.
      System.arraycopy(T, 0, T1, 0, size);

      //TODO: the follow section can be optimized to remove unnecessary array
      //I think most of them can be simplied to use just local variables.

      Arrays.fill(C, 0);
      C[0] = 1;
      C[T.length-1] = 1;

      IntArrayList f = new IntArrayList();
      f.add(0); //add first element

      for (int i = 0; i < size - 1; i++) {
        double dT = T[i+1] - T[i];
        double Q = Math.sqrt(dx + dT*dT);
        if (i < size - 2) {
          double forward = T[i + 2] - T[i + 1];
          double numerator = dT * forward;
          double denominator = Q * Math.sqrt(dx + forward * forward);
          double R = 1 - (numerator + dx) / denominator;
          if (R > 0.025) {
            C[i + 1] = 1;
            f.add(i + 1);
          }
        }
        if (i > 0) {
          cc[i] = cc[i-1] + C[i];
        }
      }

      //Add the last elements
      f.add(size - 1);
      f.add(size);
      cc[size - 1] = cc[size - 2] + C[size - 1];

      for (int i = 0; i < g.length; i++) {
        g[i] = 1.0* (i - f.get(cc[i])) / (f.get(cc[i]+1) - f.get(cc[i]));
      }

      //      int aSize = B.length * (cc[cc.length-1] + 1);
      //      double[] A = new double[aSize];
      //
      //      for (int i = 0; i < B.length; i++) {
      //        A[cc[i] * B.length + N[i]] = 1 - g[i];
      //
      //        if (i < B.length-1) {
      //          A[(cc[i]+1) * B.length + N[i]] = g[i];
      //        }
      //      }

      //int aSize = B.length * (cc[cc.length-1] + 1);
      int nA = cc[size - 1] + 1;
      double[][] A = new double[size][nA];

      for (int i = 0; i < size; i++) {
        int idx = cc[i] * size + i;
        int row = idx % size;
        int col = idx / size;

        A[row][col] = 1 - g[i];
        if (i < size-1) {
          A[row][col+1] = g[i];
        }
      }

      ols.newSampleData(B, A);
      double[] b = ols.estimateRegressionParameters();

      // It's probably worth doing this simple multiply by hand to avoid creating 2 new objects.
      // Especially since you just grab the data out of the
      SimpleMatrix st = new SimpleMatrix(A).mult(new SimpleMatrix(2, 1, true, b));
      T = st.getMatrix().getData();
    }

    return T;
  }

  /**
   * The original implementation uses a multi-dimension array for X,
   * which does not see to be necessary, as each time it is only run on
   * single index.
   *
   * //TODO: variable names followed original matlab code, change to meaningful names
   *
   * @param X
   */
  public double[] tv1DMany(double[] X) {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        A.set(j, i, i <= j ? 1.0 : 0);
      }
    }

    SimpleMatrix AtA = A.transpose().mult(A);
    SimpleMatrix ataClone = AtA.copy();

    //Orignal Matlab code use loop to iterate the multi-diemsional X
    for (int i = 0; i < size; i++) {
      f.set(i, X[i] - X[0]);
    }

    SimpleMatrix Atf = A.transpose().mult(f);

    //initialize some initial variables
    SimpleMatrix u = f;
    SimpleMatrix u1 = f.copy();
    u1.set(Double.POSITIVE_INFINITY);

    for (int i=0; i < args.nRuns; i++) {
      //System.out.println(i);

      double prev = 0;
      double curr = 0;
      for (int j = 0; j < size - 1; j++) {
        curr = args.alpha / (1e-6 + Math.abs(u.get(j+1) - u.get(j)));
        // Diagonal (L)
        AtA.set(j, j, ataClone.get(j, j) + prev + curr);
        // Off diagonals.
        AtA.set(j,     j + 1, ataClone.get(j,     j + 1) - curr);
        AtA.set(j + 1, j,     ataClone.get(j + 1, j    ) - curr);
        prev = curr;
      }
      AtA.set(size - 1, size - 1, ataClone.get(size - 1, size -1) + curr);

      u = AtA.solve(Atf);
      // Have we reach convergence?
      if (u1.minus(u).elementMaxAbs() <= args.tolerance) {
        break;
      }

      //Original Matlab code, which may not be necessary here
      //if any(isnan(u)); u=u1; break; end;
      u1 = u;
    }

    //Integration U, adding f0 back in
    u = A.mult(u);
    for (int i = 0; i < size; i++) {
      O[i] = u.get(i) + X[0];
    }

    return O;
  }
}
