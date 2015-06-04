/*
 * Copyright (c) 2015 Zhiqiang Yang, Noel Gorelick.
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
package net.larse.lcms.algorithms;

import it.unimi.dsi.fastutil.doubles.DoubleRBTreeSet;
import net.larse.lcms.coneproj.ConeProj;
import net.larse.lcms.coneproj.PolarConeProjectionResult;
import net.larse.lcms.helper.AlgorithmBase;
import org.apache.commons.math.random.GaussianRandomGenerator;
import org.apache.commons.math.random.JDKRandomGenerator;
import org.apache.commons.math.random.RandomGenerator;
import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math.stat.descriptive.moment.Variance;
import org.ejml.alg.dense.decomposition.CholeskyDecomposition;
import org.ejml.alg.dense.decomposition.DecompositionFactory;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;


import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;

/**
 * This is the Java implementation of R package ShapeSelectForest
 *
 * Given a scatterplot of (x_i,y_i), i=1,…,n, constrained least-squares spline fits are
 * obtained for all of the following shapes:
 *
 *  1. flat
 *  2. decreasing
 *  3. one-jump, i.e., decreasing, jump up, decreasing
 *  4. inverted vee (increasing then decreasing)
 *  5. vee (decreasing then increasing)
 *  6. increasing (linear)
 *  7. double-jump, i.e., decreasing, jump up, decreasing, jump up, decreasing.
 *
 *  The shape with the smallest information criterion may be considered a "best" fit.
 *  This shape-selection problem was motivated by a need to identify types of disturbances to areas of forest,
 *  given landsat signals over a number of years. The satellite signal is constant or slowly decreasing for
 *  a healthy forest, with a jump upward in the signal is caused by mass destruction of trees.
 *
 * @auther Zhiqiang Yang, 5/30/2015
 *
 * TODO:
 *   1. implement shape parameter configuration, this is in getResult() and EDF.computeEDF0()
 */
public class ShapeSelectForest {

  //region inner class Args
  static class Args extends AlgorithmBase.ArgsBase {
    @Doc(help = "Allow flat shape")
    @Optional
    boolean allowFlat = true;

    @Doc(help = "Allow decreasing shape")
    @Optional
    boolean allowDecreasing = true;

    @Doc(help = "Allow one-jump shape")
    @Optional
    boolean allowOneJump = true;

    @Doc(help = "Allow inverted-vee shape")
    @Optional
    boolean allowInvertedVee = true;

    @Doc(help = "Allow vee shape")
    @Optional
    boolean allowVee = true;

    @Doc(help = "Allow increasing shape")
    @Optional
    boolean allowIncreasing = true;

    @Doc(help = "Allow double-jump shape")
    @Optional
    boolean allowDoubleJump = true;

    @Doc(help = "Number of simulations to get edf0 vector")
    @Optional
    int nsim = 1000;

    @Doc(help = "Shape selection criterior. 1: Bayes informaiton criterion (BIC); 2: Cone information criterion (CIC)")
    @Optional
    int infoCriterion = 1;

    //TODO: The following are for internal testing should be excluded from the final interface.
    @Doc(help = "A parameter used by the maintainer to test if each shape option can be both included and excluded.")
    @Optional
    boolean testRandom = false;

    @Doc(help = "Whether to create warning message when non-convergence occures")
    @Optional
    boolean showWarning = true;
  }

  //endregion

  //region inner class EDF
  /**
   * The object "edf0s" is a 21 by 7 matrix. Each row is an edf0 vector of 7 elements corresponding
   * to the 7 shapes in this package. Such a vector will be used in the main routine "shape" to select
   * the best shape for a scatterplot. Each edf0 vector is simulated through a subroutine called "getedf0",
   * using a total of 1000 simulations with the random seed being set to 123. Each row is an edf0 vector
   * for an equally spaced x vector of n elements. From the first row to the last row, the edf0 vector
   * is for a predictor vector x of length n which is an integer ranging from 20 to 40. The matrix is
   * built for the convenience of users when they call the routine "shape".
   *
   * If the x vector is equally spaced and its number of elements n is between 20 and 40, then a corresponding
   * edf0 vector will be extracted directly from this matrix and no simulation will be done,
   * which saves a lot of time; otherwise, the subroutine "getedf0" will be called inside
   * he routine "shape" to get an edf0 vector for x. The timing depends on the number of elements in x.
   * For example, when x is an equally spaced vector of 26 elements, the timing is about 167 seconds
   * if the user allows a double-jump shape and the timing is about 12 seconds if the user doesn't
   * allow a double-jump shape. Also, when x is not an equally spaced vector,
   * no matter how many elements it has, "getedf0" will be called.
   */
  class EDF {
    private final double[][] EDF0S = {
        {1, 2.590, 6.049, 5.172, 5.172, 1.5, 9.078},
        {1, 2.648, 6.105, 5.081, 5.081, 1.5, 9.111},
        {1, 2.591, 6.163, 5.147, 5.147, 1.5, 9.128},
        {1, 2.655, 6.294, 5.381, 5.381, 1.5, 9.413},
        {1, 2.692, 6.286, 5.371, 5.371, 1.5, 9.348},
        {1, 2.680, 6.384, 5.365, 5.365, 1.5, 9.405},
        {1, 2.702, 6.362, 5.369, 5.369, 1.5, 9.407},
        {1, 2.715, 6.289, 5.289, 5.289, 1.5, 9.403},
        {1, 2.728, 6.322, 5.327, 5.327, 1.5, 9.384},
        {1, 2.664, 6.352, 5.370, 5.370, 1.5, 9.429},
        {1, 2.758, 6.388, 5.355, 5.355, 1.5, 9.418},
        {1, 2.712, 6.365, 5.323, 5.323, 1.5, 9.454},
        {1, 2.726, 6.405, 5.331, 5.331, 1.5, 9.483},
        {1, 2.715, 6.356, 5.370, 5.370, 1.5, 9.530},
        {1, 2.711, 6.299, 5.319, 5.319, 1.5, 9.485},
        {1, 2.689, 6.354, 5.352, 5.352, 1.5, 9.467},
        {1, 2.734, 6.363, 5.368, 5.368, 1.5, 9.502},
        {1, 2.734, 6.373, 5.387, 5.387, 1.5, 9.498},
        {1, 2.713, 6.394, 5.369, 5.369, 1.5, 9.531},
        {1, 2.711, 6.358, 5.359, 5.359, 1.5, 9.529},
        {1, 2.730, 6.387, 5.329, 5.329, 1.5, 9.498}};

    private double[][] edf0s;
    private boolean initialized;
//    private BitSet shapes = new BitSet(8); //shape requested

    public EDF() {
      this.edf0s = EDF0S;
      this.initialized = true;
    }

    /**
     * An edf0 vector is the estimated "null expected degrees of freedom" for shapes allowed by the user
     *
     * @param x A n by 1 predictor vector. For example, they are consecutive years.
     * @return The edf0 values for all shape options allowed by the user.
     */
    private double[] computeEdf0(double[] x) {
      return computeEdf0(x, args.allowFlat,
          args.allowDecreasing,
          args.allowOneJump,
          args.allowInvertedVee,
          args.allowVee,
          args.allowIncreasing,
          args.allowDoubleJump,
          args.nsim,
          args.testRandom,
          args.showWarning);
    }

    /**
     * An edf0 vector is the estimated "null expected degrees of freedom" for shapes allowed by the user.
     * It is an input of the main routine "shape" and it is used to select the best shape for a scatterplot.
     * See Meyer (2013a) and Meyer (2013b) for further details.
     * <p/>
     * Meyer, M. C. (2013a) Semi-parametric additive constrained regression.
     * Journal of Nonparametric Statistics 25(3), 715
     * <p/>
     * Meyer, M. C. (2013b) A simple new algorithm for quadratic programming with applications in statistics.
     * Communications in Statistics 42(5), 1126–1139.
     *
     * @param x      A n by 1 predictor vector. For example, they are consecutive years.
     * @param flat   If it is TRUE, there is a flat shape choice; otherwise, there is no flat shape choice.
     * @param dec    If it is TRUE, there is a decreasing shape choice; otherwise, there is no decreasing shape choice.
     * @param jp     If it is TRUE, there is a one-jump shape choice; otherwise, there is no one-jump shape choice.
     * @param invee  If it is TRUE, there is an inverted-vee shape choice; otherwise, there is no inverted-invee shape choice.
     * @param vee    If it is TRUE, there is a vee shape choice; otherwise, there is no vee shape choice.
     * @param inc    If it is TRUE, there is an increasing shape choice; otherwise, there is no increasing shape choice.
     * @param db     If it is TRUE, there is a double-jump option; otherwise, there is no such a shape option.
     * @param nsim   Number of simulations used to get the edf0 vector..
     * @param random A parameter used by the maintainer to test if each shape option can be both included and excluded.
     * @param msg    If msg is TRUE, then a warning message will be printed when there is a non-convergence problem;
     *               otherwise no warning message will be printed. The default is msg = TRUE
     * @return The edf0 values for all shape options allowed by the user.
     */
    private double[] computeEdf0(double[] x,
                                boolean flat,
                                boolean dec,
                                boolean jp,
                                boolean invee,
                                boolean vee,
                                boolean inc,
                                boolean db,
                                int nsim,
                                boolean random,
                                boolean msg) {

      //TODO: simple the parameter by passing in as argument of Bitset.
//      if (flat) {shapes.set(ShapeSelectForest.FLAT);}
//      if (dec) {shapes.set(ShapeSelectForest.DECREASING);}
//      if (jp) {shapes.set(ShapeSelectForest.ONE_JUMP);}
//      if (invee) {shapes.set(ShapeSelectForest.INVERTED_VEE);}
//      if (vee) {shapes.set(ShapeSelectForest.VEE);}
//      if (inc) {shapes.set(ShapeSelectForest.INCREASING);}
//      if (db) {shapes.set(ShapeSelectForest.DOUBLE_JUMP);}

//      int nsh = 0;
//      if (flat) nsh++;
//      if (dec) nsh++;
//      if (jp) nsh++;
//      if (invee) nsh++;
//      if (vee) nsh++;
//      if (inc) nsh++;
//      if (db) nsh++;

      int n = x.length;

      int k0 = (int)(4 + Math.round(Math.pow(n, 1.0 / 7)) + 2);
      int k1 = k0;
      int k = 0;

      while(k==0) {
        int pts = (n - 2) / (k1 - 1);
        int rem_pts = (n - 2) % (k1 - 1);

        if (pts > 2) {
          k = k1;
        }
        else if (pts == 2) {
          if (rem_pts / (k1 - 1) >= 1) {
            k = k1;
          }
          else {
            k1--;
          }
        }
        else {
          k1--;
        }
      }

      // in R code, nsim has to be at least 1000, this should have been checked before this point.
      // Consider force it to be 1000 all the time.
//      if (nsim < 1000) {
//        stop("We need at least 1000 simulations to get edf0!")
//      }
      int nloop = nsim < 1000 ? 1000 : nsim;

      //Assuming x is already sorted
//      xs = sort(x)
//      x = (xs - min(xs))/(max(xs) - min(xs))
      double xMin = x[0];
      double xRange = x[n-1] - xMin;
      for (int i = 0; i < n; i++) {
        x[i] = (x[i] - xMin) / xRange;
      }

      int[] kobs = new int[k];
      for (int i = 0; i < k; i++) {
        kobs[i] = i + 1;
      }

      //R: ans = bqspl(x, k, knots = NULL, pic = FALSE)
      //TODO: do we need a instance bqspl variable for this.
      bqspl.bqspl(x, k, false, true);

      SimpleMatrix delta = new SimpleMatrix(bqspl.bmat);
      SimpleMatrix qv = delta.transpose().mult(delta);

      int m0 = delta.numRows();

      //choleskydecomposition is implemented in place
      CholeskyDecomposition<DenseMatrix64F> chol = DecompositionFactory.chol(m0, false);
      if (!chol.decompose(qv.getMatrix())) {
        throw new RuntimeException("Cholesky failed!");
      }

      //since choleskydecomposition is in place, the following line may not be necessary
      SimpleMatrix umat0 = SimpleMatrix.wrap(chol.getT(null));
      SimpleMatrix uinv0 = umat0.invert();
      SimpleMatrix pmult0 = uinv0.transpose().mult(delta.transpose());
      SimpleMatrix amat1 = new SimpleMatrix(bqspl.slopes).scale(-1).mult(uinv0);

      //TODO: implement this section for random parameter
//      if (random) {
//        sz = sample(1:6, size = 1)
//        loc = sample(1:7, size = sz, replace = FALSE)
//        bool = rep(TRUE, 7)
//        bool[loc] = FALSE
//        flat = bool[1]
//        dec = bool[2]
//        jp = bool[3]
//        invee = bool[4]
//        vee = bool[5]
//        inc = bool[6]
//        db = bool[7]
//      }
//      else {
//        bool = c(flat, dec, jp, invee, vee, inc, db)
//      }

      //TODO: the following is based on sequential location, which is error prone.
      //changed it to fix array, e.g. always using size of 7 and only set the corresponding element.
      // alternatively, use a DoubleArrayList, but since it is only 7 element, maybe overkill.
      double[] edf0 = new double[7];
      if (flat) {
        edf0[0] = 1;
      }

      class randomVector {
        private SimpleMatrix vector;
        private JDKRandomGenerator randomGenerator = new JDKRandomGenerator();
        public randomVector(int n) {
          vector = new SimpleMatrix(n, 1);
        }

        private void update() {
          for (int j = 0; j < vector.numRows(); j++) {
            vector.set(j, randomGenerator.nextGaussian());
          }
        }

        public SimpleMatrix getData() {
          update();
          return vector;
        }
      }

      randomVector ysVector = new randomVector(n);

      //Decreasing
      if (dec) {
        double sd = 0.0;
        for (int i = 0; i < nloop; i++) {
          SimpleMatrix z = pmult0.mult(ysVector.getData());
          PolarConeProjectionResult ans = ConeProj.coneA(z, amat1);
          sd += ans.df;
        }
        edf0[1] = sd / nloop;
      }

      Variance variance = new Variance();
      //One Jump
      if (jp) {
        int mj = delta.numCols() + 2;
        double sd = 0.0;

        //prepare data used in the later use
        SimpleMatrix smat0 = new SimpleMatrix(k+3, mj);
        double[][] slopes = bqspl.slopes;
        for (int j = 0; j < slopes.length; j++) {
          double[] row = slopes[j];
          for (int i = 0; i < row.length; i++) {
            smat0.set(j, i, -1 * row[i]);
          }
        }
        smat0.set(k, mj-2, 1.0);

        SimpleMatrix djump0 = new SimpleMatrix(n, mj);
        for (int i = 0; i < n; i++) {
          djump0.setRow(i, 0, bqspl.bmat[i]);
        }

        for (int j = 0; j < nloop; j++) {
          SimpleMatrix ys = ysVector.getData();
          double[] yss = ysVector.getData().getMatrix().getData();
          double minsse = variance.evaluate(yss) * (yss.length - 1);

          //TODO: optimize this section
          int df = 0;
          for (int i = 0; i < n-1; i++) {
            SimpleMatrix smat = new SimpleMatrix(smat0);
            SimpleMatrix djump = new SimpleMatrix(djump0);
            double tmpMean = 0;

            double ri = (2 * (i + 1) + 1) / 2.0;
            for (int ik = 0; ik <= i; ik++) {
              djump.set(ik, mj - 2, ri - n);
              djump.set(ik, mj - 1, 0);
            }

            double halfStep = (x[i] + x[i + 1]) / 2.0;
            for (int ik = i + 1; ik < n; ik++) {
              djump.set(ik, mj - 2, ri);

              //in R, the following line use a comparison, since x is assume to be sorted,
              //we can bypass the comparison.
              //TODO: evalute this to be the case all the time.
              //djump[(i + 1):n, mj] = x[x > (x[i] + x[i + 1])/2] - (x[i] + x[i + 1])/2
              double tmp = x[ik] - halfStep;
              djump.set(ik, mj - 1, tmp);
              tmpMean += tmp;
            }
            tmpMean /= n;

            for (int ik = 0; ik < n; ik++) {
              double tmp = djump.get(ik, mj - 1);
              djump.set(ik, mj - 1, tmp - tmpMean);
            }

            //in R, not needed here.
//            kn = 1:(k + 3) < 0
//            kn[1:k] = knots > (x[i] + x[i + 1])/2

            //in R, smat[,mj] = 0, this seems redundant, as it is always 0.
            for (int ik = 0; ik < k; ik++) {
              if (bqspl.knots[ik] > halfStep) {
                smat.set(ik, mj - 1, -1);
              }
            }
            smat.set(k + 1, mj - 1, -1);

            //in R, ansi = -sl((x[i] + x[i + 1])/2, knots, slopes)
            double[] ansi = bqspl.sl(halfStep);
            for (int ik = 0; ik < ansi.length; ik++) {
              smat.set(k + 1, ik, -1 * ansi[ik]);
              smat.set(k + 2, ik, -1 * ansi[ik]);
            }

//            int useCount = k + 3;
//            boolean[] use = new boolean[useCount];
//            Arrays.fill(use, true);

            //Since x is already sorted
            double kMin = Double.POSITIVE_INFINITY;
            double kMax = Double.NEGATIVE_INFINITY;
            for (int ik = 0; ik < bqspl.knots.length; ik++) {
              double tmp = bqspl.knots[ik];
              if (tmp > halfStep) {
                kMin = kMin > tmp ? tmp : kMin;
              }
              //in R code, only < is checked.
              if (tmp <= halfStep) {
                kMax = kMax > tmp ? kMax : tmp;
              }
            }

            int sum1 = 0;
            int sum2 = 0;
            for (int ik = 0; ik < x.length; ik++) {
              if (x[ik] > halfStep && x[ik] <= kMin) {
                sum1++;
              }
              if (x[ik] < halfStep && x[ik] >= kMax) {
                sum2++;
              }
            }

            //TODO: need to optimize the following matrix manipulation
            //may be it is more efficient to use double[][]
            SimpleMatrix nsmat = smat.extractMatrix(0, k + 1, 0, SimpleMatrix.END);
            SimpleMatrix ndjump = djump.extractMatrix(0, SimpleMatrix.END, 0, SimpleMatrix.END);

            if (sum1 > 0) {
              nsmat = nsmat.combine(k + 1, 0, smat.extractMatrix(k + 1, k + 2, 0, SimpleMatrix.END));
            }
            if (sum2 > 0) {
              nsmat = nsmat.combine(k + 2, 0, smat.extractMatrix(k + 2, k + 3, 0, SimpleMatrix.END));
            }

            //in R, it use subset for both first and last index, which seems unnecessary
            //as on the first index, the element has already been dropped.
            //FIXME: URGENT!!!!

//            if (i == 1 | i == (n - 1)) {
//              smat = smat[, -mj, drop = FALSE]
//              djump = djump[, -mj, drop = FALSE]
//            }
            if (i == 0 || i == n-2) {
              nsmat = nsmat.extractMatrix(0, SimpleMatrix.END, 0, nsmat.numCols()-1);
              ndjump = ndjump.extractMatrix(0, SimpleMatrix.END, 0, ndjump.numCols()-1);
            }

            SimpleMatrix qv2 = ndjump.transpose().mult(ndjump);
            //TODO: can we reuse chol from above
            CholeskyDecomposition<DenseMatrix64F> chol2 = DecompositionFactory.chol(ndjump.numCols(), false);
            if (!chol2.decompose(qv2.getMatrix())) {
              throw new RuntimeException("Cholesky failed!");
            }

            //since choleskydecomposition is in place, the following line may not be necessary
            SimpleMatrix umat = SimpleMatrix.wrap(chol2.getT(null));
            SimpleMatrix uinv = umat.invert();
            SimpleMatrix pmult = uinv.transpose().mult(ndjump.transpose());

            SimpleMatrix z = pmult.mult(ys);
            SimpleMatrix amat = nsmat.mult(uinv);

            PolarConeProjectionResult fiti = ConeProj.coneA(z, amat);
            SimpleMatrix theta = ndjump.mult(uinv).mult(fiti.thetahat);
            SimpleMatrix diff = ys.minus(theta);
            double ssei = diff.dot(diff);

            if (ssei < minsse) {
              //R code
              //ijump = i;
              //ijump = i
              //sse3 = ssei
              //thb = theta
              df = fiti.df;
              minsse = ssei;
            }
          }
          sd += df;
        }
        edf0[2] = sd / nloop + 1.0;
      }

      //vee and invee shape
      double[] b = new double[k];
      if (vee || invee) {
        double sd = 0;
        for (int i = 0; i < nloop; i++) {
          SimpleMatrix ys = ysVector.getData();
          SimpleMatrix cv = delta.transpose().mult(ys);

          double[] yss = ysVector.getData().getMatrix().getData();
          double minsse = variance.evaluate(yss) * (yss.length - 1);

          SimpleMatrix av = new SimpleMatrix(bqspl.slopes);
          int df = 0;
          for (int j = 1; j < k; j++) {
            SimpleMatrix av1 = av.negative();
            av1 = av1.combine(j, 0, av.extractMatrix(j, k, 0, SimpleMatrix.END));

            Arrays.fill(b, 0);
            PolarConeProjectionResult qans = ConeProj.qprog(qv, cv, av1, b);

            SimpleMatrix theta = delta.mult(qans.thetahat);
            SimpleMatrix diff = ys.minus(theta);
            double sse = diff.dot(diff);

            if (sse < minsse) {
              minsse = sse;
              df = qans.df;
            }
          }
          sd += df;
        }
        double edf4 = sd / nloop + 1;
        edf0[3] = edf4;
        edf0[4] = edf4;
      }

      //increasing shape
      if (inc) {
        edf0[5] = 1.5;
      }

      //double jump
      if (db) {
        int mj = delta.numCols() + 4;
        double sd = 0;

        //prepare data used in the later use,
        //TODO: is using combine more efficient here?
        SimpleMatrix smat0 = new SimpleMatrix(k+6, mj);
        double[][] slopes = bqspl.slopes;
        for (int j = 0; j < slopes.length; j++) {
          double[] row = slopes[j];
          for (int i = 0; i < row.length; i++) {
            smat0.set(j, i, -1 * row[i]);
          }
        }
        smat0.set(k, mj-4, 1.0);

        SimpleMatrix djump0 = new SimpleMatrix(n, mj);
        for (int i = 0; i < n; i++) {
          djump0.setRow(i, 0, bqspl.bmat[i]);
        }

        for (int iloop = 0; iloop < nloop; iloop++) {
          SimpleMatrix ys = ysVector.getData();
          double[] yss = ysVector.getData().getMatrix().getData();
          double minsse = variance.evaluate(yss) * (yss.length - 1);

          int df = 0;
          for (int i = 0; i < n-4; i++) {
            for (int j = i+3; j < n-1; j++) {
              SimpleMatrix smat = new SimpleMatrix(smat0);
              SimpleMatrix djump = new SimpleMatrix(djump0);
              double tmpMean = 0;

              double ri = (2 * (i + 1) + 1) / 2.0;
              for (int ik = 0; ik <= i; ik++) {
                djump.set(ik, mj - 4, ri - n);
                djump.set(ik, mj - 3, 0);
              }

              //in R, djump[(i + 1):n, mj - 2] = x[x > (x[i] + x[i + 1])/2] - (x[i] + x[i + 1])/2
              double halfStep = (x[i] + x[i + 1]) / 2.0;
              for (int ik = i + 1; ik < n; ik++) {
                djump.set(ik, mj - 4, ri);

                double tmp = x[ik] - halfStep;
                djump.set(ik, mj - 3, tmp);
                tmpMean += tmp;
              }
              tmpMean /= n;

              for (int ik = 0; ik < n; ik++) {
                double tmp = djump.get(ik, mj - 3);
                djump.set(ik, mj - 3, tmp - tmpMean);
              }

              double rj = (2 * (j + 1) + 1) / 2.0;
              for (int ik = 0; ik <= j; ik++) {
                djump.set(ik, mj - 2, rj - n);
                djump.set(ik, mj - 1, 0);
              }

              //in R, djump[(j + 1):n, mj] = x[x > (x[j] + x[j + 1])/2] - (x[j] + x[j + 1])/2
              tmpMean = 0;
              double halfJStep = (x[j] + x[j + 1]) / 2.0;
              for (int ik = j + 1; ik < n; ik++) {
                djump.set(ik, mj - 2, rj);

                double tmp = x[ik] - halfJStep;
                djump.set(ik, mj - 1, tmp);
                tmpMean += tmp;
              }
              tmpMean /= n;

              for (int ik = 0; ik < n; ik++) {
                double tmp = djump.get(ik, mj - 1);
                djump.set(ik, mj - 1, tmp - tmpMean);
              }

              //in R, not needed here.
//              kn = 1:(k + 6) < 0
//              kn[1:k] = knots > (x[i] + x[i + 1])/2

              for (int ik = 0; ik < k; ik++) {
                if (bqspl.knots[ik] > halfStep) {
                  smat.set(ik, mj - 3, -1);
                }
                if (bqspl.knots[ik] > halfJStep) {
                  smat.set(ik, mj - 1, -1);
                }
              }
              smat.set(k + 1, mj - 3, -1);
              smat.set(k + 4, mj - 1, -1);
              smat.set(k + 4, mj - 3, -1);

              double[] ansi = bqspl.sl(halfStep);
              for (int ik = 0; ik < ansi.length; ik++) {
                smat.set(k + 1, ik, -1.0 * ansi[ik]);
                smat.set(k + 2, ik, -1.0 * ansi[ik]);
              }

              double[] ansj = bqspl.sl(halfJStep);
              for (int ik = 0; ik < ansi.length; ik++) {
                smat.set(k + 4, ik, ansi[ik]);
                smat.set(k + 5, ik, ansi[ik]);
              }
              smat.set(k + 5, mj - 3, -1);

              //in R, use = 1:(k + 6) > 0
              double kpiMin = Double.POSITIVE_INFINITY;
              double kpiMax = Double.NEGATIVE_INFINITY;
              double kpjMin = Double.POSITIVE_INFINITY;
              double kpjMax = Double.NEGATIVE_INFINITY;

              for (int ik = 0; ik < bqspl.knots.length; ik++) {
                double tmp = bqspl.knots[ik];
                if (tmp > halfStep) {
                  kpiMin = kpiMin > tmp ? tmp : kpiMin;
                }
                if (tmp <= halfStep) {
                  kpiMax = kpiMax > tmp ? kpiMax : tmp;
                }

                if (tmp > halfJStep) {
                  kpjMin = kpjMin > tmp ? tmp : kpjMin;
                }
                if (tmp <= halfJStep) {
                  kpjMax = kpjMax > tmp ? kpjMax : tmp;
                }
              }

              int iSum1 = 0;
              int iSum2 = 0;
              int jSum1 = 0;
              int jSum2 = 0;
              for (int ik = 0; ik < x.length; ik++) {
                double tmp = x[ik];
                if (tmp > halfStep && tmp <= kpiMin) {
                  iSum1++;
                }
                if (tmp < halfStep && tmp >= kpiMax) {
                  iSum2++;
                }

                if (tmp > halfJStep && tmp <= kpiMin) {
                  jSum1++;
                }
                if (tmp < halfJStep && tmp >= kpiMax) {
                  jSum2++;
                }
              }

              SimpleMatrix nsmat = smat.extractMatrix(0, k + 1, 0, SimpleMatrix.END);

              if (iSum1 > 0) {
                nsmat = nsmat.combine(k + 1, 0, smat.extractMatrix(k + 1, k + 2, 0, SimpleMatrix.END));
              }
              if (iSum2 > 0) {
                nsmat = nsmat.combine(k + 2, 0, smat.extractMatrix(k + 2, k + 3, 0, SimpleMatrix.END));
              }

              nsmat = nsmat.combine(k + 3, 0, smat.extractMatrix(k + 3, k + 4, 0, SimpleMatrix.END));

              if (jSum1 > 0) {
                nsmat = nsmat.combine(k + 4, 0, smat.extractMatrix(k + 4, k + 5, 0, SimpleMatrix.END));
              }
              if (jSum2 > 0) {
                nsmat = nsmat.combine(k + 5, 0, smat.extractMatrix(k + 5, k + 6, 0, SimpleMatrix.END));
              }

              if (i == 0) {
                SimpleMatrix tmp = nsmat.extractMatrix(0, SimpleMatrix.END, 0, mj - 3);
                nsmat = tmp.combine(0, mj - 3, nsmat.extractMatrix(0, SimpleMatrix.END, mj - 2, SimpleMatrix.END));

                tmp = djump.extractMatrix(0, SimpleMatrix.END, 0, mj - 3);
                djump = tmp.combine(0, mj - 3, djump.extractMatrix(0, SimpleMatrix.END, mj - 2, SimpleMatrix.END));
              } else if (i == n - 2) {
                SimpleMatrix tmp = nsmat.extractMatrix(0, SimpleMatrix.END, 0, mj - 1);
                nsmat = tmp.combine(0, mj - 1, nsmat.extractMatrix(0, SimpleMatrix.END, mj, SimpleMatrix.END));

                tmp = djump.extractMatrix(0, SimpleMatrix.END, 0, mj - 1);
                djump = tmp.combine(0, mj - 1, djump.extractMatrix(0, SimpleMatrix.END, mj, SimpleMatrix.END));
              }

              //in R, umat = chol(crossprod(djump))
              SimpleMatrix crossProductDJump = djump.transpose().mult(djump);
              CholeskyDecomposition<DenseMatrix64F> chol3 = DecompositionFactory.chol(djump.numCols(), false);
              if (!chol3.decompose(crossProductDJump.getMatrix())) {
                throw new RuntimeException("Cholesky failed!");
              }

              //since choleskydecomposition is in place, the following line may not be necessary
              SimpleMatrix umat = SimpleMatrix.wrap(chol3.getT(null));
              SimpleMatrix uinv = umat.invert();
              SimpleMatrix pmult = uinv.transpose().mult(djump.transpose());

              SimpleMatrix z = pmult.mult(ys);
              SimpleMatrix amat = nsmat.mult(uinv);

              PolarConeProjectionResult fiti = ConeProj.coneA(z, amat);
              SimpleMatrix theta = djump.mult(uinv).mult(fiti.thetahat);
              SimpleMatrix diff = ys.minus(theta);
              double ssei = diff.dot(diff);

              if (ssei < minsse) {
//                ijump = i
//                jjump = j
//                sse7 = ssei
//                thb = theta
                df = fiti.df;
                minsse = ssei;
              }
            }
          }
          sd += df;
        }
        edf0[6] = sd / nloop + 2;
      }
      return edf0;
    }

    private double[][] makeCopy(double[][] src) {
      double[][] result = new double[src.length][];
      for (int i = 0; i < src.length; i++) {
        result[i] = src[i].clone();
      }
      return result;
    }


    /**
     * A internal method to
     * @param ys
     */
    private void generateRandom(SimpleMatrix ys) {
      int n = ys.numCols();
    }


    public double[] getEDF0S(int index) {
      return edf0s[index];
    }
  }
  //endregion

  //region inner class Bqspl
  /**
   * Represent R function bqspl.
   * bqspl = function (x, m, knots = NULL, pic = FALSE, spl = TRUE)
   *
   * this class is a helper class.
   *
   * TODO: can we optimize this class.
   *
   */
  class Bqspl {
    boolean initialized = false;
    double[] knots;

    //the following two variables will be initialized when spl is true
    double[][] bmat;
    double[][] slopes;

    //the following two variables will be initialized when pic is true.
    double[] xpl;
    double[][] bpl;

    /**
     * The R function is bqspl = function (x, m, knots = NULL, pic = FALSE, spl = TRUE).
     * For the purpose of ShapeSelectForest, caller always use the default parameters,
     * therefore, knots, pic, and spl were not used as parameter, but are kept as defaults as
     * local variables in this implementation.
     *
     * @param x
     * @param k
     */
    public void bqspl(double[] x, int k) {
      this.bqspl(x, k, true, true);
    }

    /**
     * to call this function, bqspl() has to be call before.
     *
     * FIXME: This implementation assumes that xinterp is within range of knots.
     *
     * @param xinterp
     * @return
     */
    public double[] sl(double xinterp) {
      int nk = knots.length;
      int[] obs = new int[nk];
      for (int i = 0; i < nk; i++) {
        obs[i] = i;
      }

      int id = -1;
      for (int i = 0; i < nk-1; i++) {
        if (knots[i] - xinterp == 0) {
          id = i;
          break;
        }
        else {
          if (knots[i] < xinterp && knots[i+1] > xinterp) {
            id = i;
            break;
          }
        }
      }

      double k1 = knots[id];
      double k2 = knots[id+1];
      double[] sinterp = new double[nk+1];
      double a = (xinterp - k1) / (k2 - k1);

      for (int i = 0; i < nk+1; i++) {
        sinterp[i] = a * (slopes[id+1][i] - slopes[id][i]) + slopes[id][i];
      }

      return sinterp;
    }

    /**
     * A helper class to simulate the default R quantile function.
     */
    public class RPercentile {
      double[] data;

      public RPercentile(double[] v) {
        this.data = v;
      }

      public double evaluate(double p) {
        int n = data.length;
        double index = 1 + (n - 1) * p;
        int hi = (int) index;
        double h = index - hi;
        if (hi > n - 1)
          return data[n - 1];
        else if (hi < 0) {
          return data[0];
        } else {
          return (1 - h) * data[hi - 1] + h * data[hi];
        }
      }
    }

    public void bqspl(double[] x, int m, boolean pic, boolean spl) {
      double[] xu = new DoubleRBTreeSet(x).toDoubleArray();

      //TODO: is this necessary? If it is always equally spaced, then no need for this calculation.
      RPercentile percentile = new RPercentile(xu);
      knots = new double[m];
      for (int i = 1; i < m; i++) {
        knots[i] = percentile.evaluate(1.0 * i / (m - 1));
      }

      //TODO: combine knots with t.
      double[] t = new double[m + 4];
      t[0] = t[1] = xu[0];
      t[m + 2] = t[m + 3] = xu[xu.length - 1];
      System.arraycopy(knots, 0, t, 2, m);

      int n = x.length;

      //Initialize class variable
      bmat = new double[n][m + 1];
      slopes = new double[m][m + 1];

      //region spl
      if (spl) {
        //TODO: optimize this section by taking advantage of presorted x.
        for (int i = 0; i < n; i++) {
          double tmpX = x[i];
          if (tmpX <= t[3] && tmpX >= t[2]) {
            double r = (t[3] - tmpX) / (t[3] - t[2]) / (t[3] - t[1]);
            bmat[i][0] = (t[3] - tmpX) * r;
            bmat[i][1] = (tmpX - t[1]) * r + (t[4] - tmpX) / (t[4] - t[2]) * (tmpX - t[2]) / (t[3] - t[2]);
          }

          if (tmpX <= t[4] && tmpX >= t[3]) {
            bmat[i][1] = (t[4] - tmpX) / (t[4] - t[2]) * (t[4] - tmpX) / (t[4] - t[3]);
          }

          for (int j = 2; j < m - 1; j++) {
            if (tmpX <= t[j + 1] && tmpX >= t[j]) {
              bmat[i][j] = (tmpX - t[j]) * (tmpX - t[j]) / (t[j + 2] - t[j]) / (t[j + 1] - t[j]);
            }

            if (tmpX <= t[j + 2] && tmpX >= t[j + 1]) {
              bmat[i][j] = (tmpX - t[j]) * (t[j + 2] - tmpX) / (t[j + 2] - t[j]) / (t[j + 2] - t[j + 1]) +
                  (t[j + 3] - tmpX) * (tmpX - t[j + 1]) / (t[j + 3] - t[j + 1]) / (t[j + 2] - t[j + 1]);
            }

            if (tmpX <= t[j + 3] & tmpX >= t[j + 2]) {
              bmat[i][j] = (t[j + 3] - tmpX) * (t[j + 3] - tmpX) / (t[j + 3] - t[j + 1]) / (t[j + 3] - t[j + 2]);
            }
          }

          if (tmpX <= t[m] && tmpX >= t[m - 1]) {
            bmat[i][m - 1] = (tmpX - t[m - 1]) * (tmpX - t[m - 1]) / (t[m + 1] - t[m - 1]) / (t[m] - t[m - 1]);
          }

          if (tmpX <= t[m + 1] && tmpX >= t[m]) {
            bmat[i][m - 1] = (tmpX - t[m - 1]) * (t[m + 1] - tmpX) / (t[m + 1] - t[m - 1]) / (t[m + 1] - t[m]) +
                (t[m + 2] - tmpX) * (tmpX - t[m]) / (t[m + 2] - t[m]) / (t[m + 1] - t[m]);

            bmat[i][m] = (tmpX - t[m]) * (tmpX - t[m]) / (t[m + 2] - t[m]) / (t[m + 1] - t[m]);
          }
        }

        slopes[0][0] = -2.0 / (t[3] - t[1]);
        slopes[0][1] = 2.0 / (t[3] - t[1]);
        slopes[1][1] = -2.0 / (t[4] - t[2]);

        for (int i = 2; i < m - 1; i++) {
          slopes[i - 1][i] = 2.0 / (t[i + 2] - t[i]);
          slopes[i][i] = -2.0 / (t[i + 3] - t[i + 1]);
        }

        slopes[m - 2][m - 1] = 2.0 / (t[m + 1] - t[m - 1]);
        slopes[m - 1][m - 1] = -2.0 / (t[m + 2] - t[m]);
        slopes[m - 1][m] = 2.0 / (t[m + 2] - t[m]);
      }
      //endregion

      //region pic
      int N = 1001;
      if (pic) {
        xpl = new double[N];
        for (int i = 0; i < N; i++) {
          xpl[i] = i / (N - 1.0) * (xu[xu.length - 1] - xu[0]) + xu[0];
        }

        bpl = new double[N][m + 1];

        for (int i = 0; i < N; i++) {
          double tmpX = xpl[i];

          if (tmpX <= t[3] && tmpX >= t[2]) {
            double r = (t[3] - tmpX) / (t[3] - t[2]) / (t[3] - t[1]);
            bpl[i][0] = (t[3] - tmpX) * r;
            bpl[i][1] = (tmpX - t[1]) * r + (t[4] - tmpX) / (t[4] - t[2]) * (tmpX - t[2]) / (t[3] - t[2]);
          }

          if (tmpX <= t[4] && tmpX >= t[3]) {
            bpl[i][1] = (t[4] - tmpX) / (t[4] - t[2]) * (t[4] - tmpX) / (t[4] - t[3]);
          }

          for (int j = 2; j < m - 1; j++) {
            if (tmpX <= t[j + 1] && tmpX >= t[j]) {
              bpl[i][j] = (tmpX - t[j]) * (tmpX - t[j]) / (t[j + 2] - t[j]) / (t[j + 1] - t[j]);
            }

            if (tmpX <= t[j + 2] && tmpX >= t[j + 1]) {
              bpl[i][j] = (tmpX - t[j]) * (t[j + 2] - tmpX) / (t[j + 2] - t[j]) / (t[j + 2] - t[j + 1])
                  + (t[j + 3] - tmpX) * (tmpX - t[j + 1]) / (t[j + 3] - t[j + 1]) / (t[j + 2] - t[j + 1]);
            }

            if (tmpX <= t[j + 3] && tmpX >= t[j + 2]) {
              bpl[i][j] = (t[j + 3] - tmpX) * (t[j + 3] - tmpX) / (t[j + 3] - t[j + 1]) / (t[j + 3] - t[j + 2]);
            }
          }

          //last two
          if (tmpX <= t[m] && tmpX >= t[m - 1]) {
            bpl[i][m - 1] = (tmpX - t[m - 1]) * (tmpX - t[m - 1]) / (t[m + 1] - t[m - 1]) / (t[m] - t[m - 1]);
          }

          if (tmpX <= t[m + 1] && tmpX >= t[m]) {
            bpl[i][m - 1] = (tmpX - t[m - 1]) * (t[m + 1] - tmpX) / (t[m + 1] - t[m - 1]) / (t[m + 1] - t[m]) + (t[m + 2] - tmpX) * (tmpX - t[m]) / (t[m + 2] - t[m]) / (t[m + 1] - t[m]);

            bpl[i][m] = (tmpX - t[m]) * (tmpX - t[m]) / (t[m + 2] - t[m]) / (t[m + 1] - t[m]);
          }
        }
      }
      //endregion

      initialized = true;
    }
  }
  //endregion

  //region inner class SL
  class SL {

  }
  //endregion

  //region class constant
  //Shape selection criterion
  private static final int BIC = 1;
  private static final int CIC = 2;

  private static final int FLAT = 1;
  private static final int DECREASING = 2;
  private static final int ONE_JUMP = 3;
  private static final int INVERTED_VEE = 4;
  private static final int VEE = 5;
  private static final int INCREASING = 6;
  private static final int DOUBLE_JUMP = 7;

  //endregion

  //region instance variable
  private final Args args;
  private BitSet shapes = new BitSet(8); //shape requested
  private double[] edf0; //TODO: R code use a dynamic array, evaluate to see if need to change

  private EDF edf = new EDF();
  private Bqspl bqspl = new Bqspl();

  //endregion

  public ShapeSelectForest() {this(new Args());}

  public ShapeSelectForest(Args args) {
    this.args = args;
    if (args.allowFlat) {shapes.set(this.FLAT);}
    if (args.allowDecreasing) {shapes.set(this.DECREASING);}
    if (args.allowOneJump) {shapes.set(this.ONE_JUMP);}
    if (args.allowInvertedVee) {shapes.set(this.INVERTED_VEE);}
    if (args.allowVee) {shapes.set(this.VEE);}
    if (args.allowIncreasing) {shapes.set(this.INCREASING);}
    if (args.allowDoubleJump) {shapes.set(this.DOUBLE_JUMP);}

    System.out.println(shapes.cardinality());
  }

  public void shape() {

  }

  /**
   * Given a scatterplot of (x_i,y_i), i=1,…,n, constrained least-squares spline fits are obtained for all of the following shapes:
   *
   * 1. flat
   * 2. decreasing
   * 3. one-jump, i.e., decreasing, jump up, decreasing
   * 4. inverted vee (increasing then decreasing)
   * 5. vee (decreasing then increasing)
   * 6. increasing (linear)
   * 7. double-jump, i.e., decreasing, jump up, decreasing, jump up, decreasing.
   *
   * The "shape" routine chooses one of the seven shapes based on the minimum Bayes information criterion (BIC)
   * or the cone information criterion (CIC). It also returns the information criterion (IC) values for all
   * shape options allowed by the user. Fitting method is constrained quadratic B-splines, number of knots
   * depends on number of observations.
   *
   * TODO: in R shape implementation, user can provide edf0 and get.edf0, which controls how to get edf0.
   * need to evaluate whether we need to specify these.
   *
   * TODO: in R shape implementation, there is a random parameter which randomly choose which shape to check for.
   * which is not implemented here.
   *
   * @param x years
   * @param ymat response vector corresponding to x
   */
  public void getResult(double[] x, double[] ymat) {
    //TODO: CWN currently it is assumed that x is sorted ascending

    int n = x.length;
    double xMin = x[0];
    double xRange = x[n-1] - xMin;

    //scale x variables
    /*
    TODO: CWN consider moving this section of logic to the caller of getResult(), this can avoid repeasted calculation
     */
    boolean equalSpace = true;
    double step = x[1] - x[0];
    double prev = x[0];
    for (int i = 1; i < n; i++) {
      if (x[i]-prev != step) {
        equalSpace = false;
      }
      prev = x[i];
      x[i] = (x[i] - xMin) / xRange;
    }
    x[0] = 0;

    int nsh = shapes.cardinality();

    //TODO: this section could be in the class initialization, but the problem is that if there is missing value,
    // edf0 has to be reconstructed, which can not be set on the class ahead of time.
    if (equalSpace && n <= 40 & n >= 20) {
      edf0 = edf.getEDF0S(n - 20);
    }
    else {
      edf0 = edf.computeEdf0(x);
    }

    /***************************************
     * in R: ny = length(ymax) / n
     *
     * In this implementation, there are alays a single y,
     * but we could potentially using multiple indices here.
     *
     * TODO: evaluate possiblity of running multiple indices, when do that ny will be dynamic.
     ***************************************/
    int ny = 1;

    int k0 = (int)(4 + Math.round(Math.pow(n, 1.0/7)) + 2);

    int k1 = k0;

    int k = 0;

    while(k==0) {
      int pts = (n - 2) / (k1 - 1);
      int rem_pts = (n - 2) % (k1 - 1);

      if (pts > 2) {
        k = k1;
      }
      else if (pts == 2) {
        if (rem_pts / (k1 - 1) >= 1) {
          k = k1;
        }
        else {
          k1--;
        }
      }
      else {
        k1--;
      }
    }

    int[] kobs = new int[k];
    for (int i = 0; i < k; i++) {
      kobs[i] = i + 1;
    }

    //R: ans = bqspl(x, k, knots = NULL, pic = FALSE)
    bqspl.bqspl(x, k, false, true);

    SimpleMatrix delta = new SimpleMatrix(bqspl.bmat);
    SimpleMatrix qv = delta.transpose().mult(delta);

    int m0 = delta.numRows();

    //choleskydecomposition is implemented in place
    CholeskyDecomposition<DenseMatrix64F> chol = DecompositionFactory.chol(m0, false);
    if (!chol.decompose(qv.getMatrix())) {
      throw new RuntimeException("Cholesky failed!");
    }

    //since choleskydecomposition is in place, the following line may not be necessary
    SimpleMatrix umat0 = SimpleMatrix.wrap(chol.getT(null));
    SimpleMatrix uinv0 = umat0.invert();
    SimpleMatrix pmult0 = uinv0.transpose().mult(delta.transpose());
    SimpleMatrix amat1 = new SimpleMatrix(bqspl.slopes).scale(-1).mult(uinv0);

    //Since ny is alway 1 in current implementation
    double[] ic = new double[nsh];
    //double[][] thetab = new double[ny][n];
    double[] thetab = new double[n];
    double[][] fit = new double[nsh][n];
    double[] thb = new double[n];

    //the best shape for this pixel
    int shb = 0;

    int[] ijps0 = new int[nsh];
    int[] jjps0 = new int[nsh];
    int ijps = 0; //the position of the first jump
    int jjps = 0; //the position of the second jump

    double[] m_is0 = new double[nsh];
    double[] m_js0 = new double[nsh];
    double m_is = 0; //centering values of the first ramp edge
    double m_js = 0; //centering values of the second ramp edge

    ArrayList<double[]> bs = new ArrayList<>();
    ArrayList<double[]> bhat = new ArrayList<>();

    int ish = -1;

    SimpleMatrix y = SimpleMatrix.wrap(new DenseMatrix64F(ymat.length, 1, true, ymat));

    Variance variance = new Variance();
    double sse1 = variance.evaluate(ymat) * (ymat.length - 1);

    Mean mean = new Mean();
    double meanY = mean.evaluate(ymat);

    //check for flat shape
    if (shapes.get(this.FLAT)) {
      ish++;
      Arrays.fill(fit[ish], meanY);
      bhat.add(new double[]{0});

      if (args.infoCriterion == BIC) {
        ic[ish] = n * Math.log(sse1) + Math.log(n);
      }
      else { //use CIC
        ic[ish] = Math.log(sse1) + Math.log(2.0 / (n - 1) + 1);
      }
    }

    //check for decreasing shape
    if (shapes.get(this.DECREASING)) {
      ish++;

      SimpleMatrix z = pmult0.mult(y);
      PolarConeProjectionResult ans = ConeProj.coneA(z, amat1);

      SimpleMatrix fitted = delta.mult(uinv0).mult(ans.thetahat);
      fit[ish] = fitted.getMatrix().getData();
      bhat.add(uinv0.mult(ans.thetahat).getMatrix().getData());

      SimpleMatrix diff = y.minus(fitted);
      double sse2 = diff.dot(diff);

      if (args.infoCriterion == BIC) {
        ic[ish] = n * Math.log(sse2) + Math.log(n) * edf0[ish];
      }
      else {
        ic[ish] = Math.log(sse2) + Math.log(2.0 * (edf0[ish] + 1) / (n - 1 - 1.5 * edf0[ish]) + 1);
      }

      //TODO: In R, there is the following condition checking, do we need to handle this?
//      if ((2 * (edf0[ish] + 1)/(n - 1 - 1.5 * edf0[ish]) +
//          1) < 0) {
//        stop("The sample size n is too small to make a valid CIC for a decreasing shape!")
//      }
    }

    if (shapes.get(ONE_JUMP)) {
      ish++;
      int mj = delta.numCols() + 2;
      double minsse = sse1;

      //TODO: optimize this section
      //This section is almost identical as EDF.getEDF0() for jp section. They should be consolidated
      //into a single implementation.

      //prepare data used in the later use
      SimpleMatrix smat0 = new SimpleMatrix(k+3, mj);
      double[][] slopes = bqspl.slopes;
      for (int j = 0; j < slopes.length; j++) {
        double[] row = slopes[j];
        for (int i = 0; i < row.length; i++) {
          smat0.set(j, i, -1 * row[i]);
        }
      }
      smat0.set(k, mj-2, 1.0);

      SimpleMatrix djump0 = new SimpleMatrix(n, mj);
      for (int i = 0; i < n; i++) {
        djump0.setRow(i, 0, bqspl.bmat[i]);
      }

      //TODO: will this variable accumalate over the different shape runs?
      //it does not seem to have an effect on the result. Consider remove it.
      int df = 0;
      for (int i = 0; i < n-1; i++) {
        SimpleMatrix smat = new SimpleMatrix(smat0);
        SimpleMatrix djump = new SimpleMatrix(djump0);
        double tmpMean = 0;

        double ri = (2 * (i + 1) + 1) / 2.0;
        for (int ik = 0; ik <= i; ik++) {
          djump.set(ik, mj - 2, ri - n);
          djump.set(ik, mj - 1, 0);
        }

        double halfStep = (x[i] + x[i + 1]) / 2.0;
        for (int ik = i + 1; ik < n; ik++) {
          djump.set(ik, mj - 2, ri);

          //in R, the following line use a comparison, since x is assume to be sorted,
          //we can bypass the comparison.
          //TODO: evalute this to be the case all the time.
          //djump[(i + 1):n, mj] = x[x > (x[i] + x[i + 1])/2] - (x[i] + x[i + 1])/2
          double tmp = x[ik] - halfStep;
          djump.set(ik, mj - 1, tmp);
          tmpMean += tmp;
        }
        tmpMean /= n;

        for (int ik = 0; ik < n; ik++) {
          double tmp = djump.get(ik, mj - 1);
          djump.set(ik, mj - 1, tmp - tmpMean);
        }

        //in R, not needed here.
//            kn = 1:(k + 3) < 0
//            kn[1:k] = knots > (x[i] + x[i + 1])/2

        //in R, smat[,mj] = 0, this seems redundant, as it is always 0.
        for (int ik = 0; ik < k; ik++) {
          if (bqspl.knots[ik] > halfStep) {
            smat.set(ik, mj - 1, -1);
          }
        }
        smat.set(k + 1, mj - 1, -1);

        //in R, ansi = -sl((x[i] + x[i + 1])/2, knots, slopes)
        double[] ansi = bqspl.sl(halfStep);
        for (int ik = 0; ik < ansi.length; ik++) {
          smat.set(k + 1, ik, -1 * ansi[ik]);
          smat.set(k + 2, ik, -1 * ansi[ik]);
        }

//            int useCount = k + 3;
//            boolean[] use = new boolean[useCount];
//            Arrays.fill(use, true);

        //Since x is already sorted
        double kMin = Double.POSITIVE_INFINITY;
        double kMax = Double.NEGATIVE_INFINITY;
        for (int ik = 0; ik < bqspl.knots.length; ik++) {
          double tmp = bqspl.knots[ik];
          if (tmp > halfStep) {
            kMin = kMin > tmp ? tmp : kMin;
          }
          //in R code, only < is checked.
          if (tmp <= halfStep) {
            kMax = kMax > tmp ? kMax : tmp;
          }
        }

        int sum1 = 0;
        int sum2 = 0;
        for (int ik = 0; ik < x.length; ik++) {
          if (x[ik] > halfStep && x[ik] <= kMin) {
            sum1++;
          }
          if (x[ik] < halfStep && x[ik] >= kMax) {
            sum2++;
          }
        }

        //TODO: need to optimize the following matrix manipulation
        //may be it is more efficient to use double[][]
        SimpleMatrix nsmat = smat.extractMatrix(0, k + 1, 0, SimpleMatrix.END);
        SimpleMatrix ndjump = djump.extractMatrix(0, SimpleMatrix.END, 0, SimpleMatrix.END);

        if (sum1 > 0) {
          nsmat = nsmat.combine(k + 1, 0, smat.extractMatrix(k + 1, k + 2, 0, SimpleMatrix.END));
        }
        if (sum2 > 0) {
          nsmat = nsmat.combine(k + 2, 0, smat.extractMatrix(k + 2, k + 3, 0, SimpleMatrix.END));
        }

        //in R, it use subset for both first and last index, which seems unnecessary
        //as on the first index, the element has already been dropped.
        //FIXME: URGENT!!!!

//            if (i == 1 | i == (n - 1)) {
//              smat = smat[, -mj, drop = FALSE]
//              djump = djump[, -mj, drop = FALSE]
//            }
        if (i == 0 || i == n-2) {
          nsmat = nsmat.extractMatrix(0, SimpleMatrix.END, 0, nsmat.numCols()-1);
          ndjump = ndjump.extractMatrix(0, SimpleMatrix.END, 0, ndjump.numCols()-1);
        }

        SimpleMatrix qv2 = ndjump.transpose().mult(ndjump);
        //TODO: can we reuse chol from above
        CholeskyDecomposition<DenseMatrix64F> chol2 = DecompositionFactory.chol(ndjump.numCols(), false);
        if (!chol2.decompose(qv2.getMatrix())) {
          throw new RuntimeException("Cholesky failed!");
        }

        //since choleskydecomposition is in place, the following line may not be necessary
        SimpleMatrix umat = SimpleMatrix.wrap(chol2.getT(null));
        SimpleMatrix uinv = umat.invert();
        SimpleMatrix pmult = uinv.transpose().mult(ndjump.transpose());

        SimpleMatrix z = pmult.mult(y);
        SimpleMatrix amat = nsmat.mult(uinv);

        PolarConeProjectionResult fiti = ConeProj.coneA(z, amat);
        SimpleMatrix theta = ndjump.mult(uinv).mult(fiti.thetahat);
        SimpleMatrix diff = y.minus(theta);
        double ssei = diff.dot(diff);

        if (ssei < minsse) {
          //ijp = i
          //sse3 = ssei
          //thb = theta;
          System.arraycopy(theta.getMatrix().getData(), 0, thb, 0, thb.length);
          df = fiti.df;
          minsse = ssei;
          //b3 = bi
          //m_i = m_i0
        }
      }
      System.arraycopy(thb, 0, fit[ish], 0, thb.length);
//      bhat.add()










    }






    int jnk = 0;

  }





}

