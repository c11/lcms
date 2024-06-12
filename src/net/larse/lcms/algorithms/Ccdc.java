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
package net.larse.lcms.algorithms;

import com.google.earthengine.api.base.ArgsBase;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import net.larse.lcms.helper.ArrayHelper;
import net.larse.lcms.helper.FitGenerator;
import net.larse.lcms.helper.LassoFit;
import net.larse.lcms.helper.LassoFitGenerator;
import net.larse.lcms.helper.RobustLeastSquareBisquare;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math.stat.descriptive.rank.Min;
import org.ejml.data.DenseMatrix64F;

/** Based on CCDC 12.30 */
@SuppressWarnings("ALL")
public class Ccdc {
  public static class SolveException extends Exception {};

  // TODO(gorelick): More definition for these.
  public static final int QA_NORMAL = 0;
  public static final int QA_10 = 10;
  public static final int QA_END = 20;

  // maximum number of coefficient required
  // 2 for tri-model; 2 for bi-modal; 2 for seasonality; 2 for linear;
  private static final int MIN_NUM_COEFFS = 4;
  private static final int MID_NUM_COEFFS = 6;
  private static final int MAX_NUM_COEFFS = 8;

  private static final int ROBUST_COEFFS = 5;

  // Threshold for cloud, shadow, and snow detection
  private static final double T_CONST = 4.89;

  // max number of coefficients for the model
  // NOTE: might be redundant as it seems always equals MAX_NUM_COEFFS
  private static final int NUM_COEFFS = 8;

  // number of clear observations / number of coefficients
  private static final int N_TIMES = 3;

  // minimum number of years for model initialization
  private static final int MIN_YEARS_NEEDED = 1;

  // number of days in a year
  private double SIZE_OF_A_YEAR;

  private final double OMEGA;   // Harmonic scaling term.

  // change threshold
  //  private double threshold_change = 0.99;
  // chi-square inversed T_cg (0.99) for noise removal on 5 bands
  private double T_CG;
  // chi-square inversed T_max_cg (1e-6) for last step noise removal
  private double TMAX_CG;

  // These variables are used throughout and promoted here for ease of use.
  //
  // clrx and clry are the input data, but noisy points are filtered out in various places.
  // Rather than make copies of the arrays, we shift values down and manually track the
  // new length of the array.
  private double[] clrx;
  private double[][] clry;
  private int numObservations;
  private int numberOfBands;
  // An adjustment factor when computing rmse.  It's the median of each band.
  private double[] adjRmse;
  // The minimum number of observations required to identify a change ('consen' in the matlab code).
  private int minObs;

  // We make a copy of these band lists so the Args object can accept Object (and handle Strings),
  // but we want to work with ints in here.
  final private List<Integer> tmaskBands;
  final private List<Integer> breakpointBands;

  //a data holder for segment magnitude calculation
  private double[][] vDifMag;

  // Helper objects.
  private TMask tMask;
  private final FitGenerator fitGenerator;

  public static class Args extends ArgsBase {
    // NOTE: Caller is responsible for setting the default.
    @Doc(help = "Name or index of the bands to use for change detection. If unspecified, all "
        + "bands are used.")
    @Optional
    public List<Object> breakpointBands = null;

    @Doc(help = "Name or index of the bands to use for iterative TMask cloud detection. "
        + "These are typically the green band and the SWIR2 band.  If unspecified, TMask is "
        + "not used.")
    @Optional
    public List<Object> tmaskBands = null;

    @Doc(help = "Number of observation required to flag a change.")
    @Optional
    public int minObservations = 6;

    @Doc(help = "Chi-square probability threshold for change detection in the range of [0, 1].")
    @Optional
    public double chiSquareProbability = 0.99;

    @Doc(help = "Factors of minimum number of years to apply new fiiting.")
    @Optional
    public double minNumOfYearsScaler = 1.33;

    @Doc(help = "Which date format to use: 0 = jDays, 1 = fractional years.  The start, end and "
        + "break times for each temporal segment will be encoded in this format.")
    @Optional
    public int dateFormat = 0;

    @Doc(help = "Lambda for LASSO regression fitting. "
        + "If set to 0, regular OLS is used instead of LASSO.")
    @Optional
    public double lambda = 20;

    @Doc(help = "Maximum number of runs for LASSO regression convergence. "
        + "If set to 0, regular OLS is used instead of LASSO.")
    @Optional
    public int maxIterations = 25000;
  }

  private final Args args;

  public Ccdc(Args args) {
    // breakpointBands and tmaskBands are typed as Object lists, but the caller has already
    // made sure they're Integer lists, so convert them here so we don't have to cast below.
    breakpointBands = args.breakpointBands.stream()
        .map(o -> (Integer) o).collect(Collectors.toList());

    tmaskBands = args.tmaskBands == null
        ? null
        : args.tmaskBands.stream()
          .map((obj) -> ((Integer) obj)).collect(Collectors.toList());

    // The caller needs to make sure this is filled in.
    if (args.breakpointBands == null) {
      throw new RuntimeException("Invalid state: bands must be specified.");
    }

    this.args = args;
    if (args.lambda != 0 && args.maxIterations != 0) {
      fitGenerator = new LassoFitGenerator(args.maxIterations, args.lambda);
    } else {
      fitGenerator = new FitGenerator();
    }

    try {
      this.T_CG =
          new ChiSquaredDistributionImpl(breakpointBands.size())
              .inverseCumulativeProbability(args.chiSquareProbability);
      this.TMAX_CG =
          new ChiSquaredDistributionImpl(breakpointBands.size())
              .inverseCumulativeProbability(1 - 1e-6);
    } catch (MathException e) {
      // I don't think this can ever occur, but just in case.
      throw new RuntimeException(e.getMessage());
    }

    // To support the both time formats (julianDay and fractionalYear) we
    // need to adjust the SIZE_OF_A_YEAR "constant" and the OMEGA harmonic
    // scaling factor that uses it.
    if (args.dateFormat == 0) {
      SIZE_OF_A_YEAR = 365.25;
    } else {
      SIZE_OF_A_YEAR = 1;
    }
    OMEGA = 2.0 * Math.PI / SIZE_OF_A_YEAR;
  }

  /**
   * The Model class keeps track of the state of the current fit.
   */
  private class Model {
    int startIndex;
    int endIndex;
    boolean trained;
    boolean done;
    CcdcFit currentFit;
    private final ArrayList<CcdcFit> results;

    Model() {
      startIndex = 0;
      // start with minimum requirement of clear observations
      endIndex = N_TIMES * MIN_NUM_COEFFS - 1;
      trained = false;
      results = new ArrayList<>();
    }

    /** Skip forward until we have the minimum number of points and time span. */
    int findNextSpan() {
      while (!done()
          && (endIndex - startIndex + 1 < N_TIMES * MIN_NUM_COEFFS
          || timeSpan(endIndex, startIndex) < MIN_YEARS_NEEDED)) {
        endIndex++;
      }
      return endIndex;
    }

    /**
     * Returns true if there are no more observations to be processed.
     */
    boolean done() {
      return endIndex >= numObservations - minObs;
    }

    /** Create a new fit object */
    CcdcFit newFit() {
      CcdcFit fit = new CcdcFit(numberOfBands);
      results.add(fit);
      this.currentFit = fit;
      return fit;
    }
  }

  /**
   * run ccdc for a single pixel
   *
   * @param y [BANDS][NUM_OBSERVATIONS] array of time series of spectral values
   * @param x array of image dates corresponding to each y observation.
   */
  public List<CcdcFit> getResult(double[][] y, double[] x) throws MathException {
    this.clrx = x;
    this.clry = y;
    this.numObservations = clry[0].length;
    this.numberOfBands = clry.length;

    this.minObs = args.minObservations;
    this.adjRmse = median_variogram(clrx, clry, 1, 0);

    this.vDifMag = new double[minObs][numberOfBands];

    if (tmaskBands != null) {
      double r2 = adjRmse[tmaskBands.get(0)];
      double r5 = adjRmse[tmaskBands.get(1)];
      this.tMask = new TMask(r2, r5);
    }


    Train train = new Train();
    Monitor monitor = new Monitor();
    Model model = new Model();

    while (!model.done()) {
      model.findNextSpan();

      // Train a model.
      if (!model.trained && !model.done()) {
        train.invoke(model);
        if (!model.trained) {
          continue;
        }
      }

      // Monitor to extend the model until it doesn't work.
      // It's possible to get here with isTrained == false, if we've run out of observations.
      if (model.trained && !model.done()) {
        monitor.invoke(model);
        model.endIndex++;
      }
    }

    // % Two ways for processing the end of the time series
    if (model.trained) {
      // % 1) if no break is found at the end of the time series
      noBreakFound(model, monitor.getVecMag());
    } else {
      // % 2) if a break is found close to the end of the time series.
      finish(model);
    }

    return model.results;
  }

  private void finish(Model model) {
    // % Use [conse,min_num_c*n_times+conse) to fit curve
    if (model.results.isEmpty()) {
      // TODO(gorelick): this never triggers in the test points.
      // %first curve
      model.startIndex = 0;
    } else {
      // In training mode.
      // Find the first point after the last break.
      CcdcFit fit = model.results.get(model.results.size() - 1);
      for (int tidx = fit.numObs - 1; tidx < numObservations; tidx++) {
        if (clrx[tidx] >= fit.tBreak) {
          model.startIndex = tidx;
          break;
        }
      }
    }

    if ((numObservations - model.startIndex + 1) >= minObs) {
      if (tMask != null) {
        // Mask any outliers.
        int[] mask = tMask.autoTmask(model.startIndex, numObservations - 1).getResult();
        if (ArrayHelper.first(1, mask, 0, mask.length - minObs) != -1) { // outliers exist.
          tMask.arrayMask(model.startIndex, numObservations - minObs - 1, mask);
        }
      }
    }

    // If there's enough points left, compute the fit.
    if (numObservations - model.startIndex >= minObs) {
      // % end of fit
      CcdcFit fit = model.currentFit;
      if (fit == null) {
        fit = model.newFit();
        computeFit(fit, MIN_NUM_COEFFS, model.startIndex, numObservations - 1);
      }

      fit.tBreak = 0;
      fit.changeProb = 0;
      fit.category = QA_END + MIN_NUM_COEFFS;
    }
    return;
  }

  double timeSpan(int end, int start) {
    return (clrx[end] - clrx[start]) / SIZE_OF_A_YEAR;
  }

  private void noBreakFound(Model model, double[] vecMag) {
    // % find the last stable point.
    int idLast = minObs;
    for (int iConse = minObs; iConse > 0; iConse--) {
      if (vecMag[iConse - 1] <= T_CG) {
        idLast = iConse;
        break;
      }
    }

    CcdcFit fit = model.results.get(model.results.size() - 1);


    // % define probability of change based on minObservations
    // TODO(gorelick): What should start and end be here?
    fit.changeProb = 1.0 * (minObs - idLast) / minObs;
    fit.tEnd = clrx[numObservations - 1 - minObs + idLast];
    if (minObs > idLast) { // % > 1
      // % update time of the probable change
      fit.tBreak = clrx[numObservations - minObs + idLast];
      //update segment magnitude
      int finalIdLast = idLast;
      Arrays.setAll(fit.magnitude, band -> getMedian(vDifMag, band, finalIdLast, minObs));
    }
  }

  /**
   * Determine the time series model based on the number of points.
   */
  private static int updateCft(int count) {
    int n;
    if (count < MID_NUM_COEFFS * N_TIMES) {
      n = MIN_NUM_COEFFS;
    } else if (count < MAX_NUM_COEFFS * N_TIMES) {
      n = MID_NUM_COEFFS;
    } else {
      n = MAX_NUM_COEFFS;
    }
    return Math.min(n, NUM_COEFFS);
  }

  /**
   * Get the median of the given band.
   *
   * @param data The data upon which to compute the median.
   * @param xstart start range of values to include (inclusive)
   * @param xend end range of values to include (exclusive)
   * @return
   */
  private static double getMedian(double[][] data, int band, int xstart, int xend) {
    DescriptiveStatistics stat = new DescriptiveStatistics();
    for (int i = xstart; i < xend; i++) {
      stat.addValue(data[i][band]);
    }
    return stat.getPercentile(50);
  }

  /**
   * @param dat [NUM_OBSERVATION][BANDS]
   * @param index band index
   * @return
   */
  private static double getMedian(double[][] dat, int index) {
    DescriptiveStatistics stat = new DescriptiveStatistics();
    for (double[] refls: dat) {
      stat.addValue(refls[index]);
    }

    return stat.getPercentile(50);
  }

  private double vectorMin(double[] values, int begin, int length) {
    double min = values[begin];
    for (int i = begin; i < begin + length; i++) {
      if (!Double.isNaN(values[i])) {
        min = Math.min(min, values[i]);
      }
    }
    return min;
  }

  /**
   * Auto Trends and Seasonal Fit between breaks INPUTS: x - Julian day [1; 2; 3]; y - predicted
   * reflectances [0.1; 0.2; 0.3]; df - degree of freedom (num_c)
   *
   * <p>OUTPUTS: fitCft - fitted coefficients; General model TSModel: f1(x) = a0 + b0*x (df = 2)
   * f2(x) = f1(x) + a1*cos(x*w) + b1*sin(x*w) (df = 4) f3(x) = f2(x) + a2*cos(x*2w) + b2*sin(x*2w)
   * (df = 6) f4(x) = f3(x) + a3*cos(x*3w) + b3*sin(x*3w) (df = 8)
   *

   * @param band band index to run
   * @param df degree of freedom
   * @param idxStart startIndex (inclusive)
   * @param idxEnd endIndex (inclusive)
   * @return
   */
  private void autoTSFit(CcdcFit fit, int band, int df, int idxStart, int idxEnd) {
    int count = idxEnd - idxStart + 1;

    double[] vDif = new double[idxEnd - idxStart + 1];
    double[] fitCft = new double[MAX_NUM_COEFFS];
    double rmse;

    if (fitGenerator.isLinear()) {
      // initialize LassoFitGenerator
      fitGenerator.init(df, count);

      for (int i = idxStart; i <= idxEnd; i++) {
        // fitGenerator.setObservationValues(i - idxStart, observations[i - idxStart]);
        fitGenerator.setObservation(i - idxStart, 0, 1);
        fitGenerator.setObservation(i - idxStart, 1, clrx[i]);
        int idx = 1;
        for (int k = 1; k <= df / 2 - 1; k++) {
          fitGenerator.setObservation(i - idxStart, ++idx, Math.cos(k * OMEGA * clrx[i]));
          fitGenerator.setObservation(i - idxStart, ++idx, Math.sin(k * OMEGA * clrx[i]));
        }
        fitGenerator.setTarget(i - idxStart, clry[band][i]);
      }

      System.arraycopy(fitGenerator.linearFit(), 0, fitCft, 0, df);

      // calculate rmse from predicted value and prediction difference
      double sumSquareResidual = 0;
      for (int i = idxStart; i <= idxEnd; i++) {
        double diff = predict(clrx[i], fitCft);
        double resid = clry[band][i] - diff;
        sumSquareResidual += resid * resid;
        vDif[i - idxStart] = resid;
      }
      rmse = Math.sqrt(sumSquareResidual / (idxEnd - idxStart + 1 - df));
    } else {
      // initialize LassoFitGenerator
      fitGenerator.init(df - 1, count);

      for (int i = idxStart; i <= idxEnd; i++) {
        // fitGenerator.setObservationValues(i - idxStart, observations[i - idxStart]);
        fitGenerator.setObservation(i - idxStart, 0, clrx[i]);
        int idx = 0;
        for (int k = 1; k <= df / 2 - 1; k++) {
          fitGenerator.setObservation(i - idxStart, ++idx, Math.cos(k * OMEGA * clrx[i]));
          fitGenerator.setObservation(i - idxStart, ++idx, Math.sin(k * OMEGA * clrx[i]));
        }
        fitGenerator.setTarget(i - idxStart, clry[band][i]);
      }
      LassoFit lFit = ((LassoFitGenerator) fitGenerator).lassoFit();
      int idx = lFit.getFitByLambda(args.lambda);

      rmse = lFit.adjustedRmses[idx];
      double[] coef = lFit.getBetas(idx);
      System.arraycopy(coef, 0, fitCft, 0, coef.length);

      // calculate predicted value and prediction difference
      for (int i = idxStart; i <= idxEnd; i++) {
        double diff = predict(clrx[i], fitCft);
        double resid = clry[band][i] - diff;
        // double resid2 = resid * resid;  // This is never used.  We really want plain difference?
        vDif[i - idxStart] = resid;
      }
    }

    fit.coefs[band] = fitCft;
    fit.rmse[band] = rmse;
    fit.vDif[band] = vDif;
  }

  /**
   * Auto Trends and Seasonal Predict
   *
   * @param x
   * @param fitCft
   * @return
   */
  private double predict(double xval, double[] fitCft) {
    double prediction;
    double v = fitCft[0] + fitCft[1] * xval;

    double rx = xval * OMEGA;
    for (int h = 1; h < 4; h++) {
      v += fitCft[2 + (h - 1) * 2] * Math.cos(h * rx)
          + fitCft[2 + (h - 1) * 2 + 1] * Math.sin(h * rx);
    }
    return v;
  }

  class TMask {
    private int validCount;
    private int[] result;
    double tB2;
    double tB5;

    TMask(double tB2, double tB5) {
      this.tB2 = tB2;
      this.tB5 = tB5;
    }

    /**
     * Multitepmoral cloud, cloud shadow, & snow masks (global version) read in data with 3 more
     * consecutive clear obs & correct data
     */
    private TMask autoTmask(int start, int end) {
      double years = timeSpan(end, start);
      int year = (int) Math.ceil(years);
      int nums = end - start + 1;
      int coefsCount = year > 1 ? ROBUST_COEFFS : 3;

      // No way around making a copy; fit() wants a matrix.
      DenseMatrix64F x = new DenseMatrix64F(nums, coefsCount);
      DenseMatrix64F b2 = new DenseMatrix64F(nums, 1);
      DenseMatrix64F b5 = new DenseMatrix64F(nums, 1);

      double w2 = OMEGA / year;

      for (int i = 0; i < nums; i++) {
        x.set(i, 0, 1.0); // for intercept
        x.set(i, 1, Math.cos(OMEGA * clrx[i + start]));
        x.set(i, 2, Math.sin(OMEGA * clrx[i + start]));

        if (year > 1) {
          x.set(i, 3, Math.cos(w2 * clrx[i + start]));
          x.set(i, 4, Math.sin(w2 * clrx[i + start]));
        }

        b2.set(i, 0, clry[1][i + start]); // band 2
        b5.set(i, 0, clry[4][i + start]); // band 5
      }

      // perform robust fit
      RobustLeastSquareBisquare rls = new RobustLeastSquareBisquare(x, b2, 4.685);
      DenseMatrix64F b2coefs = new DenseMatrix64F(coefsCount, 1);
      rls.getSolution(b2coefs);    // throwing away the fit values.

      DenseMatrix64F b5coefs = new DenseMatrix64F(coefsCount, 1);
      rls.updateB(new DenseMatrix64F(b5));
      rls.getSolution(b5coefs);  // throwing away the fit values.

      double[] coefs2 = b2coefs.getData();
      double[] coefs5 = b5coefs.getData();
      double b2Threshold = T_CONST * tB2;
      double b5Threshold = T_CONST * tB5;

      validCount = 0;
      result = new int[nums];

      for (int i = 0; i < nums; i++) {
        // TODO: make this generic to avoid hard-wired index
        // + coefs2[3] * x[i][3] + coefs2[4] * x[i][4];
        double predB2 = coefs2[0] + coefs2[1] * x.get(i, 1) + coefs2[2] * x.get(i, 2);
        // + coefs5[3] * x[i][3] + coefs5[4] * x[i][4];
        double predB5 = coefs5[0] + coefs5[1] * x.get(i, 1) + coefs5[2] * x.get(i, 2);

        if (year > 1) {
          predB2 += coefs2[3] * x.get(i, 3) + coefs2[4] * x.get(i, 4);
          predB5 += coefs5[3] * x.get(i, 3) + coefs5[4] * x.get(i, 4);
        }

        if (Math.abs(predB2 - b2.get(i, 0)) > b2Threshold
            || Math.abs(predB5 - b5.get(i, 0)) > b5Threshold) {
          result[i] = 1;
        } else {
          result[i] = 0;
          validCount++;
        }
      }

      return this;
    }

    int getValidCount() {
      return validCount;
    }

    int[] getResult() {
      return result;
    }

    /**
     * Modify the data arrays by removing any values in mask that correspond to the range
     * start:end, and updates numObservations to match.
     *
     * @param start The start index of the range covered by mask.
     * @param end The end index of the range covered by mask (incl).
     * @param mask An array of flags indicating noisy observations.
     */
    private void arrayMask(int start, int end, int[] mask) {
      int count = start;
      for (int i = start; i < numObservations; i++) {
        // check data in the current processing range
        if (i <= end) {
          if (mask[i - start] == 0) { // not noise
            clrx[count] = clrx[i];
            for (int b = 0; b < numberOfBands; b++) {
              clry[b][count] = clry[b][i];
            }
            count++;
          }
        } else {
          // This final block could be done with System.arrayCopy, it's just moving everything down.
          clrx[count] = clrx[i];
          for (int b = 0; b < numberOfBands; b++) {
            clry[b][count] = clry[b][i];
          }
          count++;
        }
      }
      numObservations = count;
    }
  }


  private static double[] median_variogram(double[] x, double[][] y, int offset, int daysApart) {
    // no data being provided
    if (x.length == 0) {
      return null;
    }

    int numBands = y.length;
    double[] result = new double[numBands];

    // if there is only a single value
    if (x.length == 1) {
      for (int b = 0; b < numBands; b++) {
        result[b] = y[b][0];
      }
    } else {
      DescriptiveStatistics stat = new DescriptiveStatistics();
      for (int b = 0; b < numBands; b++) {
        stat.clear();
        for (int i = offset; i < x.length; i++) {
          if (x[i] - x[i - offset] > daysApart) {
            stat.addValue(Math.abs(y[b][i] - y[b][i - offset]));
          }
        }
        result[b] = stat.getPercentile(50);
      }
    }

    return result;
  }

  boolean computeFit(CcdcFit fit, int df, int idxStart, int idxEnd) {
    for (int b = 0; b < numberOfBands; b++) {
      autoTSFit(fit, b, df, idxStart, idxEnd);
    }
    fit.tStart = clrx[idxStart];
    fit.tEnd = clrx[idxEnd];
    fit.numObs = idxEnd - idxStart + 1;
    return true;
  }

  /**
   * A simple object to train a new model.
   * Only an object to avoid needing to recreate the tmpFit object on each iteration.
   */
  private class Train {
    CcdcFit tmpFit;

    public Train() {
      tmpFit = new CcdcFit(numberOfBands);
    }

    public void invoke(Model model) {
      int iSpan = iSpan = model.endIndex - model.startIndex + 1;
      double timeSpan;
      // step 1: noise removal (good: 0 & noise: 1)

      while (!model.trained && !model.done()) {
        if (tMask != null) {
          // Find outliers.
          int[] mask = tMask.autoTmask(model.startIndex, model.endIndex + minObs).getResult();
          // Find the limits of the good data so we can check the time-span.
          iSpan = ArrayHelper.count(0, mask, 0, -minObs);
          int firstGood = ArrayHelper.first(0, mask, 0, -minObs);
          int lastGood = ArrayHelper.last(0, mask, 0, -minObs);
          timeSpan = timeSpan(model.startIndex + lastGood, model.startIndex + firstGood);

          if (iSpan < N_TIMES * MIN_NUM_COEFFS || timeSpan < MIN_YEARS_NEEDED) {
            // not enough clear observation
            model.endIndex++;
            return;
          }

          if (iSpan < (model.endIndex - model.startIndex + 1)) {
            // There are outliers that need to be removed
            tMask.arrayMask(model.startIndex, model.endIndex, mask);
          }

          // Update with noise removed
          model.endIndex = model.startIndex + iSpan - 1;
        }

        double normVDif = 0;

        for (int b = 0; b < numberOfBands; b++) {
          autoTSFit(tmpFit, b, MIN_NUM_COEFFS, model.startIndex, model.endIndex);

          // Value of difference normalized to z-score
          if (breakpointBands.contains(b)) {
            double minRmse = Math.max(tmpFit.rmse[b], adjRmse[b]);

            // compare the first clear obs to the last one.
            // and get the anormalized slope value.
            double vStart = tmpFit.vDif[b][0];
            double vEnd = tmpFit.vDif[b][model.endIndex - model.startIndex];
            double vSlope = tmpFit.coefs[b][1] * (clrx[model.endIndex] - clrx[model.startIndex]);
            double yvDif = (Math.abs(vSlope) + Math.abs(vStart) + Math.abs(vEnd)) / minRmse;
            normVDif += yvDif * yvDif;
          }
        }

        // find stable start for each curve
        if (normVDif > T_CG) {
          // Move forward.
          model.startIndex += 1;
          model.endIndex += 1;
          return;
        }

        // find index of the previous break point, if there is one.
        int iBreak = 0;
        if (!model.results.isEmpty()) {
          CcdcFit fit = model.results.get(model.results.size() - 1);
          if (fit.tBreak > 0) {
            // tBreak is the time of the last break; find the first index after it.
            for (int tidx = fit.numObs - 1; tidx < numObservations; tidx++) {
              if (clrx[tidx] >= fit.tBreak) {
                iBreak = tidx;
                break;
              }
            }
          }
        }

        int iniConse = minObs;

        if (model.startIndex > iBreak) {
          double[] vecMag = new double[iniConse];

          // % model fit at the beginning of the time series
          // TODO(gorelick): This needs some work.
          // We're recomputing vecMag over and over, shifting it by 1 each time.
          // This could use some cleanup and explanation.
          //
          for (int ini = model.startIndex-1; ini >= iBreak; ini--) {
            iniConse = Math.min(model.startIndex - iBreak, minObs);

            computeVecMag(tmpFit, iniConse, ini, -1, vecMag);

            // % change detection\
            // NOTE: only iniConse valid values
            double minMag = vectorMin(vecMag, 0, iniConse);
            if (minMag > T_CG) { // change detected
              break;
            } else if (vecMag[0] > TMAX_CG) { // %false change
              removeNoise(ini);
              model.endIndex--;
            }

            model.startIndex = ini;
          }
        }

        if (model.results.isEmpty() && model.startIndex - iBreak >= minObs) {
          CcdcFit fit = model.newFit();
          computeFit(fit, MIN_NUM_COEFFS, iBreak, model.startIndex - 1);
          // Override tStart to be t0.
          fit.tStart = clrx[0];
          fit.tBreak = clrx[model.startIndex];
          fit.changeProb = 1;
          fit.category = QA_10 + MIN_NUM_COEFFS;

          //update segment magnitude
          int finalIniConse = iniConse;
          Arrays.setAll(fit.magnitude, band -> -1 * getMedian(vDifMag, band, 0, finalIniConse));
          // This fit is complete.
          model.currentFit = null;
        }
        model.trained = true;
        return;
      }
    }
  }

  /**
   * Compute the vector magniude.
   *
   * @param coeff The fit coefficients
   * @param count How many observations to compute.
   * @param start The index of the first observation.
   * @param dir Direction of travel.  +1 for increasing indexes; -1 for decreasing.
   * @param vecMag The result, pre-allocated.
   */
  private void computeVecMag(CcdcFit fit, int count, int start, int dir, double[] vecMag) {
    Arrays.fill(vecMag, 0);

    for (int i = 0; i < count; i++) {
      for (int b = 0; b < numberOfBands; b++) {
        double pred = predict(clrx[start + i * dir], fit.coefs[b]);
        double diff = clry[b][start + i * dir] - pred;
        vDifMag[i][b] = diff;

        if (breakpointBands.contains(b)) {
          // minimum rmse
          double minRmse = Math.max(fit.rmse[b], adjRmse[b]);
          double z = diff / minRmse; // %z-scores
          vecMag[i] += z * z;
        }
      }
    }
  }

  private class Monitor {
    private final double[] vecMag;

    public Monitor() {
      vecMag = new double[minObs];
    }

    public double[] getVecMag() {
      return vecMag;
    }

    public void invoke(Model model) {
      CcdcFit fit = model.currentFit;
      if (fit == null) {
        fit = model.newFit();
      }

      // TODO(gorelick): I don't understand what these do enough to document them.
      int oldStart = 0;
      int oldEnd = 0;
      double oldTime = 0;

      while (!model.done()) {
        int iSpan = model.endIndex - model.startIndex + 1;
        int updateNumC = updateCft(iSpan);

        // % initial model fit when there are not many obs
        if (oldTime == 0 || iSpan <= MAX_NUM_COEFFS * N_TIMES) {

          // % update oldTime at each iteration because observations may have been removed.
          oldTime = timeSpan(model.endIndex, model.startIndex);

          computeFit(fit, updateNumC, model.startIndex, model.endIndex);
          fit.tBreak = 0;
          fit.changeProb = 0;
          fit.category = QA_NORMAL + updateNumC;

          computeVecMag(fit, minObs, model.endIndex + 1, 1, vecMag);

          oldStart = model.startIndex;
          oldEnd = model.endIndex;
        } else {
          if (timeSpan(model.endIndex, model.startIndex) >= args.minNumOfYearsScaler * oldTime) {
            // % update oldTime at each iteration because observations may have been removed.
            oldTime = timeSpan(model.endIndex, model.startIndex);

            computeFit(fit, updateNumC, model.startIndex, model.endIndex);
            fit.tBreak = 0;
            fit.changeProb = 0;
            fit.numObs = model.endIndex - model.startIndex + 1;
            fit.category = QA_NORMAL + updateNumC;

            /*
              % IDs that haven't updated
            IDsOld = IDs;
            */
            oldStart = model.startIndex;
            oldEnd = model.endIndex;
          }

          // This is a candidate endTime.  We'll check to see if it's real.
          fit.tEnd = clrx[model.endIndex];

          // % use fixed number for RMSE computing
          int nRmse = N_TIMES * fit.category;
          double[] tmpcgRmse = new double[numberOfBands];

          // double[] d_rt = new double[oldI - oldIstart + 1]; not needed
          double[] dYr = new double[oldEnd - oldStart + 1];
          for (int b = 0; b < dYr.length; b++) {
            double tRt = clrx[oldStart + b] - clrx[model.endIndex + minObs];
            dYr[b] = Math.abs(Math.round(tRt / SIZE_OF_A_YEAR) * SIZE_OF_A_YEAR - tRt);
          }

          // Get the sorted indexes of the dYr array.
          int[] sortedIndx =
              IntStream.range(0, dYr.length)
                  .boxed()
                  .sorted((di, dj) -> Double.compare(dYr[di], dYr[dj]))
                  .mapToInt(ele -> ele)
                  .toArray();

          for (int b : breakpointBands) {
            double sumSquare = 0;
            int trvdIndx = 0; // index into vDif
            for (int ssi : sortedIndx) {
              if (trvdIndx++ < nRmse) {
                sumSquare += fit.vDif[b][ssi] * fit.vDif[b][ssi];
              } else {
                break;
              }
            }

            tmpcgRmse[b] = Math.sqrt(sumSquare / (nRmse - fit.category));
          }

          // Move the ith col to i-1 pos.  Fill with 0.
          vecMag[0] = 0;
          vDifMag[0] = new double[vDifMag[0].length];
          ArrayUtils.shift(vecMag, -1);
          ArrayUtils.shift(vDifMag, -1);


          for (int b = 0; b < numberOfBands; b++) {
            double pred = predict(clrx[model.endIndex + minObs], fit.coefs[b]);
            double diff = clry[b][model.endIndex + minObs] - pred;

            vDifMag[minObs-1][b] = diff;

            if (breakpointBands.contains(b)) {
              double minRmse = adjRmse[b] < tmpcgRmse[b] ? tmpcgRmse[b] : adjRmse[b];
              double z = diff / minRmse;
              vecMag[minObs - 1] += z * z;
            }
          }
        }

        // % change detection
        // if min(vecMag) > T_cg % change detected
        double minMag = new Min().evaluate(vecMag);
        if (minMag > T_CG) {
          // change detected

          // record break time
          fit.tBreak = clrx[model.endIndex + 1];
          fit.changeProb = 1;

          //update segment magnitude
          Arrays.setAll(fit.magnitude, band -> getMedian(vDifMag, band));

          // % This fit is complete.  Move on to the next one.
          model.currentFit = null;
          model.startIndex = model.endIndex + 1;
          model.trained = false;

          // We're done.
          return;
        } else if (vecMag[0] > TMAX_CG) { // false change
          removeNoise(model.endIndex + 1);
          // We just removed the last point, so no increment in end.
        } else {
          model.endIndex++;
        }
      }
    }
  }

  /** Parameters for a single segment */
  public static class CcdcFit {
    public double tStart;   // Time of the start of the model.
    public double tEnd;     // Time of the end of the model.
    public double tBreak;   // Time when the first break (change) is observed.
    public double[] rmse;      // RMSE of the each band's fit
    public double changeProb;   // Probability that a pixel has changed (0, 100). */
    public int numObs;    // The number of "good" observations; used for QA

    /**
     * Coefficients for each time series model for each spectral band. [NUM_BANDS][NUM_COEFFS]
     * NUM_BANDS=7, NUM_COEFFS=8
     */
    public double[][] coefs;

    /**
     * The quality of the model estimation.  Doesn't match the matlab docs.
     */
    public int category;

    /** The magnitude of change (difference between prediction and observation for each band). */
    public double[] magnitude;
    public double[][] vDif;

    /** Construct a new fit. */
    public CcdcFit(int nBands) {
      magnitude = new double[nBands];
      coefs = new double[nBands][];
      rmse = new double[nBands];
      vDif = new double[nBands][];
    }

    public String toString() {
      StringBuffer buffer = new StringBuffer();
      buffer.append(tStart);
      buffer.append(",");
      buffer.append(tEnd);
      buffer.append(",");
      buffer.append(tBreak);
      buffer.append(",");
      buffer.append(numObs);
      buffer.append(",");
      buffer.append(changeProb);
      buffer.append(",");
      buffer.append(category);
      buffer.append(",");
      buffer.append(arrayToString(coefs, ","));
      buffer.append(",");
      buffer.append(arrayToString(rmse, ","));
      buffer.append(",");
      buffer.append(arrayToString(magnitude, ","));

      return buffer.toString();
    }

    private String arrayToString(double[] array, String delimiter) {
      StringBuffer buffer = new StringBuffer();
      for (int i = 0; i < array.length; i++) {
        if (i > 0) {
          buffer.append(delimiter);
          if (i % 8 == 0) {
            buffer.append("\n");
          }
        }
        buffer.append(array[i]);
      }
      return buffer.toString();
    }

    private String arrayToString(double[][] array, String delimiter) {
      StringBuffer buffer = new StringBuffer();

      for (int i = 0; i < array.length; i++) {
        if (i > 0) {
          buffer.append(delimiter);
          if (i % 8 == 0) {
            buffer.append("\n");
          }
        }
        buffer.append(arrayToString(array[i], delimiter));
      }
      return buffer.toString();
    }
  }

  /** Remove an element from the data arrays by shifting everything down and updating the size */
  private void removeNoise(int index) {
    System.arraycopy(clrx, index + 1, clrx, index, numObservations - index - 1);
    for (int b = 0; b < numberOfBands; b++) {
      System.arraycopy(clry[b], index + 1, clry[b], index, numObservations - index - 1);
    }
    numObservations--;
  }
}
