/*
 * Copyright (c) 2015 LCMS Project Authors.
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
package com.google.earthengine.examples.landsat;

import com.google.earthengine.api.base.AlgorithmBase;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.LUDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;

import java.util.Arrays;

/**
 * Implements the EWMACD algorithm in the paper:
 * Brooks, E.B., R.H. Wynne, V.A. Thomas, C.E. Blinn, and J.W. Coulston. 2014.
 * On-the-fly Massively Multitemporal Change Detection
 * Using Statistical Quality Control Charts and Landsat Data,
 * IEEE Transaction on Geoscience and Remote Sensing, 52: 3316-3332
 *
 * EWMACD is an algorithm that uses Shewhart X-bar charts for change detection.
 *
 * @author Zhiqiang Yang, 2/14/2015
 *
 * TODO(YZ): consolidate the use of DoubleArrayList and RealVector
 *
 */
public final class Ewmacd {
  static class Args extends AlgorithmBase.ArgsBase {
    @Doc(help = "Threshold for vegetation. Values below this are considered non-vegetation.")
    @Required
    double vegetationThreshold = 100;

    @Doc(help = "Start year of training period, inclusive.")
    @Required
    // TODO(gorelick): Remove these testing defaults from the required args.
    int trainingStartYear = 2005;

    @Doc(help = "End year of training period, exclusive.")
    @Required
    int trainingEndYear = 2007;

    @Doc(help = "Number of harmonic function pairs (sine and cosine) used.")
    @Optional
    int harmonicCount = 2;

    @Doc(help = "Threshold for initial training xBar limit.")
    @Optional
    double xBarLimit1 = 1.5;

    @Doc(help = "Threshold for running xBar limit.")
    @Optional
    int xBarLimit2 = 20;

    @Doc(help = "EWMA lambda")
    @Optional
    double lambda = 0.3;

    @Doc(help = "EWMA number of standard deviations for marking out of control.")
    @Optional
    double lambdasigs = 3.0;

    @Doc(help = "Should rounding be performed for EWMA")
    @Optional
    boolean rounding = true;

    @Doc(help = "Minimum number of observations needed to consider a change.")
    @Optional
    int persistence = 3;
  }

  private static final int DEFAULT_VALUE = -2222;
  private final Args args;

  public Ewmacd() {
    this(new Args());
  }

  public Ewmacd(Args args) {
    this.args = args;
  }

  /**
   * The main working horse of the algorithms
   *
   * In this implementation, it is assumed that nonvalid data has already been masked.
   * Need to confirm that when Google PixelFunction read values, the masked values would
   * not be returned here.
   *
   * @param x day of year for all the observations
   * @param y spectral values
   * @param years corresponding years
   */
  public int[] getResult(double[] x, double[] y, double[] years) {
    // EWMACD results.
    int[] results = new int[x.length];

    // Build a record of used indices.
    int[] validObservations = new int[x.length];

    for (int i = 0; i < x.length; i++) {
      validObservations[i] = i;
      results[i] = DEFAULT_VALUE;
    }

    // Find out the range of training data.
    int trainingStart = 0;
    int trainingEnd = years.length - 1;
    for (int i = 0; i < years.length; i++) {
      if (years[i] == args.trainingStartYear) {
        trainingStart = i;
        break;
      }
    }
    for (int i = trainingStart; i < years.length; i++) {
      if (years[i] == args.trainingEndYear) {
        trainingEnd = i - 1;
        break;
      }
    }

    double[] trainingX = Arrays.copyOfRange(x, trainingStart, trainingEnd + 1);
    double[] trainingY = Arrays.copyOfRange(y, trainingStart, trainingEnd + 1);
    double[] betas = getTrainingHarmonic(trainingX, trainingY);

    // The following implements EWMA components
    // calculate fitted values for all the data points based on training coefficients.
    RealMatrix all = constructHarmonicMatrix(x);
    RealMatrix fitted = all.multiply(new Array2DRowRealMatrix(betas));

    // NG: Residuals probably works better as an double[].
    // YZ: Changed as suggested
    double[] residuals = new Array2DRowRealMatrix(y).subtract(fitted).getColumnVector(0).toArray();
    int trainingSize = trainingEnd-trainingStart+1;
    double[] trainingResiduals = ArrayUtils.subarray(residuals, trainingStart, trainingEnd+1);

    //first estimate historical mean of residuals (should be near 0)
    // NG: This isn't a mean.
    // YZ: This is an legacy copy fro original code, Orignal implemneted calcuated the mean, but was never used.
    // it is standard deviation that we want.
    double historicalStd = new StandardDeviation().evaluate(trainingResiduals);

    // Original comments:
    // Modifying SD estimates based on anomalous readings in the training data
    // We don't want to filter out the changes in the testing data, so xBarLimit2 is much larger!
    // The original implementation use only the first few years as training, do we need to handle
    // if training years are in the middle?
    double[] ucl0 = new double[x.length];
    for (int i = 0; i < x.length; i++) {
      if (years[i] < args.trainingEndYear && years[i] >= args.trainingStartYear) {
        ucl0[i] = args.xBarLimit1 * historicalStd;
      } else {
        ucl0[i] = args.xBarLimit2 * historicalStd;
      }
    }

    // Keeping only dates for which we have some vegetation and aren't anomalously far from 0
    // in the residuals.
    DoubleArrayList filteredResiduals = new DoubleArrayList();
    IntArrayList usedObservations = new IntArrayList(); // this should be the final list used

    for (int i = 0; i < x.length; i++) {
      if (y[i] > args.vegetationThreshold && Math.abs(residuals[i]) < ucl0[i]) {
        filteredResiduals.add(residuals[i]);
        // NG: validObservations[i] == i.
        // YZ: this should be fine and preferred: validObservations maybe a redundant variable.
        usedObservations.add(i);
      }
    }

    // Do we have enough data to work with?
    // NOTE(YZ): I added this condition here. Original implementation checks this after
    // ewma and control limit calculation, which seems inefficient.
    // NG: All the arrays below here (except filteredTrainingResiduals) are this size, so we use
    // it through-out.
    int filteredSize = filteredResiduals.size();
    if (filteredSize <= 3) {
      return adjustResult(results);
    }

    // Updating historicalMean
    DoubleArrayList filteredTrainingResiduals = new DoubleArrayList();
    for (int i = 0; i < trainingResiduals.length; i++) {
      if (trainingY[i] > args.vegetationThreshold
          && Math.abs(trainingResiduals[i]) < ucl0[i]) {
        filteredTrainingResiduals.add(trainingResiduals[i]);
      }
    }

    // Do we have enough data to continue?
    // TODO(YZ): original implementation only checks for standard deviation,
    // which can be calculated with just 2 numbers, should we make this a parameter like
    // minimum observation required?
    // Original implementation return array with default NODATA value for all the years.
    if (filteredTrainingResiduals.size() < 2) {
      return adjustResult(results);
    }
    historicalStd = new StandardDeviation().evaluate(filteredTrainingResiduals.toDoubleArray());

    // Future EWMA output.
    double[] ewma = new double[filteredSize];
    ewma[0] = filteredResiduals.get(0);

    for (int i = 1; i < filteredSize; i++) {
      ewma[i] = ewma[i - 1] * (1 - args.lambda) + args.lambda * filteredResiduals.get(i);
    }

    // EWMA upper control limit.
    // This is the threshold which dictates when the chart signals a disturbance.
    double[] upperControlLimit = new double[filteredSize];
    for (int i = 0; i < filteredSize; i++) {
      upperControlLimit[i] = historicalStd * args.lambdasigs
          * Math.sqrt(args.lambda / (2 - args.lambda) *
              (1 - Math.pow((1 - args.lambda), 2 * (i + 1))));
    }

    // Integer value for EWMA output relative to control limit (rounded towards 0).
    // A value of +/-1 represents the weakest disturbance signal
    // coded values for the "present" subset of data.
    // NG: Renamed from tmp2.
    int[] signal = new int[filteredSize];
    for (int i = 0; i < filteredSize; i++) {
      if (args.rounding) {
        signal[i] = (int) (Math.signum(ewma[i])
            * Math.floor(Math.abs(ewma[i] / upperControlLimit[i])));
      } else {
        signal[i] = (int) Math.round(ewma[i]);
      }
    }

    // Keeping only values for which a disturbance is sustained, using persistence as the threshold
    // NOTE(YZ): original implementation check this rules here, the check on tmp2.length should
    // happen earlier to improve performance it is redundant here, as I have moved it up right after
    // filtering the data.
    if (args.persistence > 1 && filteredSize > 3) {
      //TODO: optimize the following section
      // NG: Could compute array of signs in the previous loop.  Likely to be cleaner.
      // NG: Renamed from tmp3.
      int[] sustained = new int[filteredSize];
      int previousKeep = -1;
      for (int i = 0; i < filteredSize; i++) {
        int low = 0;
        int high = 0;
        int score;
        // NG: Folded.
        while (i - low >= 0 && Math.signum(signal[i]) == Math.signum(signal[i - low])) {
          low++;
        }

        while (high + i < filteredSize && Math.signum(signal[i + high]) == Math.signum(signal[i])) {
          high++;
        }
        score = low + high - 1;

        // If sustained dates are long enough, keep; otherwise set to previous sustained state.
        if (score >= args.persistence) {
          sustained[i] = signal[i];
          previousKeep = i;
        } else {
          // NOTE: the following line would cause some minor difference with the original
          // implementation.  In the original implementation, it always assign the left most
          // instance where sustained dates pass the persistence test. I think the original
          // implementation was wrong.
          // TODO(YZ): Check for correctness.
          sustained[i] = previousKeep >= 0 ? signal[previousKeep] : 0;
        }
      }
      signal = sustained;
    }

    // NG: No asserts allowed.  But its clear this is always true.
    // assert usedObservations.size() == tmp2.length;
    for (int i = 0; i < filteredSize; i++) {
      results[usedObservations.getInt(i)] = signal[i];
    }

    return adjustResult(results);
  }

  /**
   * adjust EWMACD values for final result.
   * @param results
   * @return int[] filtered ewma values
   */
  private int[] adjustResult(int[] results) {
    // If the first date of myPixel was missing/filtered, then  output a 0 (no disturbance).
    if (results[0] == DEFAULT_VALUE) {
      results[0] = 0;
    }

    // Original comment: If we have EWMA information for the first date, then for each
    // missing/filtered date in the record, fill with the last known EWMA value.
    // NOTE: with the above line, the following will always run
    for (int i = 1; i < results.length; i++) {
      if (results[i] == DEFAULT_VALUE) {
        results[i] = results[i - 1];
      }
    }

    return results;
  }

  /**
   * Derive harmonic coefficients using training data
   * NG: Why does this function only use exactly 2 harmonics and not harmonicCount?
   * YZ: it is using harmonicCount in the constructHarmonicMatrix.
   */
  private double[] getTrainingHarmonic(double[] trainingX, double[] trainingY) {
    //construct training data
    Array2DRowRealMatrix matrix = constructHarmonicMatrix(trainingX);

    //make sure the design matrix has sufficient rank
    if (trainingX.length < (1 + 2 * args.harmonicCount) && isSingular(matrix)) {
      return null;
    }

    //first pass of harmonic function
    // NG: Why is this cutting off the first column?
    // YZ: I will verify this with.
    OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
    ols.newSampleData(trainingY,
        matrix.getSubMatrix(
            0, trainingX.length - 1,
            1, matrix.getColumnDimension() - 1)
            .getData());

    //check residual from first pass of harmonic funciton
    double[] residuals = ols.estimateResiduals();
    // NG: Folded together.
    double limit = new StandardDeviation().evaluate(residuals) * args.xBarLimit1;

    //exclude observations with large residules: xBarLimit1 * sd
    int validCount = 0;
    DoubleArrayList samples2 = new DoubleArrayList();
    for (int i = 0; i < residuals.length; i++) {
      if (Math.abs(residuals[i]) <= limit) {
        samples2.add(trainingY[i]);
        double rx = trainingX[i] * 2 * Math.PI / 365;
        samples2.add(Math.sin(rx));
        samples2.add(Math.cos(rx));
        samples2.add(Math.sin(2 * rx));
        samples2.add(Math.cos(2 * rx));

        validCount++;
      }
    }

    //refit the model using filtered training data
    // NG: Shouldn't we test how many are left?
    // YZ: yes
    if (validCount < (1 + 2 * args.harmonicCount)) {
      return null;
    }
    ols.newSampleData(samples2.toDoubleArray(), validCount, 2 * args.harmonicCount);
    return ols.estimateRegressionParameters();
  }

  // NG: If getTrainingHarmonic really wants to cut off the constant, maybe
  // better to make this function not add it with a arg.
  // e.g.: (double x[], boolean constantTerm)
  protected Array2DRowRealMatrix constructHarmonicMatrix(double[] x) {
    Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(x.length, 2 * args.harmonicCount + 1);

    for (int i = 0; i < x.length; i++) {
      // NG: use setEntry instead.
      matrix.setEntry(i, 0, 1);
      double rx = x[i] * 2 * Math.PI / 365;
      //now add the harmonic terms
      for (int h = 0; h < args.harmonicCount; h++) {
        matrix.setEntry(i, h * 2 + 1, Math.sin((h + 1) * rx));
        matrix.setEntry(i, h * 2 + 2, Math.cos((h + 1) * rx));
      }
    }
    return matrix;
  }

  protected boolean isSingular(Array2DRowRealMatrix matrix) {
    double singularThreshold = 0.001;
    RealMatrix prod = matrix.transpose().multiply(matrix);
    double determinant = new LUDecompositionImpl(prod).getDeterminant();

    return Math.abs(determinant) <= singularThreshold;
  }
}
