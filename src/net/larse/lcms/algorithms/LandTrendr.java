package net.larse.lcms.algorithms;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;

import org.apache.commons.math.FunctionEvaluationException;
import org.apache.commons.math.optimization.fitting.CurveFitter;
import org.apache.commons.math.optimization.fitting.ParametricRealFunction;
import org.apache.commons.math.optimization.general.LevenbergMarquardtOptimizer;
import org.ejml.data.DenseMatrix64F;

import riso.numerical.SpecialMath;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import net.larse.lcms.helper.EEArray;
import net.larse.lcms.helper.LinearLeastSquares;
import net.larse.lcms.helper.PixelType;

/**
 * Implements the LandTrendr algorithm that was proposed in the paper:
 * Kennedy, Robert E., Yang, Zhiqiang, & Cohen, Warren B. (2010).
 * Detecting trends in forest disturbance and recovery using yearly
 * Landsat time series: 1. LandTrendr - Temporal segmentation
 * algorithms. Remote Sensing of Environment, 114, 2897-2910.
 *
 * <p> LandTrendr is an algorithm that segments a time-series of images into
 * periods of time. From the segment of each period of time, one can infer
 * times of vegetation recovery or loss.
 */
public class LandTrendr { // extends ImageConstructor<LandTrendr.Args> {
  private static final long serialVersionUID = 1L;

  @VisibleForTesting
  public final static class LandTrendrSolver {
    private final double spikeThreshold;
    private final int maxSegments;
    private final int vertexCountOvershoot;
    private final boolean preventOneYearRecovery;
    private final double recoveryThreshold;
    private final double pvalThreshold;
    private final double bestModelProportion;

    //private final int numImages;

    //YANG: we are not using the ImageValues but will use csv file
//    private final List<ImageValues> images;

    public LandTrendrSolver() {
      this.spikeThreshold = 0.9;
      this.maxSegments = 6;
      this.vertexCountOvershoot = 3;
      this.preventOneYearRecovery = true; //??
      this.recoveryThreshold = 0.25;
      this.pvalThreshold = 0.05;
      this.bestModelProportion = 0.75;
    }


    public LandTrendrSolver(double spikeThreshold,
        int maxSegments,
        int vertexCountOvershoot,
        boolean preventOneYearRecovery,
        double recoveryThreshold,
        double pvalThreshold,
        double bestModelProportion) {

        //PixelFunctionCollection[] inputs) {
      this.spikeThreshold = spikeThreshold;
      this.maxSegments = maxSegments;
      this.vertexCountOvershoot = vertexCountOvershoot;
      this.preventOneYearRecovery = preventOneYearRecovery;
      this.recoveryThreshold = recoveryThreshold;
      this.pvalThreshold = pvalThreshold;
      this.bestModelProportion = bestModelProportion;


      //this.numImages = inputs[0].size();

      /**
       * Group the band functions for each image together and sort them by year.
       */
//      images = Lists.newArrayListWithCapacity(numImages);
//      for (int image = 0; image < numImages; image++) {
//        images.add(new ImageValues(inputs, image));
//      }
//      Collections.sort(images);
    }

    /**
     *  This is the main function of LandTrendr. For each pixel at position
     *  (x, y) in the collection of images it will find the best model that
     *  describes it and calculate the values to be returned.
     *
     *  The first step to find the best model consists of the dampering of the
     *  spikes in the observations, which is achieved by the desawtooth method.
     *  After this step, maxSegments + 1 + vertexCountOvershoot vertices are
     *  identified as the most representatives vertices (the ones that
     *  represents a change) in the data. Then, this vertex count is reduced
     *  to maxSegments + 1 vertices by eliminating the vertices that forms
     *  segments with a low change in angle. Once that is done, the process of
     *  generating candidate models starts. Each model is a series of linear
     *  fits between the chosen vertices, then one of the vertices is removed
     *  and another model is fitted; this process is repeated until there are
     *  only two vertices in the model, which is the simplest possible.
     */
    //public EEArray getResult(int x, int y) {
    //public EEArray getResult(double[] x, double[] y) {
    public List<Integer> getResult(DoubleArrayList x, DoubleArrayList y) {
      Model model;
      double[] yearsArray;
      double[] rawValuesArray;

      //Yang: mask MemoryScope usage
      //try (MemoryScope scope = MemoryScope.newTransient()) {
      {
        DoubleArrayList years = new DoubleArrayList(x);
        DoubleArrayList rawValues = new DoubleArrayList(y);

        // get the data from the images (only from the first band)
//        for (ImageValues image : images) {
//          PixelFunction f = image.bandFunctions[0];
//          years.add(image.year);
//          rawValues.add(f.value().getFloat(x, y));
//        }

        int nObs = rawValues.size();
        double[] values = rawValues.toDoubleArray();
        double[] times = years.toDoubleArray();
        for (int i = 0; i < nObs; i++) {
          // subtract the minimum year (the collection was sorted before)
          times[i] = years.get(i) - years.get(0);
        }

        // pre-calculates the mean of the values.
        double valuesMean = 0.0;
        for (int i = 0; i < values.length; i++) {
          valuesMean += values[i];
        }
        valuesMean /= values.length;

        // apply the smoothing algorithm
        desawtooth(values, spikeThreshold);

        // identify the potential vertices (in total there will be
        // maxSegments + 1 + vertexCountOvershoot vertices)
        // REF: tbcd_v2.pro: find_vertices
        List<Integer> potentialVertices = identifyPotentialVertices(times,
            values, maxSegments, vertexCountOvershoot, preventOneYearRecovery);

        // prune the amount of vertices down to maxSegments + 1
        // REF: vert_verts3.pro: vet_verts3
        List<Integer> prunedVertices =
            cullByAngle(times, values, maxSegments, potentialVertices);

        // select the best model to represent the data
        // REF: tbcd_v2.pro: find_best_trace
        model = identifyBestModel(times, values, valuesMean,
            prunedVertices, recoveryThreshold, bestModelProportion);
        if (model == null || model.pValue > pvalThreshold) {
          model = identifyBestModelsUsingLevenbergMarquardt(times, values,
              valuesMean, prunedVertices, recoveryThreshold,
              bestModelProportion);
        }

        yearsArray = years.toDoubleArray();
        rawValuesArray =  rawValues.toDoubleArray();
      }
      //return toArray(yearsArray, rawValuesArray, model.yFitted, model.vertices);
      return model.vertices;
    }

    /**
     * This class is responsible for holding all the related information that a
     * model generated by the LandTrendr algorithm has. It holds the vertices
     * that represents a change, the slope and intercept of the segments, and
     * statistics such as the p-value and f-statistic.
     *
     * As there are two distinctive ways of generating a model, one by the use
     * of a series of linear regressions, and another one by the use of the
     * Levenberg-Marquardt method, this class is in fact a super class of
     * ModelNormal and ModelLM.
     */
    @VisibleForTesting
    public class Model {
      // a pointer to the x and y values.
      double[] x;
      double[] y;
      // the vertices that delimits the beginning and ending of each segment.
      public List<Integer> vertices;
      // the intercept of each segment.
      public List<Double> intercepts;
      // the slope of each segment.
      public List<Double> slopes;
      // the fitted values according to the segments.
      public double[] yFitted;
      // the f-statistic of the model.
      public double fStat;
      // the p-value of the model.
      public double pValue;
      // the max(yFitted) - min(yFitted), a value used when checking the slopes
      // of the model.
      protected double yFittedRange;
      // the mean of the original y values, that is stored here to avoid
      // recomputing it for each model.
      protected double yMean;

      public Model(double[] x, double[] y, List<Integer> vertices, double yMean) {
        this.vertices = Lists.newArrayList(vertices);

        // stores the pre-calculate mean of the y values (it is used by the
        // goodnessOfFit() method).
        this.yMean = yMean;

        this.x = x;
        this.y = y;
      }

      /**
       * Given the x values and access to the segments information stored on
       * the class, it returns the fitted values.
       */
      protected double[] getFittedValues() {
        double[] yFinal = new double[x.length];

        for (int i = 0; i < vertices.size() - 1; i++) {
          int begin = vertices.get(i);
          int end = vertices.get(i + 1);
          double slope = slopes.get(i);
          double intercept = intercepts.get(i);
          for (int j = begin; j <= end; j++) {
            yFinal[j] = (x[j] - x[begin]) * slope + intercept;
          }
        }
        return yFinal;
      }

      /**
       * Calculates the p-value and the f-statistic of the model as a way to
       * measure its goodness of fit.
       */
      protected void goodnessOfFit() {
        // Note: p-value less than this is treated as 0
        final double pzero = 1e-9;
        final double epsilon = 0.00001;
        double sumOfSquaresTotal = 0.0;
        double sumOfSquaresResidual = 0.0;
        for (int i = 0; i < y.length; i++) {
          sumOfSquaresTotal += (y[i] - yMean) * (y[i] - yMean);
          sumOfSquaresResidual += (y[i] - yFitted[i]) * (y[i] - yFitted[i]);
        }

        if (sumOfSquaresResidual > sumOfSquaresTotal) {
          sumOfSquaresResidual = sumOfSquaresTotal;
        }

        double sumOfSquaresExplained = sumOfSquaresTotal - sumOfSquaresResidual;

        int dfExplained = vertices.size() * 2 - 2;
        int dfResidual = y.length - dfExplained - 1;

        double fStat = 0.0;
        double pValue = 1.0;
        if (dfResidual > 0) {
          fStat = (sumOfSquaresExplained / dfExplained)
              / (sumOfSquaresResidual / dfResidual);

          if (fStat < epsilon) {
            fStat = epsilon / (sumOfSquaresResidual / dfResidual);
          }

          pValue = SpecialMath.incompleteBeta(dfResidual / (dfResidual
              + dfExplained * fStat), dfResidual / 2.0, dfExplained / 2.0);
        }

        this.fStat = fStat;
        this.pValue = pValue < pzero ? pzero : pValue;
      }

      /**
       * Given the fitted values of y, the slopes of the segments that generated
       * the fitted values, and the recovery threshold, returns true if the
       * model is not violating the recovery threshold and false otherwise.
       */
      public boolean checkSlopes(double recoveryThreshold) {
        for (int i = 0; i < slopes.size(); i++) {
          if (slopes.get(i) < 0.0
              && recoveryThreshold < Math.abs(slopes.get(i) / yFittedRange)) {
            return false;
          }
        }
        return true;
      }
    }

    /**
     * This class inherits from the Model class and has code to fit a model that
     * uses a series of linear regressions.
     */
    @VisibleForTesting
    public class ModelNormal extends Model {
      public ModelNormal(List<Integer> vertices,
          double[] x,
          double[] y,
          double yMean) {
        super(x, y, vertices, yMean);

        // sets the slopes and intercepts.
        identifyBestPath(x, y);

        // calculates the fitted values for this model.
        this.yFitted = getFittedValues();

        // calculate the p-value and f-statistic of the model.
        goodnessOfFit();

        // pre-calculates the range of the yFitted values for use in the
        // checkSlopes() method.
        yFittedRange = Doubles.max(yFitted) - Doubles.min(yFitted);
      }

      /**
       * This methods defines the final path of the trends given a list of the
       * vertices. The path is defined by fitting segments either by a
       * straight point-to-point model or an anchored linear regression on the
       * last point (a constraint of the path is that it must be continuous).
       * the side effect of this method is that the slopes and intercepts of the
       * object are set.
       */
      private void identifyBestPath(double[] x, double[] y) {
        this.slopes = new DoubleArrayList();
        this.intercepts = new DoubleArrayList();

        /**
         * For the first segment, as it doesn't have an anchor point, it's the
         * best (measured by MSE) of a liner regression or a point-to-point
         * linear. For every other segment it's the best of these approaches
         * with the care of the end of the last segment connecting with the
         * beginning of the next.
         */
        int endPointA = vertices.get(0);
        int endPointB = vertices.get(1);

        // series of sums used to determine the slopes of a linear regression
        double sumX, sumY, sumXX, sumXY;
        sumX = sumY = sumXY = sumXX = 0.0;
        for (int i = endPointA; i <= endPointB; i++) {
          sumX += x[i];
          sumY += y[i];
          sumXY += x[i] * y[i];
          sumXX += x[i] * x[i];
        }

        // slope and intercept of a Linear Regression (LnR).
        double segmentLength = endPointB - endPointA + 1.0;
        double slopeLnR = segmentLength * sumXY - sumX * sumY;
        slopeLnR /= (segmentLength * sumXX - sumX * sumX);
        double interceptLnR = sumY / segmentLength
            - slopeLnR * sumX / segmentLength;

        this.slopes.add(slopeLnR);
        this.intercepts.add(interceptLnR);

        // now fit the rest of the segments.
        double anchorPoint = x[endPointB] * slopeLnR + interceptLnR;
        for (int i = 1; i < vertices.size() - 1; i++) {
          endPointA = vertices.get(i);
          endPointB = vertices.get(i + 1);

          // defines the fit of the point-to-point model
          double slopePtP =
              (y[endPointB] - anchorPoint) / (x[endPointB] - x[endPointA]);
          double interceptPtP = anchorPoint;
          double yPtP = 0.0;
          double residualPtP = 0.0;
          for (int j = endPointA; j <= endPointB; j++) {
            yPtP = (x[j] - x[endPointA]) * slopePtP + interceptPtP;
            residualPtP += (y[j] - yPtP) * (y[j] - yPtP);
          }

          // defines the fit of the anchored regression model
          double xx = 0.0;
          double xy = 0.0;
          for (int j = endPointA; j <= endPointB; j++) {
            double xDisplacement = x[j] - x[endPointA];
            xx += xDisplacement * xDisplacement;
            xy += xDisplacement * (y[j] - anchorPoint);
          }
          slopeLnR = xy / xx;
          interceptLnR = anchorPoint;
          double yLnR = 0.0;
          double residualLnR = 0.0;
          for (int j = endPointA; j <= endPointB; j++) {
            yLnR = (x[j] - x[endPointA]) * slopeLnR + interceptLnR;
            residualLnR += (y[j] - yLnR) * (y[j] - yLnR);
          }

          // picks the one with least residue
          if (residualLnR >= residualPtP) {
            // best model is the point-to-point
            this.slopes.add(slopePtP);
            this.intercepts.add(interceptPtP);
            anchorPoint = yPtP;
          } else {
            // best model is the anchored linear regression
            this.slopes.add(slopeLnR);
            this.intercepts.add(interceptLnR);
            anchorPoint = yLnR;
          }
        }
      }
    }

    /**
     * This class inherits from the Model class and has specific code to fit a
     * model using the Levenberg-Marquardt method.
     */
    @VisibleForTesting
    public class ModelLM extends Model {
      public ModelLM(List<Integer> vertices,
          double[] x,
          double[] y,
          CurveFitter fitter,
          double yMean) {
        super(x, y, vertices, yMean);

        // sets the slopes and intercepts.
        identifyBestPath(x, fitter);
        // calculates the fitted values for this model.
        this.yFitted = getFittedValues();
        // calculate the p-value and f-statistic of the model.
        goodnessOfFit();

        // pre-calculates the range of the yFitted values for use in the
        // checkSlopes() method.
        yFittedRange = Doubles.max(yFitted) - Doubles.min(yFitted);
      }

      /**
       * Special constructor for a dummy model that needs to be created when
       * no suitable model is found. This dummy model has only one segment with
       * slope equal to 0.0 and intercept on the mean of y.
       */
      public ModelLM(double[] x,
          double[] y,
          List<Integer> vertices,
          double yMean) {
        super(x, y, vertices, yMean);

        this.slopes = new DoubleArrayList();
        slopes.add(0.0);

        this.intercepts = new DoubleArrayList();
        intercepts.add(yMean);

        // calculates the fitted values for this model.
        this.yFitted = getFittedValues();

        this.pValue = 1.0;

        this.fStat = 0.0;
      }

      /**
       * This class is used by the Levenberg-Marquardt optimizer to tell it
       * what it (y values) should optimize and how to do it (gradient).
       */
      public class PiecewiseLinearFunction implements ParametricRealFunction {
        private double[] verticesPositions;

        /**
         * Constructs a parameterized continuous piecewise-linear function with
         * the vertices positions. The parameters to be optimized are the
         * vertices Y values.
         */
        public PiecewiseLinearFunction(double[] verticesPositions) {
          this.verticesPositions = verticesPositions;
        }

        /**
         * For a given value of x, finds to which segment it belongs.
         */
        private int findSegment(double x) {
          int segment = Arrays.binarySearch(verticesPositions, x);
          if (segment < 0) {
            segment = Math.abs(segment + 2);
          } else if (segment > 0) {
            segment -= 1;
          }

          return segment;
        }

        /**
         * For a given value of x and the vertices y values, returns the y=f(x).
         */
        @Override public double value(double x, double[] verticesYValues) {
          int pos = findSegment(x);
          double slope = (verticesYValues[pos + 1] - verticesYValues[pos])
              / (verticesPositions[pos + 1] - verticesPositions[pos]);

          return (x - verticesPositions[pos]) * slope + verticesYValues[pos];
        }

        /**
         * Computes the gradient of the function with respect to the vertices
         * y values at the given point.
         */
        @Override public double[] gradient(double x, double[] verticesYValues)
            throws FunctionEvaluationException {
          double[] gradient = new double[verticesYValues.length];
          int pos = findSegment(x);

          // The gradient is just an array of partial derivatives with respect
          // to the vertices y values, which are all zero except for the two
          // corresponding to the end-points of the segment that contains x.
          gradient[pos] = (verticesPositions[pos + 1] - x)
              / (verticesPositions[pos + 1] - verticesPositions[pos]);
          gradient[pos + 1] = (x - verticesPositions[pos])
              / (verticesPositions[pos + 1] - verticesPositions[pos]);

          return gradient;
        }
      }

      /**
       * This methods traces the segments that compose the model by the use of
       * the Levenberg-Marquardt method. After the segments are fit, the slopes
       * and intercepts are saved.
       */
      private void identifyBestPath(double[] x, CurveFitter fitter) {
        this.slopes = new DoubleArrayList();
        this.intercepts = new DoubleArrayList();

        // the initial guess of the model is that at each one of the vertices
        // positions they have an y value of mean(y).
        double[] verticesPositions = new double[vertices.size()];
        double[] verticesYValues = new double[vertices.size()];
        for (int j = 0; j < verticesPositions.length; j++) {
          verticesPositions[j] = x[vertices.get(j)];
          verticesYValues[j] = yMean;
        }
        ParametricRealFunction prf =
            new PiecewiseLinearFunction(verticesPositions);

        try {
          // get the fit from the optimizer.
          double[] fit = fitter.fit(prf, verticesYValues);

          // from the optimized y values of the vertices positions, extract the
          // intercept and slopes of the segments.
          for (int j = 0; j < fit.length - 1; j++) {
            double slope = (fit[j + 1] - fit[j])
                / (x[vertices.get(j + 1)] - x[vertices.get(j)]);
            this.slopes.add(slope);
            this.intercepts.add(fit[j]);
          }
        } catch (Exception e) {
          throw new RuntimeException("Couldn't run the Levenberg-Marquardt "
              + "Optimizer.", e);
        }
      }
    }

    /**
     *  First step of the algorithm is the dampering of spikes in the time
     *  series. This is accomplished by successively smoothing the
     *  observations by adding a correction to their current values.
     */
    public double[] desawtooth(double[] values, double threshold) {
      final double epsilon = 1e-9;
      final double alpha = 0.3;

      // The spikes will be dampered until the loop has run for the same amount
      // of times as the number of observations or there are no spikes above a
      // certain threshold.
      // TODO(gorelick): Authors coded in a way that the correction is always
      // applied at least one time. Right now I am in touch with the authors of
      // original code to know if this is intended.
      double max = 1.0;
      int maxIdx = -1;
      for (int count = 0; count < values.length; count++) {
        double[] correction = new double[values.length];
        double[] propCorrection = new double[values.length];

        max = 0.0;
        for (int i = 1; i < values.length - 1; i++) {
          double md = Math.max(Math.abs(values[i] - values[i - 1]),
              Math.abs(values[i] - values[i + 1]));

          if (md > 0.0) {
            propCorrection[i] = 1.0 - Math.abs(values[i - 1] - values[i + 1]) / md;
            correction[i] = propCorrection[i] * (((values[i - 1] + values[i + 1]) / 2) - values[i]);
          }

          //keep a record of the largest correction proportion
          if (max < propCorrection[i] || i==1) {
            max = propCorrection[i];
            maxIdx = i;
          }
        }

        // smooth the observations by adding a correction to the current values
        // Note (yang): if always to run it once add || count==0 in the if test.
        if (max > threshold) {
            values[maxIdx] = values[maxIdx] + correction[maxIdx];
        }
        else {
          //no correction needed
          break;
        }
      }
      return values;
    }

    /**
     * Given the values that belong to a segment, return its slope, intercept
     * and mse.
     * @param x is the observations time.
     * @param y is the observations values.
     * @param begin where the segment begins.
     * @param end where the segment ends.
     * @return the MSE, slope and intercept of the segment.
     */
    public double[] linearFit(double[] x,
        double[] y,
        int begin,
        int end) {

      // run the linear least square solver
      double[] residuals = new double[1];
      DenseMatrix64F results = new DenseMatrix64F(2, 1);

      //TODO: YANG verify LinearLeastSquares
      //NOTE: Noel indicate this section is wrong
//      LinearLeastSquares lls = new LinearLeastSquares(1, 1);
//      for (int i = begin; i <= end; i++) {
//        lls.addInput(x, i, y, i);
//      }
//      lls.getSolution(results);
//      lls.getRmsResiduals(results, residuals);


      //NOTE: new code from Noel, I think this is wrong.
      LinearLeastSquares lls = new LinearLeastSquares(2, 1);
      double[] designMatrix = {1, 0};
      for (int i = begin; i <= end; i++) {
        // The first x input is always 1, so just update the second x each time.
        designMatrix[1] = x[i];
        lls.addInput(designMatrix, 0, y, i);
      }
      lls.getSolution(results);
      lls.getRmsResiduals(results, residuals);


      // retrieve the slope and intercept
      double slope = results.get(1, 0);
      double intercept = results.get(0, 0);

      // retrieve the MSE
      double mse = residuals[0] * residuals[0];

      return new double[] {mse, slope, intercept};
    }

    /**
     * Given a segment, find its largest outlier.
     * @param x is the observations time;
     * @param y is the observations values
     * @param begin indicates where the segment begins;
     * @param end indicates where the segment ends;
     * @param intercept of the segment;
     * @param slope of the segment;
     * @param preventOneYearRecovery if the largest outlier is right before a
     * year of recovery, prevents the algorithm from choosing it and instead
     * takes the next largest outlier.
     * @return the index of the largest outlier in the given segment. In the
     * case that preventOneYearRecovery is set to true and there is onyl one
     * vertex between the end-points, it's possible that the return value may be
     * -1, indicating that this segment should not be broken into two by the
     * identifyPotentialVertices() method.
     */
    public int largestOutlier(double[] x,
        double[] y,
        int begin,
        int end,
        double slope,
        double intercept,
        boolean preventOneYearRecovery) {

      double maxResidual = 0.0;
      int maxResidualIndex = -1;
      for (int i = begin + 1; i < end; i++) {
        double tmp = Math.abs(y[i] - (intercept + x[i] * slope));
        if (preventOneYearRecovery && i == end - 1 && y[i] > y[i + 1]) {
          continue;
        }

        if (tmp > maxResidual) {
          maxResidual = tmp;
          maxResidualIndex = i;
        }
      }

      return maxResidualIndex;
    }

    /**
     *  This methods implements the second step of the algorithm, which is
     *  to identify potential vertices. What it does is to fit maxSegments +
     *  vertexCountOvershoot segments in the time series. These segments are
     *  fitted by breaking the segment with largest MSE into two new by breaking
     *  on the largest outlier of the segment.
     */
    public List<Integer> identifyPotentialVertices(double[] x,
        double[] y,
        int maxSegments,
        int vertexCountOvershoot,
        boolean preventOneYearRecovery) {

      int totalSegments = Math.min(maxSegments + vertexCountOvershoot,
          x.length - 1);

      // the first segment is between the end-points
      List<Integer> vertices = Lists.newArrayList();
      vertices.add(0);
      vertices.add(x.length - 1);
      totalSegments--;

      // the other segments are determined by finding the segment with biggest
      // MSE, then the observation within that segment that has the biggest
      // residue is the new vertex that breaks the segment into two new.
      for (; totalSegments > 0; totalSegments--) {

        // the rule of prevention of one year recovery only applies to the last
        // segment, so here it's treated differently.
        int begin = vertices.get(vertices.size() - 2);
        int end = vertices.get(vertices.size() - 1);
        double[] tmpFit = linearFit(x, y, begin, end);
        double mse = tmpFit[0];
        double slope = tmpFit[1];
        double intercept = tmpFit[2];
        int vertexToBreakAt = largestOutlier(x, y, begin, end, slope, intercept,
            preventOneYearRecovery);
        double maxSegmentMSE = mse;
        if (vertexToBreakAt == -1) {
          maxSegmentMSE = 0.0;
        }

        // for every other segment just run it with preventOneYearRecovery set
        // to false (last parameter of largestOutlier).
        for (int i = 0; i < vertices.size() - 2; i++) {
          begin = vertices.get(i);
          end = vertices.get(i + 1);

          // if the segment contains no vertices between the end-points then
          // it can not be broken, so skip it.
          if (end - begin <= 1) {
            continue;
          }

          // find the segment with biggest MSE
          tmpFit = linearFit(x, y, begin, end);
          mse = tmpFit[0];
          slope = tmpFit[1];
          intercept = tmpFit[2];

          // if it's the first run OR found a segment with a bigger MSE, then
          // update.
          if (vertexToBreakAt == -1 || maxSegmentMSE < mse) {
            vertexToBreakAt = largestOutlier(x, y, begin, end, slope, intercept,
                false);
            maxSegmentMSE = mse;
          }
        }

        // if the biggest MSE is zero, we are done, even if we have not
        // fitted all the segments.
        if (maxSegmentMSE <= 0.0) {
          break;
        }

        // insert the new vertex that breaks a segment into two (keeping the
        // list sorted).
        vertices.add(vertexToBreakAt);
        Collections.sort(vertices);
      }

      return vertices;
    }

    /**
     *  Auxiliary method for the cullByAngle method. This method calculates the
     *  angle difference of two successive segments. The arguments are as
     *  follows:
     *  @param x are the observations time;
     *  @param y are the observations values;
     *  @param idx indicates which vertex has its angle being measured;
     *  @param potentialVertices is used to provide the information of which
     *  vertexes are neighbors to the one indicated by idx;
     *  @param range of the spectral values;
     *  @param weightFactor is used to weight angles.
     */
    public double angleDifference(double[] x,
        double[] y,
        int idx,
        List<Integer> potentialVertices,
        double range,
        double weightFactor) {

      int curr = potentialVertices.get(idx);
      int prev = potentialVertices.get(idx - 1);
      int next = potentialVertices.get(idx + 1);

      double yDiff1 = y[curr] - y[prev];
      double yDiff2 = y[next] - y[curr];

      double angle1 = Math.atan(yDiff1 / (x[curr] - x[prev]));
      double angle2 = Math.atan(yDiff2 / (x[next] - x[curr]));

      // original comment: weightFactor helps determine how much weight is given
      // to angles that precede a disturbance.  If set to 0, the angle
      // difference is passed straight on.
      // This factor is hard coded to 2.0 (for the original code,
      // see vet_verts3.pro and angle_diff.pro at the github).
      double scaler = Math.max(0.0, yDiff2 * weightFactor / range) + 1.0;

      return scaler * Math.max(Math.abs(angle1), Math.abs(angle2));
    }

    /**
     *  Receives a list of as much as maxSegments + 1 + vertexCountOvershoot
     *  vertices, then it reduces down to maxSegments + 1 vertices by removing
     *  those vertices that creates segments with a small angle difference.
     */
    public List<Integer> cullByAngle(double[] x,
        double[] y,
        int maxSegments,
        List<Integer> potentialVertices) {
      int maxVertices = maxSegments + 1;

      if (potentialVertices.size() > maxVertices) {
        int nObs = x.length;

        // Find the min/max values of X and Y, and rescale Y to the same range
        // as X. This is to normalize the values for the angle calculations
        // below.
        double minY = y[0];
        double maxY = y[0];
        double minX = x[0];
        double maxX = x[0];
        for (int i = 1; i < nObs; i++) {
          minY = Math.min(minY, y[i]);
          maxY = Math.max(maxY, y[i]);
          minX = Math.min(minX, x[i]);
          maxX = Math.max(maxX, x[i]);
        }

        double[] tmpY = new double[y.length];
        for (int i = 0; i < nObs; i++) {
          tmpY[i] = (maxX - minX) * (y[i] - minY) / (maxY - minY);
        }
        double range = maxX - minX;

        // while there is still vertices to be removed, take out the with
        // smallest angle difference.
        List<Double> angles = new DoubleArrayList();
        for (int i = 1; i < potentialVertices.size() - 1; i++) {
          angles.add(angleDifference(x, tmpY, i, potentialVertices, range,
              2.0));
        }

        // note that during all of the process, the first and last vertices
        // shall not be removed.
        while (true) {
          int minAngleDiffIndex = 0;
          for (int i = 0; i < angles.size(); i++) {
            if (angles.get(minAngleDiffIndex) > angles.get(i)) {
              minAngleDiffIndex = i;
            }
          }

          potentialVertices.remove(minAngleDiffIndex + 1);
          angles.remove(minAngleDiffIndex);

          if (potentialVertices.size() <= maxVertices) {
            break;
          }

          if (0 == minAngleDiffIndex) {
            angles.set(0, angleDifference(x, tmpY, 1, potentialVertices,
              maxY - minY, 2.0));
          } else if (angles.size() == minAngleDiffIndex) {
            angles.set(minAngleDiffIndex - 1, angleDifference(x, tmpY,
                minAngleDiffIndex, potentialVertices, range, 2.0));
          } else {
            angles.set(minAngleDiffIndex, angleDifference(x, tmpY,
                minAngleDiffIndex + 1, potentialVertices, range, 2.0));
            angles.set(minAngleDiffIndex - 1, angleDifference(x, tmpY,
                minAngleDiffIndex, potentialVertices, range, 2.0));
          }
        }
      }

      return potentialVertices;
    }

    /**
     * Identifies the weakest vertex according to the recovery rate criterion.
     * It favors the removal of fast recovery years, because they are probably
     * caused by inadequate shadow or cloud masking.
     *
     * <p>This method is used by the pickBestmodel() method.
     */
    public int identifyWeakestVertex(double[] x,
        double[] y,
        double[] yFitted,
        List<Integer> vertices,
        List<Double> slopes,
        double recoveryThreshold) {

      double yMax = Doubles.max(yFitted);
      double yMin = Doubles.min(yFitted);

      boolean runMSE = true;
      int weakestIndex = -1;

      int biggestRecoveryIndex = -1; //which one has the largest slopes
      double largestScaledSlope = -1;
      double[] scaledSlopes = new double[slopes.size()];
      for (int i = 0; i < slopes.size(); i++) {
        scaledSlopes[i] = Math.abs(slopes.get(i)) / (yMax - yMin);
        if (slopes.get(i) < 0.0) {
          if (scaledSlopes[i] > largestScaledSlope) {
            biggestRecoveryIndex = i;
            largestScaledSlope = scaledSlopes[i];
          }
        }
      }

      // check against recovery threshold
      // there is a violating segment
      if (largestScaledSlope > recoveryThreshold) {
        int violatorIdx = vertices.get(biggestRecoveryIndex + 1);

        weakestIndex = biggestRecoveryIndex + 1;

        if (biggestRecoveryIndex+1 == slopes.size()) {
          // the violator is the second to last segment
          yFitted[violatorIdx] = yFitted[violatorIdx - 1];

          runMSE = true;
        }
        else {
          // other violators
          double leftX = x[violatorIdx - 1];
          double rightX = x[violatorIdx + 1];
          double leftY = y[violatorIdx - 1];
          double rightY = y[violatorIdx + 1];
          double thisX = x[violatorIdx];

          yFitted[violatorIdx] = (rightY - leftY) / (rightX - leftX) * (thisX - leftX) + leftY;

          runMSE = false;
        }
      }

      // if the criterion of recovery rate did not eliminate a vertex, then
      // use the criterion of vertex that increases the MSE as minimum as
      // possible.

      if (runMSE) {
        double smallestMSE = 0.0;
        for (int i = 1; i < vertices.size() - 1; i++) {
          int begin = vertices.get(i - 1);
          int end = vertices.get(i + 1);
          double slope = (yFitted[end] - yFitted[begin]) / (x[end] - x[begin]);
          double mse = 0.0;
          for (int j = begin; j <= end; j++) {
            double tmp = y[j] - ((x[j] - x[begin]) * slope + yFitted[begin]);
            mse += tmp * tmp;
          }
          mse /= (x[end] - x[begin]);

          if (1 == i || smallestMSE > mse) {
            smallestMSE = mse;
            weakestIndex = i;
          }
        }
      }

      return weakestIndex;
    }

    /**
     * From the set of p-values from all the models, chooses the best one, that
     * is the first one that is 1.25 times over the minimum one.
     * @param pValues all the p-values from the models.
     * @return the index of the model with p-value within the threshold; or -1
     * otherwise.
     */
    public int bestModelByPValue(double[] pValues,
        double bestModelProportion) {
      double minPValue = Doubles.min(pValues);

      // selects the first model that is within the threshold
      // Note: google implementation was wrong here, updated to match landtrendr logic
      for (int i = 0; i < pValues.length; i++) {
        if (pValues[i] <= minPValue * (2-bestModelProportion)) {
          return i;
        }
      }

      return -1;
    }

    /**
     * From the set of all models, choose the best one.
     * @param models is an array of all the models.
     * @param recoveryThreshold the recovery threshold used to make sure that
     * no segment in the model is in violation of it.
     * @return the best model or null if none of them is suitable.
     */
    public Model chooseBestModel(Model[] models,
        double recoveryThreshold,
        double bestModelProportion) {
      double[] pValues = new double[models.length];
      for (int i = 0; i < models.length; i++) {
        pValues[i] = models[i].pValue;
      }

      int count = 0;
      while (true) {
        int index = bestModelByPValue(pValues, bestModelProportion);
        // if no suitable model was found, then exit and return null.
        if (-1 == index) {
          break;
        } else {
          // if the model has a p-value within the threshold tested by
          // chooseBestModel() and has valid slopes, then use this one.
          if (models[index].checkSlopes(recoveryThreshold)) {
            return models[index];
          } else {
            // if the model was not suitable, then change its p-value and
            // f-statistic so it wont be chosen again.
            pValues[index] = 1.0;
            models[index].pValue = 1.0;
            models[index].fStat = 0.0;
          }

          // to avoid infinity loop. If no model was truly suitable, then exit
          // and return null.
          if (count > pValues.length) {
            break;
          }
        }
        count++;
      }

      return null;
    }

    /**
     * This method generates all the possible models from the given set of
     * vertices and chooses the best. The initial model contains all the initial
     * vertices, then on the next iteration one is taken out and a new model is
     * generated; this is repeated until a model with only two vertices is made.
     * To chose the best model, its p-value and f-statistic are taken into
     * account.
     */
    public Model identifyBestModel(double[] x,
        double[] y,
        double yMean,
        List<Integer> vertices,
        double recoveryThreshold,
        double bestModelProportion) {

      // making a copy to guarantee no external changes.
      List<Integer> tmpVertices = Lists.newArrayList(vertices);

      // generates all the possible models and store their properties.
      int modelsCount  = 0;
      ModelNormal[] models = new ModelNormal[tmpVertices.size() - 1];
      for (int i = tmpVertices.size(); i >= 2; i--) {
        models[modelsCount] = new ModelNormal(tmpVertices, x, y, yMean);

        if (i > 2) {
          int index = identifyWeakestVertex(x, y, models[modelsCount].yFitted,
              tmpVertices, models[modelsCount].slopes, recoveryThreshold);
          tmpVertices.remove(index);
          modelsCount++;
        }
      }

      Model bestModel = chooseBestModel(models, recoveryThreshold,
          bestModelProportion);
      if (bestModel == null) {
        // if no suitable model was found, then use the one with minimum
        // f-statistic.
        int indexMinFStat = 0;
        for (int i = 0; i < modelsCount; i++) {
          if (models[i].fStat < models[indexMinFStat].fStat) {
            indexMinFStat = i;
          }
        }
        return models[indexMinFStat];
      }
      return bestModel;
    }

    /**
     * This method generates all the possible models from the given set of
     * vertices and chooses the best. The initial model contains all the initial
     * vertices, then on the next iteration one is taken out and a new model is
     * generated; this is repeated until a model with only two vertices is made.
     * To chose the best model, its p-value is taken into account.
     */
    public Model identifyBestModelsUsingLevenbergMarquardt(double[] x,
        double[] y,
        double yMean,
        List<Integer> vertices,
        double recoveryThreshold,
        double bestModelProportion) {

      // making a copy to guarantee no external changes.
      List<Integer> tmpVertices = Lists.newArrayList(vertices);

      // initialize the Levenberg-Marquardt Optimizer
      LevenbergMarquardtOptimizer optimizer = new LevenbergMarquardtOptimizer();
      CurveFitter fitter = new CurveFitter(optimizer);
      for (int i = 0; i < x.length; i++) {
        fitter.addObservedPoint(x[i], y[i]);
      }

      // generates all the possible models.
      int modelsCount  = 0;
      ModelLM[] models = new ModelLM[tmpVertices.size() - 1];
      for (int i = tmpVertices.size(); i >= 2; i--) {
        models[modelsCount] = new ModelLM(tmpVertices, x, y, fitter, yMean);

        // the weakest vertex when using the Levenberg-Mardquardt method is
        // the one that when removed increases the least the MSE.
        int index = 1;
        double leastMSE = 0.0;
        double[] fitted = models[modelsCount].yFitted;
        for (int j = 1; j < tmpVertices.size() - 1; j++) {
          int prev = tmpVertices.get(j - 1);
          int next = tmpVertices.get(j + 1);
          double slope = (fitted[next] - fitted[prev]) / (x[next] - x[prev]);

          double mse = 0.0;
          for (int k = prev; k <= next; k++) {
            double tmp = y[k] - ((x[k] - x[prev]) * slope + fitted[prev]);
            mse += tmp * tmp;
          }
          mse /= (x[next] - x[prev]);

          if (j == 1 || mse < leastMSE) {
            leastMSE = mse;
            index = j;
          }
        }

        // remove the weakest
        tmpVertices.remove(index);
        modelsCount++;
      }

      Model bestModel = chooseBestModel(models, recoveryThreshold,
          bestModelProportion);
      if (bestModel == null) {
        // if no suitable model was found, then use a dummy one, which has only
        // two vertices and a straight line crossing on the mean.
        tmpVertices.clear();
        tmpVertices.add(0);
        tmpVertices.add(x.length - 1);
        ModelLM tmpModel = new ModelLM(x, y, tmpVertices, yMean);
        return tmpModel;
      }
      return bestModel;
    }

    /**
     * Builds the 2-D matrix of 4 rows and as many columns as images that is the
     * output of the algorithm. The first two rows are the original X and Y
     * values. The third row contains the Y values fitted to the estimated
     * segments, and the 4th row contains a 1 if a corresponding point was used
     * as a segment vertex or 0 if not.
     */
    private EEArray toArray(double[] x,
        double[] y,
        double[] yFitted,
        List<Integer> vertices) {

      EEArray.Builder result;

      result = EEArray.builder(PixelType.DOUBLE, 4, x.length);
      for (int i = 0; i < vertices.size(); i++) {
        result.setDouble(1.0, vertices.get(i));
      }
      int offset = 0;
      for (int i = 0; i < x.length; i++) {
        result.setDouble(x[i], offset++);
      }
      for (int i = 0; i < y.length; i++) {
        result.setDouble(y[i], offset++);
      }
      for (int i = 0; i < yFitted.length; i++) {
        result.setDouble(yFitted[i], offset++);
      }
      for (int i = 0; i < vertices.size(); i++) {
        result.setDouble(1.0, vertices.get(i) + offset);
      }

      return result.build();
    }
  }
}
