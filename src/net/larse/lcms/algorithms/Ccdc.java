/*
 * Copyright (c) 2015 Zhiqiang Yang, Noel Gorelick
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

import it.unimi.dsi.fastutil.ints.IntArrayList;
import net.larse.lcms.helper.GEELassoFitGenerator;
//import net.larse.lcms.helper.LassoFitGenerator;
import net.larse.lcms.helper.LassoFit;
import net.larse.lcms.helper.RobustLeastSquareBisquare;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math.stat.descriptive.rank.Min;
import org.ejml.data.DenseMatrix64F;

import java.util.*;
import java.util.stream.IntStream;

/** Based on CCDC 12.30 */
@SuppressWarnings("ALL")
public class Ccdc {
  // Bands Index
  private static final int IDX_BLUE = 0;
  private static final int IDX_GREEN = 1;
  private static final int IDX_RED = 2;
  private static final int IDX_NIR = 3;
  private static final int IDX_SWIR1 = 4;
  private static final int IDX_SWIR2 = 5;
  private static final int IDX_THERMAL = 6;
  private static final int IDX_FMASK = 7;

  private static final int CFMASK_CLEAR = 0;
  private static final int CFMASK_WATER = 1;
  private static final int CFMASK_CLOUD = 2;
  private static final int CFMASK_SNOW = 3;
  private static final int CFMASK_SHADOW = 4;
  private static final int CFMASK_FILL = 255;

  // QA Flag
  private static final int QA_SNOW = 50;
  private static final int QA_FMASK_FAILED = 40;

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

  // number of days in a year
  /*
   * NOTE: Original matlab implementation use datenum to represents date, what about changing
   * all the date representation as year + julday/365.25
   */
  static final double DAYS_IN_A_YEAR = 365.25;

  // Values change with number of pixels run, probably NOT NEEDED.
  private static final double MAX_NUM_FC = 30000;
  private static final double OMEGA = 2.0 * Math.PI / DAYS_IN_A_YEAR;

  // Total input bands (1-5,7,6,Fmask)
  private static final int TOTAL_IMAGE_BANDS = 8;

  // band for multitemporal cloud/snow detection (Green band)
  private static final int CLOUD_BAND = 2;

  // band for multitemporal shadow/snow shadow detection (SWIR2)
  private static final int SHADOW_BAND = 5;

  // minimum number of years for model initialization
  private static final int MIN_YEARS_NEEDED = 1;

  // index of bands used for change detection using Java indexing
  private int[] bands;

  // number of observation required to flag a change
  private int consen;

  // change threshold
  //  private double threshold_change = 0.99;
  // chi-square inversed T_cg (0.99) for noise removal on 5 bands
  private double T_CG;
  // chi-square inversed T_max_cg (1e-6) for last step noise removal
  private double TMAX_CG;

  // no change detection for permanent snow pixels
  private double thresholdSnowProportion = 0.75;

  // fmask fails threshold
  private double thresholdClearProportion = 0.25;

  // lambda for lasso regression
  private static final double lambda = 20;
  private static final GEELassoFitGenerator fitGenerator = new GEELassoFitGenerator();

  public Ccdc() throws Exception {
    // TODO: refactor to use better indicator for bands used. e.g. band bits indicator
    this(new int[] {1, 2, 3, 4, 5}, 6, 0.99);
  }

  public Ccdc(int[] bands, int consen, double changeThreshold) throws Exception {
    this.bands = bands;
    this.consen = consen;
    this.T_CG =
            new ChiSquaredDistributionImpl(bands.length).inverseCumulativeProbability(changeThreshold);
    this.TMAX_CG =
            new ChiSquaredDistributionImpl(bands.length).inverseCumulativeProbability(1 - 1e-6);
  }

  /**
   * run ccdc for a single pixel
   *
   * <p>NOTE: refls is currently defined as refl[num_observations][TOTAL_BANDS], TODO: evaluate
   * whether refls[TOTAL_BANDS][num_observations] is more efficient? TODO: should fmask be an
   * explicit input variable for getResult?
   *
   * @param refls [NUM_OBSERVATIONS][BANDS] array of time series of spectral values [1-5,7,6,fmask]
   * @param juliandays array of juliandays
   * @return
   */
  public List<CcdcFit> getResult(double[][] refls, double[] juliandays) throws Exception {
    // ccdc result object
    List<CcdcFit> results = new ArrayList<CcdcFit>();

    // NUMber of Functional Curves
    // NOTE: matlab code assign 0 here, in this implementation, size of results is used to indicate
    // the number of fc!!!
    int numFc = 0;

    // flag for data validity
    boolean[] flagRange = new boolean[refls.length];

    // clear observation: fmask is clear or water
    boolean[] flagClr = new boolean[refls.length];

    // snow observation
    boolean[] flagSnow = new boolean[refls.length];

    // number of clear observation
    int clearCount = 0;

    // number of snow observations
    int snowCount = 0;

    // Total observation
    int total = 0;

    // Total snow or clear observation
    int snowOrClearCount = 0;

    // count of good snow pixels
    int validSnowCount = 0;

    // count the unique record, calculate it here to reduce memoery and computation
    int countUniqueClearOrSnow = 0; // for permanant snow change detection -- may not need this
    int countUniqueClearOrSnowInrange = 0; // for permanant snow change detection
    int countUniqueInrange = 0; // for backup algorithms
    int countUniqueClearInrange = 0; // for normal operation

    // Get a list of the array indexes that are valid and unique.
    IntArrayList uniqueObs = new IntArrayList();

    // Convert Kelvin to Celsius (for new espa data)
    // Note: make sure the input data for thermal is scaled by 10.
    for (int i = 0; i < refls.length; i++) {
      refls[i][6] = refls[i][6] * 10 - 27315;

      flagRange[i] = inRange(refls[i][IDX_BLUE], 0, 10000)
              && inRange(refls[i][IDX_GREEN], 0, 10000)
              && inRange(refls[i][IDX_RED], 0, 10000)
              && inRange(refls[i][IDX_NIR], 0, 10000)
              && inRange(refls[i][IDX_SWIR1], 0, 10000)
              && inRange(refls[i][IDX_SWIR2], 0, 10000)
              && inRange(refls[i][IDX_THERMAL], -9320, 7070);

      flagClr[i] = refls[i][IDX_FMASK] < CFMASK_CLOUD; // clear = 0, water = 1

      // NOTE: the matlab implementation does not check for range flag when not enough data.
      if (flagClr[i]) { // && flagRange[i]) {
        clearCount++;
        snowOrClearCount++;
      }

      flagSnow[i] = refls[i][IDX_FMASK] == CFMASK_SNOW;
      if (flagSnow[i]) {
        snowCount++;
        snowOrClearCount++;
      }

      if (refls[i][IDX_FMASK] < CFMASK_FILL) {
        total++;
      }

      // count unique records
      if (i == 0 || juliandays[i] != juliandays[i - 1]) {
        uniqueObs.add(i);
        if (flagRange[i]) {
          countUniqueInrange++;
          if (flagClr[i]) {
            countUniqueClearInrange++;
            countUniqueClearOrSnowInrange++;
          }

          if (flagSnow[i]) {
            countUniqueClearOrSnowInrange++;
          }
        }

        if (flagSnow[i] || flagClr[i]) {
          countUniqueClearOrSnow++;
        }
      }
    }

    // not enought data points
    if (snowOrClearCount < 24) {
      return results;
    }

    double snowPct = 1.0 * snowCount / snowOrClearCount;
    double clearPct = 1.0 * clearCount / total;

    double[] clrx;
    double[][] clry;

    // Not enough clear observations for change detection
    if (clearPct < thresholdClearProportion) {
      // % permanent snow pixels
      if (snowPct > thresholdSnowProportion) {
        // snow observations are good data now

        // not enough snow observations
        if (snowOrClearCount < N_TIMES * MIN_NUM_COEFFS) {
          return results;
        }

        // NOTE: In matlab implementation, spectral bands and thermal were checked separately
        // where spectral bands does not check for thermal band, and thermal band is independent
        // of spectral band. In this Java implementation, I used combined checking for spectral
        // and thermal. The main purpose is to avoid initializing clrx and clry multiple times
        // for different conditions. If it is desired to treat spectrals and thermal band
        // independely, the following section will need to be recoded.

        // TODO: refactor the following section into a method.
        // There are (snowCount + clearCount) number of observations,
        // and we will use countUniqueClearOrSnowInrange for change detection
        clrx = new double[countUniqueClearOrSnow];
        clry = new double[countUniqueClearOrSnow][];
        int[] nValid = new int[TOTAL_IMAGE_BANDS - 1];
        boolean[][] flagValid = new boolean[TOTAL_IMAGE_BANDS - 1][countUniqueClearOrSnow];
        int nSn = 0;
        for (int i : uniqueObs) {
          if (flagClr[i] || flagSnow[i]) {
            clrx[nSn] = juliandays[i];
            clry[nSn] = refls[i];

            //check valid data by band
            for (int j = 0; j < TOTAL_IMAGE_BANDS - 1; j++) {
              if (j != IDX_THERMAL) {
                flagValid[j][nSn] = clry[nSn][j] < 10000;
                if (flagValid[j][nSn]) {
                  nValid[j]++;
                }
              } else {
                //clry[nSn][j] > -9320 && clry[nSn][j] < 7070;
                flagValid[j][nSn] = inRange(clry[nSn][j], -9320, 7070);
                if (flagValid[j][nSn]) {
                  nValid[j]++;
                }
              }
            }

            nSn++;
          }
        }
        // nSn should be the same as countUniqueClearOrSnowInrange now!
        // assert(nSn == countUniqueClearOrSnowInrange);
        CcdcFit ccdc = new CcdcFit();
        ccdc.computeFitV2(clrx, clry, flagValid, nValid, MIN_NUM_COEFFS, 0, clrx.length - 1);
        ccdc.tStart = (int) clrx[0];
        ccdc.tEnd = (int) clrx[clrx.length - 1];
        ccdc.tBreak = 0;
        ccdc.changeProb = 0;
        ccdc.numObs = nSn;
        ccdc.category = QA_SNOW + MIN_NUM_COEFFS;

        results.add(ccdc);

        numFc++;
      } else {
        // NOT_ENOUGHT_OBSERVATION_CLEAR:
        // no change detection for clear observation
        // backup algorithms
        // within physical range pixels

        // Get the median manually so we don't have to make another copy.
        DescriptiveStatistics stat = new DescriptiveStatistics();
        for (int i : uniqueObs) {
          if (flagRange[i]) {
            stat.addValue(refls[i][CLOUD_BAND - 1]);
          }
        }
        double band2Median = stat.getPercentile(50); // use greenband

        clrx = new double[countUniqueInrange];
        clry = new double[countUniqueInrange][];
        int nClr = 0;
        for (int i : uniqueObs) {
          if (refls[i][CLOUD_BAND - 1] < band2Median + 400.0 && flagRange[i]) {
            clrx[nClr] = juliandays[i];
            clry[nClr] = refls[i];
            nClr++;
          }
        }

        // not enough clear observations
        if (nClr < N_TIMES * MIN_NUM_COEFFS) {
          return results;
        }

        CcdcFit ccdc = new CcdcFit();
        ccdc.computeFit(clrx, clry, MIN_NUM_COEFFS, 0, nClr - 1);
        ccdc.tStart = (int) clrx[0];
        ccdc.tEnd = (int) clrx[nClr - 1];
        ccdc.tBreak = 0;
        ccdc.changeProb = 0;
        ccdc.numObs = nClr;
        ccdc.category = QA_FMASK_FAILED + MIN_NUM_COEFFS;

        results.add(ccdc);

        numFc++;
      }
    } else {
      //Normal CCDC Operation

      double iCount = 0;

      /* calculate expected variation for observations acquired more than 30 days
       (normally in every 32 days)
      */
      // we will use countUniqueClearInrange for change detection
      clrx = new double[countUniqueClearInrange];
      // use TOTAL_IMAGE_BANDS for second dimension to enable shallow copy
      // fmask band is not needed in TOTAL_IMAGE_BANDS
      clry = new double[countUniqueClearInrange][];
      int nClr = 0;
      // TODO: refactor the following to a method
      for (int i : uniqueObs) {
        if (flagClr[i] && flagRange[i]) {
          clrx[nClr] = juliandays[i];
          clry[nClr] = refls[i];
          nClr++;
        }
      }

      // n_sn should be the same as countUniqueClearInrange now!
      if (nClr != countUniqueClearInrange) {
        throw new RuntimeException("Invalid count of observations.");
      }

      /* Matlab
      % caculate median variogram
      var_clry = clry(2:end,:)-clry(1:end-1,:);
      adjRmse = median(abs(var_clry),1);
      */
      double[] adjRmse = median_variogram(clrx, clry, 1, 0);

      // number of fitted curves
      // start with minimum requirement of clear observations
      int i = N_TIMES * MIN_NUM_COEFFS; // Match Matlab indexing, NEED TO adjust for Java indexing

      // The first observation for TSFit
      int iStart = 1; // Match Matlab indexing, NEED TO adjust for Java indexing

      // record the start of model initialization (0: initial; 1: done)
      int blTrain = 0;

      // %identify and move on for the next curve
      // NOTE: this variable does not seem necessary as ArrayList is used to store fitting result
      // and it is currently updated when a new curve is added.
      // YANG: no need to increment here, the increment here is used by Matlab for indexing
      // numFc++; //%NUM of Fitted Curves (numFc)
      numFc++;

      /* record the numFc at the beginning of each pixel */
      int recFc = numFc; // NOTE: recFc might be always 0

      /* record the start of Tmask (0=>initial;1=>done) */
      int blTmask = 0;

      // span of i
      int iSpan = 0;
      double timeSpan;

      // NOTE: GLOBAL CCDC Variables
      // NOTE: in matlab, the following variables are created on the fly.
      // % record the magnitude of change
      double[][] vDifMag = new double[consen][TOTAL_IMAGE_BANDS - 1];
      double[][] vDif = new double[consen][TOTAL_IMAGE_BANDS - 1];
      // change vector magnitude
      double[] vecMag = new double[consen];

      double[][] fitCft = new double[TOTAL_IMAGE_BANDS - 1][];
      double[] rmse = new double[TOTAL_IMAGE_BANDS - 1];

      double[][] recVDif = new double[TOTAL_IMAGE_BANDS - 1][i - iStart + 1];
      // % normal fit qa = 0
      int qa = 0;

      int oldIstart = 1;
      int oldI = 1;

      /* while loop - process till the last clear observation - CONSE */
      while (i <= clrx.length - consen) {
        // span of i
        iSpan = i - iStart + 1;

        // span of time (number of years)
        timeSpan = (clrx[i - 1] - clrx[iStart - 1]) / DAYS_IN_A_YEAR;

        /* basic requrirements: 1) enough observations; 2) enough time */
        if ((iSpan >= N_TIMES * MIN_NUM_COEFFS) && (timeSpan >= MIN_YEARS_NEEDED)) {
          // initializing model

          if (blTrain == 0) {
            // step 1: noise removal (good: 0 & noise: 1)
            int[] validCount = new int[] {0}; // initially assume all values are valid
            int[] blIds =
                autoTmask(
                    clrx,
                    clry,
                    iStart - 1,
                    i + consen - 1,
                    (clrx[i + consen - 1] - clrx[iStart - 1]) / DAYS_IN_A_YEAR,
                    adjRmse[CLOUD_BAND - 1],
                    adjRmse[SHADOW_BAND - 1],
                    T_CONST,
                    validCount); // use green and SWIR band

            /* Matlab
            % Tmask: noise removal (good => 0 & noise => 1)
            blIDs = autoTmask(clrx(i_start:i+conse),clry(i_start:i+conse,[num_B1,num_B2]),...
                (clrx(i+conse)-clrx(i_start))/num_yrs,adj_rmse(num_B1),adj_rmse(num_B2),T_const);

            % IDs to be removed
            IDs = i_start:i+conse;
            rmIDs = IDs(blIDs(1:end-conse) == 1);

            % update i_span after noise removal
            i_span = sum(~blIDs(1:end-conse));
            */

            iSpan = 0;
            for (int ii = 0; ii < blIds.length - consen; ii++) {
              if (blIds[ii] == 0) {
                iSpan++;
              }
            }

            // check if there is enough observation
            if (iSpan < N_TIMES * MIN_NUM_COEFFS) {
              // NOTE: matlab implementation did not remove the identified noise if not enough
              // observation, should they be removed?
              // move forward to the i+1th clear observation
              i++;
              // not enough clear observation
              continue;
            } else { // check if there is enough time
                /* Matlab code
                    % copy x & y
                    cpx = clrx;
                    cpy = clry;

                    % remove noise pixels between i_start & i
                    cpx(rmIDs) = [];
                    cpy(rmIDs,:) = [];
                 */
              // FIXME: not efficient, unnecessary creation of cpClrx and cpClry
              double[] cpClrx;
              double[][] cpClry;
              if (iSpan < (i - iStart + 1)) { // There are outliers that need to be removed
                int remainingN = clrx.length - blIds.length + iSpan + consen;
                cpClrx = new double[remainingN];
                cpClry = new double[remainingN][];
                int cpi = 0;
                for (int iClrx = 0; iClrx < clrx.length; iClrx++) {
                  // check data in the current processing range
                  if (iClrx >= iStart - 1 && iClrx <= i - 1) {
                    int blIdx = iClrx - iStart + 1;
                    if (blIds[blIdx] == 0) { // not noise
                      cpClrx[cpi] = clrx[iClrx];
                      cpClry[cpi] = clry[iClrx];
                      cpi++;
                    }
                  } else {
                    cpClrx[cpi] = clrx[iClrx];
                    cpClry[cpi] = clry[iClrx];
                    cpi++;
                  }
                }
              } else {
                cpClrx = clrx;
                cpClry = clry;
              }

              /*
                  % record i before noise removal
                  % This is very important as if model is not initialized
                  % the multitemporal masking shall be done again instead
                  % of removing outliers in every masking
              */
              int iRec = i;

              // update i after noise removal (iStart stays the same)
              i = iStart + iSpan - 1;

              // update span of time (num of years)
              timeSpan = (cpClrx[i - 1] - cpClrx[iStart - 1]) / DAYS_IN_A_YEAR;

              // % check if there is enough time
              if (timeSpan < MIN_YEARS_NEEDED) {
                // keep the original i
                i = iRec;
                // move forward to the i+1th clear observation
                i++;
                // not enough time
                continue;
              } else { // update with noise removed
                clrx = cpClrx;
                clry = cpClry;

                /* Matlab code
                  % STEP 2: model fitting
                  % initialize model testing variables
                  % defining computed variables
                  fit_cft = zeros(max_num_c,nbands-1);
                  % rmse for each band
                  rmse = zeros(nbands-1,1);
                  % value of differnce
                  v_dif = zeros(nbands-1,1);
                  % record the diference in all bands
                  rec_v_dif = zeros(i-i_start+1,nbands-1);
                */
                // TODO: check whether these variable creations are necessary
                fitCft = new double[TOTAL_IMAGE_BANDS - 1][];
                // rmse for each band
                rmse = new double[TOTAL_IMAGE_BANDS - 1];
                // value of difference
                double[] yvDif = new double[TOTAL_IMAGE_BANDS - 1];
                double normVDif = 0;
                // record the difference in all bands
                recVDif = new double[TOTAL_IMAGE_BANDS - 1][];

                // do not process fmask band, the last band in TOTAL_IMAGE_BANDs is fmask
                // NOTE: due to difference between java implementation fo lasso and Matlab lasso,
                // The values for fitCft and fitRmse are not the same as Matlab code, the rmse
                // value is generally smaller than the matlab code (limited testing only)
                for (int b = 0; b < TOTAL_IMAGE_BANDS - 1; b++) {
                  TsFit tsfit = autoTSFit(clrx, clry, b, MIN_NUM_COEFFS, iStart - 1, i - 1);
                  fitCft[b] = tsfit.fitCft;
                  rmse[b] = tsfit.rmse;
                  recVDif[b] = tsfit.vDif;
                }

                // normalized to z-score
                for (int bi : bands) {
                  double minRmse = adjRmse[bi] < rmse[bi] ? rmse[bi] : adjRmse[bi];

                  // compare the first clear obs
                  double vStart = recVDif[bi][0] / minRmse;

                  // compare the last clear observation
                  double vEnd = recVDif[bi][i - iStart] / minRmse;

                  // anormalized slope values between first and last clear obs
                  double vSlope = fitCft[bi][1] * (clrx[i - 1] - clrx[iStart - 1]) / minRmse;

                  yvDif[bi] = Math.abs(vSlope) + Math.abs(vStart) + Math.abs(vEnd);

                  normVDif += yvDif[bi] * yvDif[bi];
                }

                // find stable start for each curve
                if (normVDif > T_CG) {
                  // start from next clear obs
                  iStart += 1;

                  // move forward to the i+1th clear observation
                  i += 1;

                  // keep all data and move to next obs
                  continue;
                } else {
                  // model ready
                  blTrain = 1;

                  // %count difference of i & j for each iteration
                  iCount = 0;

                  // what is iBreak? possible indicate index of breaking point, make SURE it the
                  // same as matlab indexing.
                  int iBreak = 1;

                  // find previous break point
                  // if numFc == recFc
                  if (results.size() == 0) {
                    iBreak = 1; // matlab indexing
                  } else {
                    // %after the first curve
                    /* Matlab
                      iBreak = find(clrx >= rec_cg(numFc-1).tBreak);
                      iBreak = iBreak(1);
                    */
                    // YANG CcdcFit fit = results.get(results.size()-2)
                    CcdcFit fit = results.get(results.size() - 1);
                    if (fit.tBreak == 0) {
                      iBreak = 1;
                    } else {
                      for (int tidx = fit.numObs - 1; tidx < clrx.length; tidx++) {
                        if ((int) clrx[tidx] >= fit.tBreak) {
                          iBreak = tidx + 1; // use matlab indexing
                          break;
                        }
                      }
                    }
                  }

                  int iniConse = consen;
                  vDifMag = new double[iniConse][TOTAL_IMAGE_BANDS - 1];
                  vecMag = new double[iniConse];

                  if (iStart > iBreak) {
                    // % model fit at the beginning of the time series
                    for (int iIni = iStart - 1; iIni >= iBreak; iIni--) {
                      iniConse = Math.min(iStart - iBreak, consen);

                      /* Matlab code
                          % value of difference for conse obs
                          vDif = zeros(iniConse,nbands-1);
                          % record the magnitude of change
                          vDifMag = vDif;
                          % chagne vector magnitude
                          vecMag = zeros(iniConse,1);
                      */

                      for (int ii = 0; ii < vDifMag.length; ii++) {
                        Arrays.fill(vDifMag[ii], 0);
                      }
                      Arrays.fill(vecMag, 0);

                      for (int iConse = 1; iConse <= iniConse; iConse++) {
                        for (int b = 0; b < TOTAL_IMAGE_BANDS - 1; b++) {
                          double[] pred = predict(clrx, fitCft[b], iIni - iConse, iIni - iConse);
                          // %absolute difference, NOTE: not absolute difference as matlab comments
                          vDifMag[iConse - 1][b] = clry[iIni - iConse][b] - pred[0];

                          if (ArrayUtils.contains(bands, b)) {
                            // minimum rmse
                            double minRmse = adjRmse[b] < rmse[b] ? rmse[b] : adjRmse[b];
                            double z = vDifMag[iConse - 1][b] / minRmse; // %z-scores
                            vecMag[iConse - 1] += z * z;
                          }
                        }
                      }

                      // % change detection\
                      // NOTE: only iniConse valid values
                      double minMag = vectorMin(vecMag, 0, iniConse);
                      if (minMag > T_CG) { // change detected
                        break;
                      } else if (vecMag[0] > TMAX_CG) { // %false change
                        // Remove noise
                        clrx = ArrayUtils.remove(clrx, iIni - 1);
                        clry = ArrayUtils.remove(clry, iIni - 1);
                        i--;
                      }

                      iStart = iIni;
                    }
                  }

                  // if numFc == recFc ...
                  //    && iStart - iBreak >= conse
                  if (results.size() == 0 && iStart - iBreak >= consen) {
                    /* Matlab code
                        % defining computed variables
                        fitCft = zeros(max_num_c,nbands-1);
                        % rmse for each band
                        rmse = zeros(nbands-1,1);
                        % start fit qa = 10
                        qa = 10;
                    */
                    qa = 10;

                    CcdcFit fit = new CcdcFit();
                    fit.computeFit(clrx, clry, MIN_NUM_COEFFS, iBreak - 1, iStart - 2);
                    fit.tStart = (int) clrx[0];
                    fit.tEnd = (int) clrx[iStart - 2];
                    fit.tBreak = (int) clrx[iStart - 1];
                    fit.changeProb = 1;
                    fit.category = qa + MIN_NUM_COEFFS;
                    fit.numObs = iStart - iBreak;

                    // % record change magnitude
                    // fit.magnitude = - median(vDifMag,1);
                    for (int bi = 0; bi < TOTAL_IMAGE_BANDS - 1; bi++) {
                      fit.magnitude[bi] = -1 * getMedian(vDifMag, bi, 0, iniConse);
                    }

                    results.add(fit);

                    // obsolete variable
                    numFc++;
                  }
                }
              }
            }
          }

          /*
          % continuous monitoring started!!!
          if BL_train == 1
          */
          if (blTrain == 1) {
            /* Java does not have build-in array subsetting, check IDs are used, for now introduce
            int oldIstart == i_start
            int oldI = i

            % all IDs
            IDs = i_start:i;
            */
            /** NO need to keep IDs, which is always i_start and i
             int oldIstart = i_start;
             int oldI = i;
             */
            //URGENT: CHECK scope, should this be global?

            iSpan = i - iStart + 1;

            int updateNumC = updateCft(
                    iSpan, N_TIMES, MIN_NUM_COEFFS, MID_NUM_COEFFS, MAX_NUM_COEFFS, NUM_COEFFS);

            // % intial model fit when there are not many obs
            if (iCount == 0 || iSpan <= MAX_NUM_COEFFS * N_TIMES) {

              // % update iCount at each iteration
              iCount = clrx[i - 1] - clrx[iStart - 1];

              /* Matlab code
                % defining computed variables
                fit_cft = zeros(max_num_c,nbands-1);
                % rmse for each band
                rmse = zeros(nbands-1,1);
                % record the diference in all bands
                rec_v_dif = zeros(length(IDs),nbands-1);
                % normal fit qa = 0
                qa = 0;
              */

              // TODO: can fit_cft from above be reused?
              // defining computed variables
              //TODO: with Noel's refactoring, the following section might not needed. Verify!!!
              fitCft = new double[TOTAL_IMAGE_BANDS - 1][MAX_NUM_COEFFS];
              rmse = new double[TOTAL_IMAGE_BANDS - 1];
              recVDif = new double[TOTAL_IMAGE_BANDS - 1][i - iStart + 1];
              qa = 0;

              for (int b = 0; b < TOTAL_IMAGE_BANDS - 1; b++) {
                TsFit tsfit = autoTSFit(clrx, clry, b, updateNumC, iStart - 1, i - 1);
                fitCft[b] = tsfit.fitCft; // is shallow copy enough here?!
                rmse[b] = tsfit.rmse;
                recVDif[b] = tsfit.vDif;
              }

              // NOTE: YANG-NUM_FC: numFc should be defined as the next fc to be updated, if it is
              // equal to the size of results, update exising one, otherwise, create new one!!!!
              CcdcFit ccdc;
              if (numFc == results.size()) {
                ccdc = results.get(numFc - 1);
              } else {
                ccdc = new CcdcFit();
                results.add(ccdc);
              }

              //TODO: can rmse and coefficient from above be used to avoid this re-calculation?
              ccdc.computeFit(clrx, clry, updateNumC, iStart - 1, i - 1);
              ccdc.tStart = (int) clrx[iStart - 1];
              ccdc.tEnd = (int) clrx[i - 1];
              ccdc.tBreak = 0;
              ccdc.changeProb = 0;
              ccdc.numObs = i - iStart + 1;
              ccdc.category = qa + updateNumC;

              // Nothing to be done for magnitude here.
              // % record change magnitude
              // rec_cg(numFc).magnitude = zeros(1,nbands-1);

              // push the fit to result, and keep updating it until a new fit is found
              // numFc is not updated here. very important!!!
              // results.add(ccdc);

              // FIXME: the following section could be modified to similarily as how
              /* Matlab code
                % detect change
                % value of difference for conse obs. FIXME: vDif may not needed
                vDif = zeros(conse,nbands-1);
                % record the magnitude of change
                vDifMag = vDif;
                vecMag = zeros(conse,1);
              */

              // moved to refactor_ZONE1:
              vDif = new double[consen][TOTAL_IMAGE_BANDS - 1];
              vDifMag = new double[consen][TOTAL_IMAGE_BANDS - 1];
              vecMag = new double[consen];

              for (int iConse = 0; iConse < consen; iConse++) {
                // TODO: refactor this code like the commented section immediate above.

                for (int b = 0;  b < TOTAL_IMAGE_BANDS - 1; b++) {
                  //%absolute difference
                  double[] pred = predict(clrx, fitCft[b], i + iConse, i + iConse);
                  // %absolute difference, NOTE: not absolute difference as matlab comments

                  vDifMag[iConse][b] = clry[i + iConse][b] - pred[0];
                  // % normalized to z-scores
                  if (ArrayUtils.contains(bands, b)) {
                    // minimum rmse
                    double minRmse = adjRmse[b] < rmse[b] ? rmse[b] : adjRmse[b];
                    double z = vDifMag[iConse][b] / minRmse; // %z-scores
                    //vDif[iConse][b] = z;
                    vecMag[iConse] += z * z;
                  }
                }
              }

              /*
              % IDs that haven't updated
              IDsOld = IDs;
              */
              oldIstart = iStart;
              oldI = i;
            } else {
              // FIXME: make fit a global variable to represent the current fit to be worked on.
              CcdcFit fit;
              if (numFc == results.size()) {
                fit = results.get(numFc - 1);
              } else {
                //Note: Noel's refactoring removed the parameterization of running CCDC using
                // subset of bands only.
                fit = new CcdcFit(); 
                results.add(fit);
              }

              if (clrx[i - 1] - clrx[iStart - 1] >= 1.33 * iCount) {
                // % update iCount at each interation
                iCount = clrx[i - 1] - clrx[iStart - 1];

                /* Matlab code
                    % defining computed variables
                    fit_cft = zeros(max_num_c,nbands-1);
                    % rmse for each band
                    rmse = zeros(nbands-1,1);
                    % record the diference in all bands
                    rec_v_dif = zeros(length(IDs),nbands-1);
                    % normal fit qa = 0
                    qa = 0;
                */

                // YANG: use the variable created above
                fitCft = new double[TOTAL_IMAGE_BANDS - 1][];
                rmse = new double[TOTAL_IMAGE_BANDS - 1];
                recVDif = new double[TOTAL_IMAGE_BANDS - 1][i - iStart + 1];
                qa = 0;

                for (int b = 0; b < TOTAL_IMAGE_BANDS - 1; b++) {
                  TsFit tsfit = autoTSFit(clrx, clry, b, updateNumC, iStart - 1, i - 1);
                  fitCft[b] = tsfit.fitCft; // is shallow copy enough here?!
                  rmse[b] = tsfit.rmse;
                  recVDif[b] = tsfit.vDif;
                }

                fit.coefs = fitCft;
                fit.rmse = rmse;
                fit.numObs = i - iStart + 1;
                fit.category = qa + updateNumC;

                /*
                  % IDs that haven't updated
                IDsOld = IDs;
                */
                oldIstart = iStart;
                oldI = i;
              }

              // % record time of curve end
              fit.tEnd = (int) clrx[i - 1];

              // % use fixed number for RMSE computing
              int nRmse = N_TIMES * fit.category;
              double[] tmpcgRmse = new double[TOTAL_IMAGE_BANDS - 1];

              /* Matlab code
              //% better days counting for RMSE calculating
              //% relative days distance
              d_rt = clrx(IDsOld) - clrx(i+conse);
              d_yr = abs(round(d_rt/num_yrs)*num_yrs-d_rt);
              */
              // double[] d_rt = new double[oldI - oldIstart + 1]; not needed
              double[] dYr = new double[oldI - oldIstart + 1];
              for (int b = 0; b < dYr.length; b++) {
                double tRt = clrx[oldIstart - 1 + b] - clrx[i - 1 + consen];
                dYr[b] = Math.abs(Math.round(tRt / DAYS_IN_A_YEAR) * DAYS_IN_A_YEAR - tRt);
              }

              // Get the sorted indexes of the dYr array.
              int[] sortedIndx = IntStream.range(0, dYr.length)
                      .boxed()
                      .sorted((di, dj) -> Double.compare(dYr[di], dYr[dj]))
                      .mapToInt(ele -> ele)
                      .toArray();

              /* Matlab code
              for i_B = B_detect
                % temporally changing RMSE
                tmpcg_rmse(i_B) = norm(rec_v_dif(IDsOld(sorted_indx)-IDsOld(1)+1,i_B))/...
                    sqrt(n_rmse-rec_cg(num_fc).category);
              end
              */
              for (int b : bands) {
                double sumSquare = 0;
                int trvdIndx = 0; // index into recVDif
                for (int ssi : sortedIndx) {
                  if (trvdIndx++ < nRmse) {
                    sumSquare += recVDif[b][ssi] * recVDif[b][ssi];
                  } else {
                    break;
                  }
                }

                tmpcgRmse[b] = Math.sqrt(sumSquare / (nRmse - fit.category));
              }

              /* Matlab code
                % move the ith col to i-1th col
                v_dif(1:conse-1,:) = v_dif(2:conse,:);
                % only compute the difference of last consecutive obs
                v_dif(conse,:) = 0;
                % move the ith col to i-1th col
                v_dif_mag(1:conse-1,:) = v_dif_mag(2:conse,:);
                % record the magnitude of change of the last conse obs
                v_dif_mag(conse,:) = 0;
                % move the ith col to i-1th col
                vec_mag(1:conse-1) = vec_mag(2:conse);
                % change vector magnitude
                vec_mag(conse) = 0;
              */

              // Move the ith col to i-1 pos.  Fill with 0.
              Arrays.fill(vDif[0], 0);
              ArrayUtils.shift(vDif, -1);

              Arrays.fill(vDifMag[0], 0);
              ArrayUtils.shift(vDifMag, -1);

              vecMag[0] = 0;
              ArrayUtils.shift(vecMag, -1);

              /* Matlab code
                for i_B = 1:nbands-1
                  % absolute difference
                  v_dif_mag(conse,i_B) = clry(i+conse,i_B)-autoTSPred(clrx(i+conse),fit_cft(:,i_B));
                  % normalized to z-scores
                  if sum(i_B == B_detect)
                    % minimum rmse
                    mini_rmse = max(adj_rmse(i_B),tmpcg_rmse(i_B));

                    % z-scores
                    v_dif(conse,i_B) = v_dif_mag(conse,i_B)/mini_rmse;
                  end
                end
                vec_mag(conse) = norm(v_dif(end,B_detect))^2;
              */
              // TODO: does the -1 need here?
              int currIndex = i - 1 + consen;
              for (int b = 0; b < TOTAL_IMAGE_BANDS - 1; b++) {
                double[] pred = predict(clrx, fit.coefs[b], currIndex, currIndex);
                vDifMag[consen - 1][b] = clry[currIndex][b] - pred[0];

                if (ArrayUtils.contains(bands, b)) {
                  // minimum rmse
                  double minRmse = adjRmse[b] < tmpcgRmse[b] ? tmpcgRmse[b] : adjRmse[b];
                  double z = vDifMag[consen - 1][b] / minRmse;
                  vDif[consen - 1][b] = z;
                  vecMag[consen - 1] += z * z;
                }
              }
            }

            // % change detection
            // if min(vecMag) > T_cg % change detected
            double minMag = new Min().evaluate(vecMag);
            if (minMag > T_CG) {
              // change detected

              // record break time
              CcdcFit fit = results.get(results.size() - 1);
              fit.tBreak = (int) clrx[i];
              fit.changeProb = 1;
              for (int bi = 0; bi < TOTAL_IMAGE_BANDS - 1; bi++) {
                fit.magnitude[bi] = getMedian(vDifMag, bi);
              }

              // NOW increment numFc
              // % identified and move on for the next functional curve
              // This is when numFc is not the same as number items in results
              numFc++;

              // % start from i+1 for the next functional curve
              iStart = i + 1;

              // % start training again
              blTrain = 0;
            } else if (vecMag[0] > TMAX_CG) { // false change
              // % remove noise
              clrx = ArrayUtils.remove(clrx, i);
              clry = ArrayUtils.remove(clry, i);
              i--; // % stay & check again after noise removal
            }
          } // % end of continuous monitoring
        } // % end of checking basic requrirements
        // % move forward to the i+1th clear observation
        i++;
      } // % end of while iterative

      // % Two ways for processing the end of the time series
      if (blTrain == 1) {
        // % 1) if no break find at the end of the time series

        // TODO: is idLast only relevant here, what should be the default value
        int idLast = consen; // This is matlab index, adjust to java index when using it

        // % define probability of change based on conse
        for (int iConse = consen; iConse > 0; iConse--) {
          if (vecMag[iConse - 1] <= T_CG) {
            // % the last stable
            // TODO: how idLast is used, is this only local scope?
            idLast = iConse;
            break;
          }
        }

        CcdcFit fit = results.get(results.size() - 1);
        fit.changeProb = 1.0 * (consen - idLast) / consen;
        fit.tEnd = (int) clrx[clrx.length - 1 - consen + idLast];
        if (consen > idLast) { // % > 1
          // % update time of the probable change
          fit.tBreak = (int) clrx[clrx.length - consen + idLast];

          // % update magnitude of change
          for (int bi = 0; bi < TOTAL_IMAGE_BANDS - 1; bi++) {
            fit.magnitude[bi] = getMedian(vDifMag, bi, idLast, consen);
          }
        }
      } else if (blTrain == 0) {
        // % 2) if break find close to the end of the time series

        // % Use [conse,min_num_c*n_times+conse) to fit curve
        // NOTE: !!!numFc should always be the same as length of results!!!
        if (numFc == recFc) {
          // %first curve
          iStart = 1;
        } else {
          // Matlab code
          // iStart = find(clrx >= rec_cg(numFc-1).tBreak);
          // iStart = iStart(1);
          // YANG CcdcFit fit = results.get(results.size()-2)
          CcdcFit fit = results.get(results.size() - 1);
          for (int tidx = fit.numObs - 1; tidx < clrx.length; tidx++) {
            if ((int) clrx[tidx] >= fit.tBreak) {
              iStart = tidx + 1; // use matlab indexing
              break;
            }
          }
        }

        // %Tmask
        // TODO: YANG-Nov-22
        if ((clrx.length - iStart + 1) > consen) {
          int[] validCount = new int[] {0}; // initially assume all values are valid
          // use green and SWIR band
          int[] blIds = autoTmask(
                  clrx,
                  clry,
                  iStart - 1,
                  clrx.length - 1,
                  (clrx[clrx.length - 1] - clrx[iStart - 1]) / DAYS_IN_A_YEAR,
                  adjRmse[CLOUD_BAND - 1],
                  adjRmse[SHADOW_BAND - 1],
                  T_CONST,
                  validCount);

          iSpan = validCount[0];

          int nRemoval = 0;
          for (int ii = 0; ii < blIds.length - consen; ii++) {
            if (blIds[ii] == 1) {
              nRemoval++;
            }
          }

          // There are outliers to be removed
          if (nRemoval > 0) {
            // NOTE: this is similar to the begining of model checking, refactor this to reuse code.
            int remainingN = clrx.length - nRemoval;

            double[] cpClrx = new double[remainingN];
            double[][] cpClry = new double[remainingN][];

            int cpi = 0;
            for (int iClrx = 0; iClrx < clrx.length; iClrx++) {
              // check data in the current processing range
              if (iClrx >= iStart - 1 && iClrx < clrx.length - consen) {
                int blIdx = iClrx - iStart + 1;
                if (blIds[blIdx] == 0) { // not noise
                  cpClrx[cpi] = clrx[iClrx];
                  cpClry[cpi] = clry[iClrx];
                  cpi++;
                }
              } else {
                cpClrx[cpi] = clrx[iClrx];
                cpClry[cpi] = clry[iClrx];
                cpi++;
              }
            }
            clrx = cpClrx;
            clry = cpClry;
          }
        }

        // %enough data
        if (clrx.length - iStart + 1 >= consen) {

          /* Matlab code
            % defining computed variables
            fit_cft = zeros(max_num_c,nbands-1);
            % rmse for each band
            rmse = zeros(nbands-1,1);
            % end of fit qa = 20
            qa = 20;
           */

          //% end of fit qa = 20
          qa = 20;
          CcdcFit fit;
          if (numFc == results.size()) {
            fit = results.get(numFc - 1);
          } else {
            fit = new CcdcFit();
            results.add(fit);
            fit.computeFit(clrx, clry, MIN_NUM_COEFFS, iStart - 1, clrx.length - 1);
          }

          fit.tStart = (int) clrx[iStart - 1];
          fit.tEnd = (int) clrx[clrx.length - 1];
          fit.tBreak = 0;
          fit.changeProb = 0;
          fit.numObs = clrx.length - iStart + 1;
          fit.category = qa + MIN_NUM_COEFFS;
        }
//        else {
//          //not enough data, remove the last empty curve
//          //Note: this used a check on the t_start, which seems not necessary.
//          // if reached here, the last one should always be empty.
//          CcdcFit empty = results.get(results.size()-1);
//          if (empty.t_start==0) {
//            results.remove(results.size()-1);
//          }
//        }
      }
    }
    return results;
  }

  /**
   * Determine the time series model
   *
   * @param iSpan
   * @param nTimes
   * @param minNumC
   * @param midNumC
   * @param maxNumC
   * @param numC
   * @return
   */
  private int updateCft(int iSpan, int nTimes, int minNumC, int midNumC, int maxNumC, int numC) {
    int n;
    if (iSpan < midNumC * nTimes) {
      n = minNumC;
    } else if (iSpan < maxNumC * nTimes) {
      n = midNumC;
    } else {
      n = maxNumC;
    }
    return Math.min(n, numC);
  }

  /**
   * Get the median of the given band.
   * @param dat [NUM_OBSERVATION][BANDS]
   * @param index band index
   * @param xstart start NUM_OBSERVATION (inclusive)
   * @param xend end NUM_OBSERVATION (exclusive)
   * @return
   */
  private double getMedian(double[][] dat, int index, int xstart, int xend) {
    DescriptiveStatistics stat = new DescriptiveStatistics();

    for (int i = xstart; i < xend; i++) {
      stat.addValue(dat[i][index]);
    }

    return stat.getPercentile(50);
  }

  /**
   * Get the median of the given band.
   * @param dat [NUM_OBSERVATION][BANDS]
   * @param index band index
   * @return
   */
  private double getMedian(double[][] dat, int index) {
    return getMedian(dat, index, 0, dat.length);
  }

  private double vectorMin(double[] values, int begin, int length) {
    double min = values[begin];
    for (int i = begin; i < begin + length; i++) {
      if (!Double.isNaN(values[i])) {
        min = (min < values[i]) ? min : values[i];
      }
    }
    return min;
  }

  /**
   * filter out the unique element based on x
   *
   * @param x identifier
   * @param y values
   * @return number of unique record
   */
  private int filterUnique(double[] x, double[][] y) {
    // no need to do anything if there is 0 or 1 element
    if (x.length < 2) {
      return x.length;
    }

    int count = 1;
    for (int k = 1; k < x.length; k++) {
      if (x[k] != x[k - 1]) {
        x[count] = x[k];
        y[count] = y[k];
        count++;
      }
    }
    return count;
  }

  /**
   * Auto Trends and Seasonal Fit between breaks INPUTS: x - Julian day [1; 2; 3]; y - predicted
   * reflectances [0.1; 0.2; 0.3]; df - degree of freedom (num_c)
   *
   * <p>OUTPUTS: fitCft - fitted coefficients;
   * General model TSModel:
   * f1(x) = a0 + b0*x (df = 2)
   * f2(x) = f1(x) + a1*cos(x*w) + b1*sin(x*w) (df = 4)
   * f3(x) = f2(x) + a2*cos(x*2w) + b2*sin(x*2w) (df = 6)
   * f4(x) = f3(x) + a3*cos(x*3w) + b3*sin(x*3w) (df = 8)
   *
   * @param x julday
   * @param y double[observation][bands] b1-b5,b7,b6,fmask
   * @param band band index to run
   * @param df degree of freedom
   * @param idxStart startIndex (inclusive)
   * @param idxEnd endIndex (inclusive)
   * @return
   * @throws Exception
   */
  private static TsFit autoTSFit(
      double[] x, double[][] y, int band, int df, int idxStart, int idxEnd)
      throws Exception {
    int count = idxEnd - idxStart + 1;

    // initialize LassoFitGenerator
    fitGenerator.init(df - 1, count);

    for (int i = idxStart; i <= idxEnd; i++) {
      //fitGenerator.setObservationValues(i - idxStart, observations[i - idxStart]);
      fitGenerator.setObservation(i - idxStart, 0, x[i]);
      int idx = 0;
      for (int k = 1; k <= df / 2 - 1; k++) {
        fitGenerator.setObservation(i - idxStart, ++idx, Math.cos(k * OMEGA * x[i]));
        fitGenerator.setObservation(i - idxStart, ++idx, Math.sin(k * OMEGA * x[i]));
      }
      fitGenerator.setTarget(i - idxStart, y[i][band]);
    }


    // which index is lambda 20
    LassoFit fit = fitGenerator.fit(lambda);
    int idx = fit.getFitByLambda(lambda);

    TsFit result = new TsFit();
    result.rmse = fit.adjustedRmses[idx];
    double[] betas = fit.getBetas(idx);
    result.fitCft = new double[8];
    System.arraycopy(betas, 0, result.fitCft, 0, betas.length);

    // calculate predicted value and prediction difference
    result.vDif = predict(x, result.fitCft, idxStart, idxEnd);
    for (int i = idxStart; i <= idxEnd; i++) {
      double resid = y[i][band] - result.vDif[i - idxStart];
      result.vDif[i - idxStart] = resid;
    }
    return result;
  }

  private static TsFit autoTSFitV2(
      double[] x,
      double[][] y,
      boolean[] flags,
      int nValid,
      int band,
      int df,
      int idxStart,
      int idxEnd)
      throws Exception {

    // initialize LassoFitGenerator
    fitGenerator.init(df - 1, nValid);

    int vi = 0;
    for (int i = idxStart; i <= idxEnd; i++) {
      //fitGenerator.setObservationValues(i - idxStart, observations[i - idxStart]);
      if (flags[i]) {
        fitGenerator.setObservation(vi, 0, x[i]);
        int idx = 0;
        for (int k = 1; k <= df / 2 - 1; k++) {
          fitGenerator.setObservation(vi, ++idx, Math.cos(k * OMEGA * x[i]));
          fitGenerator.setObservation(vi, ++idx, Math.sin(k * OMEGA * x[i]));
        }
        fitGenerator.setTarget(vi++, y[i][band]);
      }
    }

    // which index is lambda 20
    LassoFit fit = fitGenerator.fit(lambda);
    int idx = fit.getFitByLambda(lambda);

    TsFit result = new TsFit();
    result.rmse = fit.adjustedRmses[idx];
    double[] betas = fit.getBetas(idx);
    result.fitCft = new double[8];
    System.arraycopy(betas, 0, result.fitCft, 0, betas.length);

    // calculate predicted value and prediction difference
    result.vDif = predictV2(x, result.fitCft, idxStart, idxEnd, flags, nValid);
    vi = 0;
    for (int i = idxStart; i <= idxEnd; i++) {
      if (!flags[i]) {
        continue;
      }
      double resid = y[i][band] - result.vDif[vi];
      result.vDif[vi++] = resid;
    }
    return result;
  }



  /**
   * Auto Trends and Seasonal Predict
   *
   * @param x
   * @param fitCft
   * @return
   */
  private static double[] predict(double[] x, double[] fitCft, int idxStart, int idxEnd) {
    double[] prediction = new double[idxEnd - idxStart + 1];
    for (int i = idxStart; i <= idxEnd; i++) {
      double v = fitCft[0] + fitCft[1] * x[i];

      double rx = x[i] * OMEGA;
      for (int h = 1; h < 4; h++) {
        v += fitCft[2 + (h - 1) * 2] * Math.cos(h * rx);
        v += fitCft[2 + (h - 1) * 2 + 1] * Math.sin(h * rx);
      }
      prediction[i - idxStart] = v;
    }
    return prediction;
  }

  private static double[] predictV2(
      double[] x, double[] fitCft, int idxStart, int idxEnd, boolean[] flags, int nValid) {
    double[] prediction = new double[nValid];
    int vi = 0;
    for (int i = idxStart; i <= idxEnd; i++) {
      if (flags[i]) {
        double v = fitCft[0] + fitCft[1] * x[i];

        double rx = x[i] * OMEGA;
        for (int h = 1; h < 4; h++) {
          v += fitCft[2 + (h - 1) * 2] * Math.cos(h * rx);
          v += fitCft[2 + (h - 1) * 2 + 1] * Math.sin(h * rx);
        }
        prediction[vi++] = v;
      }
    }
    return prediction;
  }

  /**
   * Multitepmoral cloud, cloud shadow, & snow masks (global version) read in data with 3 more
   * consecutive clear obs & correct data
   *
   * @return
   */
  private int[] autoTmask(
          double[] clrx,
          double[][] clry,
          int start,
          int end,
          double years,
          double tB2,
          double tB5,
          double tConst,
          int[] validCount) {

    int nums = end - start + 1;
    int year = (int) Math.ceil(years);

    int coefsCount = year > 1 ? ROBUST_COEFFS : 3;

    // No way around making a copy; fit() wants a matrix.
    DenseMatrix64F x = new DenseMatrix64F(nums, coefsCount);
    DenseMatrix64F b2 = new DenseMatrix64F(nums, 1);
    DenseMatrix64F b5 = new DenseMatrix64F(nums, 1);

    double w2 = OMEGA / year;

    for (int i = 0; i < nums; i++) {
      x.set(i, 0, 1.0); // for intercept
      x.set(i, 1, Math.cos(this.OMEGA * clrx[i + start]));
      x.set(i, 2, Math.sin(this.OMEGA * clrx[i + start]));

      if (year > 1) {
        x.set(i, 3, Math.cos(w2 * clrx[i + start]));
        x.set(i, 4, Math.sin(w2 * clrx[i + start]));
      }

      b2.set(i, 0, clry[i + start][1]); // band 2
      b5.set(i, 0, clry[i + start][4]); // band 5
    }

    // perform robust fit
    // TODO: make RobustLeastSquareBisquare as a utitly function so that no initialization is need
    // to use it.
    RobustLeastSquareBisquare rls = new RobustLeastSquareBisquare(x, b2, 4.685);
    DenseMatrix64F b2coefs = new DenseMatrix64F(coefsCount, 1);
    boolean b2fit = rls.getSolution(b2coefs);

    DenseMatrix64F b5coefs = new DenseMatrix64F(coefsCount, 1);
    rls.updateB(new DenseMatrix64F(b5));
    boolean b5fit = rls.getSolution(b5coefs);

    double[] coefs2 = b2coefs.getData();
    double[] coefs5 = b5coefs.getData();
    double b2Threshold = tConst * tB2;
    double b5Threshold = tConst * tB5;

    int valid = 0;
    int[] result = new int[nums];

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
              || Math.abs(predB5 - b5.get(i,0)) > b5Threshold) {
        result[i] = 1;
      } else {
        result[i] = 0;
        validCount[0] = ++valid;
      }
    }

    return result;
  }

  /**
   * calculate mode of an array. This method exists in Apache commons math3, but not in math2. TODO:
   * If GEE upgrade to commons math3, use org.apache.commons.math3.stat.StatUtils
   *
   * @param values
   * @return double
   */
  private double mode(double[] values, int n) {
    double[] items = new double[values.length];

    int mode;
    int max = 0, k = 0, cnt = 0;
    for (int i = 0; i < n - 1; i++) {
      mode = 0;
      for (int j = i + 1; j < n; j++) {
        if (values[i] == values[j]) {
          mode++;
        }
      }

      if (mode > max && mode != 0) {
        k = 0;
        max = mode;
        items[k] = values[i];
      } else if (mode == max) {
        items[k] = values[i];
        k++;
      }
    }

    for (int i = 0; i < n; i++) {
      if (values[i] == items[i]) {
        cnt++;
      }
    }

    // in case all values are unique, return the first value in the input array
    if (cnt == n) {
      return values[0];
    } else {
      // There could be multiple values with the same count (k>1),
      // but we are only using the first one for this application.
      return items[0];
    }
  }

  private double[] median_variogram(double[] x, double[][] y, int offset, int daysApart) {
    // no data being provided
    if (x.length == 0) {
      return null;
    }

    // number of bands including fmask
    int bandCount = y[0].length;
    // exclude fmask
    double[] result = new double[bandCount - 1];

    // if there is only single value
    if (x.length == 1) {
      result = ArrayUtils.subarray(y[0], 0, bandCount - 1);
    }

    DescriptiveStatistics stat = new DescriptiveStatistics();
    for (int i = 0; i < bandCount - 1; i++) {
      stat.clear();
      for (int j = offset; j < x.length; j++) {
        if (x[j] - x[j - offset] > daysApart) {
          stat.addValue(Math.abs(y[j][i] - y[j - offset][i]));
        }
      }
      result[i] = stat.getPercentile(50);
    }

    return result;
  }

  boolean inRange(double x, double low, double hi) {
    return x > low && x < hi;
  }

  /** Represent result from autoTSFit */
  public static class TsFit {
    public double[] fitCft;
    public double rmse;
    public double[] vDif; // TODO: is this necessary?
  }

  /** Parameters for a single segment */
  public static class CcdcFit {
    // Time when series model gets started.
    public int tStart;

    // Time when series model gets ended.
    public int tEnd;

    // Time when the first break (change) is observed.
    public int tBreak;

    /**
     * Coefficients for each time series model for each spectral band. [NUM_BANDS][NUM_COEFFS]
     * NUM_BANDS=7, NUM_COEFFS=8
     */
    public double[][] coefs;

    /** RMSE of the each band's fit */
    public double[] rmse;

    /** The probability of a pixel that have undergone change (0, 100). */
    public double changeProb;

    /** The number of "good" observations used for mode estimation. */
    public int numObs;

    /**
     * The quality of the model estimation (what model is used, what process is used). 1x:
     * persistent snow 2x: persistent water 3x: Fmask fails 4x: normal x1: mean value (1) x4: simple
     * fit (4) x6: basic fit (6) x8: full fit (8)
     */
    public int category;

    /** The magnitude of change (difference between prediction and observation for each band). */
    public double[] magnitude;

    /** Construct a new fit. */
    CcdcFit() {
      this.magnitude = new double[8];
    }

    void computeFit(double[] x, double[][] y, int df, int idxStart, int idxEnd) throws Exception {
      coefs = new double[TOTAL_IMAGE_BANDS - 1][];
      rmse = new double[TOTAL_IMAGE_BANDS - 1];
      for (int b = 0; b < TOTAL_IMAGE_BANDS - 1; b++) {
        TsFit tsfit = autoTSFit(x, y, b, df, idxStart, idxEnd);
        coefs[b] = tsfit.fitCft;
        rmse[b] = tsfit.rmse;
      }
    }

    void computeFitV2(double[] x, double[][] y,
                      boolean[][] flags, int[] nValid,
                      int df, int idxStart, int idxEnd) throws Exception {
      // FIX THIS
      coefs = new double[TOTAL_IMAGE_BANDS - 1][MAX_NUM_COEFFS];
      rmse = new double[TOTAL_IMAGE_BANDS - 1];
      for (int b = 0; b < TOTAL_IMAGE_BANDS - 1; b++) {
        if (b != Ccdc.IDX_THERMAL && nValid[b] < Ccdc.MIN_NUM_COEFFS * Ccdc.N_TIMES) {//optical band
            coefs[b][0] = 10000.0; //fixed value for saturated pixels
        }
        else {
          TsFit tsfit = autoTSFitV2(x, y, flags[b], nValid[b], b, df, idxStart, idxEnd);
          coefs[b] = tsfit.fitCft;
          rmse[b] = tsfit.rmse;
        }
      }
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
}
