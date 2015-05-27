package net.larse.lcms.algorithms;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math.stat.regression.SimpleRegression;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Implements the Vegetation Change Tracker (VCT) that was proposed in the
 * paper: Huang C, SN Goward JG Masek, N Thomas, Z Zhu, JE Vogelmann (2010). An
 * automated approach for reconstructing recent forest disturbance history using
 * dense Landsat time series stacks. Remote Sensing of Environment, 114(1),
 * 183-198
 *
 * This code was ported and adapted from Chengquan Huang's C code changeAnalysis
 * at version 3.5. It implements the "disturbRegrowth" function, but does not
 * include "removeFalseDisturbance".
 *
 * @author Matt Gregory
 *
 * May 22, 2015, Z. Yang
 * Refactor some of the implemented VCT logic.
 *
 * 1. Number of years used in the analysis would not be fixed, it would be
 *    better to keep it as a parameter.
 *
 * 2. Currently input data is passed in as List<List<Double>>, changing it
 *    to double[][] in the order of B3, B4, B5, B7, B7, NDVI, DNBR, COMP, it
 *    will be more efficient to construct this when using pixelFunctor to read
 *    the value.
 *
 * 3. Check logic when percent good is less than 50%!!!
 *
 * May 23, 2015, Z. Yang
 * 
 * 4. removed inner static class VCTSolver. Together with that, changed
 *    static final method to private methods
 *
 * May 24, 2015, Z. Yang
 * 
 * 5. replace mean, standard deviation, and linearFit
 *
 * 6. VCTOutput: it does not seem necessary to return lcType and nYears
 *
 */
public class VCT {

  // <editor-fold defaultstate="collapsed" desc=" CONSTANTS ">
  // Maximum number of years that arrays can store
  private static int MAX_YEARS = 30;

  // Quality assurance (per pixel-year) constants
  private static final int QA_BAD = 0;
  private static final int QA_GOOD = 1;

  // Band constants - note these are unorthodox because we are using a
  // subset of the TM bands for change detection
  private static final int B3 = 0;
  private static final int B4 = 1;
  private static final int B5 = 2;
  private static final int B7 = 3;
  private static final int BT = 4;
  private static final int NDVI = 5;
  private static final int DNBR = 6;
  private static final int COMP = 7;
  private static final int N_BANDS = 8;

  // Mask constants - all "fillable" categories are <= 5
  // Even though not all categories are used, we list them for metadata
  // purposes.  Note that these ARE NOT the same categories as in VCT
  // source code.
  private static final int BACKGROUND = 0;
  private static final int CLOUD = 1;
  private static final int CLOUD_EDGE = 2;
  private static final int SHADOW = 3;
  private static final int SHADOW_EDGE = 4;
  private static final int SNOW = 5;
  private static final int WATER = 6;
  private static final int CLEAR_LAND = 7;
  private static final int CORE_FOREST = 8;
  private static final int CORE_NONFOREST = 9;
  private static final int CONFIDENT_CLEAR = 10;

  private static final int FILL_CLASSES = 5;

  // Disturbance land cover types
  private static final int PERM_NON_FOREST = 1;
  private static final int PERM_FOREST = 2;
  private static final int PART_FOREST = 3;
  private static final int PERM_WATER = 4;
  private static final int INTER_DISTURB_FOREST = 5;
  private static final int JUST_DISTURBED = 6;
  private static final int TRANSITION_PERIOD = 7;

  // Regrowth land cover types
  private static final int REGROWTH_NOT_OCCURRED = 1;
  private static final int REGROWTH_OCCURRED = 2;
  private static final int REGROWTH_TO_FOREST = 3;

  // Constants that signify consecutive and non-consecutive high and low UD
  private static final int CHUD = 1;
  private static final int CLUD = 2;
  private static final int NCHUD = 3;
  private static final int NCLUD = 4;

  private static final double[] FALSE_FIT = new double[]{0.0, 25.0, 0.0, 0.0};

  //A local variable to facilitate regression
  private SimpleRegression sr = new SimpleRegression();
  // </editor-fold>

  // <editor-fold defaultstate="collapsed" desc=" VARIABLES ">

  // Input variables
  private final double maxUd;    // Maximum UD composite value for forest
  private final double minNdvi;  // Minimum NDVI value for forest
  private final double forThrMax; // Maximum threshold for forest
  private double[][] ud;         // Z-scores for all bands and indices, [NYears][B3(0), B4(1), B5(2), B7(3), BT(4), NDVI(5), DNBR(6), COMP(7)]
  private int[] mask;            // Mask (categorical) values for all years
  private int[] yearTable;       // Array to hold all years for this series
  private int numYears;          // Number of years (len of most arrays)

  // Characterization of input variables
  private int[] qFlag;           // QA flag for each year in the series
  private double fiRange;        // Range of forest index (= uRange)

  // Attributes associated with maximum forest length segment - used
  // in thresholding
  private int maxConsFor;        // Maximum consecutive forest length
  private double[] meanForUdBx;  // Mean of index values across maxConsFor
  private double[] sdForUdBx;    // SD of index values across maxConsFor

  // Chararacterization of segments based on thresholded values
  private int[] cstSeg;          // Segment characterization
  private int[] cstSegSmooth;    // Smoothed characterization

  // Disturbance variables
  private int numDist;           // Number of disturbances detected
  private int[] distFlag;        // Type of each disturbance
  private int[] distYear;        // Year of each disturbance
  private int[] distLength;      // Length of each disturbance
  private double[] distR2;       // R2 of each disturbance
  private double[] distMagn;     // Magnitude (in ud composite) of each dist.
  private double[] distMagnB4;   // Magnitude (in ud B4) of each dist.
  private double[] distMagnVi;   // Magnitude (in ud NDVI) of each dist.
  private double[] distMagnBr;   // Magnitude (in ud DNBR) of each dist.

  // Regrowth variables
  private double[] regrR2;       // R2 of each regrowth
  private double[] regrSlope;    // Slope of each regrowth
  private double[] regrRough;    // Roughness of each regrowth
  private int[] regrFlag;        // Type of each regrowth

  // Land cover/change types
  private int lcType;            // Final land cover/change type
  // </editor-fold>

  public VCT() {
    this.maxUd = 4.0;
    this.minNdvi = 0.45;
    this.forThrMax = 3.0;
    allocateArrays();
  }

  public VCT(double maxUd, double minNdvi, double forThrMax, int nYears) {
    this.maxUd = maxUd;
    this.minNdvi = minNdvi;
    this.forThrMax = forThrMax;
    this.MAX_YEARS = nYears;
    allocateArrays();
  }

  /**
   * Allocate space for all arrays so we can reuse these containers for all
   * pixels. This is predicated on setting a maximum number of possible years
   * that we will hold for any pixel. Each pixel's input arrays will actually
   * determine how much of these arrays get used.
   */
  private void allocateArrays() {
    // Integer MAX_YEARS arrays
    this.mask = new int[MAX_YEARS];
    this.yearTable = new int[MAX_YEARS];
    this.qFlag = new int[MAX_YEARS];
    this.cstSeg = new int[MAX_YEARS];
    this.cstSegSmooth = new int[MAX_YEARS];
    this.distFlag = new int[MAX_YEARS];
    this.distYear = new int[MAX_YEARS];
    this.distLength = new int[MAX_YEARS];
    this.regrFlag = new int[MAX_YEARS];

    // Double MAX_YEARS arrays
    this.distR2 = new double[MAX_YEARS];
    this.distMagn = new double[MAX_YEARS];
    this.distMagnB4 = new double[MAX_YEARS];
    this.distMagnVi = new double[MAX_YEARS];
    this.distMagnBr = new double[MAX_YEARS];
    this.regrR2 = new double[MAX_YEARS];
    this.regrSlope = new double[MAX_YEARS];
    this.regrRough = new double[MAX_YEARS];

    // Double N_BANDS arrays
    this.meanForUdBx = new double[N_BANDS];
    this.sdForUdBx = new double[N_BANDS];

    // Double N_BANDS x MAX_YEARS arrays
    this.ud = new double[N_BANDS][MAX_YEARS];
  }

  /**
   * This is the main function for determining change analysis within VCT.
   * First, time series information is read from a number of different forest
   * indexes (known as forest z-scores in the paper, ud scores throughout the
   * code) and a mask image denoting land cover type and/or image artifacts
   * (e.g. cloud, shadow). The algorithm consists of two main functions and a
   * cleanup function:
   *
   * 1) interpolationAndIndices - The ud series for all indices is first
   * filled in for all cloud/shadow pixels from neighboring years.
   *
   * 2) analyzeUDist - The filled series are analyzed to find anomalous
   * deviations from neighboring years and characterized into disturbance and
   * regrowth segments
   *
   * 3) setDisturbanceVariables - Various cleanup and clamping of variable
   * values as well as determining disturbance duration.
   *
   * Note that much of the code relies of comparing series values with known
   * or derived thresholds of "forestness" and anomalies are detected when
   * these threshold bounds are exceeded.
   *
   * @param ud    - List of all UD values across all pertinent bands and indices
   *                (B3, B4, B5, B7, thermal, NDVI, DNBR, R45). There is
   *                one List per index which contains values for all years.
   * @param mask  - List of mask values for this pixel across all years
   * @param years - List of years corresponding to the indices in ud and mask
   * @return - VCTOutput instance (currently returning distFlag and four
   *           disturbance magnitudes
   */
  public VCTOutput getResult(double[][] ud, int[] mask, int[] years) {

    //initialize instance variable for this pixel
    this.ud = ud;
    this.mask = mask;
    this.yearTable = years;
    this.numYears = years.length;

    // Read in the passed values and initialize all variables
    //TODO: (yang) change the parameter here, since each pixel could potentially
    //have different number of years, using a hardwired value is not appropriate.
    //Need to evaluate whether those initialization is necessary
    initializePixel();

    // Interpolate bad values in the time series
    interpolationAndIndices();

    // Find disturbance and recovery segments in this time series
    analyzeUDist();

    // Set disturbance tracking variables
    setDisturbanceVariables();

    // Return this pixel's disturbance metrics
    // TODO: This should all be refactored into an EEArray - for now just
    // sending back as a VCTOutput instance
    return new VCTOutput(this.lcType,
        this.numYears,
        this.yearTable,
        this.distFlag,
        this.distMagn,
        this.distMagnVi,
        this.distMagnBr,
        this.distMagnB4);
  }

  /**
   * Initialize the pixel's values from the passed arguments and calculate the
   * composite UD score as a function of bands 3, 5, and 7
   *
   */
  private void initializePixel() {
    // Initialize all variables
    this.numDist = 0;
    this.lcType = 0;
    this.maxConsFor = 0;
    this.fiRange = 0.0;

    // TODO: Keeping this as a range fill for now to be explicit that we
    // would only want to fill up to this.numYears even if we decide to
    // increase MAX_YEARS.  Overrule this if you want.
    Arrays.fill(this.qFlag, 0, this.numYears, 0);
    Arrays.fill(this.cstSeg, 0, this.numYears, 0);
    Arrays.fill(this.cstSegSmooth, 0, this.numYears, 0);
    Arrays.fill(this.distFlag, 0, this.numYears, 0);
    Arrays.fill(this.distYear, 0, this.numYears, 0);
    Arrays.fill(this.distLength, 0, this.numYears, 0);
    Arrays.fill(this.regrFlag, 0, this.numYears, 0);

    Arrays.fill(this.distR2, 0, this.numYears, 0.0);
    Arrays.fill(this.distMagn, 0, this.numYears, 0.0);
    Arrays.fill(this.distMagnB4, 0, this.numYears, 0.0);
    Arrays.fill(this.distMagnVi, 0, this.numYears, 0.0);
    Arrays.fill(this.distMagnBr, 0, this.numYears, 0.0);
    Arrays.fill(this.regrR2, 0, this.numYears, 0.0);
    Arrays.fill(this.regrSlope, 0, this.numYears, 0.0);
    Arrays.fill(this.regrRough, 0, this.numYears, 0.0);

    Arrays.fill(this.meanForUdBx, 0, N_BANDS, 0.0);
    Arrays.fill(this.sdForUdBx, 0, N_BANDS, 0.0);
  }

  /**
   * Classify the pixel's mask values into QA_GOOD and QA_BAD and fill in
   * QA_BAD values based on nearest neighbors or linear interpolation of
   * QA_GOOD values. Implements section 3.3.1 in Huang et al. (2010) paper
   */
  private void interpolationAndIndices() {
    int badCount = 0;
    
    // Flag each year's pixel as good or bad based on the mask value
    // TODO: Note that class 0 (BACKGROUND) is *not* being flagged as
    // "fillable" in the original source code (ie. SLC-off errors).
    // We need to ask Cheng about this
    //
    // Also, look for spikes and dips that may be unflagged cloud and
    // shadow and set the QA flag accordingly
    for (int i = 0; i < this.numYears; i++) {
      
      // Start by calling the value good
      this.qFlag[i] = QA_GOOD;
      
      // Check for bad mask value
      if (mask[i] != 0 && mask[i] <= FILL_CLASSES) {
        this.qFlag[i] = QA_BAD;
        badCount++;
        continue;
      }

      // Check for cloud/shadow
      if (i == 0) {
        if (isBadEndpoint(i, i + 1)) {
          this.qFlag[i] = QA_BAD;
          badCount++;
        }
      } else if (i == this.numYears - 1) {
        if (isBadEndpoint(i, i - 1)) {
          this.qFlag[i] = QA_BAD;
          badCount++;
        }
      } else if (isRelativeCloud(i) || isRelativeShadow(i)) {
        this.qFlag[i] = QA_BAD;
        badCount++;
      }
    }

    // Interpolate for bad observations indicated by qFlag
    // TODO: (yang) is there any special treatment when percentGood is less
    // than 50%?
    int i = 0;
    if (badCount <= (this.numYears / 2.0)) {
      while (i < this.numYears) {
        // Skip good observations
        if (this.qFlag[i] == QA_GOOD) {
          i += 1;
        }
        // Fill or interpolate bad observations
        else {
          // Search forward/backward to find next valid observations in
          // time series
          int prev = i - 1;
          int next = i + 1;
          while (prev >= 0 && this.qFlag[prev] == QA_BAD) {
            prev -= 1;
          }
          while (next < this.numYears && this.qFlag[next] == QA_BAD) {
            next += 1;
          }

          //YANG: will this ever happen? pctGoodObs is > 50% there has to be some good values
          // No valid QA_GOOD pixels in the time series
          if (prev < 0 && next >= this.numYears) {
            break;
          }
          // No acceptable previous QA_GOOD - use next index to fill
          // all years from 0 to next
          else if (prev < 0) {
            for (int k = 0; k < N_BANDS; k++) {
              Arrays.fill(this.ud[k], 0, next, this.ud[k][next]);
            }
          }
          // No acceptable next QA_GOOD - use prev index to fill
          // all years from prev + 1 to num_years
          else if (next >= this.numYears) {
            for (int k = 0; k < N_BANDS; k++) {
              Arrays.fill(this.ud[k], prev + 1, this.numYears, this.ud[k][prev]);
            }
          }
          // Found years acceptable for interpolation - fill between
          // prev and next
          else {
            for (int k = 0; k < N_BANDS; k++) {
              interpolateValues(this.ud[k], prev, next);
            }
          }
          i = next + 1;
        }
      }
    }

    // Get range values for the composite UD and NDVI
    this.fiRange = Doubles.max(this.ud[COMP]) - Doubles.min(this.ud[COMP]);
  }

  /**
   * Main function for determining disturbance and regrowth for this pixel's
   * trajectory. Main steps include:
   *
   * 1) Determine composite and NDVI thresholds
   * 2) Find the longest consecutive forest streak
   * 3) Find consecutive segments as either high or low UD values based
   *    on thresholds
   * 4) Smooth these segments
   * 5) Characterize segments as disturbance/regrowth classes
   * 6) Determine land cover/change type based on pattern of disturbances
   */
  private void analyzeUDist() {
    // Assign some default information to this pixel
    this.distYear[0] = this.distYear[1] = this.numYears;
    this.lcType = PART_FOREST;

    // Get the top ndvi values and the bottom two ud composite values.  If
    // the spread between the low UD values is less than FOR_THR_MAX, set
    // them equal. From this point on, min2Ud is used instead of minUd to
    // reduce impact of an anomalously low ud value
    //
    // TODO: As written, this is currently a bug in the original VCT
    // software - if the last number in the UD composite sequence is the
    // lowest number, min2Ud will stay as 9999.0 instead of the second
    // lowest number.  Keeping in here to for comparison purposes, but
    // should be changed.  The below (commented) section implements the
    // desired functionality
    //
    // double maxVi = Doubles.max(this.ud[NDVI]);
    // double minUd = Doubles.min(this.ud[COMP]);
    // double min2Ud = Doubles.max(this.ud[COMP]);
    // for (i = 0; i < this.numYears; i++) {
    //   if (this.ud[COMP] > minUd && this.ud[COMP] < min2Ud) {
    //     min2Ud = this.ud[COMP];
    //   }
    // }

    // double maxVi = Double.NEGATIVE_INFINITY; //Doubles.max(this.ud[NDVI]);
    // double minUd = Double.NEGATIVE_INFINITY; //Doubles.min(this.ud[COMP]);
    // double min2Ud = Double.POSITIVE_INFINITY; //Doubles.max(this.ud[COMP]);
    double maxVi = Doubles.max(this.ud[NDVI]);
    double minUd = 9999.0;
    double min2Ud = 9999.0;

    // Track the number of water and shadow pixels and identify if they
    // come in the first and last thirds of the time series
    int isWaterFront = 0;
    int isWaterTail = 0;
    int firstThird = this.numYears / 3;
    int lastThird = this.numYears - firstThird;
    int numWater = 0;
    int numShadow = 0;
    for (int i = 0; i < this.numYears; i++) {
      double tmp = this.ud[COMP][i];
      minUd = tmp < minUd ? tmp : minUd;
      min2Ud = (tmp > minUd && tmp < min2Ud) ? tmp : min2Ud;
      maxVi = this.ud[NDVI][i] > maxVi ? this.ud[NDVI][i] : maxVi;
      if (this.mask[i] == WATER || this.mask[i] == SHADOW) {
        numWater++;
        if (this.mask[i] == SHADOW) {
          numShadow++;
        }
        isWaterFront += i < firstThird ? 1 : 0;
        isWaterTail += i >= lastThird ? 1 : 0;
      }
    }
    double percentWater = 1.0 * numWater / this.numYears;
    double percentShadow = 1.0 * numShadow / this.numYears;

    if (min2Ud - minUd < this.forThrMax) {
      min2Ud = minUd;
    }

    // Get the maximum streak of years with forest.  The length of the
    // streak gets set in the function (this.maxConsFor) along with
    // the mean and standard deviations of the longest streak in each
    // index.  The value returned is the starting year of the streak.
    int maxForStart = getMaxForLength(min2Ud, min2Ud + this.forThrMax);

    // Set a threshold for determining change in the UD composite signal
    double changeHike = this.forThrMax;
    double adjCoeff = Math.min(this.meanForUdBx[COMP] / 5.0, 1.67);
    if (adjCoeff > 1.0) {
      changeHike *= adjCoeff;
    }
    double changeThrUd = this.meanForUdBx[COMP] + changeHike;

    // Identify consecutive high and low ud observations - this loop
    // characterizes each year in the trajectory into one of four
    // categories: consecutive low UD (CLUD), non-consecutive low UD
    // (NCLUD), consecutive high UD (CHUD), and non-consecutive high
    // UD (NCHUD). If a trajectory is above or below the dividing
    // threshold (change_thr_ud) for two or more years, it is called
    // CLUD or CHUD. If it only stays for one year, it is called NCLUD
    // or NCHUD.
    int i = 0;
    int sharpTurns = 0;
    while (i < this.numYears) {

      // Consecutive low ud - CLUD
      int j = i;
      int numCstObs = 0;
      while (j < this.numYears && this.ud[COMP][j] <= changeThrUd) {
        j += 1;
        numCstObs += 1;
      }
      Arrays.fill(this.cstSeg, i, j, numCstObs < 2 ? NCLUD : CLUD);
      if (numCstObs > 0) {
        sharpTurns += 1;
      }

      // Consecutive high ud - CHUD
      i = j;
      numCstObs = 0;
      while (j < this.numYears && this.ud[COMP][j] > changeThrUd) {
        j += 1;
        numCstObs += 1;
      }
      Arrays.fill(this.cstSeg, i, j, numCstObs < 2 ? NCHUD : CHUD);
      if (numCstObs > 0) {
        sharpTurns += 1;
      }
      i = j;
    }

    // Remove NCLUD and NCHUD labels based on adjacent labels - this
    // effectively smooths the segment labels
    System.arraycopy(this.cstSeg, 0, this.cstSegSmooth, 0, this.cstSeg.length);
    smoothSegment(this.cstSegSmooth, this.numYears, NCLUD);
    smoothSegment(this.cstSegSmooth, this.numYears, NCHUD);

    // Create an empty list of TSSegment instances to store segment
    // information.  This block uses the smoothed segment information
    // to create the segments
    i = 0;
    List<TSSegment> tsSeg = new ArrayList<>();
    while (i < this.numYears) {

      // Initialize this segment
      int j = i;

      // As long as the label for this year matches the following year's
      // label, keep growing the segment
      while (j < this.numYears - 1 
              && this.cstSegSmooth[j] == this.cstSegSmooth[j + 1]) {
        j++;
      }

      // Store this segment
      tsSeg.add(new TSSegment(this.cstSegSmooth[i], i, j - i + 1));

      // Increment for the next segment
      i = j + 1;
    }

    // Now detect changes
    // Note that ALL pixels go through this logic, although this
    // information is only used at the end where this.lcType == PART_FOREST
    for (i = 0; i < tsSeg.size(); i++) {
      TSSegment thisSeg = tsSeg.get(i);
      switch (thisSeg.segType) {

        // Consecutive high UD - signifies disturbance event
        case CHUD:

          // Characterize the disturbance and following recovery
          setDisturbance(thisSeg.startYear, thisSeg.endYear);
          
          // More convenient to set regrowth type here
          int lastDist = this.numDist - 1;
          this.regrFlag[lastDist] = REGROWTH_NOT_OCCURRED;

          // Not the last segment in this time series and is followed by
          // a forested segment
          if (i < tsSeg.size() - 1) {
            if (tsSeg.get(i + 1).segType == CLUD
                || tsSeg.get(i + 1).segType == NCLUD) {
              this.regrFlag[lastDist] = REGROWTH_TO_FOREST;
            } else {
              //TODO: (yang) when this happens, what is the right behavior?
              // Handle exception
              // String msg = "Warning: CHUD not followed by CLUD or NCLUD";
              // throw new Exception(msg);
            }
          }
          // Last segment in this time series, but high R2 and
          // negative slope indicate that regrowth is occurring
          else if (this.regrR2[lastDist] > 0.7
              && this.regrSlope[lastDist] < -0.2) {
            this.regrFlag[lastDist] = REGROWTH_OCCURRED;
          }
          break;

        // Consecutive low UD - forested
        case CLUD:
          // Mark the pixel's distFlag for the years in the segment
          setPostDisturbForest(thisSeg.startYear, thisSeg.endYear);

          // Search for low-level disturbance
          searchMinorDisturbances(thisSeg.startYear, thisSeg.endYear);
          break;

        // Non-consecutive high UD
        case NCHUD:
          // End year of this sequence is last year in time series, mark
          // as disturbance
          if (thisSeg.endYear == numYears - 1) {
            setDisturbance(thisSeg.startYear, thisSeg.endYear);
          }
          // Mark the pixel's distFlag for the years in the segment
          else {
            setPostDisturbForest(thisSeg.startYear, thisSeg.endYear);
          }
          break;

        // Non-consecutive low UD
        case NCLUD:
          // Mark the pixel's distFlag for the years in the segment
          setPostDisturbForest(thisSeg.startYear, thisSeg.endYear);
          break;

        default:
          this.lcType = PERM_NON_FOREST;
          break;
      }
    }

    // Final classification - set lcType based on calculated data
    // TODO: If this.lcType is driving the assignment and all other outputs
    // are predicated upon that designation, it seems like we can short
    // circuit the logic right after we calculate these variables

    // Water dominated pixel and present throughout the time series
    if (isWaterFront > 1 && isWaterTail > 1
        && percentWater > 0.4 && percentWater > percentShadow * 2.0) {
      this.lcType = PERM_WATER;
      return;
    }

    // Noisy time series (many sharp turns) - signifies ag or other
    // nonforest
    if (sharpTurns > (int) this.numYears * 0.33) {
      this.lcType = PERM_NON_FOREST;
      return;
    }

    // Minimum UD exceeds forest UD or maximum VI under forest NDVI
    // nonforest
    if (min2Ud > this.maxUd || maxVi < this.minNdvi) {
      this.lcType = PERM_NON_FOREST;
      return;
    }

    // Short duration segments where the maxForStart was not at the
    // beginning or end of the time series - nonforest
    if (this.maxConsFor < 3 && maxForStart > 1
        && maxForStart < (this.numYears - 1 - this.maxConsFor)) {
      this.lcType = PERM_NON_FOREST;
      return;
    }

    // Seems to signify short duration segments as well
    if (this.maxConsFor < 1) {
      if (percentWater > 0.15) {
        this.lcType = PERM_WATER;
      } else {
        this.lcType = PERM_NON_FOREST;
      }
      return;
    }

    // Only one segment identfied - persistent forest
    if (this.numDist == 0) {
      this.lcType = PERM_FOREST;
    }
  }

  /**
   * Characterize longest disturbance segment and clamp disturbance variable
   * values
   */
  private void setDisturbanceVariables() {

    // TODO: A lot of the code in here looks to be just clamping for
    // data type (unsigned byte).  Probably not necessary in GEE context

    for (int i = 0; i < this.numYears; i++) {
      if (this.distMagn[i] > 25.0) {
        this.distMagn[i] = 25.0;
      }
    }

    if (this.lcType == PART_FOREST) {
      // Find the disturbance with longest duration
      int dLength = 0;
      for (int i = 0; i < this.numDist; i++) {
        if (this.distLength[i] > dLength) {
          dLength = this.distLength[i];
        }
      }

      // Clamp pixel values
      int f = this.distYear[0];
      int l = this.distYear[this.numDist - 1];

      this.distMagn[f] = Math.max(Math.min(this.distMagn[f], 25.0), 0.0);
      this.distMagn[l] = Math.max(Math.min(this.distMagn[l], 25.0), 0.0);

      this.distMagnVi[f] = Math.max(Math.min(this.distMagnVi[f], 1.0), -1.0);
      this.distMagnVi[l] = Math.max(Math.min(this.distMagnVi[l], 1.0), -1.0);

      this.distMagnBr[f] = Math.max(Math.min(this.distMagnBr[f], 1.0), -1.0);
      this.distMagnBr[l] = Math.max(Math.min(this.distMagnBr[l], 1.0), -1.0);
    }
  }

  /**
   * Determine if a year's pixel value is relatively cloudy based on its
   * neighbors values
   *
   * @param i - index to check
   * @return - cloudiness flag
   */
  private boolean isRelativeCloud(int i) {
    return (this.ud[COMP][i] > this.ud[COMP][i - 1] + 3.5
        || this.ud[COMP][i] > this.ud[COMP][i + 1] + 3.5)
        && this.ud[COMP][i] > this.ud[COMP][i - 1] + 2.5
        && this.ud[COMP][i] > this.ud[COMP][i + 1] + 2.5
        && this.ud[BT][i] < this.ud[BT][i - 1] - 1.0
        && this.ud[BT][i] < this.ud[BT][i + 1] - 1.0
        && this.ud[BT][i] < 0.5;
  }

  /**
   * Determine if a year's pixel value is relatively shadow based on its
   * neighbors values
   *
   * @param i - index to check
   * @return - shadow flag
   */
  private boolean isRelativeShadow(int i) {
    return (this.ud[COMP][i] < this.ud[COMP][i - 1] - 3.5
        || this.ud[COMP][i] < this.ud[COMP][i + 1] - 3.5)
        && this.ud[COMP][i] < this.ud[COMP][i - 1] - 2.5
        && this.ud[COMP][i] < this.ud[COMP][i + 1] - 2.5
        && this.ud[B4][i] < 1.0
        && this.ud[B5][i] < 1.0
        && this.ud[B7][i] < 1.0;
  }

  /**
   * Determine if an endpoint is cloud/shadow based on neighbor's value
   *
   * @param i - index to check
   * @param j - neighbor's index to check against
   * @return - bad flag
   */
  private boolean isBadEndpoint(int i, int j) {
    // Likely cloud
    boolean cond1 = this.ud[COMP][i] > this.ud[COMP][j] + 3.5
        && this.ud[BT][i] < this.ud[BT][j] - 1.5
        && this.ud[BT][i] < 0.5;

    // Likely shadow
    boolean cond2 = this.ud[COMP][i] < this.ud[COMP][j] - 3.5
        && this.ud[B5][i] < 1.0
        && this.ud[B7][i] < 1.0
        && this.ud[B4][i] < 1.0;

    return cond1 || cond2;
  }

  /**
   * Get the maximum length of a pixel's forestness. This is based on a
   * consistent ud composite value within the given range
   *
   * @param minForUd - Minimum ud composite value for forest
   * @param maxForUd - Maximum ud composite value for forest
   * @return - the number of years this pixel is in a forest condition
   */
  private int getMaxForLength(double minForUd, double maxForUd) {
    // Find the longest streak (maxLength) over the time series when ud is
    // persistently within the ud range.  For this streak, also track the
    // number WATER observations
    int i = 0;
    int iStart = 0;
    int iEnd = 0;
    int waterCount = 0;
    int maxLength = 0;

    while (i < this.numYears) {
      // Reset counts
      int yearCount = 0;
      int tmpYear = i;
      int tmpWaterCount = 0;

      // Start tracking the streak
      while (tmpYear < this.numYears
          && this.ud[COMP][tmpYear] <= maxForUd
          && this.ud[COMP][tmpYear] >= minForUd) {
        if (this.mask[tmpYear] == WATER) {
          tmpWaterCount++;
        }
        tmpYear++;
        yearCount++;
      }

      // Check to see if the streak has been exceeded
      if (maxLength < yearCount) {
        maxLength = yearCount;
        waterCount = tmpWaterCount;
        iStart = i;
        iEnd = tmpYear;
      }

      // Move the year pointer ahead
      i += (yearCount > 0) ? yearCount : 1;
    }

    // Initialize the container to hold means and standard deviations of
    // the maximum forest streak
    Arrays.fill(this.meanForUdBx, 25.4);
    Arrays.fill(this.sdForUdBx, 25.4);

    // If there is a forest streak, calculate statistics on this streak
    // TODO: Mean and StandardDeviation are somehow giving very slightly
    // different values than getSliceMean / getSliceStd to the point where
    // pixels change lcType in further logic based on threshold values.
    // What to do on this?
    if (maxLength > 0) {
      // Mean m = new Mean();
      for (int j = 0; j < N_BANDS; j++) {
        // this.meanForUdBx[j] = m.evaluate(this.ud[j], iStart, iEnd - iStart);
        this.meanForUdBx[j] = getSliceMean(this.ud[j], iStart, iEnd);
      }

      if (maxLength > 1) {
        // StandardDeviation sd = new StandardDeviation(false);
        for (int j = 0; j < N_BANDS; j++) {
          // this.sdForUdBx[j] = sd.evaluate(this.ud[j], iStart, iEnd - iStart);
          this.sdForUdBx[j] = getSliceStd(this.ud[j], iStart, iEnd);
        }
      } else {
        // Calculate standard deviations using a high SD value if
        // maxLength is too short
        for (int j = 0; j < N_BANDS; j++) {
          this.sdForUdBx[j] = this.meanForUdBx[j] / 3.0;
        }
      }
    }
    this.maxConsFor = maxLength - waterCount;
    return (iStart);
  }

  /**
   * Smooth out variability in initial determination of this pixel's high/low
   * ud segments. This will remove NCLUD and NCHUD labels based on neighboring
   * values
   *
   * @param smooth   - Array in which to store smoothed segments calls
   * @param numYears - Number of years over which to run the smoothing
   * @param current  - Current label to override
   */
  private void smoothSegment(int[] smooth, int numYears, int current) {

    // Based on the vertex label (current) we're searching to change,
    // specify the new vertex label to assign as well as the border
    // condition label to find
    int newType;
    int borderType;
    if (current == NCLUD) {
      newType = CHUD;
      borderType = NCHUD;
    } else {
      newType = CLUD;
      borderType = NCLUD;
    }

    // Search forward through the time series
    int i = 1;
    while (i < numYears - 1) {
      // Skip if this isn't the target label
      if (smooth[i] != current) {
        i++;
        continue;
      }

      // If current label is between two new labels, change to newType
      if (smooth[i - 1] == newType && smooth[i + 1] == newType) {
        smooth[i] = newType;
        i++;
        continue;
      }

      // If current label is between a new label and a border label, change
      // to new and search forward for more border labels, changing them to
      // newType if found
      if (smooth[i - 1] == newType && smooth[i + 1] == borderType) {
        smooth[i] = newType;
        int j = i + 1;
        while (j < numYears && smooth[j] == borderType) {
          smooth[j] = newType;
          j += 1;
        }
        i = j;
      }
      // Current label was found but adjacent labels didn't qualify it for
      // a change
      else {
        i++;
      }
    }

    // Now, search backward through the time series, same logic as
    // above but in reverse.  There is no check for the "sandwich"
    // logic above; they have all been changed by this point
    i = numYears - 2;
    while (i > 0) {
      if (smooth[i] != current) {
        i--;
        continue;
      }
      if (smooth[i + 1] == newType && smooth[i - 1] == borderType) {
        smooth[i] = newType;
        int j = i - 1;
        while (j >= 0 && smooth[j] == borderType) {
          smooth[j] = newType;
          j--;
        }
        i = j;
      } else {
        i--;
      }
    }
  }

  /**
   * Set disturbance information for a given time range
   *
   * @param startYear - disturbance onset year
   * @param endYear   - disturbance finish year
   */
  private void setDisturbance(int startYear, int endYear) {
    //TODO: evaluate encapsulate this method within TSSegment
    // Set the disturbance year, disturbance flag and landcover type
    int distIndex = this.numDist;
    this.distYear[distIndex] = startYear;
    this.distFlag[startYear] = JUST_DISTURBED;
    this.lcType = PART_FOREST;

    // Find the local UD peak within this disturbance segment and label
    // all intermediary years as transition, year is stored in true_peak,
    // ud is stored in maxUd
    int truePeak = startYear;
    double localMaxUd = this.ud[COMP][startYear];
    if (endYear >= this.numYears) {
      endYear = this.numYears - 1;
    }
    for (int i = startYear + 1; i <= endYear; i++) {
      this.distFlag[i] = TRANSITION_PERIOD;
      if (this.ud[COMP][i] > localMaxUd && i < endYear
          && (this.ud[COMP][i] - this.ud[COMP][i - 1] < 2.0
          || this.ud[COMP][i] - this.ud[COMP][i + 1] < 2.0)) {
        localMaxUd = this.ud[COMP][i];
        truePeak = i;
      }
    }

    // Calculate the change magnitudes
    this.distMagn[startYear] = this.ud[COMP][truePeak] - this.meanForUdBx[COMP];
    this.distMagnB4[startYear] = this.ud[B4][truePeak] - this.meanForUdBx[B4];
    this.distMagnVi[startYear] = this.ud[NDVI][truePeak] - this.meanForUdBx[NDVI];
    this.distMagnBr[startYear] = this.ud[DNBR][truePeak] - this.meanForUdBx[DNBR];
  
    // Fit a recovery (regrowth) regression line (B5 vs. years) from the
    // peak year to the end year. High goodness of fit value indicates
    // a very likely change
    if (endYear - startYear < 4 || this.fiRange < 0.1) {
      this.regrR2[distIndex] = FALSE_FIT[2];
      this.regrSlope[distIndex] = FALSE_FIT[0];
      this.distR2[distIndex] = FALSE_FIT[2];
    }
    else {
      // Yang replace regression code
      // Note: if common math3 can be used, this code can be simplified.
      updateSimpleRegression(truePeak, endYear);
      this.regrR2[distIndex] = sr.getRSquare();
      this.regrSlope[distIndex] = sr.getSlope();

      // Fit a regression for the entire disturbance period
      updateSimpleRegression(startYear, endYear);
      this.distR2[distIndex] = sr.getRSquare();
    }

    //TODO: (yang) check implementation
    this.regrRough[distIndex] = fiRoughness(this.ud[COMP], startYear, endYear);
    this.distLength[distIndex] = (int) (endYear - startYear + 1);

    // Increment the disturbance index
    this.numDist += 1;
  }

  /**
   * Extract segment data in ud B5 over given time span for linear fit
   * Try to use existing array structure, but it is tightly coupled with
   * this class structure.
   *
   * This could turn into a inner class function.
   *
   * @param start start of segment inclusive
   * @param end end of segment inclusive
   * @return
   */
  private void updateSimpleRegression(int start, int end) {
    //FIXME: (yang) when will this happen?
    //should make sure endYear always have valid data where it is assigned
    if (end == this.numYears) {
      end--;
    }

    sr.clear();
    double startX = this.yearTable[start];
    for (int i = start; i <= end; i++) {
      sr.addData(this.yearTable[i]-startX, this.ud[B5][i]);
    }
  }

  /**
   * Set post-disturbance information for a given time range
   *
   * @param startYear - post-disturbance onset year
   * @param endYear   - post-disturbance finish year
   */
  private void setPostDisturbForest(int startYear, int endYear) {
    // Mark all years in this segment as INTER_DISTURB_FOREST
    for (int i = startYear; i <= endYear; i++) {
      if (i >= this.numYears) {
        break;
      }
      this.distFlag[i] = INTER_DISTURB_FOREST;
    }
  }

  /**
   * Search for minor disturbances within a given time range
   *
   * @param startYear - year to begin checking for disturbances
   * @param endYear   - year to finish checking for disturbances
   */
  private void searchMinorDisturbances(int startYear, int endYear) {
    // Skip short duration disturbances
    if (endYear < startYear + 3) {
      return;
    }

    int i = startYear;
    while (i <= endYear) {
      // Continue over bad pixels or no detected disturbances
      if (this.qFlag[i] == QA_BAD || isMinorFirst(i) == false) {
        i++;
        continue;
      }

      // Increment j as long as minor disturbance is detected
      // (is_minor_rest == 1).  If the year range is at least two years,
      // characterize it as a disturbance
      int j = i + 1;
      while (j <= endYear && this.qFlag[j] == QA_GOOD
          && isMinorRest(j) == true) {
        j++;
      }
      if (j - i > 1) {
        setDisturbance(i, j);
      }
      i = j;
    }
  }

  /**
   * Determine if a current year's pixel should be considered a minor
   * disturbance immediately after a disturbance event
   *
   * @param curr - current index to check
   * @return - minor disturbance status
   */
  private boolean isMinorFirst(int curr) {
    // TODO: These thresholds are getting recalculated each time even
    // though they are not dependent on the value of curr.  Needs to be
    // calculated once per pixel and passed

    // Define change thresholds for UD, UD_B5, NDVI and DNBR based on
    // mean, sd values
    double chgThrUd = this.meanForUdBx[COMP] + 1.5 + this.sdForUdBx[COMP];
    double chgThrDnbr = this.meanForUdBx[DNBR] - 0.15 - this.sdForUdBx[DNBR];
    double chgThrNdvi = this.meanForUdBx[NDVI] - 0.15 - this.sdForUdBx[NDVI];
    double chgThrB5 = this.meanForUdBx[B5] + 1.0 + this.sdForUdBx[B5];

    // First year of a segment
    if (curr == 0) {
      return (this.ud[COMP][curr] > chgThrUd + 1.0
          || this.ud[B5][curr] > chgThrB5 + 1.0)
          && this.ud[NDVI][curr] < this.meanForUdBx[NDVI]
          && this.ud[DNBR][curr] < this.meanForUdBx[DNBR]
          || (this.ud[NDVI][curr] < chgThrNdvi - 0.1
          || this.ud[DNBR][curr] < chgThrDnbr - 0.1);
    }
    // Other years in the segment
    else {
      return (((this.ud[COMP][curr] > chgThrUd
          || this.ud[B5][curr] > chgThrB5)
          && this.ud[NDVI][curr] < this.meanForUdBx[NDVI]
          && this.ud[DNBR][curr] < this.meanForUdBx[DNBR])
          || (this.ud[DNBR][curr] < chgThrDnbr
          || this.ud[NDVI][curr] < chgThrNdvi))
          && (this.ud[COMP][curr] > this.ud[COMP][curr - 1] + 2.0
          || this.ud[B5][curr] > this.ud[B5][curr - 1] + 2.0
          || this.ud[DNBR][curr] < this.ud[DNBR][curr - 1] - 0.2);
    }
  }

  /**
   * Determine if a current year's pixel should be considered a minor
   * disturbance following a disturbance event (but not immediately -
   * different logic for first year and other years after a disturbance event)
   *
   * @param curr - current index to check
   * @return - minor disturbance status
   */
  private boolean isMinorRest(int curr) {
    // Define change thresholds for UD, UD_B5, NDVI and DNBR based on
    // mean, sd values
    double chgThrUd, chgThrDnbr, chgThrNdvi, chgThrB5;
    chgThrUd = this.meanForUdBx[COMP] + 1.0 + this.sdForUdBx[COMP] / 2.0;
    chgThrDnbr = this.meanForUdBx[DNBR] - 0.1 - this.sdForUdBx[DNBR];
    chgThrNdvi = this.meanForUdBx[NDVI] - 0.1 - this.sdForUdBx[NDVI];
    chgThrB5 = this.meanForUdBx[B5] + 1.0 + this.sdForUdBx[B5] / 2.0;

    return ((this.ud[COMP][curr] > chgThrUd
        || this.ud[B5][curr] > chgThrB5)
        && (this.ud[NDVI][curr] < this.meanForUdBx[NDVI]
        || this.ud[DNBR][curr] < this.meanForUdBx[DNBR]))
        || (this.ud[DNBR][curr] < chgThrDnbr
        || this.ud[NDVI][curr] < chgThrNdvi);
  }

  /**
   * Class for returning information to the caller. Note that this has
   * specialized logic for choosing branching based on lcType - probably not
   * ideal.
   */
  public class VCTOutput {

    public final int[] years;
    public final int[] distFlag;
    public final double[] distMagn;
    public final double[] distMagnVi;
    public final double[] distMagnBr;
    public final double[] distMagnB4;

    public VCTOutput(int lcType,
                     int nYears,
                     int[] years,
                     int[] distFlag,
                     double[] distMagn,
                     double[] distMagnVi,
                     double[] distMagnBr,
                     double[] distMagnB4) {
      this.years = Arrays.copyOfRange(years, 0, nYears);
      if (lcType != PART_FOREST) {
        this.distFlag = new int[nYears];
        Arrays.fill(this.distFlag, lcType);
        this.distMagn = new double[nYears];
        Arrays.fill(this.distMagn, -1.0);
        this.distMagnVi = new double[nYears];
        Arrays.fill(this.distMagnVi, -1.0);
        this.distMagnBr = new double[nYears];
        Arrays.fill(this.distMagnBr, -1.0);
        this.distMagnB4 = new double[nYears];
        Arrays.fill(this.distMagnB4, -1.0);
      } else {
        this.distFlag = Arrays.copyOfRange(distFlag, 0, nYears);
        this.distMagn = Arrays.copyOfRange(distMagn, 0, nYears);
        this.distMagnVi = Arrays.copyOfRange(distMagnVi, 0, nYears);
        this.distMagnBr = Arrays.copyOfRange(distMagnBr, 0, nYears);
        this.distMagnB4 = Arrays.copyOfRange(distMagnB4, 0, nYears);
      }
    }
  }

  /**
   * Class for retaining information about segments across a time series
   */
  public static class TSSegment {

    public int segType;         // type for this segment
    public int segLength;       // length for this segment
    public int startYear;       // start year for this segment
    public int endYear;         // end year for this segment

    public TSSegment(int segType, int startYear, int segLength) {
      this.segType = segType;
      this.startYear = startYear;
      this.endYear = startYear + segLength - 1;
      this.segLength = segLength;
    }
  }

  /**
   * Given a time series and two endpoints, linearly interpolate all values
   *
   * @param ts    - Array of values to interpolate
   * @param left  - Left endpoint to use
   * @param right - Right endpoint to use
   */
  private void interpolateValues(double[] ts, int left, int right) {
    double denom = (double) (right - left);
    double slope = (ts[right] - ts[left]) / denom;
    for (int j = left + 1; j < right; j++) {
      ts[j] = ts[left] + slope * (j - left);
    }
  }

  /**
   * Calculate a measure of a time series' inter-annual variability
   *
   * @param ts    - Array of time series values
   * @param left  - Left endpoint to use
   * @param right - Right endpoint to use
   * @return - Roughness value
   */
  private double fiRoughness(double[] ts, int left, int right) {
    int numVals = right - left + 1;

    if (numVals <= 3) {
      return -1.0;
    }

    // Find all differences between consecutive data pairs
    double[] tmpData = new double[numVals - 1];
    for (int i = left; i < right - 1; i++) {
      tmpData[i - left] = ts[i + 1] - ts[i];
    }

    // Sort these differences and find the index 1/10 in from the left
    // constrained betweeen indexes 1 and 3 inclusive; 
    // return the absolute value of this pair's difference
    Arrays.sort(tmpData);
    int tmpIdx = (int) (numVals * 0.1);
    if (tmpIdx < 1) {
      tmpIdx = 1;
    }
    if (tmpIdx > 3) {
      tmpIdx = 3;
    }
    return Math.abs(tmpData[tmpIdx]);
  }

  // Almost certainly, these are pre-existing functions somewhere???
  // No error checking done here ...
  
  /**
   * Get a mean value over a slice of an array
   *
   * @param arr - Array to calculate mean
   * @param start - start index of array
   * @param end - end index of array (not included!)
   * @return - mean value of slice
   */
  private double getSliceMean(double[] arr, int start, int end) {
    double sum = 0.0;
    for (int i = start; i < end; i++) {
      sum += arr[i];
    }
    return ((double) sum / (end - start));
  }

  /**
   * Get a standard deviation value over a slice of an array
   *
   * @param arr - Array to calculate standard deviation
   * @param start - start index of array
   * @param end - end index of array (not included!)
   * @return - standard deviation value of slice
   */
  private final double getSliceStd(double[] arr, int start, int end) {
    double var = 0.0;
    double mean = getSliceMean(arr, start, end);
    for (int i = start; i < end; i++) {
      var += (arr[i] - mean) * (arr[i] - mean);
    }
    return Math.sqrt((double) var / (end - start));
  }
}
