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
 * <p/>
 * This code was ported and adapted from Chengquan Huang's C code changeAnalysis
 * at version 3.5. It implements the "disturbRegrowth" function, but does not
 * include "removeFalseDisturbance".
 *
 * @author Matt Gregory
 *
 *
 * May 22, 2015, Z. Yang
 * Refactory some of the implemented VCT logic.
 *
 * 1. Number of years used in the analysis would not be fixed, it would be better to keep it as a parameter.
 *
 * 2. Currently input data is passed in as List<List<Double>>, changing it to double[][]
 *    in the order of B3, B4, B5, B7, B7, NDVI, DNBR, COMP, it will be more efficient to construct this
 *    when using pixelFunctor to read the value.
 *
 * 3. Check logic when percent good is less than 50%!!!
 *
 *  May 23, 2015, Z. Yang
 * 4. removed inner static class VCTSolver. Together with that, changed static final method to private methods
 *
 *  May 24, 2015, Z. Yang
 * 5. replace mean, standard deviation, and linearFit
 *
 * 6. VCTOutput: it does not seem necessary to return lcType and nYears
 *
 */
public class VCT {

//  public final static class VCTSolver {

  // <editor-fold defaultstate="collapsed" desc=" CONSTANTS ">
  // Maximum number of years that arrays can store
  // TODO: Can we do something like:
  // private static final int MAX_YEARS = DateTime.now().getYear() - 1984;
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

  private static final List<Integer> UD_INDEXES = Arrays.asList(B3, B5, B7);

  // Mask constants - all "fillable" categories are <= 5
  // Even though not all categories are used, we list them for metadata
  // purposes.  Note that these ARE NOT the same categories as in VCT
  // source code.

  //TODO: (yang) consider to change these to enum.
//  public enum Mask{
//    BACKGROUND(0),
//    CLOUD(1),
//    CLOUD_EDGE(2),
//    SHADOW(3),
//    SHADOW_EDGE(4),
//    SNOW(5),
//    WATER(6),
//    CLEAR_LAND(7),
//    CORE_FOREST(8),
//    CORE_NONFOREST(9),
//    CONFIDENT_CLEAR(10),
//    FILL_CLASSES(5);
//
//    private final int value;
//
//    Mask(int value) {
//      this.value = value;
//    }
//  }

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

  // Forest threshold maximum
  //TODO: (yang) consider changing this to a parameter as it would be different for different system.
  private static final double FOR_THR_MAX = 3.0;

  // Constants that signify consecutive and non-consecutive high and low UD
  private static final int CHUD = 1;
  private static final int CLUD = 2;
  private static final int NCHUD = 3;
  private static final int NCLUD = 4;

  private static final double[] FALSE_FIT = new double[]{0.0, 25.0, 0.0, 0.0};
  // </editor-fold>

  // <editor-fold defaultstate="collapsed" desc=" VARIABLES ">

  // Input variables
  private final double maxUd;    // Maximum UD composite value for forest
  private final double minNdvi;  // Minimum NDVI value for forest
  private double[][] ud;         // Z-scores for all bands and indices, [NYears][B3(0), B4(1), B5(2), B7(3), BT(4), NDVI(5), DNBR(6), COMP(7)]
  private int[] mask;            // Mask (categorical) values for all years
  private int[] yearTable;       // Array to hold all years for this series
  private int numYears;          // Number of years (len of most arrays)

  // Characterization of input variables
  private int[] qFlag;           // QA flag for each year in the series

  //FIXME: (yang) only referenced in interpolationAndIndices(), change to local
//    private int pctGoodObs;        // Percent of good QA flags

  //FIXME: (yang) used in interpolationAndIndices() and setDisturbanceVariables(), does not seem needed this variable.
  //Evaulate more
//  private double uRange;         // Range of ud composite values

  //FIXME: (yang) variable initialized, but is never used, consider remove it!!
//    private double vRange;         // Range of ud NDVI values

  //FIXME: (yang) only used in interpolationAndIndices(), does not seem needed,
//    private double fiRough;        // Roughness of forest index from composite

  private double fiRange;        // Range of forest index (= uRange)

  //FIXME: (yang) only used in analyzeUDist(), change to local
//    private double globalUdR2;     // R2 of entire series in ud B5

  //FIXME: (yang) used in analyzeUDist() to set its value, but does not seem to be used in any real logic, consider remove!!
//    private double globalUdSlp;    // Slope of entire series in ud B5

  // Attributes associated with maximum forest length segment - used
  // in thresholding
  private int maxConsFor;        // Maximum consecutive forest length
  private double[] meanForUdBx;  // Mean of index values across maxConsFor
  private double[] sdForUdBx;    // SD of index values across maxConsFor

  // Chararacterization of segments based on thresholded values
  private int[] cstSeg;          // Segment characterization
  private int[] cstSegSmooth;    // Smoothed characterization

  //FIXME: (yang) this variable is only referenced in analyzeUDist() and was not used for anything, consider remove!!
//    private int hudSeg;            // Number of high ud segments


  //FIXME: (yang) this seem is only used in analyzeUDist() and should be a local variable.
  //private int ludSeg;            // Number of low ud segments
  //private int sharpTurns;        // Number of sharp turns between segments

  // Disturbance variables
  private int numDist;           // Number of disturbances detected

  //FIXME: (yang) only used in analyzeUDist() and not referenced in change detection, consider remove!!
//    private int numMajorDist;      // Number of major disturbances detected

  //FIXME: (yang) this is only referenced in setDisturbanceVariables() and not used anywhere, consider remove it
//    private int longestDistLength; // Length of the longest disturbance
//    private double longestDistR2;  // R2 of the longest disturbance
//    private double longestDistRough; // Roughness of the longest disturbance

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
  //FIXME: (yang) only used in calculateAgIndicator(), change to local
//    private int agIndicator;       // Agriculture indicator
  // </editor-fold>

  public VCT() {
    this.maxUd = 4.0;
    this.minNdvi = 0.45;
    allocateArrays();
  }

  public VCT(double maxUd, double minNdvi, int nYears) {
    this.maxUd = maxUd;
    this.minNdvi = minNdvi;
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
    // this.ud = new double[N_BANDS][MAX_YEARS];
  }


  /**
   * This is the main function for determining change analysis within VCT.
   * First, time series information is read from a number of different forest
   * indexes (known as forest z-scores in the paper, ud scores throughout the
   * code) and a mask image denoting land cover type and/or image artifacts
   * (e.g. cloud, shadow). The algorithm consists of two main functions and a
   * cleanup function:
   * <p/>
   * 1) interpolationAndIndices - The ud series for all indices is first
   * filled in for all cloud/shadow pixels from neighboring years.
   * <p/>
   * 2) analyzeUDist - The filled series are analyzed to find anomalous
   * deviations from neighboring years and characterized into disturbance and
   * regrowth segments
   * <p/>
   * 3) setDisturbanceVariables - Various cleanup and clamping of variable
   * values as well as determining disturbance duration.
   * <p/>
   * Note that much of the code relies of comparing series values with known
   * or derived thresholds of "forestness" and anomalies are detected when
   * these threshold bounds are exceeded.
   *
   * @param ud    - List of all UD values across all pertinent bands and indices
   *              (B3, B4, B5, B7, thermal, NDVI, DNBR, R45). There is one List per index
   *              which contains values for all years.
   * @param mask  - List of mask values for this pixel across all years
   * @param years - List of years corresponding to the indices in ud and mask
   * @return - Unspecified for now
   */
  public VCTOutput getResult(double[][] ud,
                             int[] mask, int[] years) {

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
   * //TODO: (yang) move the composite UD calcuation to VCT initialization call
   */
  private void initializePixel() {
    // Initialize all variables
    this.numDist = 0;
    this.lcType = 0;
    this.maxConsFor = 0;

    // Initialize arrays from input lists
    //
    // TODO: need better way to do this -- too fragile
    // This assumes that the input data is stored by bands and then years
    // with the following bands (or indexes): B3, B4, B5, B7, BT (thermal),
    // NDVI, DNBR.
//    for (int i = 0; i < this.numYears; i++) {
//      for (int j = 0; j < N_BANDS - 1; j++) {
//        this.ud[j][i] = ud.get(j).get(i);
//      }
//      this.mask[i] = mask.get(i);
//      this.yearTable[i] = years.get(i);
//    }
//
//    // Calculate the composite UD variable
//    for (int i = 0; i < this.numYears; i++) {
//      double sumSq = 0.0;
//      for (Integer index : UD_INDEXES) {
//        double tmp = this.ud[index][i];
//        tmp = tmp >= 0.0 ? tmp : tmp / 2.5;
//        sumSq += tmp * tmp;
//      }
//      this.ud[COMP][i] = Math.sqrt(sumSq / UD_INDEXES.size());
//    }

//      this.pctGoodObs = 0;
//      this.longestDistLength = 0;
//      this.longestDistR2 = 0.0;
//      this.longestDistRough = 0.0;
//      this.globalUdR2 = 0.0;
//      this.globalUdSlp = 0.0;

//      this.numMajorDist = 0;
//    this.uRange = 0.0;
//      this.vRange = 0.0;
//      this.fiRough = 0.0;

    //FIXME: (yang) do we need both uRange and fiRange, they seem to be the same in the code.
    this.fiRange = 0.0;
//      this.hudSeg = 0;
//      this.ludSeg = 0;
//      this.agIndicator = 0;
//      this.sharpTurns = 0;

    //TODO: (yang) initialize variables, check for necessity!!
    //which one can be local, not as class variables?
    Arrays.fill(this.qFlag, 0);
    Arrays.fill(this.cstSeg, 0);
    Arrays.fill(this.cstSegSmooth, 0);
    Arrays.fill(this.distFlag, 0);
    Arrays.fill(this.distYear, 0);
    Arrays.fill(this.distLength, 0);
    Arrays.fill(this.regrFlag, 0);

    Arrays.fill(this.distR2, 0);
    Arrays.fill(this.distMagn, 0);
    Arrays.fill(this.distMagnB4, 0);
    Arrays.fill(this.distMagnVi, 0);
    Arrays.fill(this.distMagnBr, 0);
    Arrays.fill(this.regrR2, 0);
    Arrays.fill(this.regrSlope, 0);
    Arrays.fill(this.regrRough, 0);

    Arrays.fill(this.meanForUdBx, 0);
    Arrays.fill(this.sdForUdBx, 0);
  }

  /**
   * Classify the pixel's mask values into QA_GOOD and QA_BAD and fill in
   * QA_BAD values based on nearest neighbors or linear interpolation of
   * QA_GOOD values. Implements section 3.3.1 in Huang et al. (2010) paper
   */
  private void interpolationAndIndices() {
    //YANG: changed from instance variable to local variable
    int pctGoodObs = 0;
    int badCount = 0;
    // Flag each year's pixel as good or bad based on the mask value
    // TODO: Note that class 0 (BACKGROUND) is *not* being flagged as
    // "fillable" in the original source code (ie. SLC-off errors).
    // We need to ask Cheng about this

    //YANG: refactory of the code logic to speed computation.
    int tmp = this.mask[0];
    if (tmp != 0 && tmp <= FILL_CLASSES) {
      this.qFlag[0] = QA_BAD;
      badCount++;
    } else {
      this.qFlag[0] = QA_GOOD;
    }

    tmp = this.mask[this.numYears - 1];
    if (tmp != 0 && tmp <= FILL_CLASSES) {
      this.qFlag[this.numYears - 1] = QA_BAD;
      badCount++;
    } else {
      this.qFlag[this.numYears - 1] = QA_GOOD;
    }

    // Look for spikes and dips that may be unflagged cloud and shadow and
    // set the QA flag accordingly
    for (int i = 1; i < this.numYears - 1; i++) {
      if (this.mask[i] != 0 && this.mask[i] <= FILL_CLASSES) {
        this.qFlag[i] = QA_BAD;
        badCount++;

        // Skip over pixels already labeled as QA_BAD
        continue;
      } else {
        this.qFlag[i] = QA_GOOD;
      }

      // Pixels higher in UD than neighboring years - cloud
      // Pixels lower than UD than neighboring years - shadow
      if (isRelativeCloud(i) || isRelativeShadow(i)) {
        this.qFlag[i] = QA_BAD;
        badCount++;
      }
    }

    // Now test the start/end conditions
    int n = this.numYears;
    if (this.qFlag[0] == QA_GOOD && isBadEndpoint(0, 1)) {
      this.qFlag[0] = QA_BAD;
      badCount++;
    }
    if (this.qFlag[n - 1] == QA_GOOD && isBadEndpoint(n - 1, n - 2)) {
      this.qFlag[n - 1] = QA_BAD;
      badCount++;
    }

    // Calculate bad percentage
    pctGoodObs = (int) (100.0 - ((100.0 * badCount) / this.numYears));

    // Interpolate for bad observations indicated by qFlag
    //TODO: (yang) is there any special treatment when percentGood is less than 50%?
    int i = 0;
    if (pctGoodObs > 50.0) {
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
            fillValues(0, next, next);
            i = next + 1;
          }
          // No acceptable next QA_GOOD - use prev index to fill
          // all years from prev + 1 to num_years
          else if (next >= this.numYears) {
            fillValues(prev+1, this.numYears, prev);
            i = next + 1;
          }
          // Found years acceptable for interpolation - fill between
          // prev and next
          else {
            for (int k = 0; k < N_BANDS; k++) {
              interpolateValues(this.ud[k], prev, next);
            }
            i = next + 1;
          }
        }
      }
    }

    // Get range values for the composite UD and NDVI
    this.fiRange = Doubles.max(this.ud[COMP]) - Doubles.min(this.ud[COMP]);
//      this.vRange = Doubles.max(this.ud[NDVI]) - Doubles.min(this.ud[NDVI]);

//    this.fiRange = this.uRange;
//      this.fiRough = fiRoughness(this.ud[COMP], 0, this.numYears);

    // TODO: This is a restriction to ensure that when scaled by 10.0, it
    // doesn't exceed byte length - unneeded?
//      this.fiRough = Math.min(this.fiRough, 25.4);
  }

  /**
   * A internal help method, not a good design
   * @param start start index to fill
   * @param end end index to fill
   * @param fill index of fill value
   */
  private void fillValues(int start, int end, int fill) {
    for (int k = 0; k < N_BANDS; k++) {
      Arrays.fill(this.ud[k], 0, end-start, this.ud[k][fill]);
    }
  }


  /**
   * Main function for determining disturbance and regrowth for this pixel's
   * trajectory. Main steps include:
   * <p/>
   * 1) Determine composite and NDVI thresholds
   * 2) Find the longest consecutive forest streak
   * 3) Find consecutive segments as either high or low UD values based on thresholds
   * 4) Smooth these segments
   * 5) Characterize segments as disturbance/regrowth classes
   * 6) Determine land cover/change type based on pattern of disturbances
   */
  private void analyzeUDist() {
    //YANG: changed from class varialbe to local
    int sharpTurns = 0;

    //FIXME: (yang) not used variables, consider remove it!!
//    int ludSeg = 0;
//    double globalUdR2 = 0.0;

    // Assign some default information to this pixel
    //TODO: (yang) why set distYear as the number of years?
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
    //
//    double maxVi = Doubles.max(this.ud[NDVI]);
//    double minUd = 9999.0;
//    double min2Ud = 9999.0;
//
//    for (int i = 0; i < this.numYears; i++) {
//      double cmp = this.ud[COMP][i];
//      if (cmp < minUd) {
//        minUd = cmp;
//      }
//      if (cmp > minUd && cmp < min2Ud) {
//        min2Ud = cmp;
//      }
//    }
//    if (min2Ud - minUd < FOR_THR_MAX) {
//      min2Ud = minUd;
//    }

    //Yang incorporate Matt's new logic and consolidate calculation
    //keep track of bottom two ud composite values
    double maxVi = Double.NEGATIVE_INFINITY; //Doubles.max(this.ud[NDVI]);
    double minUd = Double.NEGATIVE_INFINITY; //Doubles.min(this.ud[COMP]);
    double min2Ud = Double.POSITIVE_INFINITY; //Doubles.max(this.ud[COMP]);

    // Track the number of water and shadow pixels and identify if they
    // come in the first and last thirds of the time series
    int isWaterFront = 0;
    int isWaterTail = 0;
    int firstThird = this.numYears / 3;
    int lastThird = this.numYears - firstThird;
    double percentWater = 0.0;
    double percentShadow = 0.0;
    for (int i = 0; i < this.numYears; i++) {
      double tmp = this.ud[COMP][i];
      minUd = tmp < minUd ? tmp : minUd;
      min2Ud = (tmp > minUd && tmp < min2Ud) ? tmp : min2Ud;

      maxVi = this.ud[NDVI][i] > maxVi ? this.ud[NDVI][i] : maxVi;

      if (this.mask[i] == WATER || this.mask[i] == SHADOW) {
        percentWater += 1.0;
        if (this.mask[i] == SHADOW) {
          percentShadow += 1.0;
        }
        isWaterFront += i < firstThird ? 1 : 0;
        isWaterTail += i >= lastThird ? 1 : 0;
      }
    }
    percentWater /= this.numYears;
    percentShadow /= this.numYears;

    if (min2Ud - minUd < FOR_THR_MAX) {
      min2Ud = minUd;
    }


    // Get the maximum streak of years with forest.  The length of the
    // streak gets set in the function (this.maxConsFor) along with
    // the mean and standard deviations of the longest streak in each
    // index.  The value returned is the starting year of the streak.
    int maxForStart = getMaxForLength(min2Ud, min2Ud + FOR_THR_MAX);

    // Set a threshold for determining change in the UD composite signal
    double changeHike = FOR_THR_MAX;
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
//        ludSeg += 1;
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
//          this.hudSeg += 1;
        sharpTurns += 1;
      }
      i = j;
    }

    // Calculate fast greenup indicator of nonforest
    // TODO: Right now we are not doing anything with this information
    // comment out until we need it
    // calculateAgIndicator();

    // Remove NCLUD and NCHUD labels based on adjacent labels - this
    // effectively smooths the segment labels
    System.arraycopy(this.cstSeg, 0, this.cstSegSmooth, 0, this.cstSeg.length);

    smoothSegment(this.cstSegSmooth, this.numYears, NCLUD);
    smoothSegment(this.cstSegSmooth, this.numYears, NCHUD);

    // Create an empty list of TSSegment instances to store segment
    // information.  This block uses the smoothed segment information
    // to create the segments
    i = 0;
    List<TSSegment> tsSeg = new ArrayList<TSSegment>();
    while (i < this.numYears) {

      // Initialize this segment
      int j = i;

      // As long as the label for this year matches the following year's
      // label, keep growing the segment
      while (j < this.numYears - 1 && this.cstSegSmooth[j] == this.cstSegSmooth[j + 1]) {
        j++;
      }

      // Store this segment
      tsSeg.add(new TSSegment(this.cstSegSmooth[i], i, j-i+1));

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
//            this.numMajorDist += 1;

          // Characterize the disturbance and following recovery
          setDisturbance(thisSeg.startYear, thisSeg.endYear);

          // More convenient to set regrowth type here
          // dist_num is decremented due to increment in set_disturbance
          //FIXME: (yang) not good practice.
          int distNum = this.numDist - 1;
          this.regrFlag[distNum] = REGROWTH_NOT_OCCURRED;

          // Not the last segment in this time series and is followed by
          // a forested segment
          if (i < tsSeg.size() - 1) {
            if (tsSeg.get(i + 1).segType == CLUD
                || tsSeg.get(i + 1).segType == NCLUD) {
              this.regrFlag[distNum] = REGROWTH_TO_FOREST;
            } else {
              //TODO: (yang) when this happens, what is the right behavior?
              // Handle exception
              // String msg = "Warning: CHUD not followed by CLUD or NCLUD";
              // throw new Exception(msg);
            }
          }
          // Last segment in this time series, but high R2 and
          // negative slope indicate that regrowth is occurring
          else if (this.regrR2[distNum] > 0.7
              && this.regrSlope[distNum] < -0.2) {
            this.regrFlag[distNum] = REGROWTH_OCCURRED;
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

    // Run global linear fits across the time series, but leave off
    // portions of tails (three at both beginning and end) and
    // capture the best fit
//    double[] coeff;
//    for (i = 0; i < 3; i++) {
//      coeff = calculateLinearChangeRate(i, this.numYears - 1);
//      if (coeff[2] > globalUdR2) {
//        globalUdR2 = coeff[2];
//          this.globalUdSlp = coeff[0];
//      }
//    }

//    for (i = this.numYears - 3; i < numYears; i++) {
//      coeff = calculateLinearChangeRate(0, i);
//      if (coeff[2] > globalUdR2) {
//        globalUdR2 = coeff[2];
//          this.globalUdSlp = coeff[0];
//      }
//    }

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

//    if (this.uRange > 25.4) {
//      this.uRange = 25.4;
//    }

    for (int i = 0; i < this.numYears; i++) {
      if (this.distMagn[i] > 25.0) {
        this.distMagn[i] = 25.0;
      }
    }

    if (this.lcType == PART_FOREST) {
      //FIXME: (yang) why numDist == 0 is a problem?
      if (this.numDist < 1 || this.numDist >= this.numYears) {
        // Handle exception
        // String msg = "Error: Wrong number of disturbances";
        // throw new Exception(msg);
      }

      // Find the disturbance with longest duration
      int dLength = 0;
      int longestDistIdx = 0;
      for (int i = 0; i < this.numDist; i++) {
        if (this.distLength[i] > dLength) {
          longestDistIdx = i;
          dLength = this.distLength[i];
        }
      }
//        this.longestDistR2 = this.distR2[longestDistIdx];
//        this.longestDistRough = this.regrRough[longestDistIdx];
//        this.longestDistLength = dLength;

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

//      this.globalUdSlp = Math.max(Math.min(this.globalUdSlp, 1.0), -1.0);
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


    //TODO: (yang) verify this single for-loop to replace the double while loop,
//    int currentWaterCount = 0;
//    int currentLength = 0;
//    for (int i = 0; i < this.numYears; i++) {
//      double tmp = this.ud[COMP][i];
//      if (tmp <= maxForUd && tmp >= minForUd) {
//        currentLength++;
//        if (this.mask[i] == WATER) {
//          currentWaterCount++;
//        }
//      }
//      else {
//        if (maxLength < currentLength) {
//          maxLength = currentLength;
//          waterCount = currentWaterCount;
//          iEnd = i - 1;
//        }
//        currentWaterCount = 0;
//        currentLength = 0;
//        iStart = i;
//      }
//    }
//    //if the last segment is the longest
//    if (maxLength < currentLength) {
//      maxLength = currentLength;
//      waterCount = currentWaterCount;
//      iEnd = this.numYears - 1;
//    }

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
      if (yearCount > 0) {
        i += yearCount;
      } else {
        i++;
      }
    }

    // Initialize the container to hold means and standard deviations of
    // the maximum forest streak
    for (i = 0; i < N_BANDS; i++) {
      this.meanForUdBx[i] = 25.4;
      this.sdForUdBx[i] = 25.4;
    }

    // If there is a forest streak, calculate statistics on this streak
    if (maxLength > 0) {
      Mean m = new Mean();
      for (int j = 0; j < N_BANDS; j++) {
        this.meanForUdBx[j] = m.evaluate(this.ud[j], iStart, iEnd-iStart);
      }

      if (maxLength > 1) {
        StandardDeviation sd = new StandardDeviation();
        for (int j = 0; j < N_BANDS; j++) {
          this.sdForUdBx[j] = sd.evaluate(this.ud[j], iStart, iEnd - iStart);
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
   * Determine whether or not a pixel is indicative of agriculture - high
   * yearly variability in signal
   */
  private void calculateAgIndicator() {
    //YANG: chanded from instance variable to local variable
    int agIndicator = 0;


    double suddenChange = 3.5;

    for (int i = 1; i < this.numYears - 1; i++) {
      // Create local variables for comparison
      double curComp = this.ud[COMP][i];
      double prevComp = this.ud[COMP][i - 1];
      double nextComp = this.ud[COMP][i + 1];
      double curB4 = this.ud[B4][i];
      double prevB4 = this.ud[B4][i - 1];
      double curB5 = this.ud[B5][i];
      double prevB5 = this.ud[B5][i - 1];

      // Consective low UD (CLUD) vertices
      // If there is a sudden change in UD values in either B4, B5 or the
      // composite, calculate the number of drops (defined as the
      // magnitude of the change divided by the change threshold) in each
      // of B4, B5 and COMP.  Take the maximum of these drops, square
      // this value, and add to this.agIndicator
      if (this.cstSeg[i] == CLUD
          && (prevComp - curComp > suddenChange
          || prevB4 - curB4 > suddenChange
          || prevB5 - curB5 > suddenChange)) {
        int udAllDrop = (int) ((prevComp - curComp) / suddenChange);
        int udB5Drop = (int) ((prevB5 - curB5) / suddenChange);
        int udB4Hike = (int) ((curB4 - prevB4) / suddenChange);
        int udDropNum = udAllDrop > udB5Drop ? udAllDrop : udB5Drop;
        udDropNum = udDropNum < udB4Hike ? udB4Hike : udDropNum;
        udDropNum = Math.min(udDropNum * udDropNum, 16);
        agIndicator += udDropNum;
      }

      // Non-consective low and high UD vertices
      // If there is a sudden change in UD values in udist composite between
      // this vertex and its neighboring years, calculate the number of
      // drops (or hikes) as before.  Square this value and add to
      // this.agIndicator
      if (this.cstSeg[i] == NCLUD
          && prevComp - curComp > suddenChange
          && nextComp - curComp > suddenChange) {
        double avg = (prevComp + nextComp) / 2.0;
        int udDip = (int) ((avg - curComp) / suddenChange);
        udDip = Math.min(udDip * udDip, 16);
        agIndicator += udDip;
      }

      if (this.cstSeg[i] == NCHUD
          && curComp - prevComp > suddenChange
          && curComp - nextComp > suddenChange) {
        double avg = (prevComp + nextComp) / 2.0;
        int udHike = (int) ((curComp - avg) / suddenChange);
        udHike = Math.min(udHike * udHike, 16);
        agIndicator += udHike;
      }
    }
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
    // Set the disturbance year, disturbance flag and landcover type
    this.distYear[this.numDist] = startYear;
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

    // Calculate the change magnitudes based on the peak year
    calculateChangeMagn(startYear, truePeak);

    // Fit a recovery (regrowth) regression line (B5 vs. years) from the
    // peak year to the end year. High goodness of fit value indicates
    // a very likely change
//    double[] coeff;
    int distIndex = this.numDist;

    if (endYear - startYear < 4 || this.fiRange < 0.1) {
      this.regrR2[distIndex] = FALSE_FIT[2];
      this.regrSlope[distIndex] = FALSE_FIT[0];
      this.distR2[distIndex] = FALSE_FIT[2];
    }
    else {
      // Yang replace regression code
      // Note: if common math3 can be used, this code can be simplified.
      SimpleRegression sr = new SimpleRegression();
      sr.addData(extractSegmentValues(truePeak, endYear));
      this.regrR2[distIndex] = sr.getRSquare();
      this.regrSlope[distIndex] = sr.getSlope();

      // Fit a regression for the entire disturbance period
      sr.clear();
      sr.addData(extractSegmentValues(startYear, endYear));
      this.distR2[distIndex] = sr.getRSquare();
    }

    //TODO: (yang) check implementation
    this.regrRough[distIndex] = fiRoughness(this.ud[COMP], startYear, endYear);
    this.distLength[distIndex] = (int) (endYear - startYear + 1);

    // Increment the disturbance index
    this.numDist += 1;
  }

  /**
   * extract segment data in ud B5 over given time span for linear fit
   *
   * This is based on the logic in calculateLinearChangeRate.
   *
   * @param start start of segment inclusive
   * @param end end of segment inclusive
   * @return
   */
  private double[][] extractSegmentValues(int start, int end) {
    //FIXME: (yang) when will this happen?
    //should make sure endYear always have valid data where it is assigned
    if (end == this.numYears) {
      end--;
    }

    double[][] result = new double[end-start+1][2];
    for (int i = start; i <= end; i++) {
      result[i-start][0] = this.yearTable[i] - this.yearTable[start];
      result[i-start][1] = this.ud[B5][i];
    }
    return result;
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
   * Calculate the change magnitude (in many indexes) for a given disturbance
   * segment
   *
   * @param changeYear - year of initial disturbance onset
   * @param peakYear   - year of peak disturbance
   */
  private void calculateChangeMagn(int changeYear, int peakYear) {
    // Handle exception
    // if (changeYear < 0 || changeYear >= this.numYears
    //         || peakYear < 0 || peakYear >= this.numYears) {
    //   String msg = "Year out of range in calculateChangeMagn";
    //   throw new Exception(msg);
    // }

    // Capture the disturbance magnitudes in UD, NDVI and DNBR spaces.
    // Magnitudes are relative from the peak UD year to the mean of
    // the pixel across all years
    this.distMagn[changeYear]
        = this.ud[COMP][peakYear] - this.meanForUdBx[COMP];
    this.distMagnB4[changeYear]
        = this.ud[B4][peakYear] - this.meanForUdBx[B4];
    this.distMagnVi[changeYear]
        = this.ud[NDVI][peakYear] - this.meanForUdBx[NDVI];
    this.distMagnBr[changeYear]
        = this.ud[DNBR][peakYear] - this.meanForUdBx[DNBR];
  }

  /**
   * Calculate a linear change rate in ud B5 over a given time span using a
   * simple linear fit
   *
   * @param startYear - start year
   * @param endYear   - end year
   * @return - Regression coefficients from least squares fit
   */
//  private double[] calculateLinearChangeRate(int startYear, int endYear) {
//    // Too few observations for meaningful regression, or not much
//    // variability. Return default values
//    if (endYear - startYear < 4 || this.fiRange < 0.1) {
//      return FALSE_FIT;
//    }
//
//    // Index errors on start_year, end_year
//    // if (endYear > this.numYears || startYear < 0) {
//    //   // Handle exception
//    //   String msg = "Error: index error in calculateLinearChangeRate";
//    //   throw new Exception(msg);
//    // }
//    //FIXME: (yang) when will this happen?
//    if (endYear == this.numYears) {
//      endYear--;
//    }
//
//    // Linear regression of B5 (Y) on years since disturbance (X)
//    // TODO: Reimplement using LinearLeastSquares - for now using
//    // Cheng's code for consistency
//    //
//    // TODO: Generalize this function for any index
//    double[] x = new double[this.numYears];
//    double[] y = new double[this.numYears];
//    int numObs = endYear - startYear + 1;
//    for (int i = startYear; i <= endYear; i++) {
//      int j = i - startYear;
//      y[j] = this.ud[B5][i];
//      x[j] = this.yearTable[i] - this.yearTable[startYear];
//    }
//    return linearFit(y, x, numObs);
//  }

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

    public final List<Integer> years;
    public final List<Integer> distFlag;
    public final List<Double> distMagn;
    public final List<Double> distMagnVi;
    public final List<Double> distMagnBr;
    public final List<Double> distMagnB4;

    public VCTOutput(int lcType,
                     int nYears,
                     int[] years,
                     int[] distFlag,
                     double[] distMagn,
                     double[] distMagnVi,
                     double[] distMagnBr,
                     double[] distMagnB4) {
      this.years = Ints.asList(Arrays.copyOfRange(years, 0, nYears));
      if (lcType != PART_FOREST) {
        this.distFlag = new ArrayList<>(Collections.nCopies(nYears, lcType));
        this.distMagn = new ArrayList<>(Collections.nCopies(nYears, 0.0));
        this.distMagnVi = new ArrayList<>(Collections.nCopies(nYears, 0.0));
        this.distMagnBr = new ArrayList<>(Collections.nCopies(nYears, 0.0));
        this.distMagnB4 = new ArrayList<>(Collections.nCopies(nYears, 0.0));
      } else {
        this.distFlag = Ints.asList(Arrays.copyOfRange(distFlag, 0, nYears));
        this.distMagn = Doubles.asList(Arrays.copyOfRange(distMagn, 0, nYears));
        this.distMagnVi = Doubles.asList(Arrays.copyOfRange(distMagnVi, 0, nYears));
        this.distMagnBr = Doubles.asList(Arrays.copyOfRange(distMagnBr, 0, nYears));
        this.distMagnB4 = Doubles.asList(Arrays.copyOfRange(distMagnB4, 0, nYears));
      }
    }
  }
  //}

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
  private final double fiRoughness(double[] ts, int left, int right) {
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
   * //FIXME: (yang) replace with common math, consider delete this
   *
   * @param arr   - Array to calculate mean
   * @param start - start index of array
   * @param end   - end index of array (not included!)
   * @return - mean value of slice
   */
//  private double getSliceMean(double[] arr, int start, int end) {
//    Mean m = new Mean();
//    return m.evaluate(arr, start, end-start);
//  }

  /**
   * Get a standard deviation value over a slice of an array
   *
   * //FIXME: (yang) replace with common math, consider delete this
   *
   * @param arr   - Array to calculate standard deviation
   * @param start - start index of array
   * @param end   - end index of array (not included!)
   * @return - standard deviation value of slice
   */
//  private double getSliceStd(double[] arr, int start, int end) {
//    StandardDeviation sd = new StandardDeviation();
//    return sd.evaluate(arr, start, end-start);
//  }

  /**
   * Least-squares linear fit of Y on X
   *
   * @param y      - array of dependent values
   * @param x      - array of independent values
   * @param numObs - length of array
   * @return - slope, intercept, r2 and t value of least-squares linear regr.
   */
//  private double[] linearFit(double[] y, double[] x, int numObs) {
//    // TODO: Probably should be replaced with generic linear fitting algorithm
//    // but kept in here for comparison purposes
//
//    double sxx = 0.0;
//    double sxy = 0.0;
//    double syy = 0.0;
//
//    Mean m = new Mean();
//    double meanX = m.evaluate(x, 0, numObs);
//    double meanY = m.evaluate(y, 0, numObs);
//    for (int i = 0; i < numObs; i++) {
//      sxx += (x[i] - meanX) * (x[i] - meanX);
//      syy += (y[i] - meanY) * (y[i] - meanY);
//      sxy += (y[i] - meanY) * (x[i] - meanX);
//    }
//
//    if (sxx < 0.00001) {
//      // Handle exception
//      // String msg = "Error: divided by 0 in linear fit\n";
//      // throw new Exception(msg);
//    }
//
//    double slope = sxy / sxx;
//    double intercept = meanY - meanX * slope;
//    double r2;
//    if (syy < 0.00001) {
//      r2 = 0.0;
//    } else {
//      r2 = (sxy * sxy) / (sxx * syy);
//    }
//    double denom = r2 == 1.0 ? 0.00001 : 1.0 - r2;
//
//    // TODO: What is t?
//    double t = Math.sqrt(r2 * (numObs - 2) / denom);
//
//    return new double[]{slope, intercept, r2, t};
//  }
}
